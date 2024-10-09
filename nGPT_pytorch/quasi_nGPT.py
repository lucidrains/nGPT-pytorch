from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from einops import rearrange
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(
    t,
    dim = -1,
    norm_eps = 0.05,  # allow vectors to inhabit a small distance below and above the hypersphere if greater than 0.
    eps = 1e-10
):
    if norm_eps == 0.:
        return F.normalize(t, dim = dim, p = 2, eps = eps)

    norm = t.norm(dim = dim, keepdim = True)
    target_norm = norm.detach().clamp(min = 1. - norm_eps, max = 1. + norm_eps)
    divisor = norm / target_norm
    return t / divisor.clamp(min = eps)

# for use with parametrize

class L2Norm(Module):
    def __init__(self, dim = -1, norm_eps = 0.):
        super().__init__()
        self.dim = dim
        self.l2norm = partial(l2norm, norm_eps = norm_eps, dim = dim)

    def forward(self, t):
        return self.l2norm(t)

class NormLinear(Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm_dim_in = True,
        norm_eps = 0.
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias = False)

        parametrize.register_parametrization(
            self.linear,
            'weight',
            L2Norm(
                dim = -1 if norm_dim_in else 0,
                norm_eps = norm_eps
            )
        )

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x)

# attention and feedforward

class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        norm_qk = True,
        norm_eps = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.l2norm = partial(l2norm, norm_eps = norm_eps)

        self.to_q = NormLinear(dim, dim_inner, norm_eps = norm_eps)
        self.to_k = NormLinear(dim, dim_inner, norm_eps = norm_eps)
        self.to_v = NormLinear(dim, dim_inner, norm_eps = norm_eps)

        self.rotary_emb = RotaryEmbedding(dim_head)
        self.qk_scale = nn.Parameter(torch.ones(dim_head) * (dim_head ** 0.25))

        self.norm_qk = norm_qk
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in = False, norm_eps = norm_eps)

    def forward(
        self,
        x
    ):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(self.split_heads, (q, k, v))

        # maybe query key norm

        if self.norm_qk:
            q, k = map(self.l2norm, (q, k))

        # scaling queries and keys - this would line up with the popular use of qk rmsnorm from google deepmind and now black forest labs

        q, k = (q * self.qk_scale), (k * self.qk_scale)

        # rotary positions

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # scale is 1., as scaling factor is moved to s_qk (dk ^ 0.25) - eq. 16

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True,
            scale = 1.
        )

        out = self.merge_heads(out)
        return self.to_out(out)

class FeedForward(Module):
    def __init__(
        self,
        dim,
        *,
        expand_factor = 4,
        norm_eps = 0.
    ):
        super().__init__()
        self.dim = dim
        dim_inner = int(dim * expand_factor * 2 / 3)

        self.to_hidden = NormLinear(dim, dim_inner, norm_eps = norm_eps)
        self.to_gate = NormLinear(dim, dim_inner, norm_eps = norm_eps)

        self.hidden_scale = nn.Parameter(torch.ones(dim_inner))
        self.gate_scale = nn.Parameter(torch.ones(dim_inner))

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in = False, norm_eps = norm_eps)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)

        hidden = hidden * self.hidden_scale
        gate = gate * self.gate_scale * (self.dim ** 0.5)

        hidden = F.silu(gate) * hidden
        return self.to_out(hidden)

# classes

class QuasiNormalizedGPT(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_norm_qk = True,  # they say the query/key normalization is optional
        ff_expand_factor = 4.,
        ce_ignore_index = -1,
        residual_lerp_scale_init = None,
        norm_eps = 0.
    ):
        super().__init__()
        self.dim = dim
        self.l2norm = partial(l2norm, norm_eps = norm_eps)

        residual_lerp_scale_init = default(residual_lerp_scale_init, 1. / depth)

        self.token_embed = NormLinear(dim, num_tokens, norm_eps = norm_eps)

        self.layers = ModuleList([])
        self.residual_lerp_scales = nn.ParameterList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads, norm_qk = attn_norm_qk, norm_eps = norm_eps),
                FeedForward(dim, expand_factor = ff_expand_factor, norm_eps = norm_eps),
            ]))

            self.residual_lerp_scales.append(nn.ParameterList([
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init),
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init),
            ]))

        self.to_logits = NormLinear(dim, num_tokens, norm_eps = norm_eps)

        self.logit_scale = nn.Parameter(torch.ones(num_tokens))

        self.ignore_index = ce_ignore_index

    @torch.no_grad()
    def norm_weights_(self):
        for module in self.modules():
            if not isinstance(module, NormLinear):
                continue

            normed = module.weight
            original = module.linear.parametrizations.weight.original

            original.copy_(normed)

    def forward(
        self,
        ids,
        return_loss = False
    ):

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]        

        tokens = self.token_embed.weight[ids]

        for (attn, ff), (attn_alpha, ff_alpha) in zip(self.layers, self.residual_lerp_scales):

            attn_out = self.l2norm(attn(tokens))
            tokens = self.l2norm(tokens.lerp(attn_out, attn_alpha))

            ff_out = self.l2norm(ff(tokens))
            tokens = self.l2norm(tokens.lerp(ff_out, ff_alpha))

        logits = self.to_logits(tokens)
        logits = logits * self.logit_scale * (self.dim ** 0.5)

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss
