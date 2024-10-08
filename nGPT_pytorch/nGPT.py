from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from einops import einsum, rearrange

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

# for use with parametrize

class L2Norm(Module):
    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return l2norm(t, dim = self.dim)

class NormLinear(Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm_dim = -1
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias = False)

        parametrize.register_parametrization(
            self.linear,
            'weight',
            L2Norm(dim = norm_dim)
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
        heads = 8
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.query_weights = NormLinear(dim, dim_inner)
        self.key_weights = NormLinear(dim, dim_inner)
        self.value_weights = NormLinear(dim, dim_inner)

        self.out_weights = NormLinear(dim_inner, dim, norm_dim = -1)

    def forward(
        self,
        x
    ):
        return x

class FeedForward(Module):
    def __init__(
        self,
        dim,
        *,
        expand_factor = 4
    ):
        super().__init__()
        dim_inner = int(dim * expand_factor * 2 / 3)
        self.proj_in_weights = NormLinear(dim, dim_inner)
        self.gate_weights = NormLinear(dim, dim_inner)
        self.proj_out_weights = NormLinear(dim_inner, dim)

    def forward(self, x):
        return x

# classes

class nGPT(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_expand_factor = 4.,
        ce_ignore_index = -1
    ):
        super().__init__()

        self.token_embed = NormLinear(dim, num_tokens)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads),
                FeedForward(dim, expand_factor = ff_expand_factor),
            ]))

        self.token_unembed = NormLinear(dim, num_tokens, norm_dim = 0)
        self.ignore_index = ce_ignore_index

    def forward(
        self,
        ids,
        return_loss = False
    ):

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        tokens = self.token_embed.weight[ids]

        logits = self.token_unembed(tokens)

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss
