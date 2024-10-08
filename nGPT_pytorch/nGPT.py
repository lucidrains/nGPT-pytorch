from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

# for use with parametrize

LinearNoBias = partial(nn.Linear, bias = False)

class L2Norm(Module):
    def forward(self, t):
        return l2norm(t)

l2norm_weights = partial(
    parametrize.register_parametrization,
    tensor_name = 'weight',
    parametrization = L2Norm()
)

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
        self.query_weights = LinearNoBias(dim_inner, dim)
        self.key_weights = LinearNoBias(dim_inner, dim)
        self.value_weights = LinearNoBias(dim_inner, dim)

        self.out_weights = LinearNoBias(dim_inner, dim)

        l2norm_weights(self.query_weights)
        l2norm_weights(self.key_weights)
        l2norm_weights(self.value_weights)
        l2norm_weights(self.out_weights)

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
        self.proj_in_weights = LinearNoBias(dim_inner, dim)
        self.gate_weights = LinearNoBias(dim_inner, dim)
        self.proj_out_weights = LinearNoBias(dim_inner, dim)

        l2norm_weights(self.proj_in_weights)
        l2norm_weights(self.gate_weights)
        l2norm_weights(self.proj_out_weights)

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
        ff_expand_factor = 4.
    ):
        super().__init__()

        self.token_embed = LinearNoBias(dim, num_tokens)
        self.token_unembed = LinearNoBias(dim, num_tokens)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads),
                FeedForward(dim, expand_factor = ff_expand_factor),
            ]))

        l2norm_weights(self.token_embed)
        l2norm_weights(self.token_unembed)

    def forward(
        self,
        ids,
        return_loss = False
    ):
        tokens = self.token_embed.weight[ids]

        return tokens
