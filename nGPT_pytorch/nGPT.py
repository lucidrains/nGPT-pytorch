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

        self.token_emb = LinearNoBias(dim, num_tokens)
        l2norm_weights(self.token_emb)

    def forward(
        self,
        ids,
        return_loss = False
    ):
        tokens = self.token_emb.weight[ids]

        return tokens
