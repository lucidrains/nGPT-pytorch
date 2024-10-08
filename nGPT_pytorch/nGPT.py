import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class nGPT(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()

    def forward(
        self,
        x
    ):
        return x
