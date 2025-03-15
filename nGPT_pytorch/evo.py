from __future__ import annotations
from copy import deepcopy

import torch
from torch import cat, randperm, Tensor

from einops import rearrange
from nGPT_pytorch.nGPT import NormLinear, FeedForward, Attention

# helper functions

def exists(v):
    return v is not None

# cross over normlinear

@torch.no_grad()
def cross_over_linear(
    parent1: NormLinear,
    parent2: NormLinear,
    parent1_indices: Tensor,
    parent2_indices: Tensor,
    child: NormLinear | None = None,
    dim: int = 0
) -> NormLinear:

    if not exists(child):
        child = deepcopy(parent1)

    assert dim in {0, 1}
    assert parent1 == parent2

    w1 = parent1.weight
    w2 = parent2.weight

    if dim == 0:
        crossover_weight = cat((w1[parent1_indices], w2[parent2_indices]), dim = 0)
    else:
        crossover_weight = cat((w1[:, parent1_indices], w2[:, parent2_indices]), dim = 1)

    child.weight.copy_(crossover_weight)
    return child

# breeding feedforwards

@torch.no_grad()
def cross_over_feedforward(
    parent1: FeedForward,
    parent2: FeedForward,
    child: FeedForward | None = None
) -> FeedForward:
    assert parent1 == parent2

    if not exists(child):
        child = deepcopy(parent1)

    dim_inner = parent1.dim_inner

    parent1_w1 = parent1.to_hidden.weight
    parent2_w1 = parent2.to_hidden.weight
    child_w1 = child.to_hidden.weight

    parent1_gate = parent1.to_gate.weight
    parent2_gate = parent2.to_gate.weight
    child_gate = child.to_gate.weight

    parent1_h_scale = parent1.hidden_scale.scale
    parent2_h_scale = parent2.hidden_scale.scale
    child_h_scale = child.hidden_scale.scale

    parent1_g_scale = parent1.gate_scale.scale
    parent2_g_scale = parent2.gate_scale.scale
    child_g_scale = child.gate_scale.scale

    parent1_w2 = parent1.to_out.weight
    parent2_w2 = parent2.to_out.weight
    child_w2 = child.to_out.weight

    midpoint = dim_inner // 2
    rand_indices = randperm(dim_inner)

    # randomly select vectors from the feedforward weight matrices from both parents to constitute the child

    parent1_indices, parent2_indices = rand_indices[:midpoint], rand_indices[midpoint:]

    child_w1.copy_(cat((parent1_w1[parent1_indices], parent2_w1[parent2_indices]), dim = 0))
    child_gate.copy_(cat((parent1_gate[parent1_indices], parent2_gate[parent2_indices]), dim = 0))
    child_w2.copy_(cat((parent1_w2[:, parent1_indices], parent2_w2[:, parent2_indices]), dim = 1))

    child_h_scale.copy_(cat((parent1_h_scale[parent1_indices], parent2_h_scale[parent2_indices])))
    child_g_scale.copy_(cat((parent1_g_scale[parent1_indices], parent2_g_scale[parent2_indices])))

    return child

# breed attention

@torch.no_grad()
def cross_over_attention(
    parent1: Attention,
    parent2: Attention,
    child: Attention | None = None
) -> Attention:

    assert parent1 == parent2

    heads = parent1.heads
    assert heads > 1

    if not exists(child):
        child = deepcopy(parent1)

    split_heads_first_dim = lambda t: rearrange(t, '(h d) ... -> h d ...', h = heads)
    split_heads_last_dim = lambda t: rearrange(t, 'e (h d) -> e h d', h = heads)

    flatten_first = lambda t: rearrange(t, 'h d ... -> (h d) ...')
    flatten_last = lambda t: rearrange(t, 'e h d -> e (h d)')

    parent1_q = split_heads_first_dim(parent1.to_q.weight)
    parent2_q = split_heads_first_dim(parent2.to_q.weight)
    child_q = child.to_q.weight

    parent1_k = split_heads_first_dim(parent1.to_k.weight)
    parent2_k = split_heads_first_dim(parent2.to_k.weight)
    child_k = child.to_k.weight

    parent1_v = split_heads_first_dim(parent1.to_v.weight)
    parent2_v = split_heads_first_dim(parent2.to_v.weight)
    child_v = child.to_v.weight

    parent1_o = split_heads_last_dim(parent1.to_out.weight)
    parent2_o = split_heads_last_dim(parent2.to_out.weight)
    child_o = child.to_out.weight

    parent1_qk_scale = split_heads_first_dim(parent1.qk_scale.scale)
    parent2_qk_scale = split_heads_first_dim(parent2.qk_scale.scale)
    child_qk_scale = child.qk_scale.scale

    # randomly select heads from parents1 and parents2 for crossover

    midpoint = heads // 2
    rand_indices = randperm(heads)

    parent1_indices, parent2_indices = rand_indices[:midpoint], rand_indices[midpoint:]

    # select out the correct parameters for attention heads from parent 1 and 2

    child_q.copy_(flatten_first(cat((parent1_q[parent1_indices], parent2_q[parent2_indices]), dim = 0)))
    child_k.copy_(flatten_first(cat((parent1_k[parent1_indices], parent2_k[parent2_indices]), dim = 0)))
    child_v.copy_(flatten_first(cat((parent1_v[parent1_indices], parent2_v[parent2_indices]), dim = 0)))
    child_qk_scale.copy_(flatten_first(cat((parent1_qk_scale[parent1_indices], parent2_qk_scale[parent2_indices]), dim = 0)))

    child_o.copy_(flatten_last(cat((parent1_o[:, parent1_indices], parent2_o[:, parent2_indices]), dim = 1)))

    return child
