from __future__ import annotations
from copy import deepcopy

import torch
from torch import cat, randperm, Tensor

from einops import rearrange
from nGPT_pytorch.nGPT import NormLinear, Scale, FeedForward, Attention, nGPT

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

@torch.no_grad()
def cross_over_scale(
    parent1: Scale,
    parent2: Scale,
    parent1_indices: Tensor,
    parent2_indices: Tensor,
    child: Scale | None = None,
    dim: int = 0
) -> Scale:

    if not exists(child):
        child = deepcopy(parent1)

    assert dim in {0, 1}
    assert parent1 == parent2

    w1 = parent1.scale
    w2 = parent2.scale

    crossover_scale = cat((w1[parent1_indices], w2[parent2_indices]), dim = 0)

    child.scale.copy_(crossover_scale)
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

    midpoint = dim_inner // 2
    rand_indices = randperm(dim_inner)

    # randomly select vectors from the feedforward weight matrices from both parents to constitute the child

    parent1_indices, parent2_indices = rand_indices[:midpoint], rand_indices[midpoint:]

    cross_over_linear(parent1.to_hidden, parent2.to_hidden, parent1_indices, parent2_indices, child = child.to_hidden, dim = 0)

    cross_over_linear(parent1.to_gate, parent2.to_gate, parent1_indices, parent2_indices, child = child.to_gate, dim = 0)

    cross_over_linear(parent1.to_out, parent2.to_out, parent1_indices, parent2_indices, child = child.to_out, dim = 1)

    cross_over_scale(parent1.hidden_scale, parent2.hidden_scale, parent1_indices, parent2_indices, child = child.hidden_scale)

    cross_over_scale(parent1.gate_scale, parent2.gate_scale, parent1_indices, parent2_indices, child = child.gate_scale)

    return child

# breed attention

@torch.no_grad()
def cross_over_attention(
    parent1: Attention,
    parent2: Attention,
    child: Attention | None = None
) -> Attention:

    assert parent1 == parent2

    dim_inner = parent1.heads * parent1.dim_head

    if not exists(child):
        child = deepcopy(parent1)

    midpoint = dim_inner // 2
    rand_indices = randperm(dim_inner)

    parent1_indices, parent2_indices = rand_indices[:midpoint], rand_indices[midpoint:]

    cross_over_linear(parent1.to_q, parent2.to_q, parent1_indices, parent2_indices, child = child.to_q, dim = 0)
    cross_over_linear(parent1.to_q, parent2.to_k, parent1_indices, parent2_indices, child = child.to_k, dim = 0)
    cross_over_linear(parent1.to_q, parent2.to_v, parent1_indices, parent2_indices, child = child.to_v, dim = 0)
    cross_over_linear(parent1.to_out, parent2.to_out, parent1_indices, parent2_indices, child = child.to_out, dim = 1)

    cross_over_scale(parent1.qk_scale, parent2.qk_scale, parent1_indices, parent2_indices, child = child.qk_scale)

    return child

# breed normalized transformers

@torch.no_grad()
def cross_over_ngpt(
    ngpt1: nGPT,
    ngpt2: nGPT,
    child: nGPT | None = None
) -> nGPT:

    assert ngpt1 == ngpt2

    if not exists(child):
        child = deepcopy(ngpt1)

    num_tokens = ngpt1.num_tokens
    midpoint = num_tokens // 2
    rand_indices = randperm(num_tokens)
    parent1_indices, parent2_indices = rand_indices[:midpoint], rand_indices[midpoint:]

    cross_over_linear(ngpt1.token_embed, ngpt2.token_embed, parent1_indices, parent2_indices, child = child.token_embed)

    for (attn1, ff1), (attn2, ff2), (child_attn, child_ff) in zip(ngpt1.layers, ngpt2.layers, child.layers):
        cross_over_attention(attn1.fn, attn2.fn, child = child_attn.fn)
        cross_over_feedforward(ff1.fn, ff2.fn, child = child_ff.fn)

    cross_over_scale(ngpt1.logit_scale, ngpt2.logit_scale, parent1_indices, parent2_indices, child = child.logit_scale)

    cross_over_linear(ngpt1.to_logits, ngpt2.to_logits, parent1_indices, parent2_indices, child = child.to_logits)

    return child
