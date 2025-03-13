from copy import deepcopy

import torch
from torch import cat, randperm

from nGPT_pytorch.nGPT import FeedForward

@torch.no_grad()
def cross_over(
    parent1: FeedForward,
    parent2: FeedForward
) -> FeedForward:

    child = deepcopy(parent1)

    assert parent1.dim == parent2.dim and parent1.expand_factor == parent2.expand_factor
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
