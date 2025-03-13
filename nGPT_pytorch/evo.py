from copy import deepcopy

import torch
from torch import cat

from nGPT_pytorch.nGPT import FeedForward

@torch.no_grad()
def cross_over(
    parent1: FeedForward,
    parent2: FeedForward
) -> FeedForward:

    child = deepcopy(parent1)

    dim = parent1.dim
    assert parent1.dim == parent2.dim and parent1.expand_factor == parent2.expand_factor

    parent1_w1 = parent1.to_hidden.weight
    parent2_w1 = parent2.to_hidden.weight
    child_w1 = child.to_hidden.weight

    parent1_gate = parent1.to_gate.weight
    parent2_gate = parent2.to_gate.weight
    child_gate = child.to_gate.weight

    parent1_w2 = parent1.to_out.weight
    parent2_w2 = parent2.to_out.weight
    child_w2 = child.to_out.weight

    midpoint = dim // 2
    rand_indices = torch.randperm(dim)

    # randomly select vectors from the feedforward weight matrices from both parents to constitute the child

    parent1_indices, parent2_indices = rand_indices[:midpoint], rand_indices[midpoint:]

    child_w1.copy_(cat((parent1_w1[:, parent1_indices], parent2_w1[:, parent2_indices]), dim = 1))
    child_gate.copy_(cat((parent1_gate[:, parent1_indices], parent2_gate[:, parent2_indices]), dim = 1))
    child_w2.copy_(cat((parent1_w2[parent1_indices, :], parent2_w2[parent2_indices, :]), dim = 0))

    return child
