"""Exact event-JVP geometry on fixed endpoint-generated histories."""
from __future__ import annotations

import torch

from round11_streaming_core import smooth_rollout


def conditional_energy(tangent, probabilities):
    """Half categorical Fisher energy; trailing axes are batch,time,vocab."""
    centered = tangent - (probabilities * tangent).sum(-1, keepdim=True)
    return .5 * (probabilities * centered.square()).sum(-1)


@torch.no_grad()
def event_geometry(model, prefix_cache, n, inputs, signs, chunk_size):
    events = signs.shape[1]
    zero = torch.zeros(events, dtype=torch.float64, device=inputs.device)
    function = lambda strengths: smooth_rollout(model, prefix_cache, n, inputs, signs, strengths)
    reference, combined = torch.func.jvp(function, (zero,), (torch.ones_like(zero),))
    basis = torch.eye(events, dtype=torch.float64, device=inputs.device)
    blocks = []
    for start in range(0, events, chunk_size):
        tangent = torch.vmap(lambda direction: torch.func.jvp(function, (zero,), (direction,))[1])(
            basis[start:start + chunk_size])
        blocks.append(tangent)
    jacobian = torch.cat(blocks, 0)
    linearity_gap = float((jacobian.sum(0) - combined).abs().max())
    if linearity_gap > 1e-8:
        raise AssertionError("Combined JVP differs from event-JVP sum")
    probabilities = reference.softmax(-1)
    full = conditional_energy(combined, probabilities)
    diagonal = conditional_energy(jacobian, probabilities).sum(0)
    return reference, combined, jacobian, full, diagonal, linearity_gap


def event_gram(jacobian, probabilities):
    """Half Fisher Gram, averaged over the supplied fixed-history paths."""
    centered = jacobian - (jacobian * probabilities).sum(-1, keepdim=True)
    weighted = centered * probabilities.sqrt()
    flattened = weighted.flatten(1)
    return .5 * (flattened @ flattened.T) / probabilities.shape[0]
