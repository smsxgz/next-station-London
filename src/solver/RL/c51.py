"""Categorical action-value network and Bellman projection for C51."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dqn import ActionValueNetwork, NetworkSpec


class CategoricalQNetwork(ActionValueNetwork):
    """Predict one categorical return distribution per fixed action."""

    def __init__(
        self,
        spec: NetworkSpec = NetworkSpec(),
        *,
        atom_count: int = 51,
        value_min: float = 0.0,
        value_max: float = 20.0,
    ) -> None:
        super().__init__()
        if not spec.hidden_sizes or any(size < 1 for size in spec.hidden_sizes):
            raise ValueError("C51 hidden sizes must be positive")
        if atom_count < 2:
            raise ValueError("C51 requires at least two atoms")
        if (
            not math.isfinite(value_min)
            or not math.isfinite(value_max)
            or value_min >= value_max
        ):
            raise ValueError("C51 support must have finite increasing bounds")

        self.spec = spec
        self.atom_count = int(atom_count)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        layers: list[nn.Module] = []
        input_size = spec.observation_dim
        for hidden_size in spec.hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(
            nn.Linear(input_size, spec.action_count * self.atom_count)
        )
        self.layers = nn.Sequential(*layers)
        self.register_buffer(
            "support",
            torch.linspace(value_min, value_max, self.atom_count),
        )

    def forward(self, observations: Tensor) -> Tensor:
        logits = self.layers(observations)
        return logits.view(-1, self.spec.action_count, self.atom_count)

    def q_values(self, observations: Tensor) -> Tensor:
        probabilities = F.softmax(self(observations), dim=-1)
        return (probabilities * self.support).sum(dim=-1)

    def config(self) -> dict[str, object]:
        return {
            "observation_dim": self.spec.observation_dim,
            "action_count": self.spec.action_count,
            "hidden_sizes": list(self.spec.hidden_sizes),
            "atom_count": self.atom_count,
            "value_min": self.value_min,
            "value_max": self.value_max,
        }


def project_distribution(
    next_probabilities: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    discounts: Tensor,
    support: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project a categorical Bellman target onto a fixed evenly spaced support."""

    if next_probabilities.ndim != 2:
        raise ValueError("next probabilities must have shape [batch, atoms]")
    batch_size, atom_count = next_probabilities.shape
    if support.shape != (atom_count,):
        raise ValueError("support and probability atom counts differ")
    if any(
        tensor.shape != (batch_size,)
        for tensor in (rewards, terminated, discounts)
    ):
        raise ValueError("reward, terminal, and discount batches must align")

    value_min = support[0]
    value_max = support[-1]
    atom_delta = (value_max - value_min) / (atom_count - 1)
    raw_targets = rewards[:, None] + (
        (1.0 - terminated[:, None])
        * discounts[:, None]
        * support[None, :]
    )
    lower_clip_mass = (
        next_probabilities * (raw_targets < value_min)
    ).sum(dim=1)
    upper_clip_mass = (
        next_probabilities * (raw_targets > value_max)
    ).sum(dim=1)

    positions = (
        raw_targets.clamp(min=value_min, max=value_max) - value_min
    ) / atom_delta
    positions.clamp_(0.0, float(atom_count - 1))
    lower = positions.floor().to(torch.int64)
    upper = positions.ceil().to(torch.int64)
    same_atom = lower == upper
    lower_weights = torch.where(
        same_atom,
        torch.ones_like(positions),
        upper.to(positions.dtype) - positions,
    )
    upper_weights = torch.where(
        same_atom,
        torch.zeros_like(positions),
        positions - lower.to(positions.dtype),
    )

    offsets = (
        torch.arange(batch_size, device=next_probabilities.device)
        * atom_count
    )[:, None]
    projected = torch.zeros_like(next_probabilities)
    projected_flat = projected.view(-1)
    projected_flat.scatter_add_(
        0,
        (lower + offsets).reshape(-1),
        (next_probabilities * lower_weights).reshape(-1),
    )
    projected_flat.scatter_add_(
        0,
        (upper + offsets).reshape(-1),
        (next_probabilities * upper_weights).reshape(-1),
    )
    return projected, lower_clip_mass, upper_clip_mass
