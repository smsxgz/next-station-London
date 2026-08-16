"""Scalar afterstate continuation-value network and checkpoint loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .codec import AFTERSTATE_SCHEMA_VERSION, FEATURE_SCHEMA_HASH, OBSERVATION_DIM


@dataclass(frozen=True, slots=True)
class AfterstateNetworkSpec:
    observation_dim: int = OBSERVATION_DIM
    hidden_sizes: tuple[int, ...] = (512, 512)

    def __post_init__(self) -> None:
        if self.observation_dim < 1:
            raise ValueError("observation_dim must be positive")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw["hidden_sizes"] = list(self.hidden_sizes)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> AfterstateNetworkSpec:
        return cls(
            observation_dim=int(raw["observation_dim"]),
            hidden_sizes=tuple(int(size) for size in raw["hidden_sizes"]),
        )


class AfterstateValueNetwork(nn.Module):
    """MLP returning one normalized continuation value per afterstate."""

    def __init__(self, spec: AfterstateNetworkSpec = AfterstateNetworkSpec()) -> None:
        super().__init__()
        self.spec = spec
        layers: list[nn.Module] = []
        input_size = spec.observation_dim
        for hidden_size in spec.hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, observations: Tensor) -> Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.spec.observation_dim:
            raise ValueError(
                f"expected [batch, {self.spec.observation_dim}] observations, "
                f"got {tuple(observations.shape)}"
            )
        return self.layers(observations).squeeze(1)

    def expected_value(self, observations: Tensor) -> Tensor:
        """Return the scalar continuation value used for action selection."""

        return self(observations)

    def config(self) -> dict[str, object]:
        return self.spec.to_dict()


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def network_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> AfterstateValueNetwork:
    """Build a scalar network after validating its feature schema."""

    if checkpoint.get("model_type") != "afterstate_value":
        raise ValueError("checkpoint is not a scalar afterstate model")
    if checkpoint.get("feature_schema_hash") != FEATURE_SCHEMA_HASH:
        raise ValueError("checkpoint feature schema differs from current code")
    if checkpoint.get("afterstate_schema_version") != AFTERSTATE_SCHEMA_VERSION:
        raise ValueError("checkpoint afterstate schema differs from current code")
    if checkpoint.get("observation_dim") != OBSERVATION_DIM:
        raise ValueError("checkpoint observation dimension differs from current code")
    network = AfterstateValueNetwork(
        AfterstateNetworkSpec.from_dict(checkpoint["network_spec"])
    )
    network.load_state_dict(checkpoint["online_state_dict"])
    return network.to(device)
