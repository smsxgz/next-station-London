"""Policy/value network used by AlphaZero self-play and search."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import Tensor, nn

from ..codec import ACTION_COUNT, OBSERVATION_DIM


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    observation_dim: int = OBSERVATION_DIM
    action_count: int = ACTION_COUNT
    width: int = 1024
    residual_blocks: int = 6
    value_hidden: int = 512

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> NetworkSpec:
        return cls(
            observation_dim=int(raw["observation_dim"]),
            action_count=int(raw["action_count"]),
            width=int(raw["width"]),
            residual_blocks=int(raw["residual_blocks"]),
            value_hidden=int(raw["value_hidden"]),
        )


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.norm(inputs)
        hidden = self.activation(self.linear1(hidden))
        return inputs + self.linear2(hidden)


class PolicyValueNetwork(nn.Module):
    """A residual MLP with fixed London policy and scalar value heads."""

    def __init__(self, spec: NetworkSpec = NetworkSpec()) -> None:
        super().__init__()
        if (
            spec.observation_dim < 1
            or spec.action_count < 1
            or spec.width < 1
            or spec.residual_blocks < 1
            or spec.value_hidden < 1
        ):
            raise ValueError("all AlphaZero network dimensions must be positive")
        self.spec = spec
        self.input_layer = nn.Linear(spec.observation_dim, spec.width)
        self.input_norm = nn.LayerNorm(spec.width)
        self.blocks = nn.ModuleList(
            _ResidualBlock(spec.width) for _ in range(spec.residual_blocks)
        )
        self.trunk_norm = nn.LayerNorm(spec.width)
        self.policy_head = nn.Linear(spec.width, spec.action_count)
        self.value_head = nn.Sequential(
            nn.Linear(spec.width, spec.value_hidden),
            nn.GELU(),
            nn.Linear(spec.value_hidden, 1),
        )
        self.activation = nn.GELU()

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.activation(self.input_norm(self.input_layer(observations)))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.trunk_norm(hidden)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def config(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self.spec).items()}


def parameter_count(network: nn.Module) -> int:
    return sum(parameter.numel() for parameter in network.parameters())
