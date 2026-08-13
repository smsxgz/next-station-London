"""Masked Q-network and inference policy for Next Station: London."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from engine import Action, GameError, GameSession

from .codec import (
    ACTION_COUNT,
    OBSERVATION_DIM,
    PASS_ACTION_INDEX,
    ActionMask,
    Observation,
    encode_decision,
)


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    observation_dim: int = OBSERVATION_DIM
    action_count: int = ACTION_COUNT
    hidden_sizes: tuple[int, ...] = (512, 512)


@dataclass(frozen=True, slots=True)
class DQNDecision:
    action: Action | None
    action_index: int
    q_value: float


class ActionValueNetwork(nn.Module):
    """A network that exposes expected action values for masked selection."""

    def q_values(self, observations: Tensor) -> Tensor:
        raise NotImplementedError

    def config(self) -> dict[str, object]:
        raise NotImplementedError


class MeanActionValueNetwork(ActionValueNetwork):
    """Average action values from compatible inference networks."""

    def __init__(self, *networks: ActionValueNetwork) -> None:
        super().__init__()
        if len(networks) < 2:
            raise ValueError("a mean evaluator requires at least two networks")
        configs = [network.config() for network in networks]
        if any(config != configs[0] for config in configs[1:]):
            raise ValueError("mean evaluator networks are incompatible")
        self.networks = nn.ModuleList(networks)

    def q_values(self, observations: Tensor) -> Tensor:
        total = self.networks[0].q_values(observations)
        for network in self.networks[1:]:
            total = total + network.q_values(observations)
        return total / len(self.networks)

    def config(self) -> dict[str, object]:
        return self.networks[0].config()


class QNetwork(ActionValueNetwork):
    def __init__(self, spec: NetworkSpec = NetworkSpec()) -> None:
        super().__init__()
        if not spec.hidden_sizes or any(size < 1 for size in spec.hidden_sizes):
            raise ValueError("Q-network hidden sizes must be positive")
        self.spec = spec
        layers: list[nn.Module] = []
        input_size = spec.observation_dim
        for hidden_size in spec.hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, spec.action_count))
        self.layers = nn.Sequential(*layers)

    def forward(self, observations: Tensor) -> Tensor:
        return self.layers(observations)

    def q_values(self, observations: Tensor) -> Tensor:
        return self(observations)

    def config(self) -> dict[str, object]:
        config = asdict(self.spec)
        config["hidden_sizes"] = list(self.spec.hidden_sizes)
        return config


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def masked_argmax(q_values: Tensor, action_masks: Tensor) -> Tensor:
    if q_values.shape != action_masks.shape:
        raise ValueError("Q values and action masks must have the same shape")
    if not bool(action_masks.any(dim=1).all()):
        raise ValueError("every state must have at least one legal action")
    masked = q_values.masked_fill(~action_masks, torch.finfo(q_values.dtype).min)
    return masked.argmax(dim=1)


class GreedyBatchEvaluator:
    """Reusable host/device buffers for deterministic batched Q inference."""

    def __init__(
        self,
        network: ActionValueNetwork,
        device: torch.device,
    ) -> None:
        self.network = network
        self.device = device
        self.capacity = 0
        self._host_observations: Tensor | None = None
        self._host_masks: Tensor | None = None
        self._device_observations: Tensor | None = None
        self._device_masks: Tensor | None = None

    def buffers(
        self,
        minimum_capacity: int,
    ) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
        if minimum_capacity < 1:
            raise ValueError("minimum_capacity must be positive")
        if minimum_capacity > self.capacity:
            capacity = max(minimum_capacity, 32, self.capacity * 2)
            pinned = self.device.type == "cuda"
            self._host_observations = torch.empty(
                (capacity, OBSERVATION_DIM),
                dtype=torch.uint8,
                pin_memory=pinned,
            )
            self._host_masks = torch.empty(
                (capacity, ACTION_COUNT),
                dtype=torch.bool,
                pin_memory=pinned,
            )
            self._device_observations = torch.empty(
                (capacity, OBSERVATION_DIM),
                device=self.device,
                dtype=torch.float32,
            )
            self._device_masks = torch.empty(
                (capacity, ACTION_COUNT),
                device=self.device,
                dtype=torch.bool,
            )
            self.capacity = capacity
        if self._host_observations is None or self._host_masks is None:
            raise RuntimeError("batch evaluator buffers were not initialized")
        return (
            self._host_observations.numpy(),
            self._host_masks.numpy(),
        )

    @torch.inference_mode()
    def select(self, batch_size: int) -> NDArray[np.int64]:
        if batch_size < 1 or batch_size > self.capacity:
            raise ValueError("batch_size exceeds the prepared buffer capacity")
        if (
            self._host_observations is None
            or self._host_masks is None
            or self._device_observations is None
            or self._device_masks is None
        ):
            raise RuntimeError("batch evaluator buffers were not initialized")
        non_blocking = self.device.type == "cuda"
        observations = self._device_observations[:batch_size]
        masks = self._device_masks[:batch_size]
        observations.copy_(
            self._host_observations[:batch_size],
            non_blocking=non_blocking,
        )
        masks.copy_(
            self._host_masks[:batch_size],
            non_blocking=non_blocking,
        )
        selected = masked_argmax(
            self.network.q_values(observations),
            masks,
        )
        return selected.cpu().numpy().copy()


@torch.inference_mode()
def select_action_indices(
    network: ActionValueNetwork,
    observations: NDArray[np.uint8],
    action_masks: NDArray[np.bool_],
    *,
    device: torch.device,
    epsilon: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM:
        raise ValueError("observation batch has an incompatible shape")
    if action_masks.shape != (observations.shape[0], ACTION_COUNT):
        raise ValueError("action-mask batch has an incompatible shape")
    observation_tensor = torch.as_tensor(
        observations, device=device, dtype=torch.float32
    )
    mask_tensor = torch.as_tensor(action_masks, device=device, dtype=torch.bool)
    greedy = masked_argmax(
        network.q_values(observation_tensor), mask_tensor
    ).cpu().numpy()
    actions = greedy.astype(np.int64, copy=True)
    if epsilon == 0.0:
        return actions
    explore = rng.random(len(actions)) < epsilon
    for index in np.flatnonzero(explore):
        legal = np.flatnonzero(action_masks[index])
        actions[index] = int(rng.choice(legal))
    return actions


class DQNPolicy:
    """A deterministic masked-argmax policy loaded from a DQN checkpoint."""

    def __init__(
        self,
        network: ActionValueNetwork,
        *,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.network = network.to(self.device)
        self.network.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        weights: str = "online",
    ) -> DQNPolicy:
        if weights not in {"online", "target"}:
            raise ValueError("weights must be 'online' or 'target'")
        resolved = resolve_device(device)
        checkpoint = torch.load(
            Path(checkpoint_path), map_location=resolved, weights_only=False
        )
        if checkpoint.get("q_decomposition", "full_q") != "full_q":
            raise ValueError(
                "this checkpoint uses the retired exact-reward decomposition"
            )
        raw_spec = checkpoint["network_spec"]
        spec = NetworkSpec(
            observation_dim=int(raw_spec["observation_dim"]),
            action_count=int(raw_spec["action_count"]),
            hidden_sizes=tuple(int(size) for size in raw_spec["hidden_sizes"]),
        )
        if spec.observation_dim != OBSERVATION_DIM or spec.action_count != ACTION_COUNT:
            raise ValueError("checkpoint uses an incompatible observation/action codec")
        algorithm = str(
            checkpoint.get(
                "algorithm",
                checkpoint.get("config", {}).get("algorithm", "dqn"),
            )
        )
        if algorithm == "dqn":
            network: ActionValueNetwork = QNetwork(spec)
        elif algorithm == "c51":
            from .c51 import CategoricalQNetwork

            network = CategoricalQNetwork(
                spec,
                atom_count=int(raw_spec["atom_count"]),
                value_min=float(raw_spec["value_min"]),
                value_max=float(raw_spec["value_max"]),
            )
        else:
            raise ValueError(f"unsupported checkpoint algorithm: {algorithm}")
        network.load_state_dict(checkpoint[f"{weights}_state_dict"])
        return cls(network, device=str(resolved))

    @torch.inference_mode()
    def choose(self, game: GameSession) -> DQNDecision:
        decision = encode_decision(game)
        observation = torch.as_tensor(
            decision.observation[None, :],
            device=self.device,
            dtype=torch.float32,
        )
        mask = torch.as_tensor(
            decision.action_mask[None, :],
            device=self.device,
            dtype=torch.bool,
        )
        q_values = self.network.q_values(observation)
        action_index = int(masked_argmax(q_values, mask).item())
        if not decision.action_mask[action_index]:
            raise GameError("DQN selected an illegal action after masking")
        action = (
            None
            if action_index == PASS_ACTION_INDEX
            else decision.actions[action_index]
        )
        return DQNDecision(
            action=action,
            action_index=action_index,
            q_value=float(q_values[0, action_index].item()),
        )
