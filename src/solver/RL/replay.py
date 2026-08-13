"""N-step transition assembly and compact bit-packed replay storage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .codec import (
    ACTION_COUNT,
    OBSERVATION_DIM,
    ActionMask,
    Observation,
)


@dataclass(frozen=True, slots=True)
class Transition:
    observation: Observation
    action: int
    reward: float
    next_observation: Observation
    next_action_mask: ActionMask
    terminated: bool
    steps: int


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    indices: NDArray[np.int64]
    weights: NDArray[np.float32]
    observations: NDArray[np.uint8]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    next_observations: NDArray[np.uint8]
    next_action_masks: NDArray[np.bool_]
    terminated: NDArray[np.bool_]
    steps: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _RawTransition:
    observation: Observation
    action: int
    reward: float
    next_observation: Observation
    next_action_mask: ActionMask
    terminated: bool


class NStepAccumulator:
    """Maintain one independent n-step queue for each vector environment."""

    def __init__(self, num_envs: int, n_steps: int, gamma: float) -> None:
        if num_envs < 1 or n_steps < 1:
            raise ValueError("num_envs and n_steps must be positive")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.n_steps = n_steps
        self.gamma = gamma
        self._queues = tuple(deque() for _ in range(num_envs))

    def _build(self, queue: deque[_RawTransition]) -> Transition:
        horizon = min(self.n_steps, len(queue))
        reward = 0.0
        discount = 1.0
        final = queue[0]
        for index in range(horizon):
            final = queue[index]
            reward += discount * final.reward
            discount *= self.gamma
            if final.terminated:
                horizon = index + 1
                break
        first = queue[0]
        return Transition(
            observation=first.observation,
            action=first.action,
            reward=reward,
            next_observation=final.next_observation,
            next_action_mask=final.next_action_mask,
            terminated=final.terminated,
            steps=horizon,
        )

    def add(
        self,
        env_index: int,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Observation,
        next_action_mask: ActionMask,
        terminated: bool,
    ) -> tuple[Transition, ...]:
        queue = self._queues[env_index]
        queue.append(
            _RawTransition(
                observation=np.array(observation, copy=True),
                action=int(action),
                reward=float(reward),
                next_observation=np.array(next_observation, copy=True),
                next_action_mask=np.array(next_action_mask, copy=True),
                terminated=bool(terminated),
            )
        )
        emitted: list[Transition] = []
        if terminated:
            while queue:
                emitted.append(self._build(queue))
                queue.popleft()
        elif len(queue) >= self.n_steps:
            emitted.append(self._build(queue))
            queue.popleft()
        return tuple(emitted)


class PackedReplayBuffer:
    """A ring buffer storing binary state tensors in packed form."""

    def __init__(
        self,
        capacity: int,
        *,
        path: Path | None = None,
        resume: bool = False,
    ) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self.observation_bytes = (OBSERVATION_DIM + 7) // 8
        self.mask_bytes = (ACTION_COUNT + 7) // 8
        self.dtype = np.dtype(
            [
                ("observation", np.uint8, (self.observation_bytes,)),
                ("action", np.uint16),
                ("reward", np.float32),
                ("next_observation", np.uint8, (self.observation_bytes,)),
                ("next_action_mask", np.uint8, (self.mask_bytes,)),
                ("terminated", np.bool_),
                ("steps", np.uint8),
            ]
        )
        self.path = path
        if path is None:
            self._storage: NDArray[np.void] = np.empty(capacity, dtype=self.dtype)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            expected_bytes = capacity * self.dtype.itemsize
            if resume:
                if not path.exists() or path.stat().st_size != expected_bytes:
                    raise ValueError(
                        "replay.dat is missing or has an incompatible size"
                    )
                mode = "r+"
            else:
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite {path}")
                mode = "w+"
            self._storage = np.memmap(
                path,
                dtype=self.dtype,
                mode=mode,
                shape=(capacity,),
            )
        self.position = 0
        self.size = 0

    @property
    def allocated_bytes(self) -> int:
        return self.capacity * self.dtype.itemsize

    replay_kind = "uniform"

    def _write_transition(self, index: int, transition: Transition) -> None:
        record = self._storage[index]
        record["observation"] = np.packbits(
            transition.observation, bitorder="little"
        )
        record["action"] = transition.action
        record["reward"] = transition.reward
        record["next_observation"] = np.packbits(
            transition.next_observation, bitorder="little"
        )
        record["next_action_mask"] = np.packbits(
            transition.next_action_mask, bitorder="little"
        )
        record["terminated"] = transition.terminated
        record["steps"] = transition.steps

    def add(self, transition: Transition) -> None:
        self.add_many((transition,))

    def add_many(
        self,
        transitions: tuple[Transition, ...] | list[Transition],
    ) -> None:
        if len(transitions) > self.capacity:
            raise ValueError("one replay insertion cannot exceed capacity")
        indices = np.empty(len(transitions), dtype=np.int64)
        for offset, transition in enumerate(transitions):
            index = self.position
            indices[offset] = index
            self._write_transition(index, transition)
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        if len(indices):
            self._after_add(indices)

    def _after_add(self, indices: NDArray[np.int64]) -> None:
        return None

    def _build_batch(
        self,
        indices: NDArray[np.int64],
        weights: NDArray[np.float32],
    ) -> ReplayBatch:
        records = self._storage[indices]
        observations = np.unpackbits(
            records["observation"],
            axis=1,
            count=OBSERVATION_DIM,
            bitorder="little",
        )
        next_observations = np.unpackbits(
            records["next_observation"],
            axis=1,
            count=OBSERVATION_DIM,
            bitorder="little",
        )
        next_action_masks = np.unpackbits(
            records["next_action_mask"],
            axis=1,
            count=ACTION_COUNT,
            bitorder="little",
        ).astype(np.bool_, copy=False)
        return ReplayBatch(
            indices=np.ascontiguousarray(indices, dtype=np.int64),
            weights=np.ascontiguousarray(weights, dtype=np.float32),
            observations=np.ascontiguousarray(observations),
            actions=np.ascontiguousarray(
                records["action"], dtype=np.int64
            ),
            rewards=np.ascontiguousarray(records["reward"], dtype=np.float32),
            next_observations=np.ascontiguousarray(next_observations),
            next_action_masks=np.ascontiguousarray(next_action_masks),
            terminated=np.ascontiguousarray(
                records["terminated"], dtype=np.bool_
            ),
            steps=np.ascontiguousarray(records["steps"], dtype=np.uint8),
        )

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        beta: float = 1.0,
    ) -> ReplayBatch:
        if batch_size < 1 or self.size < batch_size:
            raise ValueError("replay does not contain a full sample batch")
        indices = rng.integers(0, self.size, size=batch_size, dtype=np.int64)
        weights = np.ones(batch_size, dtype=np.float32)
        return self._build_batch(indices, weights)

    def update_priorities(
        self,
        indices: NDArray[np.int64],
        td_errors: NDArray[np.float32],
    ) -> None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {
            "replay_kind": self.replay_kind,
            "capacity": self.capacity,
            "position": self.position,
            "size": self.size,
            "observation_dim": OBSERVATION_DIM,
            "action_count": ACTION_COUNT,
            "record_bytes": self.dtype.itemsize,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {
            "capacity": self.capacity,
            "observation_dim": OBSERVATION_DIM,
            "action_count": ACTION_COUNT,
            "record_bytes": self.dtype.itemsize,
        }
        stored_kind = state.get("replay_kind", "uniform")
        if stored_kind != self.replay_kind:
            raise ValueError("checkpoint replay kind is incompatible")
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint replay metadata is incompatible")
        position = int(state["position"])
        size = int(state["size"])
        if not 0 <= position < self.capacity or not 0 <= size <= self.capacity:
            raise ValueError("checkpoint replay cursor is invalid")
        self.position = position
        self.size = size

    def flush(self) -> None:
        if isinstance(self._storage, np.memmap):
            self._storage.flush()


class PrioritizedReplayBuffer(PackedReplayBuffer):
    """Proportional prioritized replay backed by a vectorized sum tree."""

    replay_kind = "prioritized"

    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-3,
        path: Path | None = None,
        priority_path: Path | None = None,
        resume: bool = False,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("priority alpha must be in [0, 1]")
        if priority_epsilon <= 0.0:
            raise ValueError("priority epsilon must be positive")
        super().__init__(capacity, path=path, resume=resume)
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.priority_path = priority_path
        if priority_path is None:
            self._priorities: NDArray[np.float32] = np.zeros(
                capacity, dtype=np.float32
            )
        else:
            priority_path.parent.mkdir(parents=True, exist_ok=True)
            expected_bytes = capacity * np.dtype(np.float32).itemsize
            if resume:
                if (
                    not priority_path.exists()
                    or priority_path.stat().st_size != expected_bytes
                ):
                    raise ValueError(
                        "priorities.dat is missing or has an incompatible size"
                    )
                mode = "r+"
            else:
                if priority_path.exists():
                    raise FileExistsError(f"refusing to overwrite {priority_path}")
                mode = "w+"
            self._priorities = np.memmap(
                priority_path,
                dtype=np.float32,
                mode=mode,
                shape=(capacity,),
            )
            if not resume:
                self._priorities.fill(0.0)
        self._tree_size = 1 << (capacity - 1).bit_length()
        self._sum_tree = np.zeros(self._tree_size * 2, dtype=np.float64)
        self._rebuild_tree()
        self.max_priority = 1.0

    @property
    def allocated_bytes(self) -> int:
        priority_bytes = self.capacity * np.dtype(np.float32).itemsize
        return super().allocated_bytes + priority_bytes

    def _rebuild_tree(self) -> None:
        leaves = self._sum_tree[
            self._tree_size : self._tree_size + self.capacity
        ]
        leaves[:] = self._priorities
        width = self._tree_size // 2
        while width:
            children = self._sum_tree[2 * width : 4 * width]
            self._sum_tree[width : 2 * width] = (
                children[0::2] + children[1::2]
            )
            width //= 2

    def _set_powered_priorities(
        self,
        indices: NDArray[np.int64],
        powered_priorities: NDArray[np.float32],
    ) -> None:
        unique, inverse = np.unique(indices, return_inverse=True)
        values = np.zeros(len(unique), dtype=np.float32)
        np.maximum.at(values, inverse, powered_priorities)
        self._priorities[unique] = values
        nodes = unique + self._tree_size
        self._sum_tree[nodes] = values
        while nodes.size and nodes[0] > 1:
            parents = np.unique(nodes // 2)
            self._sum_tree[parents] = (
                self._sum_tree[parents * 2]
                + self._sum_tree[parents * 2 + 1]
            )
            nodes = parents

    def _after_add(self, indices: NDArray[np.int64]) -> None:
        priorities = np.full(
            len(indices), self.max_priority, dtype=np.float32
        )
        self._set_powered_priorities(indices, priorities)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        beta: float = 1.0,
    ) -> ReplayBatch:
        if batch_size < 1 or self.size < batch_size:
            raise ValueError("replay does not contain a full sample batch")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("priority beta must be in [0, 1]")
        total = float(self._sum_tree[1])
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("prioritized replay has an invalid priority sum")
        segment = total / batch_size
        values = (np.arange(batch_size) + rng.random(batch_size)) * segment
        nodes = np.ones(batch_size, dtype=np.int64)
        for _ in range(self._tree_size.bit_length() - 1):
            left = nodes * 2
            left_sum = self._sum_tree[left]
            go_right = values >= left_sum
            values = np.where(go_right, values - left_sum, values)
            nodes = left + go_right
        indices = nodes - self._tree_size
        if np.any(indices >= self.capacity):
            raise RuntimeError("sum-tree sampling reached padding leaves")
        probabilities = self._priorities[indices].astype(np.float64) / total
        weights = np.power(self.size * probabilities, -beta)
        weights /= weights.max()
        return self._build_batch(
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(
        self,
        indices: NDArray[np.int64],
        td_errors: NDArray[np.float32],
    ) -> None:
        errors = np.asarray(td_errors, dtype=np.float32)
        if indices.shape != errors.shape:
            raise ValueError("priority indices and TD errors must align")
        if not np.isfinite(errors).all():
            raise RuntimeError("cannot store non-finite replay priorities")
        powered = np.power(
            np.abs(errors) + self.priority_epsilon,
            self.alpha,
        ).astype(np.float32)
        self.max_priority = max(self.max_priority, float(powered.max()))
        self._set_powered_priorities(indices, powered)

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state.update(
            {
                "priority_alpha": self.alpha,
                "priority_epsilon": self.priority_epsilon,
                "max_priority": self.max_priority,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        if (
            float(state.get("priority_alpha", -1.0)) != self.alpha
            or float(state.get("priority_epsilon", -1.0))
            != self.priority_epsilon
        ):
            raise ValueError("checkpoint prioritized replay parameters differ")
        self.max_priority = float(state["max_priority"])
        if not math.isfinite(self.max_priority) or self.max_priority <= 0.0:
            raise ValueError("checkpoint maximum replay priority is invalid")
        self._rebuild_tree()

    def flush(self) -> None:
        super().flush()
        if isinstance(self._priorities, np.memmap):
            self._priorities.flush()
