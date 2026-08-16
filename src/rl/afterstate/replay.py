"""Uniform and prioritized replay for canonical afterstate records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .codec import AfterstateRecord
from .replay_storage import (
    REPLAY_RECORD_DTYPE,
    replay_record_from_row,
    write_replay_record,
)

REPLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    indices: NDArray[np.int64]
    weights: NDArray[np.float32]
    records: tuple[AfterstateRecord, ...]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    terminated: NDArray[np.bool_]


class AfterstateReplayBuffer:
    """A fixed-size ring buffer with optional memory-mapped storage."""

    replay_kind = "afterstate_uniform"

    def __init__(
        self,
        capacity: int,
        *,
        path: Path | None = None,
        resume: bool = False,
    ) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.dtype = REPLAY_RECORD_DTYPE
        self.path = path
        if path is None:
            self._storage = np.empty(self.capacity, dtype=self.dtype)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            expected = self.capacity * self.dtype.itemsize
            if resume:
                if not path.exists() or path.stat().st_size != expected:
                    raise ValueError("replay.dat is missing or has incompatible size")
                mode = "r+"
            else:
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite {path}")
                mode = "w+"
            self._storage = np.memmap(
                path, dtype=self.dtype, mode=mode, shape=(capacity,)
            )
        self.position = 0
        self.size = 0

    @property
    def allocated_bytes(self) -> int:
        return self.capacity * self.dtype.itemsize

    def add(
        self,
        record: AfterstateRecord,
        *,
        action: int,
        reward: int,
    ) -> None:
        if not 0 <= action <= 155:
            raise ValueError("action index is outside the fixed action space")
        write_replay_record(
            self._storage[self.position],
            record,
            action=action,
            reward=reward,
        )
        index = np.asarray((self.position,), dtype=np.int64)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._after_add(index)

    def add_many(
        self,
        transitions: (
            list[tuple[AfterstateRecord, int, int]]
            | tuple[tuple[AfterstateRecord, int, int], ...]
        ),
    ) -> None:
        if len(transitions) > self.capacity:
            raise ValueError("one replay insertion cannot exceed capacity")
        indices = np.empty(len(transitions), dtype=np.int64)
        for offset, (record, action, reward) in enumerate(transitions):
            index = self.position
            indices[offset] = index
            write_replay_record(
                self._storage[index],
                record,
                action=action,
                reward=reward,
            )
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        if len(indices):
            self._after_add(indices)

    def _after_add(self, indices: NDArray[np.int64]) -> None:
        return None

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        beta: float = 1.0,
    ) -> ReplayBatch:
        if batch_size < 1 or self.size < batch_size:
            raise ValueError("replay does not contain a full batch")
        indices = rng.integers(0, self.size, size=batch_size, dtype=np.int64)
        rows = self._storage[indices]
        return ReplayBatch(
            indices=np.ascontiguousarray(indices),
            weights=np.ones(batch_size, dtype=np.float32),
            records=tuple(replay_record_from_row(row) for row in rows),
            actions=np.asarray(rows["action"], dtype=np.int64),
            rewards=np.asarray(rows["reward"], dtype=np.float32),
            terminated=np.asarray(rows["terminated"], dtype=np.bool_),
        )

    def update_priorities(
        self,
        indices: NDArray[np.int64],
        td_errors: NDArray[np.float32],
    ) -> None:
        return None

    def flush(self) -> None:
        if isinstance(self._storage, np.memmap):
            self._storage.flush()

    def state_dict(self) -> dict[str, object]:
        return {
            "replay_schema_version": REPLAY_SCHEMA_VERSION,
            "replay_kind": self.replay_kind,
            "capacity": self.capacity,
            "position": self.position,
            "size": self.size,
            "record_bytes": self.dtype.itemsize,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {
            "replay_schema_version": REPLAY_SCHEMA_VERSION,
            "replay_kind": self.replay_kind,
            "capacity": self.capacity,
            "record_bytes": self.dtype.itemsize,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint replay metadata is incompatible")
        position = int(state["position"])
        size = int(state["size"])
        if not 0 <= position < self.capacity or not 0 <= size <= self.capacity:
            raise ValueError("checkpoint replay cursor is invalid")
        self.position = position
        self.size = size


class PrioritizedAfterstateReplayBuffer(AfterstateReplayBuffer):
    """Proportional prioritized replay for canonical afterstate records."""

    replay_kind = "afterstate_prioritized"

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
            self._priorities: NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)
        else:
            priority_path.parent.mkdir(parents=True, exist_ok=True)
            expected = capacity * np.dtype(np.float32).itemsize
            if resume:
                if (
                    not priority_path.exists()
                    or priority_path.stat().st_size != expected
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
        self.max_priority = 1.0
        self._rebuild_tree()

    @property
    def allocated_bytes(self) -> int:
        return super().allocated_bytes + self.capacity * np.dtype(np.float32).itemsize

    def _rebuild_tree(self) -> None:
        leaves = self._sum_tree[self._tree_size : self._tree_size + self.capacity]
        leaves[:] = self._priorities
        width = self._tree_size // 2
        while width:
            children = self._sum_tree[2 * width : 4 * width]
            self._sum_tree[width : 2 * width] = children[0::2] + children[1::2]
            width //= 2

    def _set_powered_priorities(
        self,
        indices: NDArray[np.int64],
        powered: NDArray[np.float32],
    ) -> None:
        unique, inverse = np.unique(indices, return_inverse=True)
        values = np.zeros(len(unique), dtype=np.float32)
        np.maximum.at(values, inverse, powered)
        self._priorities[unique] = values
        nodes = unique + self._tree_size
        self._sum_tree[nodes] = values
        while nodes.size and nodes[0] > 1:
            parents = np.unique(nodes // 2)
            self._sum_tree[parents] = (
                self._sum_tree[parents * 2] + self._sum_tree[parents * 2 + 1]
            )
            nodes = parents

    def _after_add(self, indices: NDArray[np.int64]) -> None:
        self._set_powered_priorities(
            indices,
            np.full(len(indices), self.max_priority, dtype=np.float32),
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
        if np.any(indices >= self.capacity) or np.any(indices >= self.size):
            raise RuntimeError("prioritized sampling reached an invalid leaf")
        probabilities = self._priorities[indices].astype(np.float64) / total
        weights = np.power(self.size * probabilities, -beta)
        weights /= weights.max()
        rows = self._storage[indices]
        return ReplayBatch(
            indices=np.ascontiguousarray(indices, dtype=np.int64),
            weights=np.ascontiguousarray(weights, dtype=np.float32),
            records=tuple(replay_record_from_row(row) for row in rows),
            actions=np.asarray(rows["action"], dtype=np.int64),
            rewards=np.asarray(rows["reward"], dtype=np.float32),
            terminated=np.asarray(rows["terminated"], dtype=np.bool_),
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
        if len(powered):
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
            or float(state.get("priority_epsilon", -1.0)) != self.priority_epsilon
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
