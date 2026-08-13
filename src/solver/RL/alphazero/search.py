"""Batched chance-sampled PUCT with neural leaf evaluation and no rollout."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from time import perf_counter
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from engine import Action, GameError, GameSession

from ...state import public_state_key, sample_public_event
from ..codec import (
    ACTION_COUNT,
    OBSERVATION_DIM,
    PASS_ACTION_INDEX,
    encode_decision_into,
)
from ..dqn import resolve_device
from .network import PolicyValueNetwork


VALUE_SCALE = 200.0
_StateKey = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 256
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    def validate(self) -> None:
        if not 1 <= self.simulations <= np.iinfo(np.uint16).max:
            raise ValueError("simulations must fit in a positive uint16")
        if not math.isfinite(self.c_puct) or self.c_puct < 0.0:
            raise ValueError("c_puct must be finite and non-negative")
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0.0:
            raise ValueError("Dirichlet alpha must be finite and positive")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError("Dirichlet epsilon must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SearchResult:
    action_visits: NDArray[np.uint16]
    action_values: NDArray[np.float32]
    legal_actions: dict[int, Action]
    root_value: float
    nodes: int
    max_depth: int


@dataclass(frozen=True, slots=True)
class SearchBatchStats:
    searches: int
    simulations: int
    expanded_nodes: int
    network_evaluations: int
    inference_batches: int
    mean_inference_batch_size: float
    max_inference_batch_size: int
    chance_samples: int
    terminal_hits: int
    max_depth: int
    elapsed_seconds: float


@dataclass(slots=True)
class _Node:
    action_indices: tuple[int, ...]
    actions: tuple[Action | None, ...]
    priors: NDArray[np.float32]
    visits: NDArray[np.int32]
    value_sums: NDArray[np.float32]
    total_visits: int = 0

    @property
    def means(self) -> NDArray[np.float32]:
        return np.divide(
            self.value_sums,
            self.visits,
            out=np.zeros_like(self.value_sums),
            where=self.visits > 0,
        )


@dataclass(slots=True)
class _Search:
    root_game: GameSession
    root: _Node
    table: dict[_StateKey, _Node]
    rng: random.Random
    simulations: int = 0
    expanded_nodes: int = 1
    chance_samples: int = 0
    terminal_hits: int = 0
    max_depth: int = 0


@dataclass(slots=True)
class _PendingLeaf:
    search: _Search
    game: GameSession
    key: _StateKey
    path: list[tuple[_Node, int, float]]
    depth: int


class PolicyValueEvaluator:
    """Pinned-buffer inference for variable batches of public decisions."""

    def __init__(self, network: PolicyValueNetwork, device: torch.device) -> None:
        self.network = network
        self.device = device
        self.capacity = 0
        self._host_observations: Tensor | None = None
        self._host_masks: Tensor | None = None
        self._device_observations: Tensor | None = None
        self._device_masks: Tensor | None = None

    def _ensure_capacity(self, minimum: int) -> None:
        if minimum <= self.capacity:
            return
        capacity = max(32, minimum, self.capacity * 2)
        pinned = self.device.type == "cuda"
        self._host_observations = torch.empty(
            (capacity, OBSERVATION_DIM), dtype=torch.uint8, pin_memory=pinned
        )
        self._host_masks = torch.empty(
            (capacity, ACTION_COUNT), dtype=torch.bool, pin_memory=pinned
        )
        self._device_observations = torch.empty(
            (capacity, OBSERVATION_DIM), dtype=torch.float32, device=self.device
        )
        self._device_masks = torch.empty(
            (capacity, ACTION_COUNT), dtype=torch.bool, device=self.device
        )
        self.capacity = capacity

    @torch.inference_mode()
    def evaluate(
        self,
        games: Sequence[GameSession],
    ) -> tuple[list[_Node], NDArray[np.float32]]:
        if not games:
            return [], np.empty(0, dtype=np.float32)
        self._ensure_capacity(len(games))
        if (
            self._host_observations is None
            or self._host_masks is None
            or self._device_observations is None
            or self._device_masks is None
        ):
            raise RuntimeError("inference buffers were not initialized")

        host_observations = self._host_observations.numpy()
        host_masks = self._host_masks.numpy()
        action_maps: list[dict[int, Action]] = []
        for row, game in enumerate(games):
            action_maps.append(
                encode_decision_into(
                    game,
                    host_observations[row],
                    host_masks[row],
                )
            )

        batch_size = len(games)
        non_blocking = self.device.type == "cuda"
        observations = self._device_observations[:batch_size]
        masks = self._device_masks[:batch_size]
        observations.copy_(
            self._host_observations[:batch_size], non_blocking=non_blocking
        )
        masks.copy_(self._host_masks[:batch_size], non_blocking=non_blocking)
        amp_enabled = self.device.type == "cuda"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if amp_enabled else torch.bfloat16,
            enabled=amp_enabled,
        ):
            logits, values = self.network(observations)
        logits = logits.float().masked_fill(~masks, -1.0e9)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        value_array = values.float().cpu().numpy()

        nodes: list[_Node] = []
        for row, actions_by_index in enumerate(action_maps):
            action_indices = (PASS_ACTION_INDEX, *sorted(actions_by_index))
            actions = (
                None,
                *(actions_by_index[index] for index in action_indices[1:]),
            )
            priors = probabilities[row, list(action_indices)].astype(
                np.float32, copy=True
            )
            priors /= priors.sum()
            count = len(action_indices)
            nodes.append(
                _Node(
                    action_indices=action_indices,
                    actions=actions,
                    priors=priors,
                    visits=np.zeros(count, dtype=np.int32),
                    value_sums=np.zeros(count, dtype=np.float32),
                )
            )
        return nodes, value_array.astype(np.float32, copy=False)


def _require_standard_game(game: GameSession) -> None:
    if game.status != "playing" or game.pending is None:
        raise GameError("AlphaZero search requires a pending public card")
    if game.advanced or game.shared_objectives_enabled or game.pencil_powers_enabled:
        raise GameError("AlphaZero currently supports only the standard rules")


def _select_action(node: _Node, c_puct: float, rng: random.Random) -> int:
    q_values = node.means
    exploration = (
        c_puct
        * node.priors
        * math.sqrt(node.total_visits + 1.0)
        / (node.visits + 1.0)
    )
    scores = q_values + exploration
    best = float(scores.max())
    tied = np.flatnonzero(np.isclose(scores, best, rtol=0.0, atol=1e-12))
    return int(tied[rng.randrange(len(tied))])


def _backup(path: list[tuple[_Node, int, float]], leaf_value: float) -> None:
    value = float(leaf_value)
    for node, action_index, reward in reversed(path):
        value += reward
        node.total_visits += 1
        node.visits[action_index] += 1
        node.value_sums[action_index] += value


class BatchedPUCT:
    """Advance one simulation per independent tree between neural batches."""

    def __init__(
        self,
        network: PolicyValueNetwork,
        config: SearchConfig = SearchConfig(),
        *,
        device: str = "auto",
    ) -> None:
        config.validate()
        self.config = config
        self.device = resolve_device(device)
        self.network = network.to(self.device)
        self.network.eval()
        self.evaluator = PolicyValueEvaluator(self.network, self.device)
        self.last_stats = SearchBatchStats(0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0.0)

    def _select_until_leaf(self, search: _Search) -> _PendingLeaf | None:
        while search.simulations < self.config.simulations:
            simulation = search.root_game.copy_public_state()
            node = search.root
            path: list[tuple[_Node, int, float]] = []
            depth = 0
            while True:
                local_index = _select_action(
                    node, self.config.c_puct, search.rng
                )
                action = node.actions[local_index]
                reward = (
                    0.0
                    if action is None
                    else sum(simulation.score_delta_for_legal_action(action))
                    / VALUE_SCALE
                )
                path.append((node, local_index, reward))
                depth += 1
                simulation.apply_legal_action(action)
                if simulation.status == "finished":
                    _backup(path, 0.0)
                    search.simulations += 1
                    search.terminal_hits += 1
                    search.max_depth = max(search.max_depth, depth)
                    break

                sample_public_event(simulation, search.rng)
                search.chance_samples += 1
                key = public_state_key(simulation)
                child = search.table.get(key)
                if child is None:
                    return _PendingLeaf(search, simulation, key, path, depth)
                node = child
        return None

    def search(
        self,
        games: Sequence[GameSession],
        *,
        seeds: Sequence[int],
        add_root_noise: bool,
    ) -> tuple[SearchResult, ...]:
        if len(games) != len(seeds):
            raise ValueError("each search tree requires one independent seed")
        if not games:
            return ()
        for game in games:
            _require_standard_game(game)

        started = perf_counter()
        root_games = [game.copy_public_state() for game in games]
        root_nodes, _ = self.evaluator.evaluate(root_games)
        searches: list[_Search] = []
        for root_game, root, seed in zip(root_games, root_nodes, seeds):
            rng = random.Random(int(seed))
            if add_root_noise and len(root.priors) > 1:
                noise_rng = np.random.default_rng(rng.getrandbits(128))
                noise = noise_rng.dirichlet(
                    np.full(len(root.priors), self.config.dirichlet_alpha)
                ).astype(np.float32)
                epsilon = self.config.dirichlet_epsilon
                root.priors = (1.0 - epsilon) * root.priors + epsilon * noise
                root.priors /= root.priors.sum()
            key = public_state_key(root_game)
            searches.append(
                _Search(root_game, root, {key: root}, rng)
            )

        network_evaluations = len(root_games)
        inference_batches = 1
        inference_batch_sizes = [len(root_games)]
        while True:
            pending = [
                leaf
                for search in searches
                if (leaf := self._select_until_leaf(search)) is not None
            ]
            if not pending:
                break
            new_nodes, leaf_values = self.evaluator.evaluate(
                [leaf.game for leaf in pending]
            )
            network_evaluations += len(pending)
            inference_batches += 1
            inference_batch_sizes.append(len(pending))
            for leaf, node, value in zip(pending, new_nodes, leaf_values):
                leaf.search.table[leaf.key] = node
                leaf.search.expanded_nodes += 1
                _backup(leaf.path, float(value))
                leaf.search.simulations += 1
                leaf.search.max_depth = max(leaf.search.max_depth, leaf.depth)

        results: list[SearchResult] = []
        for search in searches:
            if search.root.total_visits != self.config.simulations:
                raise RuntimeError("root visits do not match simulation budget")
            visits = np.zeros(ACTION_COUNT, dtype=np.uint16)
            values = np.zeros(ACTION_COUNT, dtype=np.float32)
            means = search.root.means
            for local, fixed in enumerate(search.root.action_indices):
                visits[fixed] = search.root.visits[local]
                values[fixed] = means[local]
            legal_actions = {
                fixed: action
                for fixed, action in zip(
                    search.root.action_indices, search.root.actions
                )
                if action is not None
            }
            results.append(
                SearchResult(
                    action_visits=visits,
                    action_values=values,
                    legal_actions=legal_actions,
                    root_value=float(
                        search.root.value_sums.sum() / search.root.total_visits
                    ),
                    nodes=len(search.table),
                    max_depth=search.max_depth,
                )
            )

        elapsed = perf_counter() - started
        self.last_stats = SearchBatchStats(
            searches=len(searches),
            simulations=sum(item.simulations for item in searches),
            expanded_nodes=sum(item.expanded_nodes for item in searches),
            network_evaluations=network_evaluations,
            inference_batches=inference_batches,
            mean_inference_batch_size=float(np.mean(inference_batch_sizes)),
            max_inference_batch_size=max(inference_batch_sizes),
            chance_samples=sum(item.chance_samples for item in searches),
            terminal_hits=sum(item.terminal_hits for item in searches),
            max_depth=max(item.max_depth for item in searches),
            elapsed_seconds=elapsed,
        )
        return tuple(results)
