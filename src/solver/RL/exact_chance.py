"""Exact chance-node expectimax with a trained DQN at the horizon."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
import torch

from engine_cpp import (
    COLORS,
    DECK_BY_ID,
    SYMBOLS,
    Action,
    GameError,
    GameSession,
    legal_edge_mask,
)

from ..state import public_event_successors, public_state_key
from .codec import (
    ACTION_COUNT,
    ACTIVE_COLOR,
    COLOR_ORDER,
    CURRENT_CARDS,
    DRAW_COUNT,
    EVENT_FLAGS,
    LINE_EDGES,
    LINE_STATIONS,
    NUM_EDGES,
    NUM_STATIONS,
    OBSERVATION_DIM,
    PASS_ACTION_INDEX,
    REMAINING_CARDS,
    ROUND_INDEX,
    TARGET_SYMBOL,
    UNDERGROUND_COUNT,
    encode_decision,
)
from .dqn import (
    ActionValueNetwork,
    DQNPolicy,
    MeanActionValueNetwork,
    masked_argmax,
    resolve_device,
)


_SYMBOL_INDEX = {symbol: index for index, symbol in enumerate(SYMBOLS)}


def _one_hot_index(observation: NDArray[np.uint8], span: slice) -> int:
    indices = np.flatnonzero(observation[span])
    if len(indices) != 1:
        raise ValueError("exact-chance target received an invalid observation")
    return int(indices[0])


def _decode_position(
    observation: NDArray[np.uint8],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], int]:
    if observation.shape != (OBSERVATION_DIM,):
        raise ValueError("afterstate observation has an incompatible shape")
    order = tuple(
        COLORS[
            _one_hot_index(
                observation,
                slice(
                    COLOR_ORDER.start + position * len(COLORS),
                    COLOR_ORDER.start + (position + 1) * len(COLORS),
                ),
            )
        ]
        for position in range(len(COLORS))
    )
    round_index = _one_hot_index(observation, ROUND_INDEX)
    if _one_hot_index(observation, ACTIVE_COLOR) != COLORS.index(order[round_index]):
        raise ValueError("afterstate active color disagrees with color order")
    line_edge_masks: list[int] = []
    line_station_masks: list[int] = []
    for color_index in range(len(COLORS)):
        edge_start = LINE_EDGES.start + color_index * NUM_EDGES
        station_start = LINE_STATIONS.start + color_index * NUM_STATIONS
        edge_ids = np.flatnonzero(
            observation[edge_start : edge_start + NUM_EDGES]
        )
        station_ids = np.flatnonzero(
            observation[station_start : station_start + NUM_STATIONS]
        )
        line_edge_masks.append(sum(1 << int(item) for item in edge_ids))
        line_station_masks.append(sum(1 << int(item) for item in station_ids))
    return order, tuple(line_station_masks), tuple(line_edge_masks), round_index


def _decode_afterstate(observation: NDArray[np.uint8]) -> GameSession:
    """Rebuild a base-rule public state immediately before its next draw.

    Replay stores the decision state *after* the environment has drawn the
    next card.  Exact depth-2 targets need the public afterstate before that
    draw so that both chance events can be expanded with the engine's own
    transition code.  The hidden deck order is intentionally not restored.
    """

    order, line_station_masks, line_edge_masks, round_index = _decode_position(
        observation
    )
    current_ids = tuple(int(item) for item in np.flatnonzero(observation[CURRENT_CARDS]))
    remaining_ids = set(int(item) for item in np.flatnonzero(observation[REMAINING_CARDS]))
    if not current_ids or set(current_ids) & remaining_ids:
        raise ValueError("afterstate card features are inconsistent")
    remaining_mask = sum(
        1 << card_id for card_id in remaining_ids | set(current_ids)
    )
    draw_count = _one_hot_index(observation, DRAW_COUNT) - len(current_ids)
    underground_count = _one_hot_index(observation, UNDERGROUND_COUNT) - sum(
        int(DECK_BY_ID[card_id].underground) for card_id in current_ids
    )
    if draw_count < 0 or underground_count < 0:
        raise ValueError("afterstate card counters are inconsistent")
    return GameSession.from_public_state(
        order=order,
        line_station_masks=tuple(line_station_masks),
        line_edge_masks=tuple(line_edge_masks),
        remaining_mask=remaining_mask,
        round_index=round_index,
        underground_count=underground_count,
        draw_count=draw_count,
    )


def _legal_mask(
    observation: NDArray[np.uint8],
    *,
    target_symbol: str | None,
    wild: bool,
    source_any: bool,
) -> NDArray[np.bool_]:
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    mask[PASS_ACTION_INDEX] = True
    order, line_station_masks, line_edge_masks, round_index = _decode_position(
        observation
    )
    remaining_mask = sum(
        1 << int(card_id) for card_id in np.flatnonzero(observation[REMAINING_CARDS])
    )
    edge_mask = legal_edge_mask(
        order=order,
        line_station_masks=tuple(line_station_masks),
        line_edge_masks=tuple(line_edge_masks),
        remaining_mask=remaining_mask,
        round_index=round_index,
        underground_count=_one_hot_index(observation, UNDERGROUND_COUNT),
        draw_count=_one_hot_index(observation, DRAW_COUNT),
        target_symbol=target_symbol,
        wild=wild,
        source_any=source_any,
    )
    mask[:NUM_EDGES] = tuple(
        bool(edge_mask & (1 << edge_id)) for edge_id in range(NUM_EDGES)
    )
    return mask


def _exact_target_frontier(
    next_observations: NDArray[np.uint8],
    terminated: NDArray[np.bool_],
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.bool_],
    NDArray[np.float32],
    NDArray[np.int64],
]:
    """Put the visible next card back, then enumerate every possible draw."""

    observations: list[NDArray[np.uint8]] = []
    masks: list[NDArray[np.bool_]] = []
    probabilities: list[float] = []
    owners: list[int] = []
    for owner in np.flatnonzero(~terminated):
        actual = next_observations[owner]
        current_ids = tuple(int(item) for item in np.flatnonzero(actual[CURRENT_CARDS]))
        if not current_ids:
            raise ValueError("non-terminal replay state has no visible card")
        remaining_ids = tuple(
            int(item)
            for item in np.flatnonzero(
                np.logical_or(
                    actual[REMAINING_CARDS],
                    actual[CURRENT_CARDS],
                )
            )
        )
        draw_before = _one_hot_index(actual, DRAW_COUNT) - len(current_ids)
        underground_before = _one_hot_index(actual, UNDERGROUND_COUNT) - sum(
            int(DECK_BY_ID[card_id].underground) for card_id in current_ids
        )
        if draw_before < 0 or underground_before < 0:
            raise ValueError("next observation has inconsistent card counters")

        base = actual.copy()
        base[REMAINING_CARDS] = 0
        base[REMAINING_CARDS.start + np.asarray(remaining_ids)] = 1
        base[CURRENT_CARDS] = 0
        base[TARGET_SYMBOL] = 0
        base[EVENT_FLAGS] = 0
        base[DRAW_COUNT] = 0
        base[DRAW_COUNT.start + draw_before] = 1
        base[UNDERGROUND_COUNT] = 0
        base[UNDERGROUND_COUNT.start + underground_before] = 1

        first_probability = 1.0 / len(remaining_ids)
        for first_id in remaining_ids:
            first = DECK_BY_ID[first_id]
            following = (
                tuple(item for item in remaining_ids if item != first_id)
                if first.switch
                else (None,)
            )
            if not following:
                raise ValueError("switch card has no possible following card")
            probability = first_probability / len(following)
            for second_id in following:
                card_ids = (
                    (first_id, int(second_id))
                    if second_id is not None
                    else (first_id,)
                )
                target = (
                    DECK_BY_ID[int(second_id)]
                    if second_id is not None
                    else first
                )
                underground = underground_before + sum(
                    int(DECK_BY_ID[card_id].underground)
                    for card_id in card_ids
                )
                draw_count = draw_before + len(card_ids)
                wild = target.symbol is None
                source_any = first.switch

                child = base.copy()
                child[REMAINING_CARDS.start + np.asarray(card_ids)] = 0
                child[CURRENT_CARDS.start + np.asarray(card_ids)] = 1
                symbol_index = (
                    len(SYMBOLS) if wild else _SYMBOL_INDEX[target.symbol]
                )
                child[TARGET_SYMBOL.start + symbol_index] = 1
                child[EVENT_FLAGS] = (
                    int(wild),
                    int(source_any),
                    int(underground >= 5),
                )
                child[DRAW_COUNT] = 0
                child[DRAW_COUNT.start + draw_count] = 1
                child[UNDERGROUND_COUNT] = 0
                child[UNDERGROUND_COUNT.start + underground] = 1

                observations.append(child)
                masks.append(
                    _legal_mask(
                        child,
                        target_symbol=target.symbol,
                        wild=wild,
                        source_any=source_any,
                    )
                )
                probabilities.append(probability)
                owners.append(int(owner))

    if not observations:
        return (
            np.empty((0, OBSERVATION_DIM), dtype=np.uint8),
            np.empty((0, ACTION_COUNT), dtype=np.bool_),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.stack(observations),
        np.stack(masks),
        np.asarray(probabilities, dtype=np.float32),
        np.asarray(owners, dtype=np.int64),
    )


@torch.no_grad()
def exact_chance_double_dqn_values(
    online: ActionValueNetwork,
    target: ActionValueNetwork,
    next_observations: NDArray[np.uint8],
    terminated: NDArray[np.bool_],
    *,
    device: torch.device,
    inference_batch_size: int = 8192,
) -> tuple[torch.Tensor, int]:
    """Return E[Q_target(s, argmax Q_online(s))] for each replay row."""

    frontier, masks, probabilities, owners = _exact_target_frontier(
        next_observations,
        terminated,
    )
    expected = torch.zeros(len(next_observations), device=device)
    for start in range(0, len(frontier), inference_batch_size):
        stop = min(start + inference_batch_size, len(frontier))
        observations = torch.as_tensor(
            frontier[start:stop], device=device, dtype=torch.float32
        )
        legal = torch.as_tensor(
            masks[start:stop], device=device, dtype=torch.bool
        )
        actions = masked_argmax(online.q_values(observations), legal)
        values = target.q_values(observations).gather(
            1, actions[:, None]
        ).squeeze(1)
        weights = torch.as_tensor(
            probabilities[start:stop], device=device
        )
        indices = torch.as_tensor(owners[start:stop], device=device)
        expected.index_add_(0, indices, values * weights)
    return expected, len(frontier)


@torch.no_grad()
def _leaf_double_values(
    online: ActionValueNetwork,
    target: ActionValueNetwork,
    observations: list[NDArray[np.uint8]],
    masks: list[NDArray[np.bool_]],
    *,
    device: torch.device,
    reward_scale: float,
    inference_batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Get Double-DQN leaf pairs in raw score units."""

    online_values = np.empty(len(observations), dtype=np.float64)
    target_values = np.empty(len(observations), dtype=np.float64)
    batches = 0
    for start in range(0, len(observations), inference_batch_size):
        stop = min(start + inference_batch_size, len(observations))
        observation_tensor = torch.as_tensor(
            np.stack(observations[start:stop]),
            device=device,
            dtype=torch.float32,
        )
        mask_tensor = torch.as_tensor(
            np.stack(masks[start:stop]),
            device=device,
            dtype=torch.bool,
        )
        online_q = online.q_values(observation_tensor)
        actions = masked_argmax(online_q, mask_tensor)
        target_q = target.q_values(observation_tensor)
        online_values[start:stop] = (
            online_q.gather(1, actions[:, None]).squeeze(1).cpu().numpy()
            * reward_scale
        )
        target_values[start:stop] = (
            target_q.gather(1, actions[:, None]).squeeze(1).cpu().numpy()
            * reward_scale
        )
        batches += 1
    return online_values, target_values, batches


def _double_evaluate(
    expression: _Expression,
    online_leaf_values: NDArray[np.float64],
    target_leaf_values: NDArray[np.float64],
    memo: dict[int, tuple[float, float]],
    *,
    gamma: float = 1.0,
) -> tuple[float, float]:
    """Evaluate an expression with online selection and target valuation."""

    key = id(expression)
    cached = memo.get(key)
    if cached is not None:
        return cached
    if isinstance(expression, _Leaf):
        value = (
            float(online_leaf_values[expression.index]),
            float(target_leaf_values[expression.index]),
        )
    elif isinstance(expression, _Add):
        online_value, target_value = _double_evaluate(
            expression.continuation,
            online_leaf_values,
            target_leaf_values,
            memo,
            gamma=gamma,
        ) if expression.continuation is not None else (0.0, 0.0)
        value = (
            float(expression.reward) + gamma * online_value,
            float(expression.reward) + gamma * target_value,
        )
    elif isinstance(expression, _Expected):
        online_value = 0.0
        target_value = 0.0
        for probability, child in expression.outcomes:
            child_online, child_target = _double_evaluate(
                child,
                online_leaf_values,
                target_leaf_values,
                memo,
                gamma=gamma,
            )
            online_value += probability * child_online
            target_value += probability * child_target
        value = (online_value, target_value)
    elif isinstance(expression, _Maximum):
        values = [
            _double_evaluate(
                child,
                online_leaf_values,
                target_leaf_values,
                memo,
                gamma=gamma,
            )
            for child in expression.branches
        ]
        value = max(values, key=lambda pair: pair[0])
    else:
        raise TypeError(f"unknown exact-chance expression: {type(expression)!r}")
    memo[key] = value
    return value


@torch.no_grad()
def exact_chance_depth2_double_dqn_values(
    online: ActionValueNetwork,
    target: ActionValueNetwork,
    next_observations: NDArray[np.uint8],
    terminated: NDArray[np.bool_],
    *,
    device: torch.device,
    reward_scale: float,
    gamma: float = 1.0,
    inference_batch_size: int = 8192,
) -> tuple[torch.Tensor, int]:
    """Return a two-chance-event Double-DQN continuation value.

    The first draw is averaged exactly.  For every resulting decision, each
    action is evaluated through the second exact draw; online Q selects the
    intermediate action and target Q values that selected branch.  Leaf
    values are returned in the normalized units used by the learner.
    """

    if reward_scale <= 0.0 or not math.isfinite(reward_scale):
        raise ValueError("reward_scale must be finite and positive")
    if not 0.0 < gamma <= 1.0 or not math.isfinite(gamma):
        raise ValueError("gamma must be finite and in (0, 1]")
    if next_observations.ndim != 2 or next_observations.shape[1] != OBSERVATION_DIM:
        raise ValueError("next observations have an incompatible shape")
    if terminated.shape != (len(next_observations),):
        raise ValueError("terminated flags have an incompatible shape")

    builder = _ForestBuilder(depth=2)
    expressions: dict[int, _Expression] = {}
    for owner in np.flatnonzero(~terminated):
        afterstate = _decode_afterstate(next_observations[int(owner)])
        expressions[int(owner)] = builder._chance(afterstate, 2, None)

    online_leaf_values, target_leaf_values, _ = _leaf_double_values(
        online,
        target,
        builder.observations,
        builder.masks,
        device=device,
        reward_scale=reward_scale,
        inference_batch_size=inference_batch_size,
    )
    expected = torch.zeros(len(next_observations), device=device)
    memo: dict[int, tuple[float, float]] = {}
    for owner, expression in expressions.items():
        _online_value, target_value = _double_evaluate(
            expression,
            online_leaf_values,
            target_leaf_values,
            memo,
            gamma=gamma,
        )
        expected[owner] = target_value / reward_scale
    return expected, len(builder.observations)


@dataclass(frozen=True, slots=True)
class ExactChanceAction:
    action: Action | None
    immediate_reward: int
    expected_gain: float


@dataclass(frozen=True, slots=True)
class ExactChanceDecision:
    action: Action | None
    action_index: int
    immediate_reward: int
    expected_gain: float
    tied_actions: int
    estimates: tuple[ExactChanceAction, ...]


@dataclass(frozen=True, slots=True)
class ExactChanceBatchStats:
    games: int = 0
    depth: int = 0
    root_actions: int = 0
    intermediate_decisions: int = 0
    action_branches: int = 0
    chance_nodes: int = 0
    chance_outcomes: int = 0
    cache_hits: int = 0
    leaf_references: int = 0
    unique_leaf_states: int = 0
    inference_batches: int = 0
    max_inference_batch_size: int = 0
    build_seconds: float = 0.0
    inference_seconds: float = 0.0
    evaluation_seconds: float = 0.0


class ExactChanceBudgetExceeded(RuntimeError):
    """Raised before an exact search exceeds its configured branch budget."""


class _Expression:
    def evaluate(
        self,
        leaf_values: NDArray[np.float64],
        memo: dict[int, float],
    ) -> float:
        raise NotImplementedError


@dataclass(eq=False, slots=True)
class _Leaf(_Expression):
    index: int

    def evaluate(
        self,
        leaf_values: NDArray[np.float64],
        memo: dict[int, float],
    ) -> float:
        return float(leaf_values[self.index])


@dataclass(eq=False, slots=True)
class _Add(_Expression):
    reward: int
    continuation: _Expression | None

    def evaluate(
        self,
        leaf_values: NDArray[np.float64],
        memo: dict[int, float],
    ) -> float:
        key = id(self)
        if key in memo:
            return memo[key]
        value = float(self.reward)
        if self.continuation is not None:
            value += self.continuation.evaluate(leaf_values, memo)
        memo[key] = value
        return value


@dataclass(eq=False, slots=True)
class _Maximum(_Expression):
    branches: tuple[_Expression, ...]

    def evaluate(
        self,
        leaf_values: NDArray[np.float64],
        memo: dict[int, float],
    ) -> float:
        key = id(self)
        if key in memo:
            return memo[key]
        value = max(branch.evaluate(leaf_values, memo) for branch in self.branches)
        memo[key] = value
        return value


@dataclass(eq=False, slots=True)
class _Expected(_Expression):
    outcomes: tuple[tuple[float, _Expression], ...]

    def evaluate(
        self,
        leaf_values: NDArray[np.float64],
        memo: dict[int, float],
    ) -> float:
        key = id(self)
        if key in memo:
            return memo[key]
        value = sum(
            probability * child.evaluate(leaf_values, memo)
            for probability, child in self.outcomes
        )
        memo[key] = value
        return value


class _ForestBuilder:
    def __init__(
        self,
        depth: int,
        *,
        round_end: bool = False,
        max_action_branches: int | None = None,
    ) -> None:
        self.depth = depth
        self.round_end = round_end
        self.max_action_branches = max_action_branches
        self.observations: list[NDArray[np.uint8]] = []
        self.masks: list[NDArray[np.bool_]] = []
        self._leaf_indices: dict[tuple[int, ...], int] = {}
        self._decision_cache: dict[
            tuple[int, int, tuple[int, ...]], _Expression
        ] = {}
        self._chance_cache: dict[
            tuple[int, int, tuple[int, ...]], _Expression
        ] = {}
        self._boundary_cache: dict[tuple[int, ...], _Expression] = {}
        self.root_actions = 0
        self.intermediate_decisions = 0
        self.action_branches = 0
        self.chance_nodes = 0
        self.chance_outcomes = 0
        self.cache_hits = 0
        self.leaf_references = 0

    @staticmethod
    def _reward(game: GameSession, action: Action | None) -> int:
        if action is None:
            return 0
        return sum(game.score_delta_for_legal_action(action))

    def _leaf(self, game: GameSession) -> _Leaf:
        self.leaf_references += 1
        key = public_state_key(game)
        index = self._leaf_indices.get(key)
        if index is None:
            encoded = encode_decision(game)
            index = len(self.observations)
            self._leaf_indices[key] = index
            self.observations.append(encoded.observation)
            self.masks.append(encoded.action_mask)
        return _Leaf(index)

    def _action(
        self,
        game: GameSession,
        action: Action | None,
        chance_depth: int,
        stop_round: int | None,
    ) -> _Add:
        if (
            self.max_action_branches is not None
            and self.action_branches >= self.max_action_branches
        ):
            raise ExactChanceBudgetExceeded(
                f"exact search exceeded {self.max_action_branches} action branches"
            )
        self.action_branches += 1
        reward = self._reward(game, action)
        child = game.copy_for_search_action()
        child.apply_legal_action(action)
        if child.status == "finished":
            return _Add(reward, None)
        if child.pending is not None:
            raise GameError(
                "Exact-Chance DQN currently supports only the standard game"
            )
        if stop_round is not None and child.round_index != stop_round:
            return _Add(reward, self._chance_to_leaf(child))
        return _Add(reward, self._chance(child, chance_depth, stop_round))

    def _decision(
        self,
        game: GameSession,
        chance_depth: int,
        stop_round: int | None,
    ) -> _Expression:
        key = (
            chance_depth,
            -1 if stop_round is None else stop_round,
            public_state_key(game),
        )
        cached = self._decision_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.intermediate_decisions += 1
        branches = tuple(
            self._action(game, action, chance_depth, stop_round)
            for action in (None, *game.legal_actions())
        )
        expression = _Maximum(branches)
        self._decision_cache[key] = expression
        return expression

    def _chance(
        self,
        game: GameSession,
        chance_depth: int,
        stop_round: int | None,
    ) -> _Expression:
        if chance_depth < 1:
            raise ValueError("chance depth must be positive")
        key = (
            chance_depth,
            -1 if stop_round is None else stop_round,
            public_state_key(game),
        )
        cached = self._chance_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.chance_nodes += 1
        outcomes: list[tuple[float, _Expression]] = []
        probability_sum = 0.0
        for probability, child in public_event_successors(game):
            self.chance_outcomes += 1
            probability_sum += probability
            if stop_round is not None:
                continuation = self._decision(
                    child,
                    chance_depth,
                    stop_round,
                )
            else:
                continuation = (
                    self._leaf(child)
                    if chance_depth == 1
                    else self._decision(child, chance_depth - 1, None)
                )
            outcomes.append((probability, continuation))
        if not math.isclose(probability_sum, 1.0, abs_tol=1e-12):
            raise RuntimeError("chance probabilities do not sum to one")
        expression = _Expected(tuple(outcomes))
        self._chance_cache[key] = expression
        return expression

    def _chance_to_leaf(self, game: GameSession) -> _Expression:
        """Evaluate the first visible decision after a round boundary."""

        key = public_state_key(game)
        cached = self._boundary_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.chance_nodes += 1
        outcomes: list[tuple[float, _Expression]] = []
        probability_sum = 0.0
        for probability, child in public_event_successors(game):
            self.chance_outcomes += 1
            probability_sum += probability
            outcomes.append((probability, self._leaf(child)))
        if not math.isclose(probability_sum, 1.0, abs_tol=1e-12):
            raise RuntimeError("chance probabilities do not sum to one")
        expression = _Expected(tuple(outcomes))
        self._boundary_cache[key] = expression
        return expression

    def roots(
        self,
        game: GameSession,
    ) -> tuple[tuple[Action | None, int, _Expression], ...]:
        if game.status != "playing" or game.pending is None:
            raise GameError("draw a card before asking the policy for an action")
        if game.shared_objectives_enabled or game.pencil_powers_enabled:
            raise GameError(
                "Exact-Chance DQN currently supports only the standard game"
            )
        roots = []
        stop_round = game.round_index if self.round_end else None
        for action in (None, *game.legal_actions()):
            self.root_actions += 1
            reward = self._reward(game, action)
            roots.append(
                (
                    action,
                    reward,
                    self._action(game, action, self.depth, stop_round),
                )
            )
        return tuple(roots)


class ExactChanceDQNPolicy:
    """Enumerate chance exactly for ``depth`` events, then use max DQN Q."""

    def __init__(
        self,
        network: ActionValueNetwork,
        depth: int,
        *,
        reward_scale: float,
        device: str = "auto",
        leaf_batch_size: int = 8192,
        round_end: bool = False,
        max_action_branches: int | None = None,
    ) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if not math.isfinite(reward_scale) or reward_scale <= 0.0:
            raise ValueError("reward_scale must be finite and positive")
        if leaf_batch_size < 1:
            raise ValueError("leaf_batch_size must be positive")
        if max_action_branches is not None and max_action_branches < 1:
            raise ValueError("max_action_branches must be positive")
        self.depth = depth
        self.round_end = round_end
        self.max_action_branches = max_action_branches
        self.reward_scale = float(reward_scale)
        self.device = resolve_device(device)
        self.network = network.to(self.device)
        self.network.eval()
        self.leaf_batch_size = leaf_batch_size
        self.last_batch_stats = ExactChanceBatchStats(depth=depth)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        depth: int,
        *,
        device: str = "auto",
        leaf_batch_size: int = 8192,
        leaf_weights: str = "online",
        round_end: bool = False,
        max_action_branches: int | None = None,
    ) -> ExactChanceDQNPolicy:
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        reward_scale = float(
            checkpoint.get("config", {}).get("reward_scale", 1.0)
        )
        if leaf_weights == "mean-online-target":
            online = DQNPolicy.from_checkpoint(
                path, device=device, weights="online"
            )
            target = DQNPolicy.from_checkpoint(
                path, device=device, weights="target"
            )
            network: ActionValueNetwork = MeanActionValueNetwork(
                online.network,
                target.network,
            )
            resolved_device = str(online.device)
        else:
            dqn = DQNPolicy.from_checkpoint(
                path,
                device=device,
                weights=leaf_weights,
            )
            network = dqn.network
            resolved_device = str(dqn.device)
        return cls(
            network,
            depth,
            reward_scale=reward_scale,
            device=resolved_device,
            leaf_batch_size=leaf_batch_size,
            round_end=round_end,
            max_action_branches=max_action_branches,
        )

    @torch.inference_mode()
    def _leaf_values(
        self,
        observations: list[NDArray[np.uint8]],
        masks: list[NDArray[np.bool_]],
    ) -> tuple[NDArray[np.float64], int, int]:
        count = len(observations)
        values = np.empty(count, dtype=np.float64)
        batches = 0
        max_batch = 0
        for start in range(0, count, self.leaf_batch_size):
            stop = min(start + self.leaf_batch_size, count)
            observation_batch = np.stack(observations[start:stop], axis=0)
            mask_batch = np.stack(masks[start:stop], axis=0)
            observation_tensor = torch.as_tensor(
                observation_batch,
                device=self.device,
                dtype=torch.float32,
            )
            mask_tensor = torch.as_tensor(
                mask_batch,
                device=self.device,
                dtype=torch.bool,
            )
            q_values = self.network.q_values(observation_tensor)
            masked = q_values.masked_fill(
                ~mask_tensor,
                torch.finfo(q_values.dtype).min,
            )
            batch_values = masked.max(dim=1).values * self.reward_scale
            values[start:stop] = batch_values.cpu().numpy()
            batches += 1
            max_batch = max(max_batch, stop - start)
        return values, batches, max_batch

    def choose_many(
        self,
        games: tuple[GameSession, ...] | list[GameSession],
    ) -> tuple[ExactChanceDecision, ...]:
        if not games:
            raise ValueError("choose_many requires at least one game")

        build_started = perf_counter()
        builder = _ForestBuilder(
            self.depth,
            round_end=self.round_end,
            max_action_branches=self.max_action_branches,
        )
        roots = tuple(builder.roots(game) for game in games)
        build_seconds = perf_counter() - build_started

        inference_started = perf_counter()
        leaf_values, inference_batches, max_batch = self._leaf_values(
            builder.observations,
            builder.masks,
        )
        inference_seconds = perf_counter() - inference_started

        evaluation_started = perf_counter()
        memo: dict[int, float] = {}
        decisions = []
        for game_roots in roots:
            estimates = tuple(
                ExactChanceAction(
                    action=action,
                    immediate_reward=reward,
                    expected_gain=expression.evaluate(leaf_values, memo),
                )
                for action, reward, expression in game_roots
            )

            def sort_key(item: ExactChanceAction) -> tuple[object, ...]:
                if item.action is None:
                    return (-item.expected_gain, 1)
                return (-item.expected_gain, 0, *item.action.sort_key)

            ranked = tuple(sorted(estimates, key=sort_key))
            chosen = ranked[0]
            tied = sum(
                math.isclose(
                    item.expected_gain,
                    chosen.expected_gain,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for item in ranked
            )
            action_index = (
                PASS_ACTION_INDEX
                if chosen.action is None
                else chosen.action.edge_id
            )
            decisions.append(
                ExactChanceDecision(
                    action=chosen.action,
                    action_index=action_index,
                    immediate_reward=chosen.immediate_reward,
                    expected_gain=chosen.expected_gain,
                    tied_actions=tied,
                    estimates=ranked,
                )
            )
        evaluation_seconds = perf_counter() - evaluation_started

        self.last_batch_stats = ExactChanceBatchStats(
            games=len(games),
            depth=self.depth,
            root_actions=builder.root_actions,
            intermediate_decisions=builder.intermediate_decisions,
            action_branches=builder.action_branches,
            chance_nodes=builder.chance_nodes,
            chance_outcomes=builder.chance_outcomes,
            cache_hits=builder.cache_hits,
            leaf_references=builder.leaf_references,
            unique_leaf_states=len(builder.observations),
            inference_batches=inference_batches,
            max_inference_batch_size=max_batch,
            build_seconds=build_seconds,
            inference_seconds=inference_seconds,
            evaluation_seconds=evaluation_seconds,
        )
        return tuple(decisions)

    def choose(self, game: GameSession) -> ExactChanceDecision:
        return self.choose_many((game,))[0]
