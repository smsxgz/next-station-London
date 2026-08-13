"""Depth-limited expectimax with zero value beyond the search horizon."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from engine import (
    Action,
    GameError,
    GameSession,
)

from .scoring import (
    ImmediateReward,
    _RewardContext,
    _reward_context,
    _reward_for_legal_action,
)
from .state import public_event_successors, public_state_key


@dataclass(frozen=True, slots=True)
class SearchStats:
    decision_nodes: int = 0
    chance_nodes: int = 0
    chance_outcomes: int = 0
    cache_hits: int = 0


@dataclass(frozen=True, slots=True)
class LookaheadAction:
    action: Action | None
    immediate_reward: ImmediateReward
    expected_gain: float


@dataclass(frozen=True, slots=True)
class DepthKDecision:
    action: Action | None
    immediate_reward: ImmediateReward
    expected_gain: float
    tied_actions: int
    stats: SearchStats


def _clone_public_state(game: GameSession) -> GameSession:
    """Copy mutable game state without copying or consulting hidden deck order."""

    return game.copy_public_state()


def _clone_action_state(game: GameSession) -> GameSession:
    return game.copy_for_search_action()


class DepthKPolicy:
    """Exact expectimax truncated after k public card events.

    Decision nodes maximize cumulative dense score gain and chance nodes
    exactly enumerate the unknown deck without replacement. Every decision
    includes pass, and an optional Double Section follow-up stays at the same
    depth as its first section. After the kth card event, the continuation
    value is zero. Equal best root actions use system randomness.
    """

    def __init__(self, depth: int) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        self.depth = depth
        self.rng = random.SystemRandom()
        self.last_stats = SearchStats()
        self._cache: dict[tuple[object, ...], float] = {}
        self._legal_cache: dict[
            tuple[object, ...],
            tuple[Action | None, ...],
        ] = {}
        self._decision_nodes = 0
        self._chance_nodes = 0
        self._chance_outcomes = 0
        self._cache_hits = 0

    def _candidate_actions(self, game: GameSession) -> tuple[Action | None, ...]:
        pending = game.pending
        if pending is None:
            raise GameError("candidate actions require a pending card")
        key = public_state_key(game)
        cached = self._legal_cache.get(key)
        if cached is not None:
            return cached
        actions = (None, *game.legal_actions())
        self._legal_cache[key] = actions
        return actions

    @staticmethod
    def _reward(
        game: GameSession,
        action: Action | None,
        context: _RewardContext | None,
    ) -> ImmediateReward:
        if action is None:
            return ImmediateReward()
        return _reward_for_legal_action(game, action, context)

    def _apply(
        self,
        game: GameSession,
        action: Action | None,
    ) -> GameSession:
        child = _clone_action_state(game)
        # ``action`` came from this exact state before it was cloned.
        child.apply_legal_action(action)
        return child

    def _action_value(
        self,
        game: GameSession,
        action: Action | None,
        depth: int,
        context: _RewardContext | None,
    ) -> float:
        immediate = float(self._reward(game, action, context).total)
        child = self._apply(game, action)
        if child.status == "finished":
            return immediate
        if child.pending is not None:
            return immediate + self._decision_value(child, depth)
        if depth == 1:
            return immediate
        return immediate + self._chance_value(child, depth - 1)

    def _decision_value(
        self,
        game: GameSession,
        depth: int,
        reward_context: _RewardContext | None = None,
    ) -> float:
        if depth < 1 or game.status == "finished":
            return 0.0
        if game.pending is None:
            raise GameError("decision expansion requires a pending card")
        key = ("decision", depth, public_state_key(game))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._decision_nodes += 1
        actions = self._candidate_actions(game)
        context = reward_context
        if context is None and any(actions):
            context = _reward_context(game)
        value = max(
            self._action_value(game, action, depth, context)
            for action in actions
        )
        self._cache[key] = value
        return value

    def _chance_value(self, game: GameSession, depth: int) -> float:
        if depth < 1 or game.status == "finished":
            return 0.0
        if game.pending is not None:
            raise GameError("chance expansion requires the previous card to be resolved")
        key = ("chance", depth, public_state_key(game))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._chance_nodes += 1
        expected = 0.0
        probability_sum = 0.0
        reward_context = _reward_context(game)
        for probability, child in public_event_successors(game):
            self._chance_outcomes += 1
            probability_sum += probability
            expected += probability * self._decision_value(
                child,
                depth,
                reward_context,
            )
        if not math.isclose(probability_sum, 1.0, abs_tol=1e-12):
            raise RuntimeError("chance probabilities do not sum to one")
        self._cache[key] = expected
        return expected

    def _root_action_value(
        self,
        game: GameSession,
        action: Action | None,
        immediate: ImmediateReward,
        context: _RewardContext | None,
    ) -> float:
        """Evaluate one root action through the generic recursion."""

        return self._action_value(
            game,
            action,
            self.depth,
            context,
        )

    def _snapshot_stats(self) -> SearchStats:
        return SearchStats(
            decision_nodes=self._decision_nodes,
            chance_nodes=self._chance_nodes,
            chance_outcomes=self._chance_outcomes,
            cache_hits=self._cache_hits,
        )

    def rank_actions(self, game: GameSession) -> tuple[LookaheadAction, ...]:
        if game.status != "playing" or game.pending is None:
            raise GameError("draw a card before asking the policy for an action")
        self._cache.clear()
        self._legal_cache.clear()
        self._decision_nodes = 1
        self._chance_nodes = 0
        self._chance_outcomes = 0
        self._cache_hits = 0

        actions = self._candidate_actions(game)
        context = _reward_context(game) if any(actions) else None
        ranked_items: list[LookaheadAction] = []
        for action in actions:
            immediate = self._reward(game, action, context)
            expected_gain = self._root_action_value(
                game,
                action,
                immediate,
                context,
            )
            ranked_items.append(
                LookaheadAction(
                    action=action,
                    immediate_reward=immediate,
                    expected_gain=expected_gain,
                )
            )
        ranked = tuple(ranked_items)

        def sort_key(item: LookaheadAction) -> tuple[object, ...]:
            action = item.action
            if action is None:
                return (-item.expected_gain, 1)
            return (
                -item.expected_gain,
                0,
                *action.sort_key,
            )

        result = tuple(sorted(ranked, key=sort_key))
        self.last_stats = self._snapshot_stats()
        self._cache.clear()
        self._legal_cache.clear()
        return result

    def choose(self, game: GameSession) -> DepthKDecision:
        ranked = self.rank_actions(game)
        best_value = ranked[0].expected_gain
        tied = tuple(
            item
            for item in ranked
            if math.isclose(item.expected_gain, best_value, abs_tol=1e-12)
        )
        chosen = self.rng.choice(tied)
        return DepthKDecision(
            action=chosen.action,
            immediate_reward=chosen.immediate_reward,
            expected_gain=chosen.expected_gain,
            tied_actions=len(tied),
            stats=self.last_stats,
        )
