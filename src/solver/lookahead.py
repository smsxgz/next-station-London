"""Depth-limited expectimax backed by the native solver kernel."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from engine_cpp import Action, GameError, GameSession, PENCIL_POWERS

from ._native import NativeActionEstimate, native_lookahead
from .scoring import ImmediateReward


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


def _action(estimate: NativeActionEstimate) -> Action | None:
    if estimate.edge_id < 0:
        return None
    return Action(
        edge_id=estimate.edge_id,
        source=estimate.source,
        target=estimate.target,
        power=None if estimate.power < 0 else PENCIL_POWERS[estimate.power],
    )


def _sort_key(item: LookaheadAction) -> tuple[object, ...]:
    if item.action is None:
        return (-item.expected_gain, 1)
    return (-item.expected_gain, 0, *item.action.sort_key)


class DepthKPolicy:
    """Exact expectimax truncated after ``depth`` public card events."""

    _specialized_depth_two = False

    def __init__(self, depth: int) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        self.depth = depth
        self.rng = random.SystemRandom()
        self.last_stats = SearchStats()

    def rank_actions(self, game: GameSession) -> tuple[LookaheadAction, ...]:
        if game.status != "playing" or game.pending is None:
            raise GameError("draw a card before asking the policy for an action")
        estimates, native_stats = native_lookahead(
            game._handle,
            depth=self.depth,
            specialized_depth_two=self._specialized_depth_two,
        )
        ranked = tuple(
            sorted(
                (
                    LookaheadAction(
                        action=_action(estimate),
                        immediate_reward=ImmediateReward(
                            *estimate.reward_components
                        ),
                        expected_gain=estimate.value,
                    )
                    for estimate in estimates
                ),
                key=_sort_key,
            )
        )
        self.last_stats = SearchStats(
            decision_nodes=native_stats.decision_nodes,
            chance_nodes=native_stats.chance_nodes,
            chance_outcomes=native_stats.chance_outcomes,
            cache_hits=native_stats.cache_hits,
        )
        return ranked

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
