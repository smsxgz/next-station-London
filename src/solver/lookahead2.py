"""Two-card expectimax specialized for the common Lookahead-2 policy."""

from __future__ import annotations

import math

from engine import Action, GameSession, PENCIL_POWER_DOUBLE

from .lookahead import DepthKPolicy
from .scoring import ImmediateReward, _RewardContext, _reward_context
from .state import public_event_successors


class Depth2Policy(DepthKPolicy):
    """Exact depth-two expectimax with a specialized final-card expansion."""

    def __init__(self) -> None:
        super().__init__(2)

    def _final_card_action_value(
        self,
        game: GameSession,
        action: Action | None,
        context: _RewardContext | None,
    ) -> float:
        immediate = float(self._reward(game, action, context).total)
        if (
            action is None
            or game.active_power != PENCIL_POWER_DOUBLE
            or game.power_used(PENCIL_POWER_DOUBLE)
        ):
            return immediate

        child = self._apply(game, action)
        if child.status == "finished" or child.pending is None:
            return immediate
        return immediate + self._decision_value(child, 1)

    def _root_action_value(
        self,
        game: GameSession,
        action: Action | None,
        immediate: ImmediateReward,
        context: _RewardContext | None,
    ) -> float:
        immediate_value = float(immediate.total)
        child = self._apply(game, action)
        if child.status == "finished":
            return immediate_value
        if child.pending is not None:
            return immediate_value + self._decision_value(child, 2)

        self._chance_nodes += 1
        expected = 0.0
        probability_sum = 0.0
        reward_context = _reward_context(child)
        for probability, decision in public_event_successors(child):
            self._chance_outcomes += 1
            self._decision_nodes += 1
            probability_sum += probability
            actions = self._candidate_actions(decision)
            best = max(
                self._final_card_action_value(
                    decision,
                    next_action,
                    reward_context,
                )
                for next_action in actions
            )
            expected += probability * best
        if not math.isclose(probability_sum, 1.0, abs_tol=1e-12):
            raise RuntimeError("chance probabilities do not sum to one")
        return immediate_value + expected
