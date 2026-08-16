"""A one-step greedy policy for Next Station: London."""

from __future__ import annotations

from dataclasses import dataclass
import random

from engine_cpp import Action, GameError, GameSession

from .scoring import (
    ImmediateReward,
    _reward_context,
    _reward_for_legal_action,
)


@dataclass(frozen=True, slots=True)
class ScoredAction:
    action: Action
    reward: ImmediateReward


@dataclass(frozen=True, slots=True)
class GreedyDecision:
    action: Action | None
    reward: ImmediateReward
    tied_actions: int


class GreedyPolicy:
    """Maximize the exact score increment of the current action.

    Equal-valued best actions are selected uniformly with system randomness.
    The policy does not accept a seed, so game seeds only control the game.
    """

    def __init__(self) -> None:
        self.rng = random.SystemRandom()

    def rank_actions(self, game: GameSession) -> tuple[ScoredAction, ...]:
        if game.pending is None:
            raise GameError("draw a card before asking the policy for an action")
        actions = game.legal_actions()
        context = _reward_context(game) if actions else None
        scored = tuple(
            ScoredAction(action, _reward_for_legal_action(game, action, context))
            for action in actions
        )
        return tuple(
            sorted(
                scored,
                key=lambda item: (
                    -item.reward.total,
                    item.action.sort_key,
                ),
            )
        )

    def choose(self, game: GameSession) -> GreedyDecision:
        ranked = self.rank_actions(game)
        if not ranked:
            return GreedyDecision(None, ImmediateReward(), 0)

        best_value = ranked[0].reward.total
        tied = tuple(item for item in ranked if item.reward.total == best_value)
        chosen = self.rng.choice(tied)
        return GreedyDecision(chosen.action, chosen.reward, len(tied))


def deterministic_greedy_action(
    game: GameSession,
    legal: tuple[Action, ...] | None = None,
) -> Action | None:
    """Choose Greedy's best action with a stable tie order for rollouts."""

    if legal is None:
        legal = game.legal_actions()
    if not legal:
        return None

    context = _reward_context(game)
    best_action: Action | None = None
    best_value = 0
    best_order: tuple[int, ...] | None = None
    for action in legal:
        value = _reward_for_legal_action(game, action, context).total
        order = action.sort_key
        if value > best_value or (
            value == best_value
            and (best_order is None or order < best_order)
        ):
            best_action = action
            best_value = value
            best_order = order
    return best_action
