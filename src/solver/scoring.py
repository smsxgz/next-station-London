"""Dense, final-score-equivalent rewards for solver policies."""

from __future__ import annotations

from dataclasses import dataclass, field

from engine_cpp import (
    Action,
    GameError,
    GameSession,
)


@dataclass(frozen=True, slots=True)
class PositionScore:
    """Final scoring applied to every line in its current partial form."""

    route: int
    thames: int
    line_total: int
    tourist_visits: int
    tourist: int
    interchange: int
    objective: int
    total: int


@dataclass(frozen=True, slots=True)
class ImmediateReward:
    """The score gained by drawing one legal section."""

    route: int = 0
    thames: int = 0
    tourist: int = 0
    interchange: int = 0
    objective: int = 0

    @property
    def total(self) -> int:
        return (
            self.route
            + self.thames
            + self.tourist
            + self.interchange
            + self.objective
        )


@dataclass(frozen=True, slots=True)
class _RewardContext:
    action_rewards: dict[Action, ImmediateReward] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


def _reward_context(game: GameSession) -> _RewardContext:
    # The engine owns all derived scoring state.  The solver context only
    # memoizes deltas shared by equivalent chance outcomes.
    return _RewardContext()


def position_score(game: GameSession) -> PositionScore:
    """Score the current network as though all four partial lines were final.

    Future colors already contain their compulsory departure station.  Counting
    them here makes an action that reaches a future departure receive its
    guaranteed interchange value immediately.
    """

    (
        route,
        thames,
        tourist_visits,
        tourist,
        interchange,
        objective,
    ) = game.partial_score_components()

    line_total = route + thames
    return PositionScore(
        route=route,
        thames=thames,
        line_total=line_total,
        tourist_visits=tourist_visits,
        tourist=tourist,
        interchange=interchange,
        objective=objective,
        total=line_total + tourist + interchange + objective,
    )


def _reward_for_legal_action(
    game: GameSession,
    action: Action,
    context: _RewardContext | None = None,
) -> ImmediateReward:
    if context is None:
        context = _reward_context(game)
    cached = context.action_rewards.get(action)
    if cached is not None:
        return cached
    route, thames, tourist, interchange, objective = (
        game.score_delta_for_legal_action(action)
    )
    reward = ImmediateReward(
        route=route,
        thames=thames,
        tourist=tourist,
        interchange=interchange,
        objective=objective,
    )
    context.action_rewards[action] = reward
    return reward


def immediate_reward(game: GameSession, action: Action) -> ImmediateReward:
    """Return the exact final-score increment for one currently legal action."""

    if action not in game.legal_actions():
        raise GameError("the action must be legal for the pending card")
    return _reward_for_legal_action(game, action, _reward_context(game))
