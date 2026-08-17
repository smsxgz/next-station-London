"""Closed-loop chance-sampled MCTS backed by the native solver kernel."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from engine_cpp import Action, GameError, GameSession, PENCIL_POWERS

from ._native import NativeActionEstimate, native_mcts


DEFAULT_MCTS_SIMULATIONS = 5120
DEFAULT_MCTS_EXPLORATION = 22.5
DEFAULT_MCTS_SIMPLE_RANDOM_PASS_PROBABILITY = 0.05


@dataclass(frozen=True, slots=True)
class MCTSActionEstimate:
    action: Action | None
    visits: int
    mean_gain: float
    standard_error: float


@dataclass(frozen=True, slots=True)
class MCTSSearchStats:
    simulations: int = 0
    decision_nodes: int = 0
    tree_chance_samples: int = 0
    rollout_chance_samples: int = 0
    terminal_rollouts: int = 0
    rollout_decisions: int = 0
    tree_terminal_hits: int = 0
    max_tree_depth: int = 0
    mean_tree_depth: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class MCTSDecision:
    action: Action | None
    visits: int
    mean_gain: float
    standard_error: float
    tied_actions: int
    estimates: tuple[MCTSActionEstimate, ...]
    stats: MCTSSearchStats


def _action(estimate: NativeActionEstimate) -> Action | None:
    if estimate.edge_id < 0:
        return None
    return Action(
        edge_id=estimate.edge_id,
        source=estimate.source,
        target=estimate.target,
        power=None if estimate.power < 0 else PENCIL_POWERS[estimate.power],
    )


def _estimate_sort_key(estimate: MCTSActionEstimate) -> tuple[object, ...]:
    if estimate.action is None:
        return (-estimate.visits, -estimate.mean_gain, 1, 0, 0, 0)
    return (
        -estimate.visits,
        -estimate.mean_gain,
        0,
        *estimate.action.sort_key,
    )


class MCTSPolicy:
    """UCT over public decision states with a native rollout policy."""

    def __init__(
        self,
        simulations: int = DEFAULT_MCTS_SIMULATIONS,
        *,
        exploration: float = DEFAULT_MCTS_EXPLORATION,
        rollout_policy: str = "greedy",
    ) -> None:
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, int)
            or simulations < 1
        ):
            raise ValueError("simulations must be a positive integer")
        if (
            isinstance(exploration, bool)
            or not isinstance(exploration, (int, float))
            or not math.isfinite(exploration)
            or exploration < 0.0
        ):
            raise ValueError("exploration must be a finite non-negative number")
        if rollout_policy not in {"greedy", "lookahead-2", "simple-random"}:
            raise ValueError(
                "rollout_policy must be 'greedy', 'lookahead-2', or "
                "'simple-random'"
            )
        self.simulations = simulations
        self.exploration = float(exploration)
        self.rollout_policy = rollout_policy
        self.rng = random.Random()
        self.tie_rng = random.SystemRandom()
        self.last_stats = MCTSSearchStats()

    def rank_actions(self, game: GameSession) -> tuple[MCTSActionEstimate, ...]:
        if game.status != "playing" or game.pending is None:
            raise GameError("draw a card before asking the policy for an action")
        native_estimates, native_stats = native_mcts(
            game._handle,
            simulations=self.simulations,
            exploration=self.exploration,
            seed=self.rng.getrandbits(64),
            rollout_policy=self.rollout_policy,
        )
        estimates = tuple(
            sorted(
                (
                    MCTSActionEstimate(
                        action=_action(estimate),
                        visits=estimate.visits,
                        mean_gain=estimate.value,
                        standard_error=estimate.standard_error,
                    )
                    for estimate in native_estimates
                ),
                key=_estimate_sort_key,
            )
        )
        self.last_stats = MCTSSearchStats(
            simulations=native_stats.simulations,
            decision_nodes=native_stats.decision_nodes,
            tree_chance_samples=native_stats.tree_chance_samples,
            rollout_chance_samples=native_stats.rollout_chance_samples,
            terminal_rollouts=native_stats.terminal_rollouts,
            rollout_decisions=native_stats.rollout_decisions,
            tree_terminal_hits=native_stats.tree_terminal_hits,
            max_tree_depth=native_stats.max_tree_depth,
            mean_tree_depth=native_stats.mean_tree_depth,
            elapsed_seconds=native_stats.elapsed_seconds,
        )
        return estimates

    def choose(self, game: GameSession) -> MCTSDecision:
        estimates = self.rank_actions(game)
        most_visits = estimates[0].visits
        tied = tuple(item for item in estimates if item.visits == most_visits)
        chosen = self.tie_rng.choice(tied)
        return MCTSDecision(
            action=chosen.action,
            visits=chosen.visits,
            mean_gain=chosen.mean_gain,
            standard_error=chosen.standard_error,
            tied_actions=len(tied),
            estimates=estimates,
            stats=self.last_stats,
        )
