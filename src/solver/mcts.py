"""Closed-loop chance-sampled Monte Carlo tree search."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from time import perf_counter

from engine_cpp import Action, GameError, GameSession

from .greedy import deterministic_greedy_action
from .scoring import position_score
from .state import public_state_key, sample_public_event


DEFAULT_MCTS_SIMULATIONS = 5120
DEFAULT_MCTS_EXPLORATION = 22.5


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


_StateKey = tuple[object, ...]


@dataclass(slots=True)
class _ActionStats:
    visits: int = 0
    value_sum: float = 0.0
    value_square_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def standard_error(self) -> float:
        if self.visits < 2:
            return 0.0
        variance = (
            self.value_square_sum
            - self.value_sum * self.value_sum / self.visits
        ) / (self.visits - 1)
        return math.sqrt(max(0.0, variance) / self.visits)

    def update(self, value: float) -> None:
        self.visits += 1
        self.value_sum += value
        self.value_square_sum += value * value


@dataclass(slots=True)
class _DecisionNode:
    actions: tuple[Action | None, ...]
    action_stats: list[_ActionStats] = field(default_factory=list)
    visits: int = 0

    def __post_init__(self) -> None:
        if not self.action_stats:
            self.action_stats = [_ActionStats() for _ in self.actions]


def _new_node(game: GameSession) -> _DecisionNode:
    if game.status != "playing" or game.pending is None:
        raise GameError("a decision node requires a pending card")
    return _DecisionNode(actions=(None, *game.legal_actions()))


def _complete_greedy_rollout(
    game: GameSession,
    rng: random.Random,
    first_legal: tuple[Action, ...],
) -> tuple[int, int, int]:
    decisions = 0
    chance_samples = 0
    legal: tuple[Action, ...] | None = first_legal
    while game.status == "playing":
        if game.pending is None:
            sample_public_event(game, rng)
            chance_samples += 1
        action = deterministic_greedy_action(game, legal)
        legal = None
        game.apply_legal_action(action)
        decisions += 1

    final = game.final_score()
    if final is None:
        raise RuntimeError("MCTS rollout ended before final scoring")
    return final.total, decisions, chance_samples


def _select_action_index(
    node: _DecisionNode,
    exploration: float,
    rng: random.Random,
) -> int:
    unvisited = [
        index
        for index, stats in enumerate(node.action_stats)
        if stats.visits == 0
    ]
    if unvisited:
        return rng.choice(unvisited)

    log_visits = math.log(node.visits)
    values = tuple(
        stats.mean
        + exploration * math.sqrt(log_visits / stats.visits)
        for stats in node.action_stats
    )
    best = max(values)
    tied = tuple(
        index
        for index, value in enumerate(values)
        if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)
    )
    return rng.choice(tied)


def _estimate_sort_key(
    estimate: MCTSActionEstimate,
) -> tuple[object, ...]:
    action = estimate.action
    if action is None:
        return (-estimate.visits, -estimate.mean_gain, 1, 0, 0, 0)
    return (
        -estimate.visits,
        -estimate.mean_gain,
        0,
        *action.sort_key,
    )


class MCTSPolicy:
    """UCT over public decision states with deterministic Greedy rollouts."""

    def __init__(
        self,
        simulations: int = DEFAULT_MCTS_SIMULATIONS,
        *,
        exploration: float = DEFAULT_MCTS_EXPLORATION,
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
        self.simulations = simulations
        self.exploration = float(exploration)
        self.rng = random.Random()
        self.tie_rng = random.SystemRandom()
        self.last_stats = MCTSSearchStats()

    def rank_actions(self, game: GameSession) -> tuple[MCTSActionEstimate, ...]:
        if game.status != "playing" or game.pending is None:
            raise GameError("draw a card before asking the policy for an action")

        started = perf_counter()
        baseline = position_score(game).total
        root = _new_node(game)
        table: dict[_StateKey, _DecisionNode] = {public_state_key(game): root}

        tree_chance_samples = 0
        rollout_chance_samples = 0
        terminal_rollouts = 0
        rollout_decisions = 0
        tree_terminal_hits = 0
        max_tree_depth = 0
        tree_depth_sum = 0

        for _ in range(self.simulations):
            simulation = game.copy_public_state()
            node = root
            path: list[tuple[_DecisionNode, _ActionStats]] = []
            tree_depth = 0

            while True:
                action_index = _select_action_index(
                    node,
                    self.exploration,
                    self.rng,
                )
                action_stats = node.action_stats[action_index]
                action = node.actions[action_index]
                path.append((node, action_stats))
                tree_depth += 1

                simulation.apply_legal_action(action)
                if simulation.status == "finished":
                    final = simulation.final_score()
                    if final is None:
                        raise RuntimeError("finished tree state has no final score")
                    terminal_score = final.total
                    tree_terminal_hits += 1
                    break

                if simulation.pending is None:
                    sample_public_event(simulation, self.rng)
                    tree_chance_samples += 1
                child_key = public_state_key(simulation)
                child = table.get(child_key)
                if child is None:
                    child = _new_node(simulation)
                    table[child_key] = child
                    terminal_score, rollout_steps, rollout_draws = (
                        _complete_greedy_rollout(
                            simulation,
                            self.rng,
                            tuple(
                                action
                                for action in child.actions[1:]
                                if action is not None
                            ),
                        )
                    )
                    terminal_rollouts += 1
                    rollout_decisions += rollout_steps
                    rollout_chance_samples += rollout_draws
                    break
                node = child

            gain = float(terminal_score - baseline)
            for visited_node, visited_action in path:
                visited_node.visits += 1
                visited_action.update(gain)
            tree_depth_sum += tree_depth
            max_tree_depth = max(max_tree_depth, tree_depth)

        if root.visits != self.simulations:
            raise RuntimeError("root visit count does not match simulation budget")

        estimates = tuple(
            sorted(
                (
                    MCTSActionEstimate(
                        action=action,
                        visits=stats.visits,
                        mean_gain=stats.mean,
                        standard_error=stats.standard_error,
                    )
                    for action, stats in zip(root.actions, root.action_stats)
                ),
                key=_estimate_sort_key,
            )
        )
        self.last_stats = MCTSSearchStats(
            simulations=self.simulations,
            decision_nodes=len(table),
            tree_chance_samples=tree_chance_samples,
            rollout_chance_samples=rollout_chance_samples,
            terminal_rollouts=terminal_rollouts,
            rollout_decisions=rollout_decisions,
            tree_terminal_hits=tree_terminal_hits,
            max_tree_depth=max_tree_depth,
            mean_tree_depth=tree_depth_sum / self.simulations,
            elapsed_seconds=perf_counter() - started,
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
