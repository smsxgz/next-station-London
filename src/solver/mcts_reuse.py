"""Chance-sampled MCTS that reuses the realized subtree between moves."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from time import perf_counter

from engine import Action, GameError, GameSession

from .greedy import deterministic_greedy_action
from .mcts import (
    DEFAULT_MCTS_EXPLORATION,
    DEFAULT_MCTS_SIMULATIONS,
    MCTSActionEstimate,
)
from .scoring import position_score
from .state import public_state_key, sample_public_event


@dataclass(frozen=True, slots=True)
class ReuseMCTSSearchStats:
    simulations: int = 0
    decision_nodes: int = 0
    tree_nodes: int = 0
    reuse_attempted: bool = False
    reuse_hit: bool = False
    retained_nodes: int = 0
    pruned_nodes: int = 0
    retained_root_visits: int = 0
    tree_chance_samples: int = 0
    rollout_chance_samples: int = 0
    terminal_rollouts: int = 0
    rollout_decisions: int = 0
    tree_terminal_hits: int = 0
    max_tree_depth: int = 0
    mean_tree_depth: float = 0.0
    prune_seconds: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ReuseMCTSDecision:
    action: Action | None
    visits: int
    mean_gain: float
    standard_error: float
    tied_actions: int
    estimates: tuple[MCTSActionEstimate, ...]
    stats: ReuseMCTSSearchStats


_StateKey = tuple[int, ...]


@dataclass(slots=True)
class _ActionStats:
    visits: int = 0
    value_sum: float = 0.0
    value_square_sum: float = 0.0
    child_keys: set[_StateKey] | None = None

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

    def add_child(self, key: _StateKey) -> None:
        if self.child_keys is None:
            self.child_keys = {key}
        else:
            self.child_keys.add(key)


@dataclass(slots=True)
class _DecisionNode:
    actions: tuple[Action | None, ...]
    action_stats: list[_ActionStats] = field(default_factory=list)
    visits: int = 0

    def __post_init__(self) -> None:
        if not self.action_stats:
            self.action_stats = [_ActionStats() for _ in self.actions]


@dataclass(frozen=True, slots=True)
class _PreparedRoot:
    root: _DecisionNode
    baseline: int
    report_offset: float
    reuse_attempted: bool
    reuse_hit: bool
    retained_nodes: int
    pruned_nodes: int
    retained_root_visits: int
    decision_nodes: int
    prune_seconds: float


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


def _retained_subtree(
    table: dict[_StateKey, _DecisionNode],
    root_key: _StateKey,
) -> dict[_StateKey, _DecisionNode]:
    retained: dict[_StateKey, _DecisionNode] = {}
    pending = [root_key]
    while pending:
        key = pending.pop()
        if key in retained:
            continue
        node = table.get(key)
        if node is None:
            continue
        retained[key] = node
        for action_stats in node.action_stats:
            if action_stats.child_keys:
                pending.extend(action_stats.child_keys)

    retained_keys = set(retained)
    for node in retained.values():
        for action_stats in node.action_stats:
            if action_stats.child_keys is not None:
                action_stats.child_keys.intersection_update(retained_keys)
    return retained


class ReuseMCTSPolicy:
    """UCT with Greedy rollouts and realized-subtree reuse."""

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
        self.last_stats = ReuseMCTSSearchStats()
        self._game: GameSession | None = None
        self._baseline: int | None = None
        self._table: dict[_StateKey, _DecisionNode] = {}

    def reset(self) -> None:
        self._game = None
        self._baseline = None
        self._table.clear()

    def _prepare_root(
        self,
        game: GameSession,
    ) -> _PreparedRoot:
        if self._game is not game:
            self.reset()
            self._game = game

        root_key = public_state_key(game)
        reuse_attempted = bool(self._table)
        reuse_hit = reuse_attempted and root_key in self._table
        retained_nodes = 0
        retained_root_visits = 0
        prune_seconds = 0.0

        if reuse_hit:
            old_count = len(self._table)
            prune_started = perf_counter()
            self._table = _retained_subtree(self._table, root_key)
            prune_seconds = perf_counter() - prune_started
            retained_nodes = len(self._table)
            pruned_nodes = old_count - retained_nodes
            root = self._table[root_key]
            retained_root_visits = root.visits
            if self._baseline is None:
                raise RuntimeError("a retained tree has no score baseline")
            baseline = self._baseline
            decision_nodes = 0
        else:
            pruned_nodes = len(self._table)
            root = _new_node(game)
            baseline = position_score(game).total
            self._baseline = baseline
            self._table = {root_key: root}
            decision_nodes = 1

        report_offset = float(position_score(game).total - baseline)
        return _PreparedRoot(
            root=root,
            baseline=baseline,
            report_offset=report_offset,
            reuse_attempted=reuse_attempted,
            reuse_hit=reuse_hit,
            retained_nodes=retained_nodes,
            pruned_nodes=pruned_nodes,
            retained_root_visits=retained_root_visits,
            decision_nodes=decision_nodes,
            prune_seconds=prune_seconds,
        )

    def rank_actions(self, game: GameSession) -> tuple[MCTSActionEstimate, ...]:
        if game.status != "playing" or game.pending is None:
            raise GameError("draw a card before asking the policy for an action")

        started = perf_counter()
        prepared = self._prepare_root(game)
        root = prepared.root
        baseline = prepared.baseline
        decision_nodes = prepared.decision_nodes
        initial_root_visits = root.visits

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
                action_stats.add_child(child_key)
                child = self._table.get(child_key)
                if child is None:
                    child = _new_node(simulation)
                    self._table[child_key] = child
                    decision_nodes += 1
                    terminal_score, rollout_steps, rollout_draws = (
                        _complete_greedy_rollout(
                            simulation,
                            self.rng,
                            tuple(
                                candidate
                                for candidate in child.actions[1:]
                                if candidate is not None
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

        expected_visits = initial_root_visits + self.simulations
        if root.visits != expected_visits:
            raise RuntimeError(
                "root visit count does not match retained plus new visits"
            )

        estimates = tuple(
            sorted(
                (
                    MCTSActionEstimate(
                        action=action,
                        visits=stats.visits,
                        mean_gain=stats.mean - prepared.report_offset,
                        standard_error=stats.standard_error,
                    )
                    for action, stats in zip(root.actions, root.action_stats)
                ),
                key=_estimate_sort_key,
            )
        )
        self.last_stats = ReuseMCTSSearchStats(
            simulations=self.simulations,
            decision_nodes=decision_nodes,
            tree_nodes=len(self._table),
            reuse_attempted=prepared.reuse_attempted,
            reuse_hit=prepared.reuse_hit,
            retained_nodes=prepared.retained_nodes,
            pruned_nodes=prepared.pruned_nodes,
            retained_root_visits=prepared.retained_root_visits,
            tree_chance_samples=tree_chance_samples,
            rollout_chance_samples=rollout_chance_samples,
            terminal_rollouts=terminal_rollouts,
            rollout_decisions=rollout_decisions,
            tree_terminal_hits=tree_terminal_hits,
            max_tree_depth=max_tree_depth,
            mean_tree_depth=tree_depth_sum / self.simulations,
            prune_seconds=prepared.prune_seconds,
            elapsed_seconds=perf_counter() - started,
        )
        return estimates

    def choose(self, game: GameSession) -> ReuseMCTSDecision:
        estimates = self.rank_actions(game)
        most_visits = estimates[0].visits
        tied = tuple(item for item in estimates if item.visits == most_visits)
        chosen = self.tie_rng.choice(tied)
        return ReuseMCTSDecision(
            action=chosen.action,
            visits=chosen.visits,
            mean_gain=chosen.mean_gain,
            standard_error=chosen.standard_error,
            tied_actions=len(tied),
            estimates=estimates,
            stats=self.last_stats,
        )
