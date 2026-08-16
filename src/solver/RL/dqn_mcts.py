"""Chance-sampled UCT with batched deterministic Double DQN rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import random
from time import perf_counter

from engine_cpp import Action, GameError, GameSession

from ..mcts import DEFAULT_MCTS_EXPLORATION
from ..scoring import position_score
from ..state import public_state_key, sample_public_event
from .codec import PASS_ACTION_INDEX, encode_decision_into
from .dqn import (
    ActionValueNetwork,
    DQNPolicy,
    GreedyBatchEvaluator,
    QNetwork,
    resolve_device,
)


DEFAULT_DQN_MCTS_SIMULATIONS = 800


@dataclass(frozen=True, slots=True)
class DQNMCTSActionEstimate:
    action: Action | None
    visits: int
    mean_gain: float
    standard_error: float


@dataclass(frozen=True, slots=True)
class DQNMCTSSearchStats:
    simulations: int
    decision_nodes: int
    tree_nodes: int
    reuse_attempted: bool
    reuse_hit: bool
    retained_nodes: int
    pruned_nodes: int
    retained_root_visits: int
    tree_chance_samples: int
    rollout_chance_samples: int
    terminal_rollouts: int
    rollout_decisions: int
    tree_terminal_hits: int
    max_tree_depth: int
    mean_tree_depth: float
    forced_rollout_decisions: int
    dqn_network_evaluations: int
    inference_batches_participated: int
    mean_inference_batch_size: float
    max_inference_batch_size: int
    root_prepare_seconds: float
    subtree_prune_seconds: float


@dataclass(frozen=True, slots=True)
class DQNMCTSBatchStats:
    searches: int = 0
    completed_searches: int = 0
    simulations: int = 0
    rollout_decisions: int = 0
    forced_rollout_decisions: int = 0
    network_evaluations: int = 0
    inference_batches: int = 0
    mean_inference_batch_size: float = 0.0
    max_inference_batch_size: int = 0
    state_copies: int = 0
    state_keys: int = 0
    tree_phase_seconds: float = 0.0
    state_copy_seconds: float = 0.0
    state_key_seconds: float = 0.0
    encoding_seconds: float = 0.0
    inference_seconds: float = 0.0
    rollout_step_seconds: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class DQNMCTSDecision:
    action: Action | None
    visits: int
    mean_gain: float
    standard_error: float
    tied_actions: int
    estimates: tuple[DQNMCTSActionEstimate, ...]
    stats: DQNMCTSSearchStats


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


@dataclass(slots=True, eq=False)
class DQNMCTSSession:
    """Solver-owned tree state retained across one real game."""

    _rng: random.Random
    _baseline: int | None = None
    _table: dict[_StateKey, _DecisionNode] = field(default_factory=dict)
    _active: bool = False


@dataclass(slots=True, eq=False)
class DQNMCTSSearch:
    """One resumable real-decision search within a retained session."""

    session: DQNMCTSSession
    root_game: GameSession
    root_key: _StateKey
    root: _DecisionNode
    baseline: int
    report_offset: float
    simulation_budget: int
    initial_root_visits: int
    reuse_attempted: bool
    reuse_hit: bool
    retained_nodes: int
    pruned_nodes: int
    retained_root_visits: int
    root_prepare_seconds: float
    subtree_prune_seconds: float
    completed_simulations: int = 0
    decision_nodes: int = 0
    tree_chance_samples: int = 0
    rollout_chance_samples: int = 0
    terminal_rollouts: int = 0
    rollout_decisions: int = 0
    tree_terminal_hits: int = 0
    max_tree_depth: int = 0
    tree_depth_sum: int = 0
    forced_rollout_decisions: int = 0
    dqn_network_evaluations: int = 0
    inference_batches_participated: int = 0
    inference_batch_size_sum: int = 0
    max_inference_batch_size: int = 0
    pending: _PendingRollout | None = None
    finalized: bool = False

    def complete(
        self,
        path: list[tuple[_DecisionNode, _ActionStats]],
        tree_depth: int,
        terminal_score: int,
    ) -> None:
        gain = float(terminal_score - self.baseline)
        for node, action_stats in path:
            node.visits += 1
            action_stats.update(gain)
        self.completed_simulations += 1
        self.tree_depth_sum += tree_depth
        self.max_tree_depth = max(self.max_tree_depth, tree_depth)


@dataclass(slots=True)
class _PendingRollout:
    search: DQNMCTSSearch
    game: GameSession
    path: list[tuple[_DecisionNode, _ActionStats]]
    tree_depth: int


@dataclass(slots=True)
class _BatchCounters:
    rollout_decisions: int = 0
    forced_rollout_decisions: int = 0
    network_evaluations: int = 0
    inference_batches: int = 0
    inference_batch_size_sum: int = 0
    max_inference_batch_size: int = 0
    state_copies: int = 0
    state_keys: int = 0
    tree_phase_seconds: float = 0.0
    state_copy_seconds: float = 0.0
    state_key_seconds: float = 0.0
    encoding_seconds: float = 0.0
    inference_seconds: float = 0.0
    rollout_step_seconds: float = 0.0


def _new_node(game: GameSession) -> _DecisionNode:
    if game.status != "playing" or game.pending is None:
        raise GameError("a decision node requires a pending card")
    return _DecisionNode(actions=(None, *game.legal_actions()))


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
    estimate: DQNMCTSActionEstimate,
) -> tuple[float, float, int, int, int, int]:
    action = estimate.action
    if action is None:
        return (-estimate.visits, -estimate.mean_gain, 1, 0, 0, 0)
    return (
        -estimate.visits,
        -estimate.mean_gain,
        0,
        action.edge_id,
        action.source,
        action.target,
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


class DQNMCTSPolicy:
    """UCT with one in-flight simulation per tree and batched DQN rollouts."""

    def __init__(
        self,
        network: ActionValueNetwork,
        simulations: int = DEFAULT_DQN_MCTS_SIMULATIONS,
        *,
        exploration: float = DEFAULT_MCTS_EXPLORATION,
        device: str = "auto",
        profile: bool = False,
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
        self.device = resolve_device(device)
        self.network = network.to(self.device)
        self.network.eval()
        self.simulations = simulations
        self.exploration = float(exploration)
        self.profile = profile
        self.rng = random.Random()
        self.tie_rng = random.SystemRandom()
        self._evaluator = GreedyBatchEvaluator(self.network, self.device)
        self.last_batch_stats = DQNMCTSBatchStats()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        simulations: int = DEFAULT_DQN_MCTS_SIMULATIONS,
        *,
        exploration: float = DEFAULT_MCTS_EXPLORATION,
        device: str = "auto",
        profile: bool = False,
    ) -> DQNMCTSPolicy:
        policy = DQNPolicy.from_checkpoint(checkpoint_path, device=device)
        if not isinstance(policy.network, QNetwork):
            raise ValueError("DQN-MCTS requires a scalar Double DQN checkpoint")
        return cls(
            policy.network,
            simulations,
            exploration=exploration,
            device=str(policy.device),
            profile=profile,
        )

    def new_session(self) -> DQNMCTSSession:
        return DQNMCTSSession(
            _rng=random.Random(self.rng.getrandbits(128)),
        )

    def start_search(
        self,
        session: DQNMCTSSession,
        game: GameSession,
    ) -> DQNMCTSSearch:
        """Start a decision, re-rooting a retained public subtree when possible."""

        if session._active:
            raise RuntimeError("the session already has an active search")
        if game.status != "playing" or game.pending is None:
            raise GameError("draw a card before asking the policy for an action")

        prepare_started = perf_counter()
        root_game = game.copy_public_state()
        root_key = public_state_key(root_game)
        reuse_attempted = bool(session._table)
        reuse_hit = reuse_attempted and root_key in session._table
        retained_nodes = 0
        pruned_nodes = 0
        retained_root_visits = 0
        subtree_prune_seconds = 0.0

        if reuse_hit:
            old_count = len(session._table)
            prune_started = perf_counter()
            session._table = _retained_subtree(session._table, root_key)
            subtree_prune_seconds = perf_counter() - prune_started
            retained_nodes = len(session._table)
            pruned_nodes = old_count - retained_nodes
            root = session._table[root_key]
            retained_root_visits = root.visits
            if session._baseline is None:
                raise RuntimeError("a retained tree has no score baseline")
            baseline = session._baseline
            decision_nodes = 0
        else:
            pruned_nodes = len(session._table)
            root = _new_node(root_game)
            baseline = position_score(root_game).total
            session._baseline = baseline
            session._table = {root_key: root}
            decision_nodes = 1

        current_score = position_score(root_game).total
        search = DQNMCTSSearch(
            session=session,
            root_game=root_game,
            root_key=root_key,
            root=root,
            baseline=baseline,
            report_offset=float(current_score - baseline),
            simulation_budget=self.simulations,
            initial_root_visits=root.visits,
            reuse_attempted=reuse_attempted,
            reuse_hit=reuse_hit,
            retained_nodes=retained_nodes,
            pruned_nodes=pruned_nodes,
            retained_root_visits=retained_root_visits,
            root_prepare_seconds=perf_counter() - prepare_started,
            subtree_prune_seconds=subtree_prune_seconds,
            decision_nodes=decision_nodes,
        )
        session._active = True
        return search

    def _select_until_rollout(
        self,
        search: DQNMCTSSearch,
        counters: _BatchCounters,
    ) -> _PendingRollout | None:
        while search.completed_simulations < search.simulation_budget:
            counters.state_copies += 1
            if self.profile:
                copy_started = perf_counter()
                simulation = search.root_game.copy_public_state()
                counters.state_copy_seconds += perf_counter() - copy_started
            else:
                simulation = search.root_game.copy_public_state()
            node = search.root
            path: list[tuple[_DecisionNode, _ActionStats]] = []
            tree_depth = 0

            while True:
                action_index = _select_action_index(
                    node,
                    self.exploration,
                    search.session._rng,
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
                    search.tree_terminal_hits += 1
                    search.complete(path, tree_depth, final.total)
                    break

                sample_public_event(simulation, search.session._rng)
                search.tree_chance_samples += 1
                counters.state_keys += 1
                if self.profile:
                    key_started = perf_counter()
                    child_key = public_state_key(simulation)
                    counters.state_key_seconds += perf_counter() - key_started
                else:
                    child_key = public_state_key(simulation)
                action_stats.add_child(child_key)
                child = search.session._table.get(child_key)
                if child is None:
                    child = _new_node(simulation)
                    search.session._table[child_key] = child
                    search.decision_nodes += 1
                    search.terminal_rollouts += 1
                    return _PendingRollout(
                        search=search,
                        game=simulation,
                        path=path,
                        tree_depth=tree_depth,
                    )
                node = child

        return None

    def _resolve_rollout_actions(
        self,
        active: list[_PendingRollout],
        counters: _BatchCounters,
    ) -> list[tuple[_PendingRollout, Action | None]]:
        observations, masks = self._evaluator.buffers(len(active))
        resolved: list[tuple[_PendingRollout, Action | None]] = []
        inference: list[tuple[_PendingRollout, dict[int, Action]]] = []

        encode_started = perf_counter() if self.profile else 0.0
        for rollout in active:
            row = len(inference)
            actions = encode_decision_into(
                rollout.game,
                observations[row],
                masks[row],
            )
            if not actions:
                rollout.search.forced_rollout_decisions += 1
                counters.forced_rollout_decisions += 1
                resolved.append((rollout, None))
                continue
            inference.append((rollout, actions))
        if self.profile:
            counters.encoding_seconds += perf_counter() - encode_started

        if not inference:
            return resolved

        inference_started = perf_counter()
        action_indices = self._evaluator.select(len(inference))
        counters.inference_seconds += perf_counter() - inference_started

        batch_size = len(inference)
        counters.inference_batches += 1
        counters.inference_batch_size_sum += batch_size
        counters.max_inference_batch_size = max(
            counters.max_inference_batch_size,
            batch_size,
        )
        counters.network_evaluations += batch_size

        for rollout, _ in inference:
            search = rollout.search
            search.dqn_network_evaluations += 1
            search.inference_batches_participated += 1
            search.inference_batch_size_sum += batch_size
            search.max_inference_batch_size = max(
                search.max_inference_batch_size,
                batch_size,
            )

        for (rollout, actions), raw_index in zip(inference, action_indices):
            action_index = int(raw_index)
            action = (
                None
                if action_index == PASS_ACTION_INDEX
                else actions[action_index]
            )
            resolved.append((rollout, action))
        return resolved

    def _advance_rollouts(
        self,
        active: list[_PendingRollout],
        counters: _BatchCounters,
    ) -> None:
        resolved = self._resolve_rollout_actions(active, counters)
        step_started = perf_counter() if self.profile else 0.0
        for rollout, action in resolved:
            search = rollout.search
            rollout.game.apply_legal_action(action)
            search.rollout_decisions += 1
            counters.rollout_decisions += 1
            if rollout.game.status == "finished":
                final = rollout.game.final_score()
                if final is None:
                    raise RuntimeError("DQN rollout ended before final scoring")
                search.complete(rollout.path, rollout.tree_depth, final.total)
                search.pending = None
                continue
            sample_public_event(rollout.game, search.session._rng)
            search.rollout_chance_samples += 1
        if self.profile:
            counters.rollout_step_seconds += perf_counter() - step_started

    @staticmethod
    def _search_stats(search: DQNMCTSSearch) -> DQNMCTSSearchStats:
        batches = search.inference_batches_participated
        return DQNMCTSSearchStats(
            simulations=search.completed_simulations,
            decision_nodes=search.decision_nodes,
            tree_nodes=len(search.session._table),
            reuse_attempted=search.reuse_attempted,
            reuse_hit=search.reuse_hit,
            retained_nodes=search.retained_nodes,
            pruned_nodes=search.pruned_nodes,
            retained_root_visits=search.retained_root_visits,
            tree_chance_samples=search.tree_chance_samples,
            rollout_chance_samples=search.rollout_chance_samples,
            terminal_rollouts=search.terminal_rollouts,
            rollout_decisions=search.rollout_decisions,
            tree_terminal_hits=search.tree_terminal_hits,
            max_tree_depth=search.max_tree_depth,
            mean_tree_depth=search.tree_depth_sum / search.completed_simulations,
            forced_rollout_decisions=search.forced_rollout_decisions,
            dqn_network_evaluations=search.dqn_network_evaluations,
            inference_batches_participated=batches,
            mean_inference_batch_size=(
                search.inference_batch_size_sum / batches if batches else 0.0
            ),
            max_inference_batch_size=search.max_inference_batch_size,
            root_prepare_seconds=search.root_prepare_seconds,
            subtree_prune_seconds=search.subtree_prune_seconds,
        )

    def _decision(self, search: DQNMCTSSearch) -> DQNMCTSDecision:
        estimates = tuple(
            sorted(
                (
                    DQNMCTSActionEstimate(
                        action=action,
                        visits=stats.visits,
                        mean_gain=stats.mean - search.report_offset,
                        standard_error=stats.standard_error,
                    )
                    for action, stats in zip(
                        search.root.actions,
                        search.root.action_stats,
                    )
                ),
                key=_estimate_sort_key,
            )
        )
        most_visits = estimates[0].visits
        tied = tuple(item for item in estimates if item.visits == most_visits)
        chosen = self.tie_rng.choice(tied)
        return DQNMCTSDecision(
            action=chosen.action,
            visits=chosen.visits,
            mean_gain=chosen.mean_gain,
            standard_error=chosen.standard_error,
            tied_actions=len(tied),
            estimates=estimates,
            stats=self._search_stats(search),
        )

    def _finish_search(self, search: DQNMCTSSearch) -> DQNMCTSDecision:
        if search.finalized:
            raise RuntimeError("search was already finalized")
        if search.pending is not None:
            raise RuntimeError("cannot finalize a search with an active rollout")
        if search.completed_simulations != search.simulation_budget:
            raise RuntimeError("search did not consume its simulation budget")
        expected_visits = search.initial_root_visits + search.simulation_budget
        if search.root.visits != expected_visits:
            raise RuntimeError(
                "root visit count does not match retained plus new visits"
            )
        classified = (
            search.forced_rollout_decisions
            + search.dqn_network_evaluations
        )
        if classified != search.rollout_decisions:
            raise RuntimeError("rollout decision accounting is inconsistent")
        search.finalized = True
        search.session._active = False
        return self._decision(search)

    def advance_many(
        self,
        searches: list[DQNMCTSSearch] | tuple[DQNMCTSSearch, ...],
    ) -> tuple[tuple[DQNMCTSSearch, DQNMCTSDecision], ...]:
        """Advance independent searches until at least one decision finishes."""

        if not searches:
            raise ValueError("advance_many requires at least one search")
        if len({id(search) for search in searches}) != len(searches):
            raise ValueError("advance_many received the same search twice")
        if any(search.finalized for search in searches):
            raise ValueError("advance_many received a finalized search")

        started = perf_counter()
        before_simulations = sum(
            search.completed_simulations for search in searches
        )
        counters = _BatchCounters()

        while True:
            completed: list[tuple[DQNMCTSSearch, DQNMCTSDecision]] = []
            for search in searches:
                if search.pending is not None:
                    continue
                if self.profile:
                    tree_started = perf_counter()
                    rollout = self._select_until_rollout(search, counters)
                    counters.tree_phase_seconds += perf_counter() - tree_started
                else:
                    rollout = self._select_until_rollout(search, counters)
                if rollout is None:
                    completed.append((search, self._finish_search(search)))
                else:
                    search.pending = rollout

            if completed:
                break

            active = [
                search.pending
                for search in searches
                if search.pending is not None
            ]
            if not active:
                raise RuntimeError("search scheduler made no progress")
            self._advance_rollouts(active, counters)

        elapsed = perf_counter() - started
        after_simulations = sum(
            search.completed_simulations for search in searches
        )
        batch_count = counters.inference_batches
        self.last_batch_stats = DQNMCTSBatchStats(
            searches=len(searches),
            completed_searches=len(completed),
            simulations=after_simulations - before_simulations,
            rollout_decisions=counters.rollout_decisions,
            forced_rollout_decisions=counters.forced_rollout_decisions,
            network_evaluations=counters.network_evaluations,
            inference_batches=batch_count,
            mean_inference_batch_size=(
                counters.inference_batch_size_sum / batch_count
                if batch_count
                else 0.0
            ),
            max_inference_batch_size=counters.max_inference_batch_size,
            state_copies=counters.state_copies,
            state_keys=counters.state_keys,
            tree_phase_seconds=counters.tree_phase_seconds,
            state_copy_seconds=counters.state_copy_seconds,
            state_key_seconds=counters.state_key_seconds,
            encoding_seconds=counters.encoding_seconds,
            inference_seconds=counters.inference_seconds,
            rollout_step_seconds=counters.rollout_step_seconds,
            elapsed_seconds=elapsed,
        )
        return tuple(completed)

    def choose_many(
        self,
        games: list[GameSession] | tuple[GameSession, ...],
    ) -> tuple[DQNMCTSDecision, ...]:
        """Compatibility API using fresh, non-retained sessions."""

        if not games:
            raise ValueError("choose_many requires at least one game")
        searches = [
            self.start_search(self.new_session(), game)
            for game in games
        ]
        started = perf_counter()
        active = list(searches)
        decisions: dict[int, DQNMCTSDecision] = {}
        batches: list[DQNMCTSBatchStats] = []
        while active:
            for search, decision in self.advance_many(tuple(active)):
                decisions[id(search)] = decision
            batches.append(self.last_batch_stats)
            active = [search for search in active if not search.finalized]
        inference_batches = sum(item.inference_batches for item in batches)
        self.last_batch_stats = DQNMCTSBatchStats(
            searches=len(searches),
            completed_searches=len(searches),
            simulations=sum(item.simulations for item in batches),
            rollout_decisions=sum(item.rollout_decisions for item in batches),
            forced_rollout_decisions=sum(
                item.forced_rollout_decisions for item in batches
            ),
            network_evaluations=sum(
                item.network_evaluations for item in batches
            ),
            inference_batches=inference_batches,
            mean_inference_batch_size=(
                sum(
                    item.mean_inference_batch_size * item.inference_batches
                    for item in batches
                )
                / inference_batches
                if inference_batches
                else 0.0
            ),
            max_inference_batch_size=max(
                (item.max_inference_batch_size for item in batches),
                default=0,
            ),
            state_copies=sum(item.state_copies for item in batches),
            state_keys=sum(item.state_keys for item in batches),
            tree_phase_seconds=sum(item.tree_phase_seconds for item in batches),
            state_copy_seconds=sum(item.state_copy_seconds for item in batches),
            state_key_seconds=sum(item.state_key_seconds for item in batches),
            encoding_seconds=sum(item.encoding_seconds for item in batches),
            inference_seconds=sum(item.inference_seconds for item in batches),
            rollout_step_seconds=sum(
                item.rollout_step_seconds for item in batches
            ),
            elapsed_seconds=perf_counter() - started,
        )
        return tuple(decisions[id(search)] for search in searches)

    def choose(self, game: GameSession) -> DQNMCTSDecision:
        return self.choose_many((game,))[0]
