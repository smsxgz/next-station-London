"""Executable checks for native search policies."""

from __future__ import annotations

import math

from engine_cpp import COLORS, GameSession

from .lookahead import DepthKPolicy
from .lookahead2 import Depth2Policy
from .mcts import MCTSPolicy
from .mcts_reuse import ReuseMCTSPolicy


def _action_set(game: GameSession) -> set[object]:
    return {None, *game.legal_actions()}


def _public_signature(game: GameSession) -> tuple[object, ...]:
    return (
        game.status,
        game.order,
        game.round_index,
        game.underground_count,
        game.draw_count,
        game.remaining_card_mask,
        game.pending,
        tuple(
            (game.lines[color].station_mask, game.lines[color].edge_mask)
            for color in COLORS
        ),
        game.shared_objectives,
        game.shared_objective_mask,
        tuple(sorted(game.pencil_powers.items())),
        game.used_power_mask,
        game.completed_objectives,
        game.double_section_pending,
        game.double_target_symbol,
        tuple(game.round_scores),
        game.partial_score_components(),
        game.score_summary(),
    )


def _check_lookahead(*, advanced: bool) -> None:
    game = GameSession(seed=7123, advanced=advanced)
    game.draw()
    before = game.copy_public_state()

    depth_one = DepthKPolicy(1)
    ranked_one = depth_one.rank_actions(game)
    if {item.action for item in ranked_one} != _action_set(game):
        raise AssertionError("native depth-one action set is incomplete")
    if any(
        item.expected_gain != item.immediate_reward.total for item in ranked_one
    ):
        raise AssertionError("native depth-one values are not immediate rewards")
    if depth_one.last_stats.decision_nodes != 1:
        raise AssertionError("native depth-one node accounting changed")

    depth_two = Depth2Policy()
    ranked_two = depth_two.rank_actions(game)
    if {item.action for item in ranked_two} != _action_set(game):
        raise AssertionError("native depth-two action set is incomplete")
    if depth_two.last_stats.chance_outcomes < 1:
        raise AssertionError("native depth-two search did not expand chance events")
    if _public_signature(game) != _public_signature(before):
        raise AssertionError("native lookahead mutated its root game")


def _check_mcts(*, advanced: bool) -> None:
    game = GameSession(seed=9817, advanced=advanced)
    game.draw()
    before = _public_signature(game)
    policy = MCTSPolicy(64)
    policy.rng.seed(314159)
    first = policy.rank_actions(game)
    first_stats = policy.last_stats
    policy.rng.seed(314159)
    second = policy.rank_actions(game)
    second_stats = policy.last_stats
    if first != second:
        raise AssertionError("native MCTS is not deterministic for a fixed search seed")
    if first_stats.simulations != 64 or sum(item.visits for item in first) != 64:
        raise AssertionError("native MCTS root visits do not match its budget")
    if first_stats.terminal_rollouts + first_stats.tree_terminal_hits != 64:
        raise AssertionError("native MCTS simulations do not terminate exactly once")
    if first_stats.decision_nodes != first_stats.terminal_rollouts + 1:
        raise AssertionError("native MCTS node accounting is inconsistent")
    if {item.action for item in first} != _action_set(game):
        raise AssertionError("native MCTS action set is incomplete")
    comparable_first = (
        first_stats.simulations,
        first_stats.decision_nodes,
        first_stats.tree_chance_samples,
        first_stats.rollout_chance_samples,
        first_stats.terminal_rollouts,
        first_stats.rollout_decisions,
        first_stats.tree_terminal_hits,
        first_stats.max_tree_depth,
        first_stats.mean_tree_depth,
    )
    comparable_second = (
        second_stats.simulations,
        second_stats.decision_nodes,
        second_stats.tree_chance_samples,
        second_stats.rollout_chance_samples,
        second_stats.terminal_rollouts,
        second_stats.rollout_decisions,
        second_stats.tree_terminal_hits,
        second_stats.max_tree_depth,
        second_stats.mean_tree_depth,
    )
    if comparable_first != comparable_second:
        raise AssertionError("native MCTS stats changed for a fixed search seed")
    if not math.isfinite(second_stats.elapsed_seconds):
        raise AssertionError("native MCTS returned a non-finite elapsed time")
    if _public_signature(game) != before:
        raise AssertionError("native MCTS mutated its root game")


def _check_reuse_compatibility() -> None:
    game = GameSession(seed=2468)
    game.draw()
    decision = ReuseMCTSPolicy(8).choose(game)
    if decision.action is not None and decision.action not in game.legal_actions():
        raise AssertionError("ReuseMCTS returned an illegal action")


def main() -> None:
    _check_lookahead(advanced=False)
    _check_lookahead(advanced=True)
    _check_mcts(advanced=False)
    _check_mcts(advanced=True)
    _check_reuse_compatibility()
    print("solver checks: ok")


if __name__ == "__main__":
    main()
