"""Selective exact depth-3 search with a bounded round-end option."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch

from engine import Action, GameSession

from .dqn import ActionValueNetwork, DQNPolicy
from .exact_chance import (
    ExactChanceBudgetExceeded,
    ExactChanceDecision,
    ExactChanceDQNPolicy,
)


DEFAULT_DEPTH2_GAP = 0.5
DEFAULT_MAX_ACTION_BRANCHES = 75_000
DEFAULT_ROUND_END_REMAINING = 4


@dataclass(frozen=True, slots=True)
class SelectiveDepth3Decision:
    action: Action | None
    action_index: int
    expected_gain: float
    selected_search: str
    trigger_reasons: tuple[str, ...]
    depth1_action_index: int
    depth2_action_index: int
    depth2_gap: float | None
    selected_gap: float | None
    action_changed: bool
    budget_exceeded: bool
    action_branches: int
    unique_leaf_states: int
    search_seconds: float


@dataclass(frozen=True, slots=True)
class SelectiveDepth3BatchStats:
    games: int = 0
    triggered: int = 0
    depth3_attempts: int = 0
    depth3_completed: int = 0
    round_end_attempts: int = 0
    round_end_completed: int = 0
    budget_exceeded: int = 0
    action_changes: int = 0
    action_branches: int = 0
    unique_leaf_states: int = 0
    search_seconds: float = 0.0


def _decision_gap(decision: ExactChanceDecision) -> float | None:
    if len(decision.estimates) < 2:
        return None
    return float(
        decision.estimates[0].expected_gain
        - decision.estimates[1].expected_gain
    )


class SelectiveDepth3Policy:
    """Use depth-2 normally and deepen only ambiguous public decisions."""

    def __init__(
        self,
        network: ActionValueNetwork,
        *,
        reward_scale: float,
        device: str = "auto",
        leaf_batch_size: int = 8192,
        depth2_gap: float = DEFAULT_DEPTH2_GAP,
        max_action_branches: int = DEFAULT_MAX_ACTION_BRANCHES,
        round_end_remaining: int | None = DEFAULT_ROUND_END_REMAINING,
        use_depth_triggers: bool = True,
    ) -> None:
        if not math.isfinite(depth2_gap) or depth2_gap < 0.0:
            raise ValueError("depth2_gap must be finite and non-negative")
        if max_action_branches < 1:
            raise ValueError("max_action_branches must be positive")
        if round_end_remaining is not None and round_end_remaining < 0:
            raise ValueError("round_end_remaining must be non-negative")
        self.depth2_gap = float(depth2_gap)
        self.max_action_branches = max_action_branches
        self.round_end_remaining = round_end_remaining
        self.use_depth_triggers = use_depth_triggers
        common = {
            "reward_scale": reward_scale,
            "device": device,
            "leaf_batch_size": leaf_batch_size,
        }
        self.depth1 = ExactChanceDQNPolicy(network, 1, **common)
        self.depth2 = ExactChanceDQNPolicy(network, 2, **common)
        self.depth3 = ExactChanceDQNPolicy(
            network,
            3,
            max_action_branches=max_action_branches,
            **common,
        )
        self.round_end = ExactChanceDQNPolicy(
            network,
            1,
            round_end=True,
            max_action_branches=max_action_branches,
            **common,
        )
        self.last_batch_stats = SelectiveDepth3BatchStats()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        leaf_batch_size: int = 8192,
        depth2_gap: float = DEFAULT_DEPTH2_GAP,
        max_action_branches: int = DEFAULT_MAX_ACTION_BRANCHES,
        round_end_remaining: int | None = DEFAULT_ROUND_END_REMAINING,
        use_depth_triggers: bool = True,
    ) -> SelectiveDepth3Policy:
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        reward_scale = float(
            checkpoint.get("config", {}).get("reward_scale", 1.0)
        )
        dqn = DQNPolicy.from_checkpoint(path, device=device)
        return cls(
            dqn.network,
            reward_scale=reward_scale,
            device=str(dqn.device),
            leaf_batch_size=leaf_batch_size,
            depth2_gap=depth2_gap,
            max_action_branches=max_action_branches,
            round_end_remaining=round_end_remaining,
            use_depth_triggers=use_depth_triggers,
        )

    def choose_many(
        self,
        games: tuple[GameSession, ...] | list[GameSession],
    ) -> tuple[SelectiveDepth3Decision, ...]:
        roots = list(games)
        if not roots:
            raise ValueError("choose_many requires at least one game")
        depth1_decisions = self.depth1.choose_many(roots)
        depth2_decisions = self.depth2.choose_many(roots)

        decisions: list[SelectiveDepth3Decision] = []
        triggered = depth3_attempts = depth3_completed = 0
        round_end_attempts = round_end_completed = 0
        exceeded = changes = branches = leaves = 0
        search_seconds = 0.0

        for game, decision1, decision2 in zip(
            roots,
            depth1_decisions,
            depth2_decisions,
        ):
            gap2 = _decision_gap(decision2)
            reasons: list[str] = []
            if self.use_depth_triggers:
                if decision1.action_index != decision2.action_index:
                    reasons.append("depth-disagreement")
                if (
                    self.depth2_gap > 0.0
                    and gap2 is not None
                    and gap2 <= self.depth2_gap
                ):
                    reasons.append("small-gap")

            pending = game.pending
            can_search_round_end = (
                self.round_end_remaining is not None
                and pending is not None
                and not pending.final_card
                and len(game.remaining_card_ids()) <= self.round_end_remaining
            )
            if can_search_round_end:
                reasons.append("round-end")

            selected = decision2
            selected_search = "depth-2"
            budget_exceeded = False
            selected_branches = selected_leaves = 0
            selected_seconds = 0.0
            if reasons:
                triggered += 1
                if can_search_round_end:
                    round_end_attempts += 1
                    try:
                        selected = self.round_end.choose(game)
                    except ExactChanceBudgetExceeded:
                        budget_exceeded = True
                    else:
                        selected_search = "round-end"
                        round_end_completed += 1
                        stats = self.round_end.last_batch_stats
                        selected_branches = stats.action_branches
                        selected_leaves = stats.unique_leaf_states
                        selected_seconds = (
                            stats.build_seconds
                            + stats.inference_seconds
                            + stats.evaluation_seconds
                        )

                if selected_search == "depth-2" and self.use_depth_triggers:
                    depth3_attempts += 1
                    try:
                        selected = self.depth3.choose(game)
                    except ExactChanceBudgetExceeded:
                        budget_exceeded = True
                        selected = decision2
                    else:
                        selected_search = "depth-3"
                        depth3_completed += 1
                        stats = self.depth3.last_batch_stats
                        selected_branches = stats.action_branches
                        selected_leaves = stats.unique_leaf_states
                        selected_seconds = (
                            stats.build_seconds
                            + stats.inference_seconds
                            + stats.evaluation_seconds
                        )

            changed = selected.action_index != decision2.action_index
            changes += int(changed)
            exceeded += int(budget_exceeded)
            branches += selected_branches
            leaves += selected_leaves
            search_seconds += selected_seconds
            decisions.append(
                SelectiveDepth3Decision(
                    action=selected.action,
                    action_index=selected.action_index,
                    expected_gain=selected.expected_gain,
                    selected_search=selected_search,
                    trigger_reasons=tuple(reasons),
                    depth1_action_index=decision1.action_index,
                    depth2_action_index=decision2.action_index,
                    depth2_gap=gap2,
                    selected_gap=_decision_gap(selected),
                    action_changed=changed,
                    budget_exceeded=budget_exceeded,
                    action_branches=selected_branches,
                    unique_leaf_states=selected_leaves,
                    search_seconds=selected_seconds,
                )
            )

        self.last_batch_stats = SelectiveDepth3BatchStats(
            games=len(roots),
            triggered=triggered,
            depth3_attempts=depth3_attempts,
            depth3_completed=depth3_completed,
            round_end_attempts=round_end_attempts,
            round_end_completed=round_end_completed,
            budget_exceeded=exceeded,
            action_changes=changes,
            action_branches=branches,
            unique_leaf_states=leaves,
            search_seconds=search_seconds,
        )
        return tuple(decisions)

    def choose(self, game: GameSession) -> SelectiveDepth3Decision:
        return self.choose_many((game,))[0]


__all__ = [
    "DEFAULT_DEPTH2_GAP",
    "DEFAULT_MAX_ACTION_BRANCHES",
    "DEFAULT_ROUND_END_REMAINING",
    "SelectiveDepth3BatchStats",
    "SelectiveDepth3Decision",
    "SelectiveDepth3Policy",
]
