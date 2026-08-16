"""Greedy afterstate policy and batched candidate inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from engine_cpp import Action, GameSession

from .codec import Candidate, make_candidates
from .network import AfterstateValueNetwork
from .target import CudaFeatureWorkspace, TargetStats, exact_afterstate_targets


@dataclass(frozen=True, slots=True)
class AfterstateDecision:
    action: Action | None
    action_index: int
    immediate_reward: int
    expected_gain: float
    candidate: Candidate


@dataclass(slots=True)
class BellmanImprovementStats:
    """Aggregate diagnostics collected on Bellman-improved decisions."""

    decisions: int = 0
    agreements: int = 0
    normalized_regret_sum: float = 0.0
    root_candidates: int = 0
    chance_outcomes: int = 0
    expanded_candidates: int = 0
    online_batches: int = 0
    target_batches: int = 0

    @property
    def agreement_rate(self) -> float:
        return self.agreements / self.decisions if self.decisions else 0.0

    @property
    def mean_normalized_regret(self) -> float:
        return self.normalized_regret_sum / self.decisions if self.decisions else 0.0


class AfterstatePolicy:
    """Select actions using exact immediate reward plus scalar W."""

    def __init__(
        self,
        network: AfterstateValueNetwork,
        *,
        device: torch.device,
        reward_scale: float = 10.0,
    ) -> None:
        if reward_scale <= 0.0 or not np.isfinite(reward_scale):
            raise ValueError("reward_scale must be finite and positive")
        self.network = network.to(device)
        self.network.eval()
        self.device = device
        self.reward_scale = float(reward_scale)

    @torch.inference_mode()
    def select_groups(
        self,
        candidate_groups: Sequence[Sequence[Candidate]],
    ) -> tuple[AfterstateDecision, ...]:
        if not candidate_groups:
            raise ValueError("select_groups requires at least one group")
        if any(not group for group in candidate_groups):
            raise ValueError("candidate groups cannot be empty")
        all_candidates = [
            candidate for group in candidate_groups for candidate in group
        ]
        features = torch.as_tensor(
            np.stack([candidate.features for candidate in all_candidates]),
            device=self.device,
            dtype=torch.float32,
        )
        values = self.network.expected_value(features).cpu().numpy()
        for index, candidate in enumerate(all_candidates):
            if candidate.record.terminated:
                values[index] = 0.0
        decisions: list[AfterstateDecision] = []
        cursor = 0
        for group in candidate_groups:
            count = len(group)
            group_values = values[cursor : cursor + count]
            cursor += count
            scores = np.asarray(
                [candidate.reward / self.reward_scale for candidate in group],
                dtype=np.float64,
            ) + group_values.astype(np.float64, copy=False)
            # make_candidates is sorted by action index; np.argmax keeps the
            # first exact tie, matching the fixed deterministic tie rule.
            selected_position = int(np.argmax(scores))
            selected = group[selected_position]
            decisions.append(
                AfterstateDecision(
                    action=selected.action,
                    action_index=selected.action_index,
                    immediate_reward=selected.reward,
                    expected_gain=float(scores[selected_position]),
                    candidate=selected,
                )
            )
        return tuple(decisions)

    def choose_many(
        self,
        games: Sequence[GameSession],
    ) -> tuple[AfterstateDecision, ...]:
        if not games:
            raise ValueError("choose_many requires at least one game")
        return self.select_groups(tuple(make_candidates(game) for game in games))

    def choose(self, game: GameSession) -> AfterstateDecision:
        return self.choose_many((game,))[0]


class BellmanImprovedAfterstatePolicy:
    """Select actions after one exact chance/action Bellman backup."""

    def __init__(
        self,
        online: AfterstateValueNetwork,
        target: AfterstateValueNetwork,
        *,
        device: torch.device,
        reward_scale: float = 10.0,
        gamma: float = 1.0,
        inference_batch_size: int = 8192,
    ) -> None:
        if reward_scale <= 0.0 or not np.isfinite(reward_scale):
            raise ValueError("reward_scale must be finite and positive")
        if not 0.0 < gamma <= 1.0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and in (0, 1]")
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        self.online = online.to(device)
        self.target = target.to(device)
        self.online.eval()
        self.target.eval()
        self.device = device
        self.reward_scale = float(reward_scale)
        self.gamma = float(gamma)
        self.inference_batch_size = int(inference_batch_size)
        self.feature_workspace = (
            CudaFeatureWorkspace(device) if device.type == "cuda" else None
        )
        self.stats = BellmanImprovementStats()

    @torch.inference_mode()
    def select_groups(
        self,
        candidate_groups: Sequence[Sequence[Candidate]],
    ) -> tuple[AfterstateDecision, ...]:
        if not candidate_groups:
            raise ValueError("select_groups requires at least one group")
        if any(not group for group in candidate_groups):
            raise ValueError("candidate groups cannot be empty")

        all_candidates = [
            candidate for group in candidate_groups for candidate in group
        ]
        features = torch.as_tensor(
            np.stack([candidate.features for candidate in all_candidates]),
            device=self.device,
            dtype=torch.float32,
        )
        raw_values = self.online.expected_value(features).cpu().numpy()
        terminated = np.asarray(
            [candidate.record.terminated for candidate in all_candidates],
            dtype=np.bool_,
        )
        raw_values[terminated] = 0.0
        backed_up, target_stats = exact_afterstate_targets(
            self.online,
            self.target,
            tuple(candidate.record for candidate in all_candidates),
            terminated,
            device=self.device,
            reward_scale=self.reward_scale,
            gamma=self.gamma,
            inference_batch_size=self.inference_batch_size,
            feature_workspace=self.feature_workspace,
        )
        backed_up_values = backed_up.cpu().numpy()

        decisions: list[AfterstateDecision] = []
        cursor = 0
        for group in candidate_groups:
            count = len(group)
            rewards = np.asarray(
                [candidate.reward / self.reward_scale for candidate in group],
                dtype=np.float64,
            )
            raw_scores = rewards + raw_values[cursor : cursor + count].astype(
                np.float64, copy=False
            )
            improved_scores = rewards + backed_up_values[
                cursor : cursor + count
            ].astype(np.float64, copy=False)
            cursor += count

            raw_position = int(np.argmax(raw_scores))
            improved_position = int(np.argmax(improved_scores))
            selected = group[improved_position]
            self.stats.decisions += 1
            self.stats.agreements += int(raw_position == improved_position)
            self.stats.normalized_regret_sum += max(
                0.0,
                float(
                    improved_scores[improved_position] - improved_scores[raw_position]
                ),
            )
            self.stats.root_candidates += count
            decisions.append(
                AfterstateDecision(
                    action=selected.action,
                    action_index=selected.action_index,
                    immediate_reward=selected.reward,
                    expected_gain=float(improved_scores[improved_position]),
                    candidate=selected,
                )
            )

        self._add_target_stats(target_stats)
        return tuple(decisions)

    def _add_target_stats(self, stats: TargetStats) -> None:
        self.stats.chance_outcomes += stats.chance_outcomes
        self.stats.expanded_candidates += stats.candidate_actions
        self.stats.online_batches += stats.online_batches
        self.stats.target_batches += stats.target_batches

    def choose_many(
        self,
        games: Sequence[GameSession],
    ) -> tuple[AfterstateDecision, ...]:
        if not games:
            raise ValueError("choose_many requires at least one game")
        return self.select_groups(tuple(make_candidates(game) for game in games))

    def choose(self, game: GameSession) -> AfterstateDecision:
        return self.choose_many((game,))[0]
