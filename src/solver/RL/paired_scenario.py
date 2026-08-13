"""Paired-scenario root rollouts with a batched DQN continuation policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from engine import Action, GameError, GameSession

from ..scoring import position_score
from ..state import sample_public_event
from .codec import PASS_ACTION_INDEX, encode_decision_into
from .dqn import DQNPolicy, GreedyBatchEvaluator


DEFAULT_PAIRED_SCENARIOS = 32


@dataclass(frozen=True, slots=True)
class PairedScenarioActionEstimate:
    action: Action | None
    samples: int
    mean_gain: float
    standard_error: float
    paired_mean_difference: float
    paired_standard_error: float


@dataclass(frozen=True, slots=True)
class PairedScenarioDecision:
    action: Action | None
    action_index: int
    mean_gain: float
    standard_error: float
    paired_mean_difference: float
    paired_standard_error: float
    tied_actions: int
    dqn_root_action_index: int
    estimates: tuple[PairedScenarioActionEstimate, ...]


@dataclass(frozen=True, slots=True)
class PairedScenarioBatchStats:
    games: int = 0
    scenarios_per_action: int = 0
    root_actions: int = 0
    trajectories: int = 0
    rollout_decisions: int = 0
    chance_samples: int = 0
    inference_batches: int = 0
    max_inference_batch_size: int = 0
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class _Rollout:
    game: GameSession
    rng: random.Random
    owner: int
    action: int
    scenario: int


def _standard_error(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def _estimate_sort_key(
    estimate: PairedScenarioActionEstimate,
) -> tuple[object, ...]:
    if estimate.action is None:
        return (-estimate.mean_gain, 1)
    return (-estimate.mean_gain, 0, *estimate.action.sort_key)


class PairedScenarioDQNPolicy:
    """Compare every root action on the same sampled future card orders.

    Each root action receives ``scenarios_per_action`` complete continuations.
    Within one scenario, every action uses an independent RNG initialized from
    the same seed, so the future public card sequence is paired exactly. The
    root action is forced once; all later decisions use deterministic DQN.
    """

    def __init__(
        self,
        dqn: DQNPolicy,
        scenarios_per_action: int = DEFAULT_PAIRED_SCENARIOS,
    ) -> None:
        if (
            isinstance(scenarios_per_action, bool)
            or not isinstance(scenarios_per_action, int)
            or scenarios_per_action < 1
        ):
            raise ValueError("scenarios_per_action must be a positive integer")
        self.dqn = dqn
        self.scenarios_per_action = scenarios_per_action
        self.scenario_rng = random.Random()
        self.tie_rng = random.SystemRandom()
        self.evaluator = GreedyBatchEvaluator(dqn.network, dqn.device)
        self.last_batch_stats = PairedScenarioBatchStats(
            scenarios_per_action=scenarios_per_action
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        scenarios_per_action: int = DEFAULT_PAIRED_SCENARIOS,
        *,
        device: str = "auto",
    ) -> PairedScenarioDQNPolicy:
        return cls(
            DQNPolicy.from_checkpoint(checkpoint_path, device=device),
            scenarios_per_action,
        )

    def _root_dqn_actions(self, games: list[GameSession]) -> np.ndarray:
        observations, masks = self.evaluator.buffers(len(games))
        for index, game in enumerate(games):
            encode_decision_into(game, observations[index], masks[index])
        return self.evaluator.select(len(games))

    def _complete_rollouts(
        self,
        active: list[_Rollout],
        scores: list[np.ndarray],
        baselines: list[int],
    ) -> tuple[int, int, int, int]:
        rollout_decisions = 0
        chance_samples = 0
        inference_batches = 0
        max_batch = 0

        while active:
            for rollout in active:
                if rollout.game.pending is None:
                    sample_public_event(rollout.game, rollout.rng)
                    chance_samples += 1

            observations, masks = self.evaluator.buffers(len(active))
            action_maps: list[dict[int, Action]] = []
            for index, rollout in enumerate(active):
                action_maps.append(
                    encode_decision_into(
                        rollout.game,
                        observations[index],
                        masks[index],
                    )
                )
            action_indices = self.evaluator.select(len(active))
            inference_batches += 1
            max_batch = max(max_batch, len(active))

            survivors: list[_Rollout] = []
            for rollout, action_map, raw_index in zip(
                active,
                action_maps,
                action_indices,
            ):
                action_index = int(raw_index)
                action = (
                    None
                    if action_index == PASS_ACTION_INDEX
                    else action_map[action_index]
                )
                rollout.game.apply_legal_action(action)
                rollout_decisions += 1
                if rollout.game.status == "playing":
                    survivors.append(rollout)
                    continue

                final = rollout.game.final_score()
                if final is None:
                    raise RuntimeError("paired rollout ended without a final score")
                scores[rollout.owner][rollout.action, rollout.scenario] = (
                    final.total - baselines[rollout.owner]
                )
            active = survivors

        return (
            rollout_decisions,
            chance_samples,
            inference_batches,
            max_batch,
        )

    def choose_many(
        self,
        games: tuple[GameSession, ...] | list[GameSession],
    ) -> tuple[PairedScenarioDecision, ...]:
        if not games:
            raise ValueError("choose_many requires at least one game")
        roots = list(games)
        for game in roots:
            if game.status != "playing" or game.pending is None:
                raise GameError("draw a card before asking for a paired rollout")
            if game.shared_objectives_enabled or game.pencil_powers_enabled:
                raise GameError("paired DQN rollouts currently support base rules only")

        started = perf_counter()
        dqn_action_indices = self._root_dqn_actions(roots)
        root_actions = [(None, *game.legal_actions()) for game in roots]
        baselines = [position_score(game).total for game in roots]
        scores = [
            np.empty(
                (len(actions), self.scenarios_per_action),
                dtype=np.float64,
            )
            for actions in root_actions
        ]
        active: list[_Rollout] = []

        for owner, (game, actions) in enumerate(zip(roots, root_actions)):
            scenario_seeds = tuple(
                self.scenario_rng.getrandbits(64)
                for _ in range(self.scenarios_per_action)
            )
            for action_index, action in enumerate(actions):
                after_action = game.copy_for_search_action()
                after_action.apply_legal_action(action)
                if after_action.status == "finished":
                    final = after_action.final_score()
                    if final is None:
                        raise RuntimeError("finished root action has no final score")
                    scores[owner][action_index, :] = (
                        final.total - baselines[owner]
                    )
                    continue
                for scenario, seed in enumerate(scenario_seeds):
                    active.append(
                        _Rollout(
                            game=after_action.copy_public_state(),
                            rng=random.Random(seed),
                            owner=owner,
                            action=action_index,
                            scenario=scenario,
                        )
                    )

        trajectories = sum(matrix.size for matrix in scores)
        (
            rollout_decisions,
            chance_samples,
            inference_batches,
            max_batch,
        ) = self._complete_rollouts(active, scores, baselines)

        decisions: list[PairedScenarioDecision] = []
        for owner, (actions, matrix, raw_dqn_index) in enumerate(
            zip(root_actions, scores, dqn_action_indices)
        ):
            dqn_index = int(raw_dqn_index)
            dqn_root = next(
                index
                for index, action in enumerate(actions)
                if (
                    dqn_index == PASS_ACTION_INDEX
                    and action is None
                    or action is not None
                    and action.edge_id == dqn_index
                )
            )
            reference = matrix[dqn_root]
            estimates = tuple(
                PairedScenarioActionEstimate(
                    action=action,
                    samples=self.scenarios_per_action,
                    mean_gain=float(values.mean()),
                    standard_error=_standard_error(values),
                    paired_mean_difference=float((values - reference).mean()),
                    paired_standard_error=_standard_error(values - reference),
                )
                for action, values in zip(actions, matrix)
            )
            ranked = tuple(sorted(estimates, key=_estimate_sort_key))
            best_mean = ranked[0].mean_gain
            tied = tuple(
                estimate
                for estimate in ranked
                if math.isclose(
                    estimate.mean_gain,
                    best_mean,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            chosen = self.tie_rng.choice(tied)
            chosen_index = (
                PASS_ACTION_INDEX
                if chosen.action is None
                else chosen.action.edge_id
            )
            decisions.append(
                PairedScenarioDecision(
                    action=chosen.action,
                    action_index=chosen_index,
                    mean_gain=chosen.mean_gain,
                    standard_error=chosen.standard_error,
                    paired_mean_difference=chosen.paired_mean_difference,
                    paired_standard_error=chosen.paired_standard_error,
                    tied_actions=len(tied),
                    dqn_root_action_index=dqn_index,
                    estimates=ranked,
                )
            )

        self.last_batch_stats = PairedScenarioBatchStats(
            games=len(roots),
            scenarios_per_action=self.scenarios_per_action,
            root_actions=sum(len(actions) for actions in root_actions),
            trajectories=trajectories,
            rollout_decisions=rollout_decisions,
            chance_samples=chance_samples,
            inference_batches=inference_batches,
            max_inference_batch_size=max_batch,
            elapsed_seconds=perf_counter() - started,
        )
        return tuple(decisions)

    def choose(self, game: GameSession) -> PairedScenarioDecision:
        return self.choose_many((game,))[0]
