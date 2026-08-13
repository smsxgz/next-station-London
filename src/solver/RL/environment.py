"""Decision-to-decision environments for batched DQN data collection."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from engine import GameError, GameSession

from ..scoring import position_score
from .codec import (
    ACTION_COUNT,
    OBSERVATION_DIM,
    PASS_ACTION_INDEX,
    ActionMask,
    EncodedDecision,
    Observation,
    encode_decision,
)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    game_seed: int
    score: int
    reward_sum: int
    initial_score: int
    decisions: int
    sections: int
    passes: int
    color_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StepResult:
    reward: int
    terminated: bool
    next_observation: Observation
    next_action_mask: ActionMask
    episode: EpisodeResult | None = None


@dataclass(frozen=True, slots=True)
class VectorStepResult:
    rewards: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    next_observations: NDArray[np.uint8]
    next_action_masks: NDArray[np.bool_]
    completed_episodes: tuple[EpisodeResult, ...]


_TERMINAL_OBSERVATION = np.zeros(OBSERVATION_DIM, dtype=np.uint8)
_TERMINAL_ACTION_MASK = np.zeros(ACTION_COUNT, dtype=np.bool_)
_TERMINAL_ACTION_MASK[PASS_ACTION_INDEX] = True


class DecisionEnv:
    """A game wrapper whose states always wait for one player action."""

    def __init__(self, stream_seed: int, *, verify: bool = False) -> None:
        self._seed_stream = random.Random(stream_seed)
        self.verify = verify
        self.game: GameSession
        self.game_seed = 0
        self.initial_score = 0
        self.reward_sum = 0
        self.decisions = 0
        self.sections = 0
        self.passes = 0
        self._decision: EncodedDecision
        self.reset()

    @property
    def observation(self) -> Observation:
        return self._decision.observation

    @property
    def action_mask(self) -> ActionMask:
        return self._decision.action_mask

    def seed_stream_state(self) -> object:
        return self._seed_stream.getstate()

    def restore_seed_stream_state(self, state: object) -> None:
        self._seed_stream.setstate(state)

    def reset(self, game_seed: int | None = None) -> tuple[Observation, ActionMask]:
        if game_seed is None:
            game_seed = self._seed_stream.getrandbits(63)
        self.game_seed = int(game_seed)
        self.game = GameSession(seed=self.game_seed)
        self.game.draw()
        self.initial_score = position_score(self.game).total
        self.reward_sum = 0
        self.decisions = 0
        self.sections = 0
        self.passes = 0
        self._decision = encode_decision(self.game)
        return self.observation, self.action_mask

    def step(self, action_index: int) -> StepResult:
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, (int, np.integer))
            or not 0 <= int(action_index) < ACTION_COUNT
        ):
            raise GameError("RL action index is outside the fixed action space")
        action_index = int(action_index)
        if not self.action_mask[action_index]:
            raise GameError("RL policy selected an illegal action")

        before = position_score(self.game).total if self.verify else 0
        if action_index == PASS_ACTION_INDEX:
            action = None
            reward = 0
            self.passes += 1
        else:
            action = self._decision.actions[action_index]
            reward = sum(self.game.score_delta_for_legal_action(action))
            self.sections += 1

        self.game.apply_legal_action(action)
        self.reward_sum += reward
        self.decisions += 1

        if self.verify:
            after = position_score(self.game).total
            if after - before != reward:
                raise RuntimeError(
                    "dense reward mismatch: "
                    f"cached={reward}, score delta={after - before}"
                )

        if self.game.status == "finished":
            final = self.game.final_score()
            if final is None:
                raise RuntimeError("finished RL episode has no final score")
            if self.initial_score + self.reward_sum != final.total:
                raise RuntimeError(
                    "dense rewards do not telescope to the final score: "
                    f"{self.initial_score} + {self.reward_sum} != {final.total}"
                )
            episode = EpisodeResult(
                game_seed=self.game_seed,
                score=final.total,
                reward_sum=self.reward_sum,
                initial_score=self.initial_score,
                decisions=self.decisions,
                sections=self.sections,
                passes=self.passes,
                color_order=self.game.order,
            )
            return StepResult(
                reward=reward,
                terminated=True,
                next_observation=_TERMINAL_OBSERVATION,
                next_action_mask=_TERMINAL_ACTION_MASK,
                episode=episode,
            )

        self.game.draw()
        self._decision = encode_decision(self.game)
        return StepResult(
            reward=reward,
            terminated=False,
            next_observation=self.observation,
            next_action_mask=self.action_mask,
        )


class VectorDecisionEnv:
    """Synchronous independent games feeding one batched network request."""

    def __init__(
        self,
        num_envs: int,
        seed: int,
        *,
        verify: bool = False,
    ) -> None:
        if (
            isinstance(num_envs, bool)
            or not isinstance(num_envs, int)
            or num_envs < 1
        ):
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        stream_factory = random.Random(seed)
        self.envs = tuple(
            DecisionEnv(stream_factory.getrandbits(63), verify=verify)
            for _ in range(num_envs)
        )
        self.observations = np.stack(
            [env.observation for env in self.envs], axis=0
        )
        self.action_masks = np.stack(
            [env.action_mask for env in self.envs], axis=0
        )

    def seed_stream_states(self) -> tuple[object, ...]:
        return tuple(env.seed_stream_state() for env in self.envs)

    def restore_seed_stream_states(self, states: Sequence[object]) -> None:
        if len(states) != self.num_envs:
            raise ValueError("checkpoint has a different number of env seed streams")
        for env, state in zip(self.envs, states):
            env.restore_seed_stream_state(state)
            env.reset()
        self._refresh_current_batches()

    def _refresh_current_batches(self) -> None:
        self.observations = np.stack(
            [env.observation for env in self.envs], axis=0
        )
        self.action_masks = np.stack(
            [env.action_mask for env in self.envs], axis=0
        )

    def step(self, action_indices: NDArray[np.integer]) -> VectorStepResult:
        indices = np.asarray(action_indices)
        if indices.shape != (self.num_envs,):
            raise ValueError(
                f"expected {self.num_envs} action indices, got {indices.shape}"
            )

        rewards = np.empty(self.num_envs, dtype=np.float32)
        terminated = np.empty(self.num_envs, dtype=np.bool_)
        next_observations = np.empty(
            (self.num_envs, OBSERVATION_DIM), dtype=np.uint8
        )
        next_action_masks = np.empty(
            (self.num_envs, ACTION_COUNT), dtype=np.bool_
        )
        completed: list[EpisodeResult] = []

        for index, (env, action_index) in enumerate(zip(self.envs, indices)):
            result = env.step(int(action_index))
            rewards[index] = result.reward
            terminated[index] = result.terminated
            next_observations[index] = result.next_observation
            next_action_masks[index] = result.next_action_mask
            if result.episode is not None:
                completed.append(result.episode)
                env.reset()

        self._refresh_current_batches()
        return VectorStepResult(
            rewards=rewards,
            terminated=terminated,
            next_observations=next_observations,
            next_action_masks=next_action_masks,
            completed_episodes=tuple(completed),
        )
