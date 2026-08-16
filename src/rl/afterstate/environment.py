"""Game collection environments for afterstate-value training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from engine_cpp import GameError, GameSession
from solver.scoring import position_score

from .codec import Candidate, afterstate_key, make_candidates


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


class AfterstateEnv:
    """One environment that always exposes a pending decision."""

    def __init__(
        self,
        stream_seed: int,
        *,
        verify: bool = False,
        initial_game_seed: int | None = None,
    ) -> None:
        self._seed_stream = random.Random(stream_seed)
        self.verify = verify
        self.game: GameSession
        self.game_seed = 0
        self.initial_score = 0
        self.reward_sum = 0
        self.decisions = 0
        self.sections = 0
        self.passes = 0
        self.reset(initial_game_seed)

    def seed_stream_state(self) -> object:
        return self._seed_stream.getstate()

    def restore_seed_stream_state(self, state: object) -> None:
        self._seed_stream.setstate(state)

    def next_game_seed(self) -> int:
        return self._seed_stream.getrandbits(63)

    def reset(self, game_seed: int | None = None) -> None:
        if game_seed is None:
            game_seed = self.next_game_seed()
        self.game_seed = int(game_seed)
        self.game = GameSession(seed=self.game_seed, advanced=False)
        self.game.draw()
        self.initial_score = position_score(self.game).total
        self.reward_sum = 0
        self.decisions = 0
        self.sections = 0
        self.passes = 0

    def candidates(self) -> tuple[Candidate, ...]:
        return make_candidates(self.game)

    def step(self, candidate: Candidate) -> EpisodeResult | None:
        if self.game.status != "playing" or self.game.pending is None:
            raise GameError("environment is not waiting for an action")
        if (
            candidate.action is not None
            and candidate.action not in self.game.legal_actions()
        ):
            raise GameError("candidate is not legal for the current decision")
        expected_reward = (
            0
            if candidate.action is None
            else sum(self.game.score_delta_for_legal_action(candidate.action))
        )
        if expected_reward != candidate.reward:
            raise GameError("candidate reward does not match the current decision")
        before = position_score(self.game).total if self.verify else 0
        action = candidate.action
        self.game.apply_legal_action(action)
        self.reward_sum += candidate.reward
        self.decisions += 1
        if action is None:
            self.passes += 1
        else:
            self.sections += 1
        if self.verify:
            after = position_score(self.game).total
            if after - before != candidate.reward:
                raise RuntimeError(
                    f"dense reward mismatch: {after - before} != {candidate.reward}"
                )
        if afterstate_key(self.game) != afterstate_key(candidate.afterstate):
            raise RuntimeError("real action and candidate afterstate disagree")
        if self.game.status != "finished":
            self.game.draw()
            return None
        final = self.game.final_score()
        if final is None:
            raise RuntimeError("finished game has no final score")
        if self.initial_score + self.reward_sum != final.total:
            raise RuntimeError(
                f"dense rewards do not telescope: {self.initial_score} + "
                f"{self.reward_sum} != {final.total}"
            )
        return EpisodeResult(
            game_seed=self.game_seed,
            score=final.total,
            reward_sum=self.reward_sum,
            initial_score=self.initial_score,
            decisions=self.decisions,
            sections=self.sections,
            passes=self.passes,
            color_order=self.game.order,
        )

    def state_dict(self) -> dict[str, object]:
        """Snapshot collector-only state, including hidden RNG/deck order."""

        return {
            "seed_stream": self._seed_stream.getstate(),
            "game": self.game,
            "game_seed": self.game_seed,
            "initial_score": self.initial_score,
            "reward_sum": self.reward_sum,
            "decisions": self.decisions,
            "sections": self.sections,
            "passes": self.passes,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._seed_stream.setstate(state["seed_stream"])
        self.game = state["game"]
        self.game_seed = int(state["game_seed"])
        self.initial_score = int(state["initial_score"])
        self.reward_sum = int(state["reward_sum"])
        self.decisions = int(state["decisions"])
        self.sections = int(state["sections"])
        self.passes = int(state["passes"])


class VectorAfterstateEnv:
    """Independent environments used by the synchronous collector."""

    def __init__(self, num_envs: int, seed: int, *, verify: bool = False) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        stream = random.Random(seed)
        self.envs = tuple(
            AfterstateEnv(stream.getrandbits(63), verify=verify)
            for _ in range(num_envs)
        )
        self.num_envs = num_envs

    def seed_stream_states(self) -> tuple[object, ...]:
        return tuple(env.seed_stream_state() for env in self.envs)

    def restore_seed_stream_states(self, states: Sequence[object]) -> None:
        if len(states) != self.num_envs:
            raise ValueError("checkpoint has a different number of env seed streams")
        for env, state in zip(self.envs, states):
            env.restore_seed_stream_state(state)
            env.reset()

    def state_dict(self) -> tuple[dict[str, object], ...]:
        return tuple(env.state_dict() for env in self.envs)

    def load_state_dict(self, states: Sequence[dict[str, object]]) -> None:
        if len(states) != self.num_envs:
            raise ValueError("checkpoint has a different number of environments")
        for env, state in zip(self.envs, states):
            env.load_state_dict(state)

    def candidate_groups(
        self, count: int | None = None
    ) -> tuple[tuple[Candidate, ...], ...]:
        actual = self.num_envs if count is None else int(count)
        if not 0 < actual <= self.num_envs:
            raise ValueError("candidate count is outside the environment batch")
        return tuple(self.envs[index].candidates() for index in range(actual))

    def step(
        self,
        candidates: Sequence[Candidate],
    ) -> tuple[EpisodeResult | None, ...]:
        if len(candidates) > self.num_envs:
            raise ValueError("too many candidates")
        results: list[EpisodeResult | None] = []
        for index, candidate in enumerate(candidates):
            result = self.envs[index].step(candidate)
            results.append(result)
            if result is not None:
                self.envs[index].reset()
        return tuple(results)
