"""Small executable correctness checks; deliberately not a pytest suite."""

from __future__ import annotations

import random
from time import perf_counter
from typing import Any

import numpy as np
import torch

from engine_cpp import COLORS, DECK, GameSession
from solver.scoring import position_score
from solver.state import public_event_successors

from .codec import PASS_ACTION_INDEX, decode_afterstate, make_candidates
from .group_replay import DecisionGroupReplayBuffer
from .group_training import GroupAfterstateLearner, GroupTrainConfig
from .network import AfterstateNetworkSpec, AfterstateValueNetwork, resolve_device
from .replay import AfterstateReplayBuffer, PrioritizedAfterstateReplayBuffer
from .target import exact_afterstate_targets
from .training import AfterstateLearner, TrainConfig


def _check_candidates(games: int) -> int:
    checked = 0
    for seed in range(games):
        game = GameSession(seed=seed, advanced=False)
        game.draw()
        for _ in range(8):
            candidates = make_candidates(game)
            if candidates[-1].action_index != PASS_ACTION_INDEX:
                raise AssertionError("Pass is not the fixed final candidate")
            for candidate in candidates:
                decoded = decode_afterstate(candidate.record)
                if (
                    decoded.partial_score_components()
                    != candidate.afterstate.partial_score_components()
                ):
                    raise AssertionError("decoded score cache differs from candidate")
                if (
                    position_score(decoded).total
                    != position_score(candidate.afterstate).total
                ):
                    raise AssertionError(
                        "decoded position score differs from candidate"
                    )
                real = game.copy_public_state()
                real.apply_legal_action(candidate.action)
                if real.remaining_card_mask != candidate.afterstate.remaining_card_mask:
                    raise AssertionError("candidate changed hidden card mask")
                if real.lines != candidate.afterstate.lines:
                    raise AssertionError("candidate board differs from real action")
                checked += 1
            chosen = candidates[0]
            game.apply_legal_action(chosen.action)
            if game.status == "finished":
                break
            game.draw()
    return checked


def _check_switch() -> int:
    switch_id = next(card.id for card in DECK if card.switch)
    game = GameSession(seed=17, advanced=False)
    # Leave exactly the switch and one other card in the public pile so both
    # the two-card draw and source_any semantics are exercised directly.
    keep = {switch_id, next(card.id for card in DECK if card.id != switch_id)}
    game = GameSession.from_public_state(
        order=game.order,
        line_station_masks=tuple(game.lines[color].station_mask for color in COLORS),
        line_edge_masks=tuple(game.lines[color].edge_mask for color in COLORS),
        remaining_mask=sum(1 << card_id for card_id in keep),
        round_index=0,
        underground_count=0,
        draw_count=0,
    )
    outcomes = tuple(public_event_successors(game))
    if not np.isclose(sum(probability for probability, _ in outcomes), 1.0):
        raise AssertionError("switch chance probabilities do not sum to one")
    switch_outcomes = [
        child for _, child in outcomes if switch_id in child.pending.card_ids
    ]
    if len(switch_outcomes) != 1:
        raise AssertionError("switch outcome was not enumerated exactly once")
    event = switch_outcomes[0].pending
    if event is None or not event.source_any or len(event.card_ids) != 2:
        raise AssertionError("switch event did not set source_any or consume two cards")
    return len(outcomes)


def _check_target(device: torch.device) -> dict[str, float | int]:
    game = GameSession(seed=29, advanced=False)
    game.draw()
    record = make_candidates(game)[0].record
    online = AfterstateValueNetwork(AfterstateNetworkSpec(hidden_sizes=(16, 16))).to(
        device
    )
    target = AfterstateValueNetwork(AfterstateNetworkSpec(hidden_sizes=(16, 16))).to(
        device
    )
    target.load_state_dict(online.state_dict())
    values, stats = exact_afterstate_targets(
        online,
        target,
        (record,),
        np.asarray([record.terminated], dtype=np.bool_),
        device=device,
        inference_batch_size=128,
    )
    if not torch.isfinite(values).all():
        raise AssertionError("target produced a non-finite value")
    config = TrainConfig(
        seed=31,
        num_envs=1,
        total_transitions=1,
        replay_capacity=2,
        batch_size=1,
        warmup_transitions=1,
        hidden_sizes=(16, 16),
        validation_seeds=(),
        device=str(device),
    )
    learner = AfterstateLearner(config, device)
    replay = AfterstateReplayBuffer(2)
    replay.add(record, action=0, reward=0)
    update = learner.update(replay.sample(1, np.random.default_rng(0)))
    if not np.isfinite(update.metrics["loss"]):
        raise AssertionError("learner update produced a non-finite loss")
    return {
        "target": float(values.item()),
        "chance_outcomes": stats.chance_outcomes,
        "candidate_actions": stats.candidate_actions,
        "loss": update.metrics["loss"],
    }


def _check_prioritized_replay() -> float:
    """Exercise the afterstate PER storage path."""

    game = GameSession(seed=0xA51, advanced=False)
    game.draw()
    record = make_candidates(game)[0].record
    replay = PrioritizedAfterstateReplayBuffer(16)
    replay.add(record, action=0, reward=0)
    if not np.isfinite(replay._priorities[:1]).all() or replay._sum_tree[1] <= 0.0:
        raise AssertionError("afterstate PER did not initialize a new priority")
    rng = np.random.default_rng(7)
    batch = replay.sample(1, rng, beta=0.4)
    replay.update_priorities(batch.indices, np.asarray([2.0], dtype=np.float32))
    resampled = replay.sample(1, rng, beta=1.0)
    if not np.isfinite(resampled.weights).all():
        raise AssertionError("afterstate PER returned non-finite weights")
    return float(replay._sum_tree[1])


def _check_group_training(device: torch.device) -> dict[str, float | int]:
    game = GameSession(seed=0x6A09_E667, advanced=False)
    game.draw()
    candidates = make_candidates(game)
    replay = DecisionGroupReplayBuffer(4, 256)
    replay.add(candidates)
    batch = replay.sample(1, np.random.default_rng(9))
    if batch.candidate_count != len(candidates):
        raise AssertionError("group replay changed the candidate count")
    if not np.array_equal(
        batch.actions,
        np.asarray([candidate.action_index for candidate in candidates]),
    ):
        raise AssertionError("group replay changed candidate actions")

    config = GroupTrainConfig(
        seed=33,
        source_checkpoint="unused",
        num_envs=1,
        total_transitions=1,
        group_capacity=4,
        candidate_capacity=256,
        group_batch_size=1,
        warmup_groups=1,
        hidden_sizes=(16, 16),
        validation_seeds=(),
        device=str(device),
    )
    learner = GroupAfterstateLearner(config, device)
    update = learner.update(batch)
    if not np.isfinite(update.metrics["loss"]):
        raise AssertionError("group learner update produced a non-finite loss")
    return {
        "candidates": batch.candidate_count,
        "loss": update.metrics["loss"],
    }


def run_self_check(*, games: int = 4, device_name: str = "auto") -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be positive")
    random.seed(0)
    device = resolve_device(device_name)
    return {
        "candidate_afterstates": _check_candidates(games),
        "switch_outcomes": _check_switch(),
        "target_smoke": _check_target(device),
        "per_priority_sum": _check_prioritized_replay(),
        "group_smoke": _check_group_training(device),
        "device": str(device),
    }


def benchmark_profile(
    *,
    env_count: int = 128,
    transitions: int = 1_024,
    batch_size: int = 512,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Measure the agreed full-copy collector and exact-target path."""

    if env_count < 1 or transitions < 1 or batch_size < 1:
        raise ValueError("benchmark sizes must be positive")
    device = resolve_device(device_name)
    network = AfterstateValueNetwork(AfterstateNetworkSpec(hidden_sizes=(512, 512))).to(
        device
    )
    target = AfterstateValueNetwork(AfterstateNetworkSpec(hidden_sizes=(512, 512))).to(
        device
    )
    target.load_state_dict(network.state_dict())
    from .environment import VectorAfterstateEnv
    from .policy import AfterstatePolicy

    env = VectorAfterstateEnv(env_count, 0xBEEF, verify=False)
    policy = AfterstatePolicy(network, device=device)
    records = []
    started = perf_counter()
    while len(records) < transitions:
        groups = env.candidate_groups()
        decisions = policy.select_groups(groups)
        selected = tuple(decision.candidate for decision in decisions)
        env.step(selected)
        records.extend(candidate.record for candidate in selected)
    collector_elapsed = perf_counter() - started
    records = tuple(records[:transitions])
    terminal = np.asarray([record.terminated for record in records], dtype=np.bool_)
    target_started = perf_counter()
    _, stats = exact_afterstate_targets(
        network,
        target,
        records[:batch_size],
        terminal[:batch_size],
        device=device,
        inference_batch_size=8192,
    )
    target_elapsed = perf_counter() - target_started
    return {
        "device": str(device),
        "env_count": env_count,
        "collector_transitions": transitions,
        "collector_seconds": collector_elapsed,
        "collector_transitions_per_second": transitions / max(collector_elapsed, 1e-9),
        "target_batch_size": min(batch_size, len(records)),
        "target_seconds": target_elapsed,
        "chance_outcomes": stats.chance_outcomes,
        "candidate_actions": stats.candidate_actions,
        "online_batches": stats.online_batches,
        "target_batches": stats.target_batches,
    }
