"""Executable correctness and throughput checks without a test framework."""

from __future__ import annotations

from pathlib import Path
import random
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from ..codec import ACTION_COUNT
from ..environment import DecisionEnv
from .network import NetworkSpec, PolicyValueNetwork, parameter_count
from .replay import AlphaZeroReplay, ReplayRecord
from .search import BatchedPUCT, SearchConfig
from .training import TrainConfig, collect_generation


def run_self_check(device: str = "auto") -> dict[str, object]:
    spec = NetworkSpec(width=64, residual_blocks=2, value_hidden=32)
    network = PolicyValueNetwork(spec)
    envs = [DecisionEnv(index, verify=True) for index in range(4)]
    search = BatchedPUCT(
        network,
        SearchConfig(simulations=16),
        device=device,
    )
    results = search.search(
        [env.game for env in envs],
        seeds=(101, 102, 103, 104),
        add_root_noise=True,
    )
    for env, result in zip(envs, results):
        if int(result.action_visits.sum(dtype=np.uint64)) != 16:
            raise RuntimeError("self-check search visits do not sum to budget")
        if np.any(result.action_visits[~env.action_mask]):
            raise RuntimeError("self-check search visited an illegal action")
        if result.nodes > 17:
            raise RuntimeError("self-check search exceeded its node bound")

    records = [
        ReplayRecord(
            observation=np.array(env.observation, copy=True),
            action_mask=np.array(env.action_mask, copy=True),
            visits=np.array(result.action_visits, copy=True),
            value=0.5,
        )
        for env, result in zip(envs, results)
    ]
    with TemporaryDirectory() as directory:
        replay = AlphaZeroReplay(8, Path(directory) / "replay.dat")
        replay.add_many(records)
        batch = replay.sample(4, np.random.default_rng(5))
        if batch.observations.shape[1] != spec.observation_dim:
            raise RuntimeError("self-check replay observation shape changed")
        if not np.allclose(batch.policy_targets.sum(axis=1), 1.0):
            raise RuntimeError("self-check replay policies are not normalized")
        replay.close()

    full_config = TrainConfig.fresh(
        num_envs=4,
        simulations=8,
        total_positions=1,
        replay_capacity=64,
        batch_size=32,
        network_width=64,
        residual_blocks=2,
        value_hidden=32,
        validation_interval=1,
    )
    generation = collect_generation(
        network,
        full_config,
        device=device,
        game_seed_stream=random.Random(full_config.run_seed),
        collector_rng=np.random.default_rng(full_config.run_seed ^ 0x1234),
        verify=True,
    )
    if len(generation.scores) != 4 or not generation.records:
        raise RuntimeError("self-check did not finish every complete game")
    if not all(np.isfinite(record.value) for record in generation.records):
        raise RuntimeError("self-check produced a non-finite value target")

    return {
        "status": "ok",
        "device": str(search.device),
        "small_network_parameters": parameter_count(network),
        "searches": search.last_stats.searches,
        "simulations": search.last_stats.simulations,
        "expanded_nodes": search.last_stats.expanded_nodes,
        "max_depth": search.last_stats.max_depth,
        "complete_games": len(generation.scores),
        "training_positions": len(generation.records),
    }


def benchmark_generation(
    *,
    num_envs: int,
    simulations: int,
    device: str,
    width: int = 1024,
    residual_blocks: int = 6,
) -> dict[str, object]:
    config = TrainConfig.fresh(
        num_envs=num_envs,
        simulations=simulations,
        total_positions=1,
        replay_capacity=2048,
        batch_size=2048,
        network_width=width,
        residual_blocks=residual_blocks,
        validation_interval=1,
    )
    network = PolicyValueNetwork(config.network_spec)
    stream = random.Random(config.run_seed)
    rng = np.random.default_rng(config.run_seed ^ 0x1234)
    started = perf_counter()
    result = collect_generation(
        network,
        config,
        device=device,
        game_seed_stream=stream,
        collector_rng=rng,
    )
    elapsed = perf_counter() - started
    return {
        "num_envs": num_envs,
        "simulations": simulations,
        "parameters": parameter_count(network),
        "positions": len(result.records),
        "games": len(result.scores),
        "mean_score": float(np.mean(result.scores)),
        "elapsed_seconds": elapsed,
        "positions_per_second": len(result.records) / elapsed,
        "simulations_per_second": result.simulations / elapsed,
        "mean_inference_batch_size": result.mean_inference_batch_size,
        "max_inference_batch_size": result.max_inference_batch_size,
        "cuda_peak_mib": (
            torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available() and device != "cpu"
            else 0.0
        ),
    }
