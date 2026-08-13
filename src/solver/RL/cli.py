"""Command-line interface for deep RL checks, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import secrets

from .dqn import DQNPolicy
from .training import (
    TrainConfig,
    benchmark_vector_envs,
    evaluate_network,
    generate_validation_seeds,
    load_config,
    run_self_check,
    train,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _load_manifest_seeds(path: Path, games: int | None) -> tuple[int, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in raw.get("game_seeds", ()))
    if not seeds:
        raise ValueError(f"{path} does not contain game_seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path} contains duplicate game seeds")
    if games is None:
        return seeds
    if games > len(seeds):
        raise ValueError(f"requested {games} games from only {len(seeds)} seeds")
    return seeds[:games]


def _add_device(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = "auto",
) -> None:
    parser.add_argument(
        "--device", default=default, help="auto, cpu, cuda, or cuda:N"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Next Station masked deep RL")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="run in-memory correctness checks")
    check.add_argument("--games", type=_positive_int, default=16)
    check.add_argument(
        "--algorithm", choices=("dqn", "c51"), default="dqn"
    )
    _add_device(check)

    benchmark = commands.add_parser(
        "benchmark-env", help="benchmark batched inference plus engine stepping"
    )
    benchmark.add_argument(
        "--env-counts", type=_positive_int, nargs="+", default=(16, 32, 64, 128)
    )
    benchmark.add_argument(
        "--transitions", type=_positive_int, default=50_000
    )
    benchmark.add_argument("--seed", type=_non_negative_int)
    _add_device(benchmark)

    training = commands.add_parser("train", help="train or resume a DQN run")
    training.add_argument(
        "--run-dir", type=Path, default=Path("artifacts/dqn/uniform")
    )
    training.add_argument("--resume", action="store_true")
    training.add_argument(
        "--algorithm", choices=("dqn", "c51"), default="dqn"
    )
    training.add_argument("--num-envs", type=_positive_int, default=128)
    training.add_argument("--total-transitions", type=_positive_int)
    training.add_argument("--n-steps", type=_positive_int, default=1)
    training.add_argument(
        "--target-depth", type=int, choices=(0, 1, 2), default=0
    )
    training.add_argument("--init-checkpoint", type=Path)
    training.add_argument("--eval-seeds-from", type=Path)
    training.add_argument("--validation-games", type=_positive_int, default=200)
    training.add_argument(
        "--validation-interval", type=_positive_int, default=250_000
    )
    training.add_argument("--evaluation-envs", type=_positive_int, default=128)
    training.add_argument("--log-interval", type=_positive_int, default=100_000)
    training.add_argument(
        "--checkpoint-interval", type=_positive_int, default=250_000
    )
    training.add_argument(
        "--replay-capacity", type=_positive_int, default=1_000_000
    )
    training.add_argument(
        "--replay-kind",
        choices=("uniform", "prioritized"),
        default="uniform",
    )
    training.add_argument("--priority-alpha", type=_unit_float, default=0.6)
    training.add_argument(
        "--priority-beta-start", type=_unit_float, default=0.4
    )
    training.add_argument(
        "--priority-beta-end", type=_unit_float, default=1.0
    )
    training.add_argument(
        "--priority-epsilon", type=_positive_float, default=1e-3
    )
    training.add_argument(
        "--warmup-transitions", type=_positive_int, default=25_000
    )
    training.add_argument("--batch-size", type=_positive_int, default=512)
    training.add_argument("--replay-ratio", type=_positive_float, default=8.0)
    training.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    training.add_argument(
        "--target-update-interval", type=_positive_int, default=1_000
    )
    training.add_argument("--epsilon-initial", type=_unit_float, default=1.0)
    training.add_argument("--epsilon-final", type=_unit_float, default=0.05)
    training.add_argument("--verify-env", action="store_true")
    _add_device(training, default=None)

    evaluate = commands.add_parser(
        "evaluate", help="evaluate a checkpoint on manifest game seeds"
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--seeds-from", type=Path, required=True)
    evaluate.add_argument("--games", type=_positive_int)
    evaluate.add_argument("--num-envs", type=_positive_int, default=64)
    _add_device(evaluate)
    return parser


def _fresh_config(args: argparse.Namespace) -> TrainConfig:
    seed = secrets.randbits(63)
    validation_seeds = (
        _load_manifest_seeds(args.eval_seeds_from, args.validation_games)
        if args.eval_seeds_from is not None
        else generate_validation_seeds(seed, args.validation_games)
    )
    return TrainConfig(
        seed=seed,
        algorithm=args.algorithm,
        num_envs=args.num_envs,
        total_transitions=args.total_transitions or 10_000_000,
        n_steps=args.n_steps,
        target_depth=args.target_depth,
        init_checkpoint=(
            str(args.init_checkpoint.resolve())
            if args.init_checkpoint is not None
            else None
        ),
        replay_capacity=args.replay_capacity,
        replay_kind=args.replay_kind,
        priority_alpha=args.priority_alpha,
        priority_beta_start=args.priority_beta_start,
        priority_beta_end=args.priority_beta_end,
        priority_epsilon=args.priority_epsilon,
        batch_size=args.batch_size,
        replay_ratio=args.replay_ratio,
        learning_rate=args.learning_rate,
        target_update_interval=args.target_update_interval,
        epsilon_initial=args.epsilon_initial,
        epsilon_final=args.epsilon_final,
        warmup_transitions=args.warmup_transitions,
        validation_seeds=validation_seeds,
        validation_interval=args.validation_interval,
        evaluation_envs=args.evaluation_envs,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device or "auto",
    )


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "check":
        result = run_self_check(
            games=args.games,
            device_name=args.device,
            algorithm=args.algorithm,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    if args.command == "benchmark-env":
        results = benchmark_vector_envs(
            args.env_counts,
            transitions_per_count=args.transitions,
            device_name=args.device,
            seed=args.seed,
        )
        for result in results:
            print(
                f"envs={result['num_envs']}; transitions={result['transitions']}; "
                f"elapsed={result['elapsed_seconds']:.2f}s; "
                f"rate={result['transitions_per_second']:.0f}/s; "
                f"games={result['completed_games']}",
                flush=True,
            )
        return

    if args.command == "train":
        if args.resume:
            config = load_config(args.run_dir)
            if args.total_transitions is not None:
                config = TrainConfig.from_dict(
                    {**config.to_dict(), "total_transitions": args.total_transitions}
                )
            if args.device is not None:
                config = TrainConfig.from_dict(
                    {**config.to_dict(), "device": args.device}
                )
        else:
            config = _fresh_config(args)
        print(f"training_seed={config.seed}", flush=True)
        train(config, args.run_dir, resume=args.resume, verify_env=args.verify_env)
        return

    if args.command == "evaluate":
        seeds = _load_manifest_seeds(args.seeds_from, args.games)
        policy = DQNPolicy.from_checkpoint(args.checkpoint, device=args.device)
        result = evaluate_network(
            policy.network,
            seeds,
            device=policy.device,
            num_envs=args.num_envs,
        )
        print(
            f"games={len(result.scores)}; mean={result.mean:.2f}; "
            f"SE={result.standard_error:.2f}; min={result.minimum}; "
            f"median={result.median:.1f}; max={result.maximum}; "
            f"elapsed={result.elapsed_seconds:.2f}s",
            flush=True,
        )
        return

    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
