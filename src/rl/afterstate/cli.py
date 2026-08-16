"""Command line entry points for the afterstate-value experiment."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from .checks import benchmark_profile, run_self_check
from .evaluation import (
    evaluate_afterstate_checkpoint,
    generate_independent_seeds,
    generate_validation_seeds,
    load_manifest_seeds,
    run_final_evaluation,
)
from .group_training import GroupTrainConfig, train_groups
from .training import TrainConfig, train

DEFAULT_VALIDATION_MANIFEST = Path("benchmark_results/current_200/manifest.json")
DEFAULT_BASELINE = Path("artifacts/dqn/exact_chance_depth1_lr1e4_4m/best.pt")
DEFAULT_GROUP_SOURCE = Path("artifacts/afterstate/value_10m_cpp/latest.pt")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _load_validation_seeds(path: Path | None, seed: int, games: int) -> tuple[int, ...]:
    if path is None:
        return generate_validation_seeds(seed, games)
    return load_manifest_seeds(path, games)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Next Station afterstate-value RL")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="run executable correctness checks")
    check.add_argument("--games", type=_positive_int, default=4)
    check.add_argument("--device", default="auto")

    benchmark = commands.add_parser(
        "benchmark", help="profile collector and exact target"
    )
    benchmark.add_argument("--env-count", type=_positive_int, default=128)
    benchmark.add_argument("--transitions", type=_positive_int, default=1024)
    benchmark.add_argument("--batch-size", type=_positive_int, default=512)
    benchmark.add_argument("--device", default="auto")

    train_parser = commands.add_parser(
        "train", help="train or resume an afterstate run"
    )
    train_parser.add_argument(
        "--run-dir", type=Path, default=Path("artifacts/afterstate/value_10m")
    )
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--seed", type=_non_negative_int)
    train_parser.add_argument("--num-envs", type=_positive_int, default=128)
    train_parser.add_argument("--total-transitions", type=_positive_int)
    train_parser.add_argument(
        "--eval-seeds-from", type=Path, default=DEFAULT_VALIDATION_MANIFEST
    )
    train_parser.add_argument("--validation-games", type=_positive_int, default=200)
    train_parser.add_argument(
        "--validation-interval", type=_positive_int, default=250_000
    )
    train_parser.add_argument("--evaluation-envs", type=_positive_int, default=64)
    train_parser.add_argument("--log-interval", type=_positive_int, default=100_000)
    train_parser.add_argument(
        "--checkpoint-interval", type=_positive_int, default=250_000
    )
    train_parser.add_argument(
        "--replay-capacity", type=_positive_int, default=1_000_000
    )
    train_parser.add_argument(
        "--replay-kind", choices=("uniform", "prioritized"), default="uniform"
    )
    train_parser.add_argument("--priority-alpha", type=float, default=0.6)
    train_parser.add_argument("--priority-beta-start", type=float, default=0.4)
    train_parser.add_argument("--priority-beta-end", type=float, default=1.0)
    train_parser.add_argument("--priority-epsilon", type=float, default=1e-3)
    train_parser.add_argument("--batch-size", type=_positive_int, default=512)
    train_parser.add_argument(
        "--warmup-transitions", type=_positive_int, default=25_000
    )
    train_parser.add_argument("--replay-ratio", type=float, default=8.0)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument(
        "--target-update-interval", type=_positive_int, default=1_000
    )
    train_parser.add_argument("--epsilon-initial", type=float, default=1.0)
    train_parser.add_argument("--epsilon-final", type=float, default=0.05)
    train_parser.add_argument(
        "--epsilon-decay-transitions", type=_positive_int, default=1_000_000
    )
    train_parser.add_argument(
        "--inference-batch-size", type=_positive_int, default=8192
    )
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument(
        "--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE
    )
    train_parser.add_argument("--warm-start-checkpoint", type=Path)

    group_train = commands.add_parser(
        "train-groups",
        help="train scalar W from complete decision groups",
    )
    group_train.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/afterstate/value_group_5m_cpp"),
    )
    group_train.add_argument("--resume", action="store_true")
    group_train.add_argument("--seed", type=_non_negative_int)
    group_train.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_GROUP_SOURCE,
    )
    group_train.add_argument("--num-envs", type=_positive_int, default=128)
    group_train.add_argument("--total-transitions", type=_positive_int)
    group_train.add_argument(
        "--eval-seeds-from",
        type=Path,
        default=DEFAULT_VALIDATION_MANIFEST,
    )
    group_train.add_argument("--validation-games", type=_positive_int, default=200)
    group_train.add_argument(
        "--validation-interval", type=_positive_int, default=250_000
    )
    group_train.add_argument("--evaluation-envs", type=_positive_int, default=64)
    group_train.add_argument("--log-interval", type=_positive_int, default=100_000)
    group_train.add_argument(
        "--checkpoint-interval", type=_positive_int, default=250_000
    )
    group_train.add_argument("--group-capacity", type=_positive_int, default=1_000_000)
    group_train.add_argument(
        "--candidate-capacity", type=_positive_int, default=5_000_000
    )
    group_train.add_argument("--group-batch-size", type=_positive_int, default=500)
    group_train.add_argument("--warmup-groups", type=_positive_int, default=25_000)
    group_train.add_argument("--replay-ratio", type=float, default=8.0)
    group_train.add_argument("--learning-rate", type=float, default=3e-4)
    group_train.add_argument(
        "--target-update-interval", type=_positive_int, default=1_000
    )
    group_train.add_argument("--epsilon", type=float, default=0.05)
    group_train.add_argument("--inference-batch-size", type=_positive_int, default=8192)
    group_train.add_argument("--device", default=None)

    evaluate = commands.add_parser("evaluate", help="evaluate an afterstate checkpoint")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--seeds-from", type=Path, required=True)
    evaluate.add_argument("--games", type=_positive_int)
    evaluate.add_argument("--num-envs", type=_positive_int, default=64)
    evaluate.add_argument("--device", default="auto")

    final = commands.add_parser(
        "final-evaluate", help="compare afterstate and exact depth-1"
    )
    final.add_argument("--checkpoint", type=Path, required=True)
    final.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    final.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST
    )
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--games", type=_positive_int, default=200)
    final.add_argument("--seed", type=_non_negative_int, default=0xA57E_2000)
    final.add_argument("--num-envs", type=_positive_int, default=64)
    final.add_argument("--baseline-num-envs", type=_positive_int, default=8)
    final.add_argument("--device", default="auto")
    return parser


def _train_config(args: argparse.Namespace) -> TrainConfig:
    seed = secrets.randbits(63) if args.seed is None else int(args.seed)
    validation_seeds = _load_validation_seeds(
        args.eval_seeds_from, seed, args.validation_games
    )
    return TrainConfig(
        seed=seed,
        num_envs=args.num_envs,
        total_transitions=args.total_transitions or 10_000_000,
        replay_capacity=args.replay_capacity,
        replay_kind=args.replay_kind,
        priority_alpha=args.priority_alpha,
        priority_beta_start=args.priority_beta_start,
        priority_beta_end=args.priority_beta_end,
        priority_epsilon=args.priority_epsilon,
        batch_size=args.batch_size,
        warmup_transitions=args.warmup_transitions,
        replay_ratio=args.replay_ratio,
        learning_rate=args.learning_rate,
        target_update_interval=args.target_update_interval,
        epsilon_initial=args.epsilon_initial,
        epsilon_final=args.epsilon_final,
        epsilon_decay_transitions=args.epsilon_decay_transitions,
        validation_seeds=validation_seeds,
        validation_interval=args.validation_interval,
        evaluation_envs=args.evaluation_envs,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        inference_batch_size=args.inference_batch_size,
        device=args.device or "auto",
        baseline_checkpoint=str(args.baseline_checkpoint.resolve()),
        warm_start_checkpoint=(
            None
            if args.warm_start_checkpoint is None
            else str(args.warm_start_checkpoint.resolve())
        ),
    )


def _group_train_config(args: argparse.Namespace) -> GroupTrainConfig:
    seed = secrets.randbits(63) if args.seed is None else int(args.seed)
    validation_seeds = _load_validation_seeds(
        args.eval_seeds_from,
        seed,
        args.validation_games,
    )
    return GroupTrainConfig(
        seed=seed,
        source_checkpoint=str(args.source_checkpoint.resolve()),
        num_envs=args.num_envs,
        total_transitions=args.total_transitions or 5_000_000,
        group_capacity=args.group_capacity,
        candidate_capacity=args.candidate_capacity,
        group_batch_size=args.group_batch_size,
        replay_ratio=args.replay_ratio,
        warmup_groups=args.warmup_groups,
        learning_rate=args.learning_rate,
        target_update_interval=args.target_update_interval,
        epsilon=args.epsilon,
        validation_seeds=validation_seeds,
        validation_interval=args.validation_interval,
        evaluation_envs=args.evaluation_envs,
        inference_batch_size=args.inference_batch_size,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device or "auto",
    )


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "check":
        print(
            json.dumps(
                run_self_check(games=args.games, device_name=args.device),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "benchmark":
        print(
            json.dumps(
                benchmark_profile(
                    env_count=args.env_count,
                    transitions=args.transitions,
                    batch_size=args.batch_size,
                    device_name=args.device,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "train":
        if args.resume:
            config_path = args.run_dir / "config.json"
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            config = TrainConfig.from_dict(raw)
            if (
                args.total_transitions is not None
                and args.total_transitions != config.total_transitions
            ):
                config = TrainConfig.from_dict(
                    {**config.to_dict(), "total_transitions": args.total_transitions}
                )
            if args.device is not None and args.device != config.device:
                config = TrainConfig.from_dict(
                    {**config.to_dict(), "device": args.device}
                )
        else:
            config = _train_config(args)
        print(f"training_seed={config.seed}", flush=True)
        train(config, args.run_dir, resume=args.resume)
        return
    if args.command == "train-groups":
        if args.resume:
            config_path = args.run_dir / "config.json"
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            config = GroupTrainConfig.from_dict(raw)
            if (
                args.total_transitions is not None
                and args.total_transitions != config.total_transitions
            ):
                config = GroupTrainConfig.from_dict(
                    {**config.to_dict(), "total_transitions": args.total_transitions}
                )
            if args.device is not None and args.device != config.device:
                config = GroupTrainConfig.from_dict(
                    {**config.to_dict(), "device": args.device}
                )
        else:
            config = _group_train_config(args)
        print(f"training_seed={config.seed}", flush=True)
        train_groups(config, args.run_dir, resume=args.resume)
        return
    if args.command == "evaluate":
        seeds = load_manifest_seeds(args.seeds_from, args.games)
        result = evaluate_afterstate_checkpoint(
            args.checkpoint, seeds, device_name=args.device, num_envs=args.num_envs
        )
        print(json.dumps(result.summary_dict(), ensure_ascii=False, indent=2))
        return
    if args.command == "final-evaluate":
        source = load_manifest_seeds(args.source_manifest)
        seeds = generate_independent_seeds(source, seed=args.seed, count=args.games)
        report = run_final_evaluation(
            args.checkpoint,
            args.baseline_checkpoint,
            seeds,
            args.output_dir,
            device_name=args.device,
            num_envs=args.num_envs,
            baseline_num_envs=args.baseline_num_envs,
        )
        print(
            json.dumps(
                {key: value for key, value in report.items() if key not in {"paired"}},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
