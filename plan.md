# Evaluation Plan

## Seed Sets

- `benchmark_results/current_200/manifest.json` is the fixed development set.
- `benchmark_results/large_4096/manifest.json` is the fixed confirmation set.
- The 4,096 seeds are unique and disjoint from `current_200`.
- Every policy in a comparison uses the same ordered seed list.

## Search Baselines

Run on both seed sets:

- random
- greedy
- lookahead-2, lookahead-3, lookahead-4
- MCTS-5120

## Full-Q

- current_200: raw DQN, exact chance depth-1, and exact chance depth-2.
- large_4096: raw DQN and exact chance depth-1.
- The large-sample depth-2 run is intentionally omitted because of its
  disproportionate runtime.

Checkpoint:
`artifacts/dqn/fullq_n1_uniform_10m/best.pt`.

## Afterstate

Evaluate `greedy` (raw), `online-online` (main exact backup), and
`online-target` (target-lag diagnostic) for:

- group-5p75m
- group-7m
- group-9p75m
- group-10m
- scalar-9m
- scalar-10m

The group checkpoint paths and SHA-256 values are recorded in each output
manifest. All afterstate evaluations use 64 environments and an inference
batch size of 8,192.

## Outputs

- `benchmark_results/current_200/summary.md`
- `benchmark_results/current_200/games/`
- `benchmark_results/large_4096/summary.md`
- `benchmark_results/large_4096/games/`

Each root `manifest.json` defines the seed set and records checkpoint metadata.
`benchmark_results/summarize.py` discovers result files under `games/`, refreshes
the manifest index, and rebuilds the unified summary.

Report means, standard errors, and paired score differences. The benchmark
solver policies retain their existing unseeded agent-randomness semantics;
game seeds remain fixed and shared within each comparison.
