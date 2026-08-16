# Current Experiment: 1024-Seed Afterstate Checkpoint Comparison

## Goal

Measure the raw and exact-backup performance of every distinct retained
checkpoint on one shared, previously unused set of 1,024 game seeds. No
training or checkpoint selection is performed in this experiment.

## Seeds

- Manifest: `benchmark_results/afterstate_checkpoints_new_1024/seeds.json`.
- Generator seed: `0xA57E1024`.
- All 1,024 seeds are unique.
- They are disjoint from every `game_seeds` list in the existing benchmark
  manifests, including `current_200` and the previous independent 200.
- Every checkpoint and inference mode uses the same ordered seeds.

## Checkpoints

1. Original scalar 9M best:
   `artifacts/afterstate/value_10m_cpp/best.pt`.
2. Original scalar 10M latest and decision-group source:
   `artifacts/afterstate/value_10m_cpp/latest.pt`.
3. Decision-group 5.75M best raw:
   `artifacts/afterstate/value_group_5m_workspace_cpp/best_raw.pt`.
4. Decision-group 7M best backup:
   `artifacts/afterstate/value_group_5m_workspace_cpp/best.pt`.
5. Decision-group 9.75M best backup:
   `artifacts/afterstate/value_group_10m_workspace_cpp/best.pt`.
6. Decision-group 10M best raw:
   `artifacts/afterstate/value_group_10m_workspace_cpp/best_raw.pt`.

The original decision-group 5M checkpoint no longer exists and is therefore
not included. Duplicate `latest.pt` files with identical weights are omitted.

## Inference Modes

For each checkpoint evaluate:

1. `greedy`: raw `argmax_a [r_a / 10 + W_online(o_a)]`.
2. `online-online`: one exact Bellman improvement with online selection and
   online continuation evaluation. This is the main deployed policy.
3. `online-target`: the same improvement with target continuation evaluation,
   retained only as a target-lag diagnostic.

Use CUDA, 64 parallel environments, inference batches of 8,192, and 10 native
C++ expansion threads.

## Output

- Directory: `benchmark_results/afterstate_checkpoints_new_1024`.
- Save every ordered score vector, checkpoint SHA-256, summary statistics,
  action agreement, and regret.
- Report means and standard errors for all checkpoints and main modes.
- Use paired score differences on the shared seeds when comparing checkpoints;
  do not infer progress from unpaired standard errors alone.
