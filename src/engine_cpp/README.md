# C++ game engine

This directory owns the London game state, random draws, base rules, Shared
Objectives, Pencil Powers, scoring, public copies, and native serialization.
Python exposes the existing object API through `engine_cpp.GameSession`; solver,
experiment, and RL entry points use that adapter.

Solver-specific native search code lives in `src/solver/cpp`. It links this
engine but is not part of the engine's rules or state ownership.

The C ABI also exports immutable map and deck metadata plus standard-game legal
edge masks. Python builds lightweight value views from those exports; it does
not reconstruct map geometry, card layout, leaf stations, or legal moves.

The implementation stores each line's stations in one `uint64_t`, its 155
sections in three `uint64_t` words, and card/district state in fixed-width
masks. The C ABI also provides the 1041-dimensional afterstate feature schema
and batched exact-target expansion.

## Build

From an x64 Visual Studio developer prompt:

```powershell
cmake -S src/engine_cpp -B build/engine-cpp `
  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/engine-cpp --parallel
build/engine-cpp/next_station_engine_check.exe
build/engine-cpp/solver-cpp/next_station_solver_check.exe
```

The Python adapter searches `build/engine-cpp` and its `Release` subdirectory.
`NEXT_STATION_NATIVE_LIBRARY` can select another compatible Release library.

Exact-target expansion runs across native C++ worker threads. Set
`NEXT_STATION_NATIVE_THREADS` to a positive integer to cap the workers; when it
is unset, the engine uses the reported hardware concurrency, limited by the
input batch size. On the current 20-logical-processor development machine,
10 workers give the best end-to-end target time without occupying threads that
do not improve throughput.

The exact-target C ABI also provides `ns_expansion_select_candidates` and
`ns_expansion_reduce_targets`. They move the variable-length first-argmax and
the owner-wise probability reduction into C++ while preserving candidate order,
first-tie selection, terminal handling, reward scaling, discounting, and the
original floating-point reduction order. Python still performs the online and
target network inference; the native calls only replace the deterministic
array operations around those inferences.

Run the cross-language rule and serialization checks without pytest:

```powershell
$env:PYTHONPATH = "src"
python -m engine_cpp.checks
python -m solver.checks
python -m rl.afterstate check --games 4 --device cpu
python -m solver.RL check --games 4 --device cpu
```

Seeded games are reproducible within this engine. The C++ random generator is
different from the retired Python backend, so old and new seed numbers do not
identify the same hidden card order.
