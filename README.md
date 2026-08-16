# Next Station: London solver

这是一个面向算法研究的项目，以 C++ 实现《Next Station: London》游戏引擎，并用 Python 实现决策策略和训练流程。

本仓库只包含核心引擎、solver 和实验代码。`src/web/` 与 `src/helper/` 仅保留在本地开发工作区，不纳入版本控制，也不会随仓库同步。

当前规则范围：

- 四种颜色的完整顺序在开局随机确定并公开；
- 每轮独立洗 11 张 Station cards，翻到第 5 张 Underground card 后结束；
- 始发站计入线路的区域数和站点数；
- Tourist、Thames 和 Interchange 正常计分；
- pass 始终合法；
- 基础规则默认关闭 Shared Objectives 和 Pencil Powers；
- C++ engine、传统 solver 和实验入口可独立开启任一进阶模块，或同时开启两者。

官方规则及地图资料保存在 `rule/`。

## 目录

```text
src/
  engine_cpp/   # C++ 游戏状态、随机抽牌、规则、计分与薄 Python 接口
  solver/       # 公开状态上的搜索与决策算法
    RL/         # batched env、replay、DQN 和训练入口
  experiments/  # 完整对局、并行实验、JSONL 和统计比较
benchmark_results/  # 已完成实验的数据与摘要
rule/               # 官方规则和地图资料
```

依赖方向保持单向：`solver -> engine_cpp`、`experiments -> solver + engine_cpp`。C++ engine 不知道任何算法，solver 也不处理进程、文件或整批实验。

## C++ Engine

solver、实验入口和 RL collector 默认使用 C++ 后端。Windows 下从 x64 Visual
Studio developer prompt 构建：

```powershell
cmake -S src/engine_cpp -B build/engine-cpp `
  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/engine-cpp --parallel
build/engine-cpp/next_station_engine_check.exe
```

Python 默认从 `build/engine-cpp` 加载动态库；也可以通过
`NEXT_STATION_NATIVE_LIBRARY` 指定兼容的 Release library。C++ RNG 对同一个
seed 可复现。旧 Python 规则引擎已经删除；其 seed 与 C++ 后端不对应同一组
隐藏牌序，因此旧 benchmark 需要在新后端上重跑。

## 本地工具

网页和 helper 不属于仓库内容，只需在本地工作区直接启动：

```powershell
$env:PYTHONPATH = "src"
python -m web --port 8000
```

```powershell
$env:PYTHONPATH = "src"
python -m helper play
```

Seed 只控制真实对局的颜色顺序和洗牌。Random player 的动作、Greedy 同分以及 MCTS 内部采样不由该 seed 控制。

## Solver

- `GreedyPolicy`：最大化当前动作的精确即时计分增量，同分随机。
- `DepthKPolicy`：使用统一递归枚举未来 `k` 次公开翻牌及其概率，不附加叶节点估值。
- `Depth2Policy`：与 `DepthKPolicy(2)` 数学等价的显式特化实现，避免通用递归在最常用深度上的额外开销。
- `MCTSPolicy`：公共状态上的 chance-sampled UCT；每个节点包含 pass，未知抽牌只从当前剩余牌中采样，首次到达的状态使用确定性 Greedy 完成对局。
- `solver.RL`：London 基础规则固定动作空间上的 Masked Double DQN 和批量 DQN-rollout MCTS；只编码公开信息，不读取隐藏牌序，当前不支持进阶模块。

Shared Objectives 与 Pencil Powers 是两个独立开关。共同目标在首次达成时产生与终局 `+10` 完全等价的即时奖励；Circle 只把所圈车站在当前线路对应区域的站数中额外计一次，不会制造真实站点或帮助共同目标。Double Section 的第二段作为同一次翻牌下的后续决策，不额外消耗 Lookahead 深度。

MCTS 的默认配置是调参后选定的：

```python
MCTSPolicy(simulations=5120, exploration=22.5)
```

## 强化学习

强化学习依赖是可选的，不会影响 C++ engine、传统 solver 或网页：

```powershell
python -m pip install -e ".[rl]"
```

先运行内存自检，再测量不同 env batch 的完整决策吞吐：

```powershell
python -m solver.RL check --games 32 --device cuda
python -m solver.RL benchmark-env --env-counts 16 32 64 128 --device cuda
```

默认训练配置是 128 个同步 env、batch 512、1-step return、`gamma=1`、100 万容量的 bit-packed uniform replay，以及 1000 万 transitions。训练命令不接受指定 seed；每个新 run 会从系统随机源生成内部 RNG seed 并写入配置和 checkpoint。训练局来自持续生成的随机 seed 流，下面仅把现有 200 seeds 用于评估：

```powershell
python -m solver.RL train `
  --run-dir artifacts/dqn/fullq_n1_uniform `
  --n-steps 1 `
  --eval-seeds-from benchmark_results/current_200/manifest.json `
  --device cuda
```

训练会持续更新 `metrics.jsonl`、`latest.pt`、`best.pt`、`summary.md` 和 `replay.dat`。中断后可以提高或保持总预算并恢复：

```powershell
python -m solver.RL train `
  --run-dir artifacts/dqn/fullq_n1_uniform `
  --resume `
  --total-transitions 12000000 `
  --device cuda
```

在固定 seed manifest 上评估 checkpoint：

```powershell
python -m solver.RL evaluate `
  --checkpoint artifacts/dqn/fullq_n1_uniform/best.pt `
  --seeds-from benchmark_results/current_200/manifest.json `
  --games 200 `
  --device cuda
```

标量网络直接拟合标准 Double DQN target：

```text
Q(s, a) <- r + gamma * Q_target(s', argmax_a Q_online(s', a))
```

即时奖励只作为 transition 的 `r`，不参与动作分数的额外拼接。`reward_scale=10` 仅用于数值缩放；网络仍然学习完整 Q，而不是未来价值残差。

### DQN-MCTS

`DQNMCTSPolicy` 保留 chance-sampled UCT 的树策略，并使用确定性的 Double
DQN 完成首次到达节点后的 rollout。每个真实对局由一个 solver session
持有搜索树；行动和真实抽牌后，如果对应公开状态已经被搜索过，就将该节点
重定为新根并只保留其可达子图。每次决策仍新增固定数量的 simulation。

多棵独立搜索树共用预分配的编码和 GPU 推理缓冲。每棵树同时最多只有一条
未完成 simulation，因此下一次 selection 一定能看到上一条 simulation 的
真实回传结果，不使用 virtual loss。完成根搜索的对局会立即行动并开始下一次
搜索，不需要等待同一批中的其他对局。

在固定 seed manifest 上运行独立、可恢复的实验：

```powershell
python -m experiments.dqn_mcts `
  --checkpoint artifacts/dqn/fullq_n1_uniform_10m/best.pt `
  --seeds-from benchmark_results/current_200/manifest.json `
  --games 200 `
  --parallel-games 32 `
  --simulations 800 `
  --exploration 22.5 `
  --device cuda `
  --output benchmark_results/dqn_mcts/games.jsonl
```

这里的 `--parallel-games` 是共享一次 DQN 推理流的独立树数量，不是进程数，
也不是每棵树的并行 simulation 数。`--profile` 可以额外报告 tree、状态复制、
状态键、编码、GPU 推理和 rollout step 的分阶段耗时；正常实验不启用它，以免
细粒度计时影响吞吐。

当前基准中的 DQN-MCTS-800 结果 `155.86 +/- 1.06` 来自引入异步调度和
子树复用之前的实现，耗时 1:53:57，平均物理推理 batch 为 `17.4`。新实现
使用不同的 policy 名称写结果，避免与旧 JSONL 混合。

RTX 4060 Ti 上的 10M-transition、1-step、uniform replay 实验如下；checkpoint 在现有 200 seeds 上选择并评估：

| Algorithm | Best step | Best | Final | Time |
| --- | ---: | ---: | ---: | ---: |
| Full-Q Double DQN | 9.0M | `144.64 +/- 1.07` | `144.12 +/- 1.08` | 31.4 min |

## 实验

实验编排全部位于 `src/experiments/`。下面的命令复用当前 manifest 中完全相同的对局 seed，并让 16 个完整对局并行：

```powershell
python -m experiments.mcts `
  --seeds-from benchmark_results/current_200/manifest.json `
  --games 200 `
  --workers 16 `
  --output benchmark_results/new_agent/games/mcts.jsonl
```

默认使用 `5120 / 22.5`。每局完成后立即追加 UTF-8 JSONL；再次执行同一命令会跳过已有 seed。单 policy runner 只写 JSONL，统计结果打印到终端。

实验入口可追加 `--shared-objectives` 或 `--pencil-powers` 独立开启模块；`--advanced` 等价于同时开启两者。例如：

```powershell
python -m experiments.mcts `
  --seeds-from benchmark_results/current_200/manifest.json `
  --games 200 `
  --workers 16 `
  --advanced `
  --output benchmark_results/advanced/games/mcts.jsonl
```

重跑当前维护的全部策略时使用：

```powershell
python -m experiments.benchmark `
  --seeds-from benchmark_results/current_200/manifest.json `
  --games 200 `
  --workers 16 `
  --output-dir benchmark_results/new_benchmark
```

完整 benchmark 保留各 policy 的 JSONL、`manifest.json` 和唯一一份总体 `summary.md`。

比较两个策略时必须使用相同 seed：

```powershell
python -m experiments.compare `
  benchmark_results/baseline.jsonl `
  benchmark_results/candidate.jsonl
```

工具会报告配对均值差、标准误、95% 区间、胜/平/负，以及颜色顺序和抽牌序列是否一致。

## 当前基准与目标

当前 200-seed 重跑结果为：Simple random `94.98`、Greedy `125.16`、Lookahead-2 `137.99`、Lookahead-4 `148.40`、DQN-MCTS-800 `155.86`、默认 MCTS `157.05 +/- 1.03`；MCTS 最高单局为 `196`。汇总保存在 `benchmark_results/current_200/summary.md`。

下一阶段应以固定 seed 集上的均分为主指标：

- 近期目标：独立验证均分达到 `165`；
- 延伸目标：稳定达到 `170`；
- 单局里程碑：在有利牌序上突破 `200`，但不把幸运高分当作策略强度。

官方只把 `>150` 列为最高单人评价档，并未给出基础规则的理论最大值。公开资料中能找到 `222` 分的截图，但没有说明是否启用了高级模块，因此不能把它当作当前规则的已验证最优解。
