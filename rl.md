# Review 后的强化学习实验总结

更新时间：2026-08-13

## 结论

`review.md` 建议的第一条主线已经得到验证：将已知的抽牌概率精确展开，明显优于直接使用 DQN。

- 原始 DQN 为 `144.64 +/- 1.07`。
- 同一网络加 exact-chance depth-1 后为 `150.26 +/- 1.10`。
- 同一网络加 exact-chance depth-2 后为 `153.40 +/- 1.07`。
- 使用 depth-1 exact-chance Bellman target 继续训练后，最佳纯 DQN 提升到当前代码复评的 `152.89 +/- 1.11`。
- 最佳 exact-target 网络加 exact-chance depth-2 后达到 `157.62 +/- 1.12`。
- Paired-scenario rollout 从 32 提高到 128 scenarios/action 后，由 `153.51` 提升到 `158.01 +/- 1.13`；采样噪声确实是第一版失败的主因。
- 用 5,000 个全新 teacher 对局进行第一轮 depth-2 Q 蒸馏后，raw student 没有变强，但 student 作为 depth-2 leaf 在旧 200 局开发集上达到 **`158.52 +/- 1.04`**，比母网络同配置提高 `0.91 +/- 0.43`。
- 在与旧评测及 teacher 数据均不重合的新 100 局上，母网络 depth-2 为 `157.43 +/- 1.52`，student 为 `158.06 +/- 1.62`；配对差值 `+0.63 +/- 0.73`，95% CI `[-0.79, +2.05]`。提升方向复现了，但证据还不足以确认稳定提升。
- 第三轮 `best.pt` 内 online/target Q 均值作为 depth-2 leaf 达到 `158.17 +/- 1.10`，比单独 online 高 `0.55 +/- 0.31`，但没有超过蒸馏 student。
- 在蒸馏 student 的 200 局 depth-2 轨迹上，depth-1 与 depth-2 只在 `587/7316 = 8.0%` 的状态选择不同；分歧更集中在后两个回合、每轮后半段和动作数较多的状态。
- 五种选择性 depth-3/精确回合尾局配置在前 64 个开发 seeds 上都没有超过固定 depth-2，因此没有扩跑 200 局。

目前最值得保留的组合是：

```text
distill_depth2_5000/student/best.pt
    + ExactChanceDQNPolicy(depth=2)
    = 158.52 +/- 1.04
```

这是旧 200 局开发集上目前最强的 RL 系列组合。它与 Paired-128 的 `158.01 +/- 1.13` 和 Greedy-rollout MCTS-5120 的 `157.05 +/- 1.03` 在统计上相当，但推理方式与成本不同。当前 200 seeds 已反复参与模型选择和算法开发，因此这些数字不能当作最终泛化成绩；新 100 局的独立确认应当优先于开发集排名解读。

## 统一评估口径

除特别说明外，以下结果都使用 `benchmark_results/current_200/manifest.json` 中相同的 200 个基础规则 seeds：

- 不开启共同目标；
- 不开启铅笔能力；
- 对局 seed 同时决定颜色顺序和真实牌序；
- DQN 采用合法动作上的确定性 `argmax`；
- 表中的 `SE` 是 200 局最终分数均值的标准误差。

基础规则的合法动作和计分在进阶模块加入前后没有变化，因此结果可以直接横向比较。

## 总成绩

| 网络 / 策略 | Mean | SE | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 DQN | 144.64 | 1.07 | 103 | 147.0 | 185 |
| 原始 DQN + exact depth-1 | 150.26 | 1.10 | 102 | 151.5 | 194 |
| 原始 DQN + exact depth-2 | 153.40 | 1.07 | 106 | 154.0 | 191 |
| 第一轮 exact-target DQN | 148.65 | 1.02 | 101 | 148.0 | 183 |
| 第二轮最佳 DQN | 150.79 | 1.05 | 110 | 150.5 | 194 |
| 第二轮最佳 + exact depth-1 | 155.20 | 1.08 | 110 | 156.0 | 191 |
| 第二轮最佳 + exact depth-2 | 156.87 | 1.11 | 116 | 158.0 | 191 |
| 第三轮最佳 DQN | 152.89 | 1.11 | 109 | 155.0 | 201 |
| 第三轮最佳 + exact depth-1 | 155.47 | 1.08 | 109 | 157.0 | 203 |
| 第三轮最佳 + exact depth-2 | 157.62 | 1.12 | 109 | 159.0 | 203 |
| 第三轮最佳 + paired scenario 32 | 153.51 | 1.06 | 106 | 155.0 | 187 |
| 第三轮最佳 + paired scenario 128 | 158.01 | 1.13 | 112 | 159.5 | 194 |
| 纯 Q 蒸馏 student | 151.46 | 1.12 | 106 | 154.0 | 195 |
| 纯 Q 蒸馏 student + exact depth-1 | 155.66 | 1.04 | 110 | 156.5 | 193 |
| 纯 Q 蒸馏 student + exact depth-2 | **158.52** | **1.04** | 109 | 160.0 | 193 |
| 排序蒸馏 student | 152.12 | 1.06 | 115 | 153.0 | 197 |
| 排序蒸馏 student + exact depth-2 | 158.19 | 1.11 | 109 | 160.0 | 195 |
| 第三轮 best online/target Q 均值 + exact depth-2 | 158.17 | 1.10 | 109 | 159.0 | 202 |

作为非 RL 对照：

| 策略 | Mean | SE |
| --- | ---: | ---: |
| DQN-MCTS-800 | 155.86 | 1.06 |
| Greedy-rollout MCTS-5120 | 157.05 | 1.03 |

第三轮 exact depth-2 相比 MCTS-5120 为 `+0.57 +/- 0.82`，95% CI 为 `[-1.05, +2.18]`，没有显著差异；相比 DQN-MCTS-800 为 `+1.76 +/- 0.79`，95% CI 为 `[+0.21, +3.30]`。

## 1. Exact-Chance DQN 推理

首先直接复用原始 Full-Q、1-step、uniform replay DQN：

```text
artifacts/dqn/fullq_n1_uniform_10m/best.pt
```

在每个决策动作之后，不读取真实隐藏牌序，而是精确枚举下一次所有公开抽牌事件及其概率。depth-1 的核心形式是：

```text
Q1(s, a) = r(s, a) + E_o[max_b Q_DQN(s[a, o], b)]
```

depth-2 再展开一层动作和机会节点，并只在边界使用 DQN。Pass 始终属于候选动作。

200 局结果：

| Policy | Mean | 相比 raw DQN | 配对 95% CI | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| Raw DQN | 144.64 | - | - | - |
| Exact depth-1 | 150.26 | +5.62 | [+3.92, +7.33] | 136/3/61 |
| Exact depth-2 | 153.40 | +8.76 | [+7.14, +10.38] | 154/7/39 |

depth-2 相比 depth-1 仍增加 `3.13 +/- 0.82`，95% CI `[+1.52, +4.75]`。因此 review 对“网络边界价值有用，但直接 argmax 没有充分利用已知 chance model”的判断成立。

结果目录：`benchmark_results/exact_chance_dqn_200/`。

## 2. Depth-1 Exact-Chance Target 训练

训练目标改为一阶精确期望的 Double DQN target：

```text
y = r / 10
    + E_o[Q_target(s'_o, argmax_b Q_online(s'_o, b))]
```

其中 `o` 枚举动作后下一次所有可能的公开抽牌事件。在线网络负责选动作，target 网络负责估值；`gamma=1.0`，因为这是有限时域、无折扣的总分最大化问题。训练仍为普通 Full-Q Double DQN，没有 reward decomposition。

共同设置：

- `128` 个同步环境；
- batch size `512`；
- uniform replay，容量 `1,000,000`；
- replay ratio `8`；
- 1-step return；
- target network 每 `1,000` updates 同步；
- 每 `250k` transitions 在旧 200 seeds 上验证；
- 每轮从上一轮最佳网络初始化，但 replay 和 optimizer 重新开始。

### 第一轮：2M，标准探索率

目录：`artifacts/dqn/exact_chance_depth1_2m/`

- 初始化：原始 `144.64` DQN；
- learning rate：`3e-4`；
- epsilon：`1.0 -> 0.05`，前 1M transitions 线性下降；
- transitions：`2,000,000`；
- updates：`30,860`；
- 用时：`5,857.8s`；
- 最佳和最终验证：`148.65 +/- 1.02`。

训练初期曾降至约 `141.7`，之后恢复并超过初始化网络。高探索率会生成大量弱行为数据，但最终仍获得 `+4.01` 的 raw DQN 提升。

### 第二轮：2M，固定低探索率

目录：`artifacts/dqn/exact_chance_depth1_low_epsilon_2m/`

- 初始化：第一轮 `best.pt`；
- learning rate：`3e-4`；
- epsilon：始终 `0.05`；
- transitions：`2,000,000`；
- updates：`30,860`；
- 用时：`6,043.1s`；
- 最佳验证：`150.795`，位于约 1.75M transitions；
- 最终验证：`150.235 +/- 1.14`。

对本轮最佳 checkpoint 做完整搜索评估：

| Policy | Mean | 相比 raw DQN |
| --- | ---: | ---: |
| Raw DQN | 150.79 | - |
| Exact depth-1 | 155.20 | +4.41 |
| Exact depth-2 | 156.87 | +6.07 |

低探索率延续训练有效，而且 exact search 仍然能在新网络上继续改善决策。

### 第三轮：4M，降低学习率

目录：`artifacts/dqn/exact_chance_depth1_lr1e4_4m/`

- 初始化：第二轮 `best.pt`；
- learning rate：`1e-4`；
- epsilon：始终 `0.05`；
- transitions：`4,000,000`；
- updates：`62,110`；
- 用时：`12,028.3s`；
- 最佳验证：约 3.0M transitions 时 `152.83`；
- 最终验证：`151.775 +/- 1.11`。

用当前代码重新加载磁盘 checkpoint 后：

| Checkpoint | Mean | SE |
| --- | ---: | ---: |
| `best.pt` | 152.89 | 1.11 |
| `latest.pt` | 151.69 | 1.12 |

逐 seed 的 `latest - best` 为 `-1.20 +/- 0.68`，95% CI `[-2.53, +0.13]`，W/T/L 为 `75/32/93`。后 1M transitions 的确呈退化倾向，因此后续必须始终保留并使用最佳 checkpoint。

`152.83` 与 `152.89` 的微小差异已经定位。训练进程启动时，早期 Railroad Switch 的 `source_any` 观察位仍使用旧编码；后来该位改成始终如实表示 Switch。基础规则下两种编码产生完全相同的合法动作，但网络输入有一个冗余 bit 不同。用旧观察编码复评 `best.pt` 可精确复现 `152.83`；当前代码的可复现结果是 `152.89`。这不是基础规则或计分变化。

第三轮最佳 checkpoint 的完整结果：

| Policy | Mean | 相比 raw DQN | 配对 95% CI | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| Raw DQN | 152.89 | - | - | - |
| Exact depth-1 | 155.47 | +2.58 | [+1.22, +3.93] | 116/19/65 |
| Exact depth-2 | 157.62 | +4.72 | [+3.20, +6.25] | 125/16/59 |

depth-2 相比 depth-1 为 `+2.15 +/- 0.55`，95% CI `[+1.07, +3.23]`。随着 raw DQN 变强，搜索带来的绝对增益从 `+8.76` 缩小到 `+4.72`，但仍然稳定为正。

### 为什么 loss 继续下降而游戏均分下降

这不是矛盾。训练 loss 衡量网络对当前 replay 分布和不断移动的 Bellman target 的拟合；最终分数由约 35--40 次离散 `argmax` 共同决定。Q 值很小的排序变化就可能改变整条轨迹，而不需要让平均 Huber loss 上升。

当前训练还同时包含：

- bootstrapping 和 target network 带来的非平稳目标；
- 策略改善后不断变化的 replay 状态分布；
- 固定容量 replay 对旧数据的替换；
- 函数逼近造成的动作间误差传播。

所以“更多 SGD updates”不保证策略单调改善。第三轮曲线是先从 `150.80` 提升到 `152.83`，再回落，而不是从头持续下降。正确处理方式是 checkpoint selection 和 early stopping，而不是假设最后权重最好。

## 3. Paired-Scenario DQN Rollout

实现目录：

- `src/solver/RL/paired_scenario.py`
- `src/experiments/paired_scenario_dqn.py`

每个根动作都面对完全相同的一组未来随机牌序：根动作强制执行一次，后续始终由确定性 DQN 行动。共同随机数应当消除“某副未来牌本身好或坏”的大部分比较噪声。第一版固定为每个动作 `32` 个完整 continuation，不做逐轮淘汰。

使用第三轮最佳网络，在 200 seeds 上：

```text
Mean:   153.51 +/- 1.06
Range:  106..187
Median: 155.0
```

配对比较：

| 对照 | Paired-32 差值 | 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Raw DQN | +0.62 +/- 0.84 | [-1.03, +2.27] | 103/7/90 |
| Exact depth-1 | -1.96 +/- 0.80 | [-3.51, -0.40] | 86/7/107 |
| Exact depth-2 | -4.11 +/- 0.81 | [-5.69, -2.52] | 78/10/112 |
| DQN-MCTS-800 | -2.35 +/- 0.84 | [-4.00, -0.70] | 78/11/111 |
| MCTS-5120 | -3.54 +/- 0.91 | [-5.33, -1.75] | 77/9/114 |

结果目录：`benchmark_results/paired_scenario_dqn_200/`。

最初 10 局的累计均分曾是 `163.00`，但这批 seeds 本身偏容易：同 10 局 raw DQN 为 `160.20`，exact depth-2 为 `159.70`，MCTS 为 `159.90`。其余 190 局 Paired-32 只有 `153.01`。因此“均值一路下降”主要是 seed 顺序造成的累计曲线假象；完整 200 局结果才有意义。

当前不能仅凭全局分数判断失败来自哪里。两种主要可能是：

1. `32` scenarios/action 不足以稳定辨别相差几分的根动作；
2. 即使估计精确，强制根动作后由 raw DQN rollout 的目标也不如 exact depth-2 的短视野精确 backup。

随后用固定状态诊断区分这两者，而不是直接盲跑完整的 128/256-scenario 200 局。

### 固定状态稳定性诊断

固定公开状态的 paired-scenario 稳定性诊断已经完成。选取 8 个开发 seeds，并在每条 raw DQN 轨迹的早期、中期、后期各取一个状态，共 24 个状态。结果目录为 `benchmark_results/paired_scenario_diagnostics/`。

使用 Paired-2048 作为高预算参考。在全部 24 个状态上，Paired-2048 与 exact depth-2 的最优动作完全相同。

| Budget | Repeats | 高预算动作一致率 | 重复中始终稳定的状态 | 高预算估值下平均 regret |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 94.3% | 19/24 | 0.061 |
| 128 | 4 | 96.9% | 22/24 | 0.014 |
| 256 | 2 | 97.9% | 23/24 | 0.002 |

这说明第一版 Paired-32 的主要问题确实包含抽样噪声，而不是完整 DQN rollout 系统性偏向了与 exact depth-2 不同的动作。32 samples 在 5/24 个状态上会随 scenario seed 改变选择；128 将其降到 2/24，256 降到 1/24。

这个诊断也有边界：状态来自 raw DQN 轨迹，样本只有 24 个，而且 Paired-2048 本身仍是 Monte Carlo 参考，不能由此证明两种策略在所有状态完全等价。

随后运行完整的 Paired-128 / 200 seeds。先选 128 而不是 256，是因为它已经消除了大部分可观察的不稳定性；256 的固定状态一致率只再提高约 1 个百分点，计算量却再翻倍。

完整结果：

```text
Paired-128: 158.01 +/- 1.13
range:      112..194
median:     159.5
```

配对比较：

| 对照 | Paired-128 差值 | 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Paired-32 | +4.50 +/- 0.82 | [+2.89, +6.11] | 127/5/68 |
| Raw DQN | +5.12 +/- 0.84 | [+3.47, +6.77] | 133/8/59 |
| Exact depth-1 | +2.55 +/- 0.71 | [+1.15, +3.94] | 117/8/75 |
| Exact depth-2 | +0.40 +/- 0.68 | [-0.94, +1.73] | 92/20/88 |
| MCTS-5120 | +0.96 +/- 0.81 | [-0.63, +2.55] | 102/7/91 |

结果目录：`benchmark_results/paired_scenario_dqn_128_200/`。

因此 128 samples 已经显著修复 32 samples 的噪声，但还不能证明它强于 exact depth-2 或 MCTS-5120。由于 Paired-256 在固定状态上相比 128 只多约 1 个百分点一致率、计算量却翻倍，本轮没有继续跑完整 Paired-256。

## 4. Exact Depth-2 Teacher 蒸馏

实现：

- `src/solver/RL/distillation.py`：单数据库标签存储和监督训练；
- `src/experiments/exact_chance_distillation.py`：随机 teacher 对局生成；
- teacher 直接复用 `ExactChanceDQNPolicy(depth=2)` 的全部根动作估值，没有复制搜索算法。

### Teacher 数据

使用第三轮最佳网络作为 depth-2 leaf，生成 5,000 个全新随机基础对局；旧 200 seeds 被显式排除，且不指定训练 seed。每个决策状态保存 observation、legal mask 和所有合法动作的 teacher Q。每 10 个完整对局中有 1 个整体进入验证集，因此同一局相邻状态不会跨 split。

所有数据保存在单个 SQLite 数据库中：

```text
artifacts/dqn/distill_depth2_5000/teacher.sqlite
```

数据统计：

| 项目 | 数值 |
| --- | ---: |
| Teacher games | 5,000 |
| Total positions | 183,111 |
| Train positions | 164,833 |
| Validation positions | 18,278 |
| Teacher mean score | 158.03 |
| Teacher range | 94..207 |
| Database size | 152,481,792 bytes |
| Generation time | 12,260.1s（约 3h 24m） |

数据库通过 SQLite integrity、唯一 ordinal/seed、外键、split 和 blob 尺寸检查。teacher 平均每局产生约 36.62 个决策状态。

### 纯 Q 回归

第一版从母网络初始化，对每个状态的所有合法动作等权拟合：

```text
L = mean_legal Huber(Q_student, Q_teacher)
```

设置：batch `2048`、learning rate `1e-4`、30 epochs，以验证集 teacher 动作一致率为主选择 checkpoint。最佳为 epoch 7：

| 离线指标 | 初始化母网络 | 最佳 student |
| --- | ---: | ---: |
| Teacher action agreement | 85.55% | 85.87% |
| Mean teacher regret | 0.161 | 0.155 |
| Q RMSE | 0.163 | 0.143 |

实际 200 局：

| Policy | Mean | SE | 相比对应母网络 |
| --- | ---: | ---: | ---: |
| Raw student | 151.46 | 1.12 | -1.43 +/- 0.74 |
| Student + exact depth-1 | 155.66 | 1.04 | +0.20 +/- 0.52 |
| Student + exact depth-2 | **158.52** | **1.04** | **+0.91 +/- 0.43** |

depth-2 的母子网络配对 95% CI 为 `[+0.06, +1.75]`，W/T/L 为 `64/90/46`。改进很小，但在这批开发 seeds 上配对显著。`latest.pt` 的验证 Q/advantage RMSE 更低，却只有 `158.18`，没有超过 epoch-7 `best.pt`；继续压低全局回归误差不保证搜索策略提高。

关键结论不是“蒸馏出了更强 raw policy”：raw student 实际退化了。成立的是另一件事：蒸馏后的 Q 函数作为 depth-2 搜索叶值，比母网络更好，完成了第一轮小幅 policy improvement。

### 排序感知回归

为了直接强调合法动作的相对排序，又使用同一数据库训练：

```text
L = L_absolute_Q
  + L_centered_advantage
  + 0.25 * KL(teacher_policy || student_policy)
```

policy temperature 为 2 分。最佳 epoch 19 的动作一致率提高到 `86.34%`，teacher regret 降到 `0.146`。实际结果：

| Policy | Mean | SE |
| --- | ---: | ---: |
| Raw ranked student | 152.12 | 1.06 |
| Ranked student + exact depth-1 | 155.01 | 1.07 |
| Ranked student + exact depth-2 | 158.19 | 1.11 |

它的 raw policy 比纯 Q student高 `0.66 +/- 0.57`，但仍未超过母网络；depth-2 比纯 Q student低 `0.33 +/- 0.40`。因此这组系数改善了离线排序指标，却没有改善当前最强搜索分数。

### 独立 100 局确认

为了避免继续在已反复参与模型选择的旧 200 seeds 上得出过强结论，又随机生成 100 个确认 seeds。它们与旧 200 局、5,000 局 teacher 对局的重合数都为 0；母网络和 student 在完全相同的对局上配对比较。

| Policy | 母网络 | Student | Student - 母网络 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Raw DQN | 152.66 +/- 1.63 | 151.79 +/- 1.62 | -0.87 +/- 1.05 | [-2.92, +1.18] |
| Exact depth-1 | 156.15 +/- 1.51 | 156.81 +/- 1.45 | +0.66 +/- 0.81 | [-0.93, +2.25] |
| Exact depth-2 | 157.43 +/- 1.52 | **158.06 +/- 1.62** | +0.63 +/- 0.73 | [-0.79, +2.05] |

结果目录：`benchmark_results/distillation_confirmation_100/`。

这个结果与开发集呈现相同结构：raw student 较弱，但用于 exact-chance 搜索的 leaf 略好。新集样本只有 100 局，三个置信区间都跨 0，因此正确表述是“搜索叶值存在可复现的正向信号”，而不是“已经证明提高约 1 分”。

### 当前判断

第一轮 teacher distillation 值得保留：开发集上 depth-2 从 `157.62` 提高到 `158.52`，独立集上也保留了 `+0.63` 的同方向差值；但它还没有达到原计划中“把搜索能力压回 raw 网络”的目标，也没有在独立集上达到统计显著。验证集上的 teacher 动作分歧本来就很小：母网络已与 teacher 一致 `85.55%`，选择不同动作造成的平均 teacher regret 只有约 `0.16` 分/状态。普通回归很容易改善数值误差，却难以显著改变整局 argmax 轨迹。

下一步不应直接增加第一代同分布数据。更有价值的是：

- 用第一代 student + depth-2 生成第二代 on-policy teacher 状态，验证第二轮 Expert Iteration 是否还能带来配对提升；
- 同时增加靠近决策边界状态的权重，或只对 teacher 与 student 分歧状态做 DAgger 式补充；
- 保留这批新 100 seeds 只做确认，不再用它们调蒸馏超参数；
- Paired-256 暂不优先，除非需要一个更昂贵的 teacher 对照。

尚未进行的 review 建议包括 sequential halving 和第二轮 Expert Iteration。今晚没有直接生成第二代 5,000 局：独立确认虽然方向为正，但 raw student 仍退步，当前证据更支持先改进决策排序或分歧状态采样，而不是单纯追加同类数据。第一轮结果说明搜索 teacher 路线可行，但提升幅度还远未达到 165 目标。

## 5. Leaf Evaluator Sweep

Double DQN checkpoint 同时包含 online network 和每 1,000 次更新同步一次的 target network。固定 exact depth-2 的树与 backup 不变，只替换最外层 leaf Q：

- `best online`：第三轮 `best.pt` 的 online 权重，也是原来的默认；
- `best target`：同一 checkpoint 的 target 权重；
- `best mean`：逐动作取 `(Q_online + Q_target) / 2`；
- `latest online`：第三轮训练结束时 `latest.pt` 的 online 权重。

先用当前 200 seeds 的前 64 局筛选：

| Leaf | Depth-2 mean | 相比 best online |
| --- | ---: | ---: |
| Best online | 158.27 | - |
| Best target | 158.70 | +0.44 +/- 0.89 |
| Best online/target mean | **159.27** | **+1.00 +/- 0.55** |
| Latest online | 158.81 | +0.55 +/- 1.01 |
| Distilled student | 159.78 | +1.52 |

只有 best mean 有足够信号扩到完整 200 局：

| Leaf | Mean | SE | 相比 best online | 相比 distilled student |
| --- | ---: | ---: | ---: | ---: |
| Best online | 157.62 | 1.12 | - | -0.91 |
| Best online/target mean | 158.17 | 1.10 | +0.55 +/- 0.31 | -0.36 +/- 0.43 |
| Distilled student | **158.52** | **1.04** | +0.91 | - |

Best mean 相对 best online 的 95% CI 为 `[-0.05, +1.15]`，W/T/L `24/161/15`。均值显著减少了策略差异，大多数对局完全相同，并提供了小幅正向信号；但当前最强 leaf 仍是纯 Q 蒸馏 student。

结果目录：`benchmark_results/leaf_sweep_64/` 和 `benchmark_results/leaf_sweep_200/`。

## 6. Depth-1 / Depth-2 分歧分析

使用当前最强的 distilled student，同时计算 depth-1 与 depth-2，但始终执行 depth-2 动作。完整 200 局精确复现 `158.52 +/- 1.04`，共记录 `7,316` 个决策状态。

```text
动作不同：587 / 7,316 = 8.0%
depth-2 对 depth-1 动作的平均 root regret：0.046 分/状态
只看动作不同时：0.576 分/状态
```

这里的 regret 是 depth-2 搜索自身对两个根动作的估值差，用来定位可能重要的状态；它不是可以直接累加成最终对局提升的真实分数差。

按回合：

| Round | 状态数 | 分歧率 | Regret 占比 |
| ---: | ---: | ---: | ---: |
| 1 | 1,826 | 6.2% | 11.7% |
| 2 | 1,842 | 8.8% | 26.7% |
| 3 | 1,831 | 8.5% | 29.9% |
| 4 | 1,817 | 8.6% | 31.7% |

按每轮 draw count，`8/9/10` 三组贡献约 `48.4%` 的总 regret。按动作数，`7-10` 和 `11+` 状态的分歧率分别为 `15.6%` 和 `15.8%`，明显高于 `2-3` 动作状态的 `3.9%`。

Depth-2 top-2 gap 也具有筛选价值：

| Gap threshold | 触发状态比例 | 覆盖分歧 | 覆盖 depth-2 regret |
| ---: | ---: | ---: | ---: |
| <= 0.25 | 8.4% | 41.7% | 11.2% |
| <= 0.5 | 14.8% | 64.7% | 28.2% |
| <= 1.0 | 26.7% | 86.9% | 59.6% |
| <= 2.0 | 46.2% | 96.8% | 84.8% |

因此“只加深少数状态”在计算量上成立，但 gap 小不等于加深后一定能改善最终策略。结果目录：`benchmark_results/exact_chance_disagreements_200/`。

## 7. 选择性 Depth-3 与精确回合尾局

搜索器新增两项受限能力：

- 固定 depth-3；
- 精确搜索到当前颜色回合结束，然后在下一回合第一次公开决策处使用 neural leaf。

每次深搜最多建立 `75,000` 个动作分支，超出就回退 depth-2。结构剖析显示：一个早期状态的 depth-3 约 `46,861` 分支、`136,848` 个唯一 leaf，耗时约 4.6 秒；一个中期状态约 `2,871` 分支、`5,964` leaf，约 0.3 秒。无条件搜索到回合结束在早期和中期均超过 50 万分支，因此只允许在剩余牌很少时使用。

所有消融使用同一前 64 个开发 seeds；固定 depth-2 基线为 `159.78 +/- 1.93`：

| 策略 | Mean | 相比 depth-2 | 95% CI | 深搜触发 | 改变动作 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gap <= 0.5 / 分歧 depth-3 + round-end <= 4 | 159.11 | -0.67 +/- 0.94 | [-2.51, +1.17] | 944 | 128 |
| Gap <= 0.5 / 分歧 depth-3 | 159.12 | -0.66 +/- 0.84 | [-2.31, +0.99] | 439 | 105 |
| Round-end <= 4 | 159.58 | -0.20 +/- 0.72 | [-1.61, +1.20] | 703 | 70 |
| Depth-1/depth-2 分歧时 depth-3 | 159.25 | -0.53 +/- 0.49 | [-1.49, +0.43] | 201 | 29 |
| Round-end <= 3 | **159.59** | **-0.19 +/- 0.33** | **[-0.83, +0.45]** | 482 | 43 |

没有一个配置表现出正向均值，因此没有扩跑 200 局。结果不证明 depth-3 的估值更差，但说明当前的 gap、depth disagreement 和剩余牌数都不是足以可靠选择“何时相信更深搜索”的门控信号。更深搜索频繁改变动作，却没有转化为最终均分提升，并带来约数倍到数十倍的局部计算成本。

当前保留 `SelectiveDepth3Policy` 作为受预算约束的研究工具，但不把它列为推荐 solver。接下来不应继续微调这些阈值；更合理的是使用这些分歧状态生成训练数据，或先建立更强、更可靠的深层 teacher 再设计门控。
