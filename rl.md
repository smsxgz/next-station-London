# Afterstate Value 训练计划

## 1. 范围与约束

这是新实验的算法设计文档，目标是实现一个独立的 `src/rl/afterstate/` 包。
本阶段只确定算法和实验协议，不开始写训练代码。现有 `src/solver/RL/` 的
Full-Q 实现保持可用。新实验使用基础规则：

- `advanced=False`；不启用 shared objectives 和 pencil powers；
- 训练预算为 10,000,000 个决策 transitions，与原 DQN 的预算口径一致；
- 不写 pytest，使用现有检查入口和小规模 smoke run 验证；
- 所有公开状态都不恢复隐藏的真实牌序，chance 由当前剩余牌集合精确展开。

Actor-Critic 暂不作为第一版。动作数很小，立即奖励是引擎可精确计算的，
而且当前牌已经在决策状态中确定；只学习 afterstate 的 continuation value
就可以得到确定性的策略。Actor-Critic 会额外引入 policy head、熵项和另一套
训练稳定性问题，不能直接利用这里已有的精确 reward/chance 结构。

## 2. 状态时序和符号

一次决策的时序固定为：

```text
o_t --draw c_t--> s_t --choose a_t--> o_{t+1}
```

其中：

- `s_t` 是等待动作的 public decision state，包含当前公开 card event、活动
  颜色、四条线、剩余牌和所有计分相关公开信息；
- `a_t` 是当前 event 下的合法 section，或 `Pass`；
- `r(s_t, a_t)` 是这次动作带来的 dense score delta。Pass 的 reward 为 0；
- `o_{t+1}=T(s_t,a_t)` 是动作完成、当前 pending event 清除之后的
  public afterstate，尚未抽下一张牌；
- `c_t` 是从 afterstate 的剩余牌中产生的下一个公开 event，
  `s_t=D(o_t,c_t)`。

对剩余牌集合 `R`，普通首牌 event 的概率为 `1/|R|`；如果首牌是 Switch，
则对每个不同的第二张牌形成一个有序 event，概率为
`1/(|R| * (|R|-1))`。Switch 本身不作为一个只消费一张牌的 outcome。
所有 outcome 必须通过 `draw_known_cards()`/同等 engine transition 生成，
不能直接拼 observation；展开后的概率总和必须检查为 1。

`GameSession.draw()` 的当前语义直接作为唯一规则来源。Railroad Switch
消费下一张牌，目标 symbol/wild 由第二张牌决定，两张牌的 `draw_count` 和
`underground_count` 都要计入。

Action 完成一轮时，afterstate 可能有两种结果：

1. 最后一轮完成，`status="finished"`，后续价值定义为 0；
2. 非最后一轮完成，`round_index` 前进、剩余牌重置为整副牌，仍然是一个
   没有 pending card 的非终止 afterstate，下一步 chance 从新牌堆展开。

因此不能把“没有 pending card”当成 terminal；terminal 只能由
`status == "finished"` 判断。

## 3. Afterstate value 定义

定义 afterstate continuation value：

```text
W(o) = E_c [ V(D(o, c)) ]
Q(s, a) = r(s, a) + W(T(s, a))
V(s) = max_a Q(s, a)
```

terminal afterstate 的 `W(o)` 为 0。第一阶段固定 `gamma=1`，但实现接口仍
保留 discount，使公式和代码以后可以扩展。

实现中的 `continuation(o)` 必须是：`0`（若 `o.status == "finished"`），否则
才调用网络 `W(features(o))`。这个 terminal override 同时用于 collector 的
动作选择、target 中的 online `b*` 选择和 target 网络估值；不能只依靠一个
terminal 输入特征让网络自行学出零值。

对一个 replay afterstate `o`，训练 target 使用用户提出的 Double-DQN 形式：

```text
b*(c) = argmax_b [ r(s_c, b) + gamma W_online(T(s_c, b)) ]

y(o) = E_c [ r(s_c, b*(c))
              + gamma W_target(T(s_c, b*(c))) ]
```

这就是 chance 深度为 1 的 exact backup：从当前 afterstate 只展开一次
公开抽牌，再在每个 successor decision 上做一次动作最大化。`online` 只
负责选择 `b*`，选中的 afterstate 由 `target` 估值；reward 始终由引擎精确
计算，不由网络估计。它和现有 Full-Q 的 exact depth-1 是同一类边界处理，
但新网络的输出对象是 `W(o)`，所以不再有 156 个 action output。

如果网络输出的是归一化值 `w=W/reward_scale`，实际 target 为：

```text
y_norm = E_c [ r(s_c,b*) / reward_scale
               + gamma w_target(T(s_c,b*)) ]
```

当前建议沿用原训练的 `reward_scale=10`，loss 预测归一化 continuation，
日志再乘回 10 显示原始分数。注意：replay 中当前动作的 `r_t` 不应再加到
`W(o_{t+1})` 的 target 里；它已经属于到达这个 afterstate 之前的那一步。

## 4. Afterstate 特征

网络输入必须同时保留游戏的 Markov 信息和显式计分信息。完整棋盘复制是
候选动作的状态转换方式；特征只是把复制后的 public afterstate 提供给网络，
不能用一个压缩的 action embedding 替代棋盘状态。

### 4.1 棋盘、牌堆和回合信息

第一版保留现有 Full-Q 中对评分和合法性有用的 public 部分：

- 每种颜色的 155 条 edge occupancy；
- 每种颜色的 53 个 station occupancy；
- 当前 round 的 11 张 remaining-card mask；
- `order` 的 16 维 permutation one-hot、`round_index` 的 4 维 one-hot；
- `active_color`（虽然可由 order 和 round 推导，仍建议显式保留）；
- `underground_count` 的 6 维 one-hot、`draw_count` 的 12 维 one-hot；
- 一个 terminal flag。afterstate 不编码当前 card、target symbol 或 pending
  event，因为它们已经被动作消费。

这些字段足以从 replay 记录重新建立没有隐藏 deck order 的 `GameSession`，
并再次调用引擎的 `public_event_successors()`。

decoder 需要重建的不只是 bit mask：它还要按 `order[:round_index]` 从四条线
重建已结束回合的 `round_scores` 和 tourist 累计，并由整张棋盘重新计算
`_line_metrics`、network masks、interchange counts、partial score caches。
这些量不作为容易失真的独立 replay 真相；若实现选择直接存缓存，也必须在
`check` 中和重算结果逐字段相等。

### 4.2 每条线的计分特征

对四种颜色分别记录以下字段：

| 字段 | 维度/含义 |
| --- | --- |
| `station_counts` | 13 维，各 district 的 station 数 |
| `district_mask` | 建议额外保留 13 个 touched-district bit；edge 覆盖的 district 也会计入 |
| `district_count` | `district_mask.bit_count()` |
| `max_stations` | 13 个 district count 的最大值 |
| `route` | 当前 line 的 route 分数 |
| `thames` | 当前 line 的 Thames 分数，即 `2 * thames_crossings` |
| `tourist_visits` | 当前 line 已覆盖的 tourist station 数 |

`district_mask` 不是为了改变规则，而是避免网络从 155 条 edge 重新学习
“edge 所覆盖 district”的映射。`route` 仍以 engine 的 `_line_metrics` 为
准；不能只用 station 所在 district 重算，因为 edge 也可能扩展 district。
如需诊断，可以同时输出 raw `thames_crossings`，但训练特征以 points 为主。

### 4.3 全局计分和边际特征

使用 `partial_score_components()` 的 dense、final-score-equivalent 值，不用
未结束回合的 `score_summary()` 作为当前分数。至少包含：

- `partial_route`、`partial_thames`；
- `partial_tourist_visits`、当前 tourist tier（11 档 one-hot）和
  `tourist_points`；
- `interchange_counts[0..4]`，即每个站点当前被 0、1、2、3、4 条线经过的
  数量；
- `interchange_bonus`；
- `current_total = partial_route + partial_thames + tourist_points
  + interchange_bonus`（基础规则 objective 为 0）；
- `tourist_next_gain = track[v+1] - track[v]`，其中 `v` 是当前
  `partial_tourist_visits`，达到 track 上限后为 0；
- `interchange_unit_gain[k] = I[k+1]-I[k]`，`k=0..3`，表示一个 k 重站点
  再接入一条线的分数增益；同时记录
  `interchange_gain_mass[k] = interchange_counts[k] *
  interchange_unit_gain[k]`，表示当前全局的该档位潜力。`k=4` 没有下一档，
  不生成 gain。第一版可以只把二者作为一组固定顺序的连续特征，不能把它们
  和某个具体的未知下一动作混淆。

这里的“边际增益”是 afterstate 的全局潜力摘要，不是把下一张未知牌或某个
具体 action 偷渡进网络输入。当前 decision 上某个 action 的精确增益仍由
`score_delta_for_legal_action()` 返回，并在 `r + W` 中单独使用。

所有连续特征使用固定、可复现的常数归一化到大致 `[0,1]`，不使用训练集
均值/方差。归一化常数和字段顺序写入 `feature_schema`，并在 checkpoint 中
保存版本与 hash；one-hot 和 mask 保持 0/1。最终 observation dimension
由 extractor 生成并写入 schema，不在文档中手工猜一个数字。

## 5. 候选动作和 afterstate 生成

第一版明确采用最直接、容易核对的转换：**每个候选动作复制一份完整的
public `GameSession`**。

对当前 decision `game`：

1. 取 `game.legal_actions()`，再加入 `Pass`，动作索引沿用 edge id 和
   `PASS_ACTION_INDEX=155`；每个候选记录固定为
   `(action_index, action_or_none, reward, afterstate)`；
2. 对每个 section action，先在原 state 上调用
   `score_delta_for_legal_action()` 得到精确 reward；Pass reward 为 0；
3. 调用 `game.copy_public_state()`，在副本上
   `apply_legal_action(action)`；第一版不依赖只复制活动线的局部 helper，
   让每个候选都明确拥有一份完整 public 棋盘；
4. 副本 pending 被清除，副本就是候选 afterstate；从副本提取 canonical
   state 和网络特征；
5. 用一条稳定的动作顺序处理 tie。建议按现有 action index 的升序，Pass
   放在固定的 155 号；与现有 `masked_argmax` 一样，数值相同时选择较小
   index。

选动作时计算：

```text
score(a) = reward(a) / reward_scale + continuation_online(afterstate_a)
```

训练采集使用 epsilon-greedy：以 `epsilon` 概率在合法 section 加 Pass 中
均匀随机，否则取上式 argmax。评估和最终报告使用完全 greedy，不额外做
chance search。选中的副本只作为要保存的 afterstate；真实环境仍在原始
`GameSession` 上执行同一个 action，并检查两者 public state 一致，这样不会
把真实游戏的隐藏 deck RNG 带入 replay。

## 6. Replay 的状态表示

一个 afterstate transition 的核心训练对象只有 `o`，不需要 Full-Q 的
`next_observation` 或 action mask。建议第一版记录：

- canonical afterstate：四条线的 station/edge masks、remaining-card mask、
  order、round/counter、status；
- `terminated`；
- 到达该 afterstate 的 `action_index` 和 `immediate_reward`（只用于日志、
  score telescope 和诊断，不进入 W target）；
- 可选的 game/episode 标识，便于统计，不参与网络输入。

canonical state 是 replay 的事实来源，采样时解码为 `GameSession`，再生成
特征。这样不会因为以后调整归一化方式而让 replay 中的 feature 和 exact
chance decoder 不一致，也能保证 target 需要的 remaining pile 始终存在。
建议使用固定 dtype 的结构化记录，而不是把 Python 对象写进 checkpoint：
每条线的 155 条 edge 可用 3 个 `uint64`（四条线共 12 个），每条线的 53 个
station 可用 1 个 `uint64`，其余是小整数和一个 `uint16` remaining mask。
具体 record bytes 在实现时由 schema 计算并写入 metadata。

不建议第一版只存 feature vector：那样网络可以前向，但 exact target 无法
可靠恢复下一次抽牌和合法动作。也不建议把当前 afterstate 的全部未来值或
动作 reward 拼进 W 的输入；reward 已在 Bellman 式中显式出现。

## 7. 训练流程（完整时序）

以下流程是第一版的基准流程，所有计数均以“完成一个 decision 并保存一个
afterstate”为一个 transition：

### 7.1 初始化

1. 创建 `W_online` 和结构相同的 `W_target`（建议 hidden sizes 先沿用
   `(512, 512)`），target 初始复制 online；
2. 初始化 Adam、replay ring、collector RNG、Torch RNG 和一组同步
   `GameSession` 环境；
3. 每个环境创建新游戏并先 draw 一次，使其停在 decision state；
4. 在 `current_200` manifest 上评估随机初始化的策略，写入 step 0 的
   validation 记录，作为最佳 checkpoint 的初始基线；
5. 固定 `gamma=1`、`reward_scale=10`、1,000,000 容量的 uniform replay、
   batch 512、warmup 25,000、replay ratio 8、target 每 1,000 次 update 同步一次。
   这些默认值先与原 DQN 对齐，不能在没有 smoke/profile 结果时自行改预算。

### 7.2 每个 collector step

对每个尚未结束的环境执行：

```text
decision = current GameSession with pending event
candidates = []  # Candidate(index, action, reward, afterstate)
for a in legal_sections(decision) + [Pass]:
    r = exact immediate reward of a
    child = full public clone of decision
    child.apply_legal_action(a)
    candidates.append((action_index(a), a, r, child))

if random() < epsilon:
    chosen = uniform random legal candidate
else:
    batch W_online(features(child) for child in candidates)
    chosen = argmax_a(r / scale + continuation_online(child))

execute chosen action on the real GameSession
assert canonical_public(real_game) == canonical_public(chosen.afterstate)
replay.add(afterstate=chosen.afterstate,
           terminated=(real_game.status == "finished"),
           action=chosen.action_index, reward=chosen.reward)

if terminal:
    finish score accounting, assert initial_score + reward_sum == final_score,
    and reset this environment
else:
    real_game.draw()  # hidden seeded order only affects data collection
    continue at the next decision
```

Switch event作为一个 decision 消费两张牌，不能拆成两个 transition。Pass
也必须生成 afterstate，因为它可能触发 final-card 的换轮或终局。

collector 的 epsilon 建议先沿用原配置：从 1.0 线性降到 0.05，覆盖前
1,000,000 transitions。transition 计数在 replay insertion 时递增；最终
循环应准确停在 10M（最后一个 vector batch 只提交尚未超过预算的环境槽），
避免用环境数造成不可控的预算偏差。

### 7.3 每次 SGD update

warmup 满足后，按 replay ratio 为本次 collector step 发放 update credit。
每次更新：

1. uniform sample 一批 canonical afterstates，解码并提取 `features(o_i)`；
2. 用 `W_online(features(o_i))` 得到预测值；
3. 对每个非终止 `o_i` 精确枚举 `public_event_successors(o_i)`；
4. 对每个 chance outcome `c`，完整复制 decision state 的每个合法 action，
   得到候选 `(r, o')`；
5. 批量前向所有候选的 `W_online`，以
   `r/reward_scale + gamma*W_online(o')` 选出 `b*(c)`；
6. 对每个选中的 `o'` 用 `W_target` 前向，按 chance probability 加权求和；
7. 终止 afterstate 的 target 直接置 0；非终止样本得到 `y_norm`；
8. 对 `W_online(o_i)` 和 `y_norm` 做 elementwise Smooth L1 loss，反向传播，
   gradient clip 后更新 online；
9. 每 1,000 次 update 将 online state_dict 复制到 target；target 全程
   `eval()` 且不参与梯度。

target 的等价伪代码：

```text
target(o):
    if o.status == finished:
        return 0
    total = 0
    for p, decision in exact_public_event_successors(o):
        branches = []
        for b in legal_sections(decision) + [Pass]:
            r_b = exact_reward(decision, b)
            o_b = full_public_clone_and_apply(decision, b)
            u_b = r_b / scale + gamma * continuation_online(o_b)
            branches.append((u_b, r_b, o_b))
        _, r_star, o_star = argmax_with_fixed_tie(branches)
        total += p * (r_star / scale + gamma * continuation_target(o_star))
    return total
```

所有 network calls 都放在 `no_grad` target block 内；只有第 1 步的
`W_online(features(o_i))` 参与 loss。online 选择和 target 估值不能交换，
否则就不再是计划中的 Double-DQN backup。

### 7.4 日志和 checkpoint 时序

- 每隔固定 transitions 记录 loss、TD error、W 均值、chance outcome 数、
  candidate action 数和 collector/update throughput；
- 每隔 validation interval 在 `current_200` 上运行 greedy afterstate policy，
  只用 mean 选择 `best.pt`，同分保留较早 checkpoint；
- `latest.pt` 保存可 resume 的 online/target、optimizer、replay cursor、
  RNG、environment seed streams 和所有 schema/config；
- `best.pt` 至少保存同一套 schema 和 online/target 权重，不能只保存裸
  `state_dict`；
- 新 checkpoint 带有明确的 `model_type="afterstate_value"`、
  `afterstate_schema_version`、`feature_schema_hash`、`replay_schema_version`
  和 `observation_dim`。旧 Full-Q loader 不读取新 checkpoint；旧 Full-Q
  checkpoint 也不被新 loader 静默当成 afterstate。

## 8. 批量化、缓存和其他性能取舍

这些是实现前需要确认的工程选择，不能先擅自改成更复杂的编码。

| 项目 | 第一版建议 | 语义影响/理由 |
| --- | --- | --- |
| 候选动作转换 | 每个 action 复制完整 `GameSession` | 最容易逐步核对；保留引擎作为唯一 transition 来源 |
| 候选动作编码 | 不做 compact action embedding；仍用 edge id/Pass + 完整 afterstate 特征 | 避免把 action-dependent score 偷塞进 W |
| NN 批量化 | 建议做；把同一 decision 或同一 target batch 的候选特征 stack 后分块前向 | 只改变调用方式，不改变复制和 argmax 语义 |
| global cache | 第一版不做 | 网络每次 update 都变，跨 update 缓存值会失效且占内存 |
| local dedup/cache | profile 后再考虑，只在一次 target call 内按 canonical public key 去重 | 可减少相同 chance/action 分支的前向，不改变结果 |
| replay 存储 | canonical afterstate 结构化记录，采样时解码 | exact chance 必须恢复 remaining/line masks；feature 可重算 |
| 环境并行数 | 初始沿用 128 以便与原 DQN 对比 | 先测候选复制和 target 展开吞吐，再讨论调整 |

预计平均合法 action 数较小，但 target 每个样本还要展开多个 chance
outcome；因此先跑短 smoke/profile，记录“每个 batch 的 chance outcomes、
candidate copies、NN 前向耗时”。只有拿到数据后，才决定是否加入 local cache、
减少并行环境或改变 replay 表示。任何这类性能改动都必须保持上述 target
数值完全一致，并在文档中单独记录。

## 9. CLI、实验目录和评估协议

新代码放在 `src/rl/afterstate/`，不修改旧 `solver.RL` 的 Full-Q CLI。建议
入口形态为：

```text
python -m rl.afterstate train \
  --run-dir artifacts/afterstate/value_10m \
  --eval-seeds-from benchmark_results/current_200/manifest.json \
  --total-transitions 10000000
```

因为当前 `pyproject.toml` 的 package discovery 还没有包含 `rl`，实现时要
显式加入 `rl`、`rl.*`（并保留 `solver`、`solver.*`），再添加对应的
`__init__.py`/`__main__.py`。可以增加独立的 `next-station-afterstate`
console entry point，但不能替换现有 `next-station-rl`。

另提供 `check`、`benchmark` 和 `evaluate` 子命令；具体参数名在实现前
沿用仓库已有 CLI 风格。训练目录至少包含 `config.json`、`metrics.jsonl`、
`latest.pt`、`best.pt`、`replay.dat`、`summary.md`。

训练集/验证集协议：

1. 训练期间和 checkpoint selection 只使用
   `benchmark_results/current_200/manifest.json` 的 200 个 seed；
2. 10M 完成后，用独立的确定性 seed stream 生成新的 200 个 seed，至少保证
   与 current_200 不重复，并把 manifest 保存到新实验目录；
3. 在这同一批新 seed 上比较：
   - 新 afterstate `best.pt` 的 greedy policy；
   - 原来的“蒸馏前最佳网络 + existing exact depth-1” baseline。

baseline checkpoint 路径作为 CLI 参数保存，不能在代码中静默猜测。按当前
实验记录，“蒸馏前最佳网络”更可能是第三轮 exact-target 训练的 raw checkpoint
`artifacts/dqn/exact_chance_depth1_lr1e4_4m/best.pt`，然后在它上面使用已有
`ExactChanceDQNPolicy(depth=1)`；`artifacts/dqn/fullq_n1_uniform_10m/best.pt`
是原始 Full-Q 网络，属于另一个可选对照。开始训练前确认到底采用哪一个，
并把实际路径写入最终 manifest。两种 policy 必须使用同一批 seed、同一基础
规则和同一验证开关。

最终报告至少输出每个 policy 的 mean、standard error、min/median/max，及
逐 seed 的 paired difference、95% CI 和 W/T/L。current_200 的分数只能作为
开发/选点结果，不能冒充独立最终测试。

## 10. 不使用 pytest 的验收检查

实现后用仓库已有命令和少量脚本做以下检查：

1. 旧路径：现有 `python -m solver.RL check` 仍能运行，Full-Q checkpoint
   加载行为不变；
2. 新 `afterstate check`：复制并 apply 每个合法 action 后，真实 game 和
   candidate 的 public state 一致；reward 与 dense score delta 一致；
3. Switch outcome：枚举概率总和为 1，第二张牌被消费，候选动作与引擎一致；
4. round boundary：非最后一轮结束后的 full-pile afterstate 可继续展开，
   最后一轮 terminal target 为 0；
5. feature check：逐字段和 engine 的 line metrics、partial score、tourist
   track、interchange track 对齐；current total 能由组件重建；
6. target smoke：随机生成少量 afterstates，online/target target 为有限值，
   mask 不为空，梯度能完成一次 update；
7. 运行一个很短的 collector（例如 10k transitions）并成功保存、重新加载
   `latest.pt` 后 resume；最后再开始 10M 实验。

## 11. 开始实现前需要确认的点

算法主线已经固定，以下三个工程选择已确认：

1. 采用“canonical afterstate replay、采样时解码”，不把 feature vector 作为
   replay 的唯一事实来源；
2. 采用语义中立的候选 NN batching，第一版不做 global/local cache；
3. baseline 为第三轮 exact-target 的
   `artifacts/dqn/exact_chance_depth1_lr1e4_4m/best.pt`，配合现有
   `ExactChanceDQNPolicy(depth=1)`。

除这三点外，第一版不引入 Actor-Critic、n-step、prioritized replay、
depth-2 target、隐式 action encoding 或其它未讨论的优化。

实现顺序：先完成 canonical codec/decoder、feature extractor 和单步
afterstate transition；再加入批量候选推理与 exact target；随后接入 replay、
checkpoint/CLI，最后做短 smoke、current_200 选点和 10M 训练。

## 12. 实施记录

上述流程已落地到 `src/rl/afterstate/`。当前实现的 observation dimension 为
`1041`，feature schema hash 为
`3e2bba7b62a20e19c2582480f1eabbf5795738852eaede0202f7329453946cef`；replay
使用 canonical structured record，候选动作仍逐个复制完整 public
`GameSession`，网络候选前向做 batch，第一版没有 cache。checkpoint 明确写入
afterstate、feature 和 replay schema metadata，旧 `solver.RL` Full-Q loader
保持独立。

已完成的验收包括：随机完整轨迹的 encode/decode 和 terminal score 对齐、
Switch 概率、换轮后 full-pile afterstate、terminal target 为零、target
梯度 smoke、短训练以及 checkpoint resume。验证器使用
manifest 中的 seed 直接初始化 `GameSession`，因此 current_200 和最终 paired
评估不会再经过额外 seed stream。

第一轮正式实验目录为
`artifacts/afterstate/value_10m_seedfixed/`，配置为 10M transitions、128
collector environments、batch 512、replay ratio 8、hidden `(512,512)`，
baseline 路径为
`artifacts/dqn/exact_chance_depth1_lr1e4_4m/best.pt`。进程启动后先在
current_200 做 step-0 validation。该 run 在 500,096 transitions 处停止：后续
决定让真实对局的 RNG 和采集也全部进入 C++，因此不在同一 run 中混用两套
seed-to-deck 映射。旧目录保留作开发记录，不参与最终 checkpoint 选择。

## 13. C++ 等价加速记录

训练的 replay observation、一次 chance 展开、完整候选棋盘复制以及 1041 维
候选特征生成最初接入了 C++ Release 库；源码现已统一迁移到
`src/engine_cpp/`。online
网络仍选择动作，target 网络仍估计被选 afterstate；reward、outcome 顺序、
固定 tie 规则、replay ratio 和 batch 不变。

32 条完整轨迹的 oracle 检查覆盖 1,176 个 afterstate、11,615 个 chance
outcome、61,423 个候选动作和 62,599 条 feature vector，所有 1041 维 float32
逐位一致。同一组随机 online/target 网络在 96 个 replay afterstate 上产生的
target 也和接入前逐位一致。

在同一台机器和 CUDA 设备上，512 样本 exact-target 从约 4.70 秒降到
0.249 秒，约快 18.9 倍。使用正式参数的 100k 短训练在 warmup 后三个区间为
260.2、258.7、259.8 transitions/s，稳定均值约 259.5/s；上一版向量化 Python
路径的最好稳定区间为 21.67/s，因此端到端约快 12.0 倍。按该吞吐估算，10M
核心训练约 10.7 小时；计入 current_200 验证和 checkpoint，预算约 11 小时。

## 14. 全 C++ 游戏后端与重训

标准 London 的颜色顺序、每轮洗牌、基础规则、Shared Objectives、四种 Pencil
Powers、计分、公开复制和隐藏 RNG 序列化现均由 C++ `GameState` 持有。
`engine_cpp.GameSession` 只提供现有 Python 对象接口；solver、experiments、
Full-Q env 和 afterstate collector 都改用该后端。C++ RNG 对同一 seed 可复现，
但不要求与旧 Python RNG 产生相同牌序，因此所有 benchmark 和最终 200-seed
结果都要在新后端上重跑。

删除旧实现前，差分矩阵用相同显式牌事件逐状态比较两端，覆盖全部 5 种
Shared Objectives 的实际完成时刻、全部 4 种 Pencil Powers、24 种 power
assignment、所有合法动作及其五项即时奖励、公开复制、每轮与终局计分；全部
一致。随后已删除 `src/engine`，伦敦静态地图和值对象移入 `engine_cpp`，全仓及
本地 `web/helper` 均无旧 engine import。

逐事件对照使用相同显式牌事件和动作，覆盖基础/高级规则、合法动作、五项即时
奖励、Double Section 第二阶段、Wild/Switch/Circle、Objectives、每轮与终局
计分、公开复制、历史和 checkpoint 后 RNG 续跑；当前全部相等。新的非 pytest
验收入口为 `python -m engine_cpp.checks`，同时保留 `rl.afterstate check` 和
`solver.RL check`。

优化前的短 profile（128 env、4096 transitions、CUDA）得到 collector
`1663.3/s`，512 样本 exact target 为 `0.266s`。但细粒度兼容 API 的跨语言往返使简单完整
对局只有旧 Python 后端的 `0.75x`，Depth-2 约 `0.31x`。下一项待确认的纯工程
优化是一次返回 chance node 的全部公开后继，或 decision 的全部
`(action, reward, afterstate)`；递归、概率、argmax/tie 和网络仍留在 Python，
不加入 cache，也不改变算法语义。在确认前不实现该批量 solver ABI。

新的正式目录为 `artifacts/afterstate/value_10m_cpp/`。step-0 current_200 为
`123.895 +/- 0.992`；250,112、500,096、750,080 transitions 分别为
`129.210 +/- 1.029`、`132.165 +/- 0.959`、`135.055 +/- 1.052`，每次均刷新
best。warmup 后各 100k 区间的真实吞吐约为 `240-256 transitions/s`；最初
`313.4/s` 的区间包含 warmup，不能用于总时长外推。750,080 checkpoint 已保存
完整环境隐藏牌堆、C++ RNG、replay、optimizer 和网络状态。

为利用多核但保持训练语义不变，exact-target expansion 现由一次 C ABI 调用
创建 owner-ordered immutable 结果；C++ 先并行计数，再按固定 prefix offset
并行直接写最终数组。没有 cache、候选重排或近似计算。固定 750k replay 样本
包含 512 records、5,021 chance outcomes 和 26,349 candidates；owner、概率、
offset/count、reward、terminal 以及 `26,349 x 1041` float32 features 均与优化前
DLL 逐字节一致，不同线程数产生的最终 target 也逐元素一致。

该固定样本的 native expansion 从约 `72.3ms` 降到 10 线程约 `20.9ms`；完整
target 中位数由单线程 `161.8ms` 降到 10 线程 `115.0ms`。8-16 线程已经进入
平台区，20 线程略慢，因此正式训练固定
`NEXT_STATION_NATIVE_THREADS=10`。同口径短 profile 的 collector 为
`1673.2/s`，512-sample exact target 为 `0.189s`。从 750,080 恢复后的
800k、900k、1,000,064 区间分别为 `339.2/s`、`330.6/s`、`324.3/s`；两个
完整 100k 区间均值约 `327.5/s`，比优化前稳定均值约 `248.4/s` 提升
`31.8%`。1,000,064 的 current_200 为 `138.295 +/- 1.047`，再次刷新 best。
实际 10M 全程（含验证和 checkpoint）为 `33,027.7s`，约 9 小时 10 分；恢复后
随着 replay 后期状态分布变化，长期吞吐稳定在约 `300-310/s`。这取代所有早期
基于 warmup 的时长估算。

## 15. 10M 结果与独立 200-seed 对照

正式训练完成 10,000,000 transitions 和 155,860 次 learner update。current_200
从 step-0 的 `123.895 +/- 0.992` 持续提升；最佳 checkpoint 位于 9,000,064
transitions，为 `151.065 +/- 1.088`。10M 最后一步为
`150.080 +/- 1.044`，因此按预先约定选择 9M 的 `best.pt`，不使用 latest。

最终评估使用固定生成器产生 200 个新 seed；它们全部唯一，且与 current_200
的交集为零。两种 policy 按完全相同的 seed 顺序在全 C++ 游戏后端上重跑：

| Policy | Mean | SE | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| afterstate best (9,000,064) | 152.090 | 1.045 | 112 | 153.5 | 190 |
| pre-distillation Full-Q + exact depth-1 | 155.805 | 1.016 | 110 | 157.0 | 192 |

paired difference（afterstate - baseline）为 `-3.715 +/- 0.774`，95% CI
`[-5.232, -2.198]`，W/T/L 为 `63/12/125`。因此第一阶段 afterstate value
训练明显学到有效策略，但在独立 seeds 上仍显著落后既有 depth-1 baseline。
完整 seed、逐局分数和统计保存在
`artifacts/afterstate/value_10m_cpp/final_200/manifest.json`，简表保存在同目录
`summary.md`；本计划中的第一阶段训练、current_200 选点和最终独立对照均已
完成。

进一步的语义不变性能方向仍需讨论后再实现。剩余方向是让下一批纯状态
expansion 与当前批 GPU 工作做单元素预取重叠，最后再设计 128-env collector
的批量 C ABI。
增大训练 batch 会改变 update 数量和梯度统计，不属于本轮“严格语义不变”范围。

## 16. 第 1 项语义不变优化：原生 argmax 与 expectation reduction

已实现并接入 `src/rl/afterstate/target.py`。`ns_expansion_select_candidates`
在每个 chance outcome 的原始候选区间内完成

`argmax_b [ reward(s_c,b) / reward_scale + gamma * W_online(T(s_c,b)) ]`，

候选顺序和严格的 first-tie 规则保持不变；`ns_expansion_reduce_targets` 按
每个 replay owner 的原始 outcome 顺序完成概率加权和。owner 之间可以并行，
但单个 owner 内不重排、不改变浮点加法顺序。terminal owner、`reward_scale`
和 `gamma` 的处理与原 Python 路径一致。online/target 网络前向、批大小、
replay 抽样、optimizer 更新和 target 同步均未改动。

验证分三层完成：

1. 固定 10M replay、latest checkpoint 和固定抽样批次上，native selected
   indices、targets、loss、TD errors，以及执行一次 Adam update 后的整个
   online state dict 均逐字节一致；C++ expansion 的 owner/outcome/candidate
   数量和 feature buffer 也一致。
2. 8 局 afterstate self-check、8 局旧 Full-Q self-check、C++ executable
   check 和 `python -m engine_cpp.checks` 全部通过。
3. 同一固定 512-record 批次、同一网络状态交替测量 15 次：旧 Python
   selection/reduction 中位数 `119.293 ms`，native 中位数 `64.849 ms`，
   为 `1.84x`（每次 update 节省约 `54.4 ms`）。正式参数的短 profile 中，
   exact target 从约 `188.7 ms` 降至 `102.1 ms`，collector 保持约
   `1,673 transitions/s`，说明 collector 语义和成本未被这项改动影响。

在最终 10M checkpoint/replay 上，完整 learner update（feature 生成、两次
网络推理、native target、反向和 Adam）CUDA 中位数为 `70.9 ms`（21 次，
`NEXT_STATION_NATIVE_THREADS=10`）。这只是性能优化，尚未启动新的正式训练；
已有 10M 结果和独立 200-seed 对照不应因这项优化重新解释。下一步仍需单独
讨论是否值得继续训练，以及是否实现预取或批量 collector ABI。

## 17. 10M 到 11M 的有界延长对照

为检验“loss 仍下降是否意味着同一算法值得继续训练”，从正式目录的
`latest.pt`（10,000,000 transitions、155,860 updates）复制出独立目录
`artifacts/afterstate/value_10m_cpp_ext_11m/`，保留同一 replay、optimizer、
环境状态、随机状态、网络和 native reduction，实现严格恢复。目标设为 11M，
但按预先规则在两个 validation 点没有可靠连续改善时提前停止。

10 个 native worker 下，100k 区间吞吐约 `436-459 transitions/s`。在
10,250,112 transitions 的 current_200 validation 为 `151.185 +/- 1.105`，
在 10,500,096 为 `150.480 +/- 1.039`。前者相对原 9M best 的固定
current_200 配对差为 `+0.120 +/- 0.847`，95% CI 为 `[-1.540, 1.780]`，
W/T/L 为 `92/11/97`；因此没有可识别的真实提升，按规则在 10.5M 停止。

这次延长的 loss 和 TD error 仍略有下降，但策略分数没有同步改善，进一步
支持“继续跑同一配置”不是当前最有价值的方向。没有为这个未胜出的 checkpoint
生成新的独立 200-seed 结果；完整日志和 checkpoint 保存在上述实验目录。

## 18. C51 与 PER 实验协议

在完成语义不变的 native expansion/reduction 后，新增两组独立的 afterstate
实验。先运行 C51，再运行 PER。C51 从零训练 10M；PER 根据 C51 完成后的讨论，
改为从 scalar afterstate 的 9M best warm-start，追加 3M transitions。两者都
使用同一 128-env collector、batch 512、replay ratio 8、`gamma=1`、
`reward_scale=10`、`hidden=(512,512)` 和同一个 `current_200` manifest。每组
都只从自己的新目录读取和保存 replay、optimizer、环境 RNG 与 checkpoint，
不覆盖 scalar 正式 run。

C51 网络在每个 afterstate 输出 51 个 logits，固定归一化 support `[0,20]`。
动作选择使用 `E_b[r(s_c,b)/10 + E[W(T(s_c,b))]]`，其中期望来自 online
分布；选中的 action 由 target 分布提供 continuation，先加即时 reward 并乘
`gamma`，再按 C51 线性投影回 support，最后按 native chance probability
混合各 outcome。terminal candidate/owner 的 continuation 分布分别是 zero
delta，因而不引入额外终局价值。训练损失是 projected distribution 的交叉
熵；日志同时记录期望值 TD、上下 support clipping mass。C51 的 epsilon 从
1.0 在前 3M transitions 线性衰减到 0.05，之后保持不变。

PER 只替换 afterstate replay 的抽样：优先级为
`(|expected-value TD| + 1e-3)^0.6`，按 proportional sum-tree 分层抽样，
importance-sampling beta 从 0.4 线性升到 1.0；目标、网络和 collector
语义保持 scalar afterstate 版本不变。每次 learner update 后用该 batch 的
absolute expected-value TD 更新优先级，新写入记录使用当前最大优先级。
warm-start 保留 9M best 的 online、target、Adam、环境状态和 RNG，但使用全新
PER replay，并把本轮新增 transition 计数从 0 开始；25k warmup 后更新。epsilon
固定为 0.05，不重新执行高 epsilon 探索阶段。这样 beta 在新增 3M 内完整退火，
同时只改变 replay sampling，不丢掉原策略已经学到的能力。

每组训练在 current_200 的 validation checkpoint 中选择最高 mean；该集合只
用于选点。训练结束后对选中的 checkpoint 各生成一组与 current_200 不相交的
新 200 seeds，并与原始 pre-distillation Full-Q + exact depth-1 baseline 在
完全相同的 seeds 上做 paired 对照，报告 mean/SE、95% CI 和 W/T/L。

### C51 实际结果

C51 正式目录为 `artifacts/afterstate/value_10m_c51_cpp/`。训练完成
10,000,000 transitions 和 155,860 次更新，总耗时 30,746.3 秒（约 8 小时
32 分）。current_200 的最佳 checkpoint 位于最终 10M，mean 为
`148.690 +/- 1.092`；9.75M 为 `148.485 +/- 1.124`。训练早期的 support
upper clipping mass 曾随分布尚未收敛而升高，最终降到约 `0.000047`，没有
持续的边界截断问题。

在新生成且与 current_200 不相交的 200 seeds 上，C51 afterstate 为
`148.890 +/- 1.153`，pre-distillation Full-Q + exact depth-1 baseline 为
`155.715 +/- 1.126`。paired difference 为 `-6.825 +/- 0.926`，95% CI
`[-8.640, -5.010]`，W/T/L 为 `55/6/139`。完整逐局结果保存在
`artifacts/afterstate/value_10m_c51_cpp/final_200/manifest.json`。C51 没有超过
scalar afterstate，也显著落后 baseline，因此不继续延长；下一步按协议运行
scalar afterstate + PER。
