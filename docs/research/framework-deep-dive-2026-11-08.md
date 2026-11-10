# vLLM / DeepInfer 六课源码研究记录

日期：2026-11-08。用途：为新的三节 vLLM 与三节 DeepInfer 中文课程提供经过源码核对的事实、数值例子、离线摘录定位，以及现有 08–14 课纠错依据。

## 研究边界与核验状态

- 本地 vLLM：`vendor/vllm`，`git rev-parse HEAD` = `ef9339e64d458b10ab49b2c9af35645eff6388fd`，origin = `vllm-project/vllm`。
- 本地 DeepInfer：`vendor/deepinfer`，HEAD = `a679bb86ce36e883a98a97733492b686e5b9a08e`，origin = `deepseek-ai/DeepInfer`。
- 两个 vendor checkout 的 `git status --short` 均为空。本次只读 vendor 与 lessons，不改源码或课程；只新增本文件。
- 曾调用 `web.run` 搜索 vLLM 官方文档和 DeepInfer 官方仓库，但工具没有返回可引用结果。因此**没有完成当前线上文档核验**，不把固定提交称为“最新版”。本文链接是固定提交 permalink，不是本次线上抓取结果。
- 下列数值例子是**按真实代码规则手算的缩小示例**，不是 GPU 实测、完整运行轨迹或性能 benchmark。DSpark 耗时曲线明确为人为设定。
- 源码中的注释也需核对可执行语句：DeepInfer decode 抢占的注释与实际代码有一处相反，下文专门说明。

## 六节新课建议：每节 7 个可教、可验证的知识点

### V1：预算与请求状态，沿一次 step 走完

1. 从 `EngineCore.step` 的闭环讲起：schedule → 异步提交 execute_model → grammar bitmask → 等执行结果 / 需要时 sample_tokens → 处理 abort → update_from_output。普通 `step` 中存在等待；`non_block=True` 本身不等于已经调度下一轮。[V-LOOP]
2. 定义三个请求量：`num_tokens` 是 prompt 加已提交输出；`num_tokens_with_spec` 再加 draft；`num_computed_tokens` 是调度账本的进度，异步时包含乐观计入的 in-flight tokens，不能永远翻译成“GPU 已经完成”。[V-REQUEST] [V-ADVANCE]
3. token 模型统一的是“待追赶数量”，不是删除所有特殊分支。基本需求为 `num_tokens_with_spec + num_output_placeholders - num_computed_tokens`；再受长 prefill 阈值、全局预算、输入预算、模型长度等限制。[V-BUDGET]
4. 两份预算不同：`token_budget` 扣本轮主模型调度 token；`input_budget` 每安排一个请求会扣 `num_new_tokens + draft_slots`。本轮验证的 draft 已在 `num_new_tokens` 中；`draft_slots` 是额外 drafting 预留，不能再把它当作本轮验证 token 数。[V-BUDGET] [V-ALLOC]
5. running 优先、waiting 用剩余预算；running 里也可能有未完成的 chunked prefill，不能把 running 等同 decode。KV 不足时默认牺牲 running 尾部，priority 分支按 `(priority, arrival_time)` 选最大项。[V-ORDER] [V-ALLOC]
6. 构造 SchedulerOutput 后先推进 computed/in-flight 账本，返回时再扣 rejected draft。以“已提交 token、乐观计算进度、物理 KV 可用边界”三条线讲异步，避免把它们混成一个数。[V-ADVANCE] [V-OUTPUT]
7. SchedulerOutput 是跨进程契约：新请求传完整数据，缓存请求传差量，`num_scheduled_tokens` 是本轮 token **总数**，`scheduled_spec_decode_tokens` 标明其中哪些是 draft。输出更新还要处理已取消请求、stale 输出、停止与缓存登记。[V-CONTRACT] [V-OUTPUT]

### V2：KV 块从创建、命中、引用到复用

1. 请求创建和追加输出时增量计算完整块的 token hash；算出 hash 不等于该 KV 已完成、已可共享。[V-REQUEST] [V-CACHE]
2. hash 链含前一块 hash、本块 token IDs、extra keys；extra keys 包括 LoRA、多模态标识、cache salt、prompt embeddings。块池再加 KV group ID，不能用“相同 token 子串”直接推出可复用。[V-HASH] [V-GROUP]
3. `get_computed_blocks` 是查连续可用前缀；常规生成要留下最后位置重算 logits，块对齐会令一次“只差最后一个 token”变成重算整个最后块。[V-CACHE]
4. `allocate_slots` 把主模型所需 token、lookahead、已有本地/外部 cache、reserved/watermark 等折成块需求。`full_sequence_must_fit` 检查当前已知序列能否容纳，并非保证未知的未来最大输出一定能容纳。[V-FIT]
5. `ref_cnt>0` 的块有活跃所有者；`ref_cnt==0` 的已缓存块可以留在 free queue，同时保留 hash，等待命中或淘汰。“free”是可复用资格，不等于已清零或已丢失 KV。[V-POOL]
6. 命中后的 `touch` 增加引用计数，必要时从 free queue 摘除；释放减计数，降为 0 后才回队列。默认 cached blocks 放队尾，非 cached blocks 放队首；新分配从队首取，并删除旧 hash。[V-POOL]
7. LRU 适合解释 cached blocks 的常规淘汰行为，但不是整个队列一条无例外规则：无 hash 块优先复用、`evict_first` 可以改变优先级，多 group / sparse-retention 边界还要协调。[V-POOL] [V-CACHE]

### V3：执行元数据与测量边界

1. Scheduler 交出的是计划，GPUModelRunner 维护持久请求/批状态，再把计划压平为一次前向输入。新旧请求的传输契约可从 SchedulerOutput 开始追。[V-CONTRACT] [V-PACK]
2. 按各请求本轮 token 数重复 req index；累加得到 `query_start_loc`；位置等于该请求本轮起点加局部 query offset。这里可以手算，不要跳过数组。[V-PACK]
3. `seq_lens` 是请求已有上下文加本轮 query 长度；`query_start_loc` 分割本轮扁平 token；`block_table` 把逻辑块定位到物理块；`slot_mapping` 决定本轮 KV 写入位置。四者不能互相替代。[V-META] [V-SLOT]
4. 异步 speculative decode 时 CPU 进度可能偏乐观；GPU 用上一轮实际接受数修正 computed token，再生成 positions、seq_lens、slot mapping。批内请求换序还要用 `prev_positions` 对回上一轮。[V-META]
5. 常规无 spec 情况的 logits index 为每条 query 段最后位置；未完成的 prefill chunk 虽可走采样路径，结果会由 discard mask 丢弃。一次前向处理 N 个 token，不代表生成 N 个输出。[V-PACK] [V-META]
6. 区分 runtime mode（NONE/PIECEWISE/FULL）与策略（含 FULL_DECODE_ONLY/FULL_AND_PIECEWISE）。FULL 能否覆盖 mixed/prefill 取决于 attention backend；不是天然仅 decode。该 runner 的常见路径在 `_model_forward` 后单独计算 logits，不能以 FULL 名称断言 logits 在图内。[V-GRAPH-ENUM] [V-GRAPH] [V-FORWARD]
7. 教学测量应单列：真实输出 token / 墙钟时间、输入 token、draft 验证 token、accepted draft、TTFT/后续间隔、prefill/decode 批组成、图 padding。模拟只检查数组和账本；性能要在固定模型、硬件、并发和配置下实测。这个测量清单是教学建议，不是固定提交的性能结论；为何区分 input/output/spec 可直接由上述执行/输出路径验证。[V-OUTPUT] [V-META]

### D1：公开边界与 Python → C++ → Worker 调度链

1. 开源包定位是离线推理；完整服务还需要外部 KV cache system、VE/Prefill/Decode disaggregation 和 DP replicas 间负载均衡。仓库存在 BlockManager 和部分 engine hooks，因此“不包含完整 KV 服务系统”不等于“没有 KV cache”。[D-README]
2. 固定提交声明支持 DeepSeek-V4.1-Flash、NVIDIA SM100、Ascend 950。课程应说“本文覆盖这一提交的支持矩阵”，不能扩张为所有硬件或未来模型；Linux 等条件也是该提交 README 的要求。[D-README]
3. 请求链：Python `add_request` 构造 Sequence，必要时等待结构化输出 matcher，然后 `scheduler.add_sequence`；pybind 对应 C++ `add_sequence`，写 arrival time 并推入 waiting。[D-ADD] [D-BIND] [D-QUEUE]
4. C++ `_schedule` 有 PROMPT、COMPLETION、NORMAL 三种 engine；NORMAL 用 DP ALL/ANY 投票保证本轮组内同为 prefill 或 decode，可能有 rank 跑空 prompt step。它并未照搬 vLLM 的“无阶段”调度抽象。[D-TYPE]
5. prefill 取 waiting 队首，受 token budget、CP 对齐、chunk checkpoints、多模态切分、最小 lookahead、块/状态容量约束，生成 `seqlens_q`。`FlopsEstimator.calc_flops` 的调用在 `if(log_stats)` 内，用于指标；不能说该公开循环按 FLOPs limit 解预算。[D-PREFILL] [D-METADATA]
6. decode 先接纳 prompted，再将 running 按 output_len 降序排，取尾部较短输出请求推进；不足时取头部较长输出请求抢占。之后 scheduled 按总序列长度降序排列，再由 MTPVarlen 决定 query 长度。**L557 注释说“抢占 output 最短”，与 L624–632 实际语句相反，应讲代码。**[D-DECODE]
7. `schedule` 生成 `(seq_metadata_list, scheduler_outputs)`，其中 `seqlen_k = cache.num_tokens + total_inflight_tokens + seqlen_q`。Worker 顺序为当前 forward → 上一轮 post_process → 当前 sample；ZOS decode 把本轮后处理留到下一轮。CPU 调度与 GPU 执行重叠须同时解释这些 in-flight/metadata 修正。[D-METADATA] [D-STEP] [D-WORKER]

### D2：MoE dispatch / combine 与 EPLB 的真实优化目标

1. logical expert 是模型语义；physical expert 是一个驻留副本。物理数等于逻辑数加冗余数，每 rank 可放多个 physical experts，并非“每卡只容一个专家”。[D-POLICY]
2. gate 接收 logical-to-physical map 与副本数；dispatch 接收 physical top-k 与权重，将 token 发到对应专家，同时累积本地 expert token stats。[D-MOE]
3. 非 MegaMoE 路径为 gate → dispatch → wait handle → routed_experts_forward → shared expert → combine；combine 使用 dispatch 的 handle 恢复原 token 布局，并处理共享专家输出。Megamoe 是另一路，不应把这一串当所有路径逐次 CPU 调用的保证。[D-MOE]
4. EPLB 输入是 `[history_steps, layers, logical_experts]` 的 token 统计。副本分配优化历史每一步的最大每副本负载之和；不是按最近单步 top-1 机械复制，也不是改变 router 的模型选择语义。[D-POLICY]
5. 放置时按每副本历史总负载降序遍历 physical experts；尝试放到每个未满 rank，最小化 `Σ_step max_rank(rank_load)`，同分取较小 rank ID。它是贪心，不能称全局最优解。[D-POLICY]
6. 物理槽位重排后，要用 inverse permutation 更新 logical-to-physical map，保留 -1 无效位；运行时先聚合统计/all-reduce，算新布局，搬专家权重，再复制三张映射/计数表。[D-POLICY] [D-REBALANCE]
7. 触发由 interval 与 prefill 配置决定；已读触发函数没有“偏斜超过阈值才触发”的判定。代价包括通信、权重搬运、同步与额外 buffer；不能凭想象添加“迁 KV”“暂停接收新请求”的服务实现。[D-TRIGGER] [D-REBALANCE]

### D3：DSpark、验证回退、SWA 与确定性

1. ConfidenceHead 估计 draft 接受概率；draft 预测既生成 token，也产生 confidence。confidence 用于“分配验证工作”，不是直接决定 token 是否可接受。真正接受/拒绝和 residual resampling 在 sampler。[D-CONFIDENCE] [D-SAMPLER]
2. 使用每位置 confidence 的累乘形成前缀接受率；前面的 draft 没接受，后面不能成为有效输出。CPU 更新把 raw confidence 与累计 accept_rates 分开保存。[D-RATES]
3. MTPVarlen 按全部请求的累计接受率排序，给每条请求保留 1 个基础 token；枚举总 query 预算，按预估有效输出 / 预估时间选择最佳总量，再分配每请求长度。[D-MTP]
4. 时间估计含 profiled graph-size forward 曲线、基于运行反馈的 forward slope、sampler 估计；总预算用 N−2 confidence，每请求长度使用 N−1 信息（ZOS 时在 GPU 按最新上一轮结果修正）。不要简写成“本轮看到 confidence 就立刻定所有长度”。[D-MTP] [D-GPU-RATES]
5. 验证后是 accepted prefix + 最后一个采样 token。C++ 先乐观推进 cache.num_tokens，再按 token_cnt 回填；遇到 EOS 也会截断 token_cnt。被拒绝的 draft 不能被当成已提交 token 或合法 KV 前缀。[D-COMMIT] [D-SAMPLER]
6. YOCO bounded replay 只重建有界后缀的 decoder SWA 状态；部署说明明确这是近似重建，并非自动等价于全量重放。公开配置分 off/mask/compact，compact 才移除早期 query；保留后缀长度必须覆盖 window + max backtracking。[D-SWA] [D-SWA-CONFIG]
7. 固定提交 README 的确定性配方是 NVIDIA 上 `BATCH_INVARIANT=1`、`ENABLE_MTP_VARLEN=0`、固定每请求 seq_seed；默认 seq_seed 会随机生成。batch invariance、固定 seed 和统计分布正确性是不同层次；不能宣称跨硬件/版本无条件 bitwise 相同，也不能把“非确定性”直接等同 RL 梯度错误。[D-README] [D-SEED]

## vLLM 的三个数值例子

### V-E1：先算预算，再算 accepted 与 computed

假设无 spec、无并发在途、无 prefix hit、KV 充足，已启用 chunked prefill，token/input budget 均为 8，长 prefill 阈值不限制：

| 请求 | 队列 | num_tokens | computed | 本轮需求 | 实际调度 |
|---|---|---:|---:|---:|---:|
| A | running | 10 | 9 | 1 | 1 |
| B | running（未完成 prefill） | 10 | 4 | 6 | 6 |
| C | waiting | 5 | 0 | 5 | 1 |

总数 8，剩余预算 0。发出计划后 computed 成为 A=10、B=10、C=1；C 还不能提交新输出，A/B 若各采样一个，则 num_tokens 成为 11，computed 仍为 10。**采样出的最后一个 token 要在后续步骤作为输入计算 KV。** 本例是按规则投影，省略 encoder/Mamba/grammar 等附加约束。[V-BUDGET] [V-ORDER] [V-ADVANCE] [V-META]

再看一个独立 spec 分支：原 `num_tokens=11, computed=10, spec_tokens=3`，需求 4（上轮最后 token + 3 drafts）。假设都调度，computed 乐观变 14；验证返回“接受 1 draft + 新采样 1 token”，输出列表长 2，故 accepted=2−1=1、rejected=3−1=2，computed 修正为 12，num_tokens 增到 13。不要把 4 解释成“4 个正常输入再加 3 drafts”。[V-OUTPUT]

**预算变体**：若 `token_budget=8,input_budget=12,draft_slots=2`，安排请求 X 的 4 个验证 token 后，剩余为 4 与 6；再安排 Y 的 4 个 token，剩余为 0 与 0。这里每请求多扣的 2 是 drafting 预留，不能等同物理 KV 块数。[V-ALLOC]

### V-E2：32 token 的“全前缀命中”为何仍算 16 token

设单一 full-attention group、块大小 16、prompt 长 32，两块都曾被算好且 hash 可命中。为了拿最后位置 logits，常规生成的最大命中长度为 31；只能复用完整块时命中 16，需重算剩下 16，而不是只算 1。课程图应画“可复用 16 / 本轮输入 16”。[V-CACHE]

独立的块池轨迹：free queue 初始为 `[U5, C3, C4]`，U 无 hash，C 有 hash，三者 ref_cnt 均为 0。另一个请求命中 C3，touch 后队列为 `[U5,C4]`、C3.ref_cnt=1；该请求释放 C3 后队列为 `[U5,C4,C3]`。之后分配两个新块取 U5 和 C4；C4 的旧 hash 被删除，C3 仍可命中。**缓存可用性与活跃引用是两个维度。** 此例忽略 null block、跨 pool 与 `evict_first`。[V-POOL]

### V-E3：手算 worker 的六个 token

设无 CP、无 spec、单 KV group，块大小 4，三个请求本轮 token 数 `[2,1,3]`、开始 computed `[4,10,0]`：

| 项目 | 值 |
|---|---|
| req_indices | `[0,0,1,2,2,2]` |
| 局部 query offset | `[0,1,0,0,1,2]` |
| positions | `[4,5,10,0,1,2]` |
| query_start_loc | `[0,2,3,6]` |
| seq_lens | `[6,11,3]` |
| 常规 logits_indices | `[1,2,5]` |

若 A 块表 `[7,2]`，B `[1,3,5]`，C `[8]`，则 `slot = physical_block * 4 + position % 4`，得到 `[8,9,22,32,33,34]`。只在本例无 CP 的前提下可直接用该公式；真实 kernel 有 CP interleave 和 PAD_ID 分支。[V-PACK] [V-META] [V-SLOT]

**三项关键易错点**：computed 不是永远已完成；free 不等于不可命中；packed input token 数不等于输出 token 数。让学生每次指出自己正在讨论哪一层状态。[V-REQUEST] [V-POOL] [V-META]

## DeepInfer 的三个数值例子

### D-E1：请求进入 C++ batch 的最短轨迹

为隔离 ZOS 和 chunking，设单 DP rank、NORMAL、关闭 MTP（nextn=0）、关闭 ZOS、CP size=1、KV 充足。A prompt=6，B prompt=3，prefill token budget=16。

1. Python 构造 A/B 的 Sequence，C++ `add_sequence` 后 waiting 为 `[A,B]`。
2. 满足 prefill 尝试点和投票条件时，`_schedule_prefill` 得 `scheduled=[A,B]`、`seqlens_q=[6,3]`、`num_batched_tokens=9`。两条完整 prompt 标记 RUNNING；这里不会把 decode 请求混入同一 prompt batch。
3. 元数据为 A `(q=6,k=6)`，B `(q=3,k=3)`；无上轮 in-flight。完成 prefill 各采样 1 token，cache.num_tokens 为 6/3，已提交总长度为 7/4。
4. 下一轮设走 decode：两者 q 均为 1；k 为 7/4，num_batched_tokens=2。decode 后无拒绝情况下 cache.num_tokens 为 7/4，总长度为 8/5。

这不是完整 scheduler 单元测试：省略了 dp 投票节奏、前缀命中、请求上限、lookahead、abort、停止等分支。它保留了“输入位置已算 KV / 最后采样 token 还没算 KV”的一位差。[D-ADD] [D-TYPE] [D-PREFILL] [D-METADATA] [D-COMMIT]

加一个抢占观察题：output_len 为 `[2,7,5]`，真实排序后为 `[7,5,2]`；先尝试尾部 2；若必须腾块，取头部 7 为 victim。读代码能纠正 L557 的旧注释，不能背“LCFS”名称来推断全部行为。[D-DECODE]

### D-E2：三逻辑专家、四物理槽位、两张卡

一个历史 step、一个 MoE layer，logical token load `[100,20,20]`，4 physical slots、每 rank capacity=2。复制 E0 后每副本最大负载由 100 降到 50；复制 E1 或 E2 则最大值仍为 100，所以冗余给 E0。

- 分配副本后旧 physical-to-logical 为 `[0,1,2,0]`，logical_count 为 `[2,1,1]`。
- 每副本负载为 `[50,20,20,50]`；稳定降序旧槽位顺序为 `[0,3,1,2]`。
- 贪心放置：槽 0 → rank0（50）；槽 3 → rank1（50）；槽 1 → rank0（70，平手取较小 rank）；槽 2 → rank1（70）。
- 最终 physical-to-logical 为 `[0,1,0,2]`，logical-to-physical 为 `E0:[0,2]，E1:[1,-1]，E2:[3,-1]`，每 rank 预测负载 70。

“一个副本分一半流量”是该策略估计模型中的 load/count，不承诺每个瞬时 batch 都精确 50/50；真实执行还需要 gate 使用映射、dispatch 将 token 送到物理副本、权重 shuffle 完成。[D-POLICY] [D-MOE] [D-REBALANCE]

### D-E3：DSpark 按总吞吐选择验证长度，并修正回退账本

设 2 个请求，每个 2 drafts；raw confidence：A=`[0.9,0.5]`，B=`[0.8,0.25]`。累计前缀接受率 A=`[0.9,0.45]`，B=`[0.8,0.2]`，全局排序 `A1 0.9, B1 0.8, A2 0.45, B2 0.2`。为手算设前后历史 confidence 相同、slope=0、sampler_ms=0；人为设 graph-size 曲线 `(2,1ms),(4,1.2ms),(6,2ms)`。[D-RATES] [D-MTP]

| total_q | 验证 draft 数 | 预期输出 token | 对应图耗时 | 预期 token/ms |
|---:|---:|---:|---:|---:|
| 2 | 0 | 2.00 | 1.0 | 2.000 |
| 3 | 1 | 2.90 | 1.2 | 2.417 |
| 4 | 2 | 3.70 | 1.2 | 3.083 |
| 5 | 3 | 4.15 | 2.0 | 2.075 |
| 6 | 4 | 4.35 | 2.0 | 2.175 |

所以 best total_q=4，每请求各得 1 个 draft，seqlens_q=`[2,2]`。看得到第 5 token 引起 graph bucket 跳变，不能用固定 `p×3−1` 阈值替代真实目标。真实值来自 profiling/运行反馈，此表不是性能宣传。[D-MTP]

单独看一次回退：之前 cache.num_tokens=100，已提交长度=101，本轮 q=4（基础 token+3 drafts），乐观 cache=104。若只接受 1 draft，再采样 1 token，token_cnt=2，commit 后 cache=102，已提交长度=103，仍差最后一个 token 的 KV。需要纠正的是逻辑边界，不能把未来读者指向已拒绝的 draft 位置。[D-COMMIT] [D-CACHE-COUNTER]

SWA 约束另算：window=128、max backtracking=6，则最低 keep=134，README 的 keep=140 满足要求，keep=128 不满足。并不需要“回退距离超过整个窗口”才有问题：即便只向后回退 1，在仅存最近 128 个槽位的环形缓存里，也可能已经覆盖回退后窗口所需的最旧位置。这个数值说明是由窗口区间推导；源码实际做的是 `window + backtracking <= keep` 检查，并有扩大的环形保存容量。[D-SWA-CONFIG] [D-SWA-ASSERT] [D-SWA-RING]

**三项关键易错点**：confidence 是预测，不是接受凭证；EPLB 复制/放置不改变模型的 logical expert 定义；离线 engine 不等于完整在线服务。确定性配方与性能配置必须分别标注。[D-SAMPLER] [D-POLICY] [D-README]

## 现有 08–14 课具体纠错与可替换段落

以下仅为建议，未修改 lessons。每项既包含原文定位，也给出可直接改写的正文。

### 08：统一模型不等于没有分支

文件 `lessons/0008-vllm-v1-architecture.html` L119–121 原文：“chunked prefill 不需要特殊分支”“投机解码不需要特殊分支”“前缀缓存命中不需要特殊分支”。

建议替换：

> vLLM 用同一套“请求还差多少 token”的账本表达 chunked prefill、投机解码和前缀命中。统一的是调度单位；真实实现仍分别处理长 prefill 限额、draft 验证、前缀查找、encoder 和混合 KV 状态。读完基本公式后，要继续追踪这些约束如何改变本轮可调度数量。

依据：[V-BUDGET] [V-ALLOC] [V-CACHE]。

同文件 L146 的“1.5 秒一次 decode 的小模型”和 L171 的“前端挂了不影响正在跑的请求（反之亦然）”未有本次源码证据支持。建议删除未经测量的时间，改成“进程分离减少前端工作对核心循环的直接干扰；具体故障恢复语义要另查”。不要把设计目的写成无条件容错保证。

### 09：补全预算、命中粒度和淘汰例外

文件 `lessons/0009-vllm-scheduling-kv.html` L43 把 draft_slots 解释为 KV 块预留；L91 只讲“重算最后一个 token”；L105 讲整个块队列纯 LRU。

建议替换：

> 本轮已调度 token 数包含主模型对 draft 的验证输入。调度器另外对每个请求从 input_budget 预留 draft_slots，KVManager 则通过 num_lookahead_tokens 等参数计算物理块需求，这两层不能混为一谈。前缀缓存必须为最后位置 logits 留下计算机会；由于块对齐，可能需要重算整个最后块。空闲块中仍可有有效缓存：命中会 touch 并移出 free queue，释放后缓存块通常进入队尾，无 hash 块优先放在队首复用。

依据：[V-BUDGET] [V-ALLOC] [V-FIT] [V-CACHE] [V-POOL]。

该课“full_sequence_must_fit 能保证整条序列跑完”还应限定为**当前已知 request.num_tokens 对应的序列**，不包含全部未知未来生成。128 块 prompt 与剩余 100 块的例子能说明 full-attention 无法容纳当前输入，但“40 条一次各分 1–2 块”并非由所给 8192 token budget 自动推出的真实调度，须标为概念反例并补明确 chunk 阈值。[V-FIT]

### 10：FULL 图范围与 logits

文件 `lessons/0010-vllm-execution-layer.html` L108 原文：“只有纯 decode 批次满足”；L113 原文：“vLLM 的 FULL 模式会尽量把 logits 也算进去”。

建议替换：

> FULL 表示以完整模型 forward 为捕获单位，不表示仅支持纯 decode。是否能捕获 mixed/prefill 批次由 attention backend 的 CUDA Graph 能力和形状分桶决定；不支持时配置会降级。这个提交的 GPUModelRunner 在 _model_forward 返回后选择 logits_indices，再调用 compute_logits，因此讲图范围时应画出模型 forward、logits 和 sampling 的真实边界。

依据：[V-GRAPH] [V-FORWARD]。该课相关选择题里“prefill 时 FULL 无从谈起”应同步删除绝对表述。

### 11：draft 已包含在调度数里

文件 `lessons/0011-vllm-advanced.html` L44 原文：“num_scheduled_tokens 个正常 token，外加这几个 draft token”。

建议替换：

> num_scheduled_tokens 给出这条请求本轮主模型输入的总数，其中可能已经包含 draft 验证位置。scheduled_spec_decode_tokens 标识需要验证的 draft，而不是在总数之外再加一批 token。例如上轮最后 token 加 3 个 draft，总数就是 4。返回后按实际接受的 draft 数回退 computed 账本，并提交 accepted prefix 与新的采样 token。

依据：[V-BUDGET] [V-ALLOC] [V-OUTPUT]。

拒绝采样的数学正确性应与具体配置区分；“所有方法、所有运行模式严格数值一致”不是由一般拒绝采样证明自动得到的。课程可保留理想分布证明，但不要把它等同 bitwise deterministic replay。

### 12：FlopsEstimator 与公开服务边界

文件 `lessons/0012-deepinfer-overview.html` L73/L94 原文：“估算 FLOPs，用于调度决策”“tflops_limit=get_bf16_tflops_limit() * 2”。

建议替换：

> Python LLMEngine 创建 C++ Scheduler，并传入 FlopsEstimator。这个固定提交中，calc_flops 在调度结果已经形成之后、log_stats 分支内计算统计指标；公开 prefill 调度主要按 token、上下文长度、容量、对齐和 chunk 约束组成 batch。部署 QA 另外介绍了内部服务的时间预算与二次成本模型，不能把它直接当成这份开源 scheduler 的现行实现。

依据：[D-PREFILL] [D-METADATA]，另见 `doc/inference_deployment_qa.md` L139–145（内部部署叙述）。

L99 的“Python 占 20–30%”未发现对应 benchmark；改为“降低每步 Python 对象处理与解释器开销，具体收益需测量”。GIL 也应改为“C++ 调度入口显式释放 GIL”，而不是“Python 线程什么都做不了”；源码锚点 `_schedule()` 第一行 `py::gil_scoped_release`。[D-TYPE]

该课 README 边界原文可以保留，但“最容易被接受的 PR 方向/社区需求最大”没有证据，不宜教成事实。`DeepSelect=专家路由 top-k` 也未在已读本地集成路径找到支持，先删除或查 DeepSelect 自身原始文档再恢复；vendor 子模块目录当前没有可读 README，不能仅凭名字猜职责。

### 13：循环、触发与迁移成本

文件 `lessons/0013-deepinfer-moe-eplb.html` L37 原文：“实际上每张卡只有一个专家的容量”；L110：“没有 Python 循环遍历 expert”；L120：“只有负载偏斜超过一定比例才触发”。

建议替换：

> 每个 rank 有 capacity 个物理专家槽位。策略先给逻辑专家分配冗余副本，再按每副本负载顺序遍历物理专家，将它放入能够降低历史峰值目标的 rank。实现有一个遍历 num_physical 的 Python for 循环，循环内对 layer、rank、history step 做张量计算。运行时按 rearrangement interval 和 prefill 相关配置触发重排；本提交的触发函数未设置负载偏斜阈值。重排需要统计通信、专家权重搬运与映射更新。

依据：[D-POLICY] [D-TRIGGER] [D-REBALANCE]。

L116–117 的“搬 KV”“暂停服务”建议改成“同步和权重搬运影响迭代时间，具体是否暂停请求接纳属于服务层设计，本次源码未证明”。复杂度不能只写 physical×rank 而忽略 layers/history；若只给定性描述，讲“逐物理副本尝试各 rank，按历史步峰值打分”更准确。[D-POLICY]

### 14：置信度不是固定阈值，SWA 不是无损回退补丁

文件 `lessons/0014-dspark-batch-invariance.html` L84–91 以“回退距离超过窗口”解释 bounded replay；L129 原文：“真实阈值会比 1/3 略低”。

建议替换：

> DSpark 先将逐 draft confidence 累乘，估计接受到每个位置的概率。调度器结合 graph-size 耗时曲线，在跨请求排序后的候选预算中最大化预计有效 token / 预计耗时，再给每条请求分配验证长度。固定收益阈值只能做一个独立的玩具模型，不能推出真实阈值必然更低。异步模式下还要区分选择总预算所用的历史 confidence 与 GPU 修正每请求长度时所用的信息。

依据：[D-RATES] [D-MTP] [D-GPU-RATES]。

建议替换 SWA 段落：

> Decoder bounded replay 的目标是减少 YOCO consumer 层 prefill 计算：只用有界后缀近似重建 SWA 状态。compact 模式真正移除早期 query，mask 模式保留 query 但收窄可见域。为了让回退后的窗口仍有足够 KV，keep_tokens 必须至少覆盖 window_size + max_backtracking。默认模式是 off；140 是 README 对默认 V4.1 设置给出的启用建议，不是该环境变量的默认值，也不意味着与全量重放严格等价。

依据：[D-SWA] [D-SWA-CONFIG]。

L99/L142 原文：“对聊天应用无所谓”“否则梯度就是错的”“rollout 和重算的 reward 对不上”；L112：“极致吞吐=全默认”。

建议替换：

> 确定性有利于回归测试、调试、结果审计和需要可复现轨迹的训练工作流；是否需要 bitwise 相同取决于算法与评测目标，不能由少量浮点差异直接推出梯度必错。固定提交给出的确定性配置是 NVIDIA 的 batch-invariant 路径、关闭 MTP variable-length verification，并固定 seq_seed。最快配置则取决于硬件和请求分布，必须测量，不能把默认配置称为吞吐最优。

依据：配置条件见 [D-README] [D-SEED]；关于梯度/吞吐的建议是删除超出来源的推断。

## 离线源码摘录清单：精确路径、行范围、锚点

以下路径均已在对应固定提交本地打开并核对。表内范围适合摘录；链接引用为同一提交。较大的源码文件可按同一行的多个短范围拆开显示。

### vLLM

| ID | 本地文件 | 行范围 | 搜索锚点 |
|---|---|---|---|
| V-LOOP | `vendor/vllm/vllm/v1/engine/core.py` | 710–740 | `def step(self)` |
| V-REQUEST | `vendor/vllm/vllm/v1/request.py` | 165–186；235–240；284–335 | `num_in_flight_tokens` / `append_output_token_ids` / `num_tokens_with_spec` |
| V-BUDGET | `vendor/vllm/vllm/v1/core/sched/scheduler.py` | 706–729；846–864 | `token_budget =` / `request.num_tokens_with_spec` |
| V-ORDER | 同上 | 800–829；1080–1102；1333–1384 | `First, schedule the RUNNING` / `request_token_budget` |
| V-ALLOC | 同上 | 922–1028 | `Preempt the lowest-priority request` / `input_budget -=` |
| V-ADVANCE | 同上 | 1888–1907 | `Advance the number of computed tokens` |
| V-OUTPUT | 同上 | 2349–2403；2441–2455 | `num_in_flight_tokens -=` / `num_rejected` / `_update_request_with_output` |
| V-CONTRACT | `vendor/vllm/vllm/v1/core/sched/output.py` | 245–264 | `class SchedulerOutput` |
| V-HASH | `vendor/vllm/vllm/v1/core/kv_cache_utils.py` | 640–678；710–740 | `generate_block_hash_extra_keys` / `hash_block_tokens` |
| V-GROUP | `vendor/vllm/vllm/v1/core/block_pool.py` | 175–201 | `def get_cached_block` / `make_block_hash_with_group_id` |
| V-CACHE | `vendor/vllm/vllm/v1/core/kv_cache_manager.py` | 266–328 | `get_computed_blocks` / `must recompute the last token` |
| V-FIT | 同上 | 530–587 | `if full_sequence_must_fit` / `num_tokens_need_slot` |
| V-POOL | `vendor/vllm/vllm/v1/core/block_pool.py` | 681–714；744–830 | `get_new_blocks` / `touch` / `free_blocks` |
| V-PACK | `vendor/vllm/vllm/v1/worker/gpu_model_runner.py` | 1923–1982；2031–2064 | `def _prepare_inputs` / `query_start_loc` / `discard_request_mask` |
| V-META | 同上 | 2102–2151；2187–2196；2356–2372 | `update_num_computed_tokens_for_batch_change` / `CommonAttentionMetadata` |
| V-SLOT | `vendor/vllm/vllm/v1/worker/block_table.py` | 348–376 | `virtual_block_size` / `slot_ids =` |
| V-FORWARD | `vendor/vllm/vllm/v1/worker/gpu_model_runner.py` | 4136–4190 | `set_forward_context` / `sample_hidden_states` |
| V-GRAPH | `vendor/vllm/docs/design/cuda_graphs.md` | 40–56；145–169 | `` `FULL` — `` / `CUDA Graphs always supported` |
| V-GRAPH-ENUM | `vendor/vllm/vllm/config/compilation.py` | 130–146 | `class CUDAGraphMode` |

### DeepInfer

| ID | 本地文件 | 行范围 | 搜索锚点 |
|---|---|---|---|
| D-README | `vendor/deepinfer/README.md` | 3–52 | `offline inference` / `Deterministic sampling` / `not included` |
| D-ADD | `vendor/deepinfer/deepinfer/engine/llm_engine.py` | 591–631 | `seq = Sequence` / `self.scheduler.add_sequence` |
| D-BIND | `vendor/deepinfer/csrc/engine/pybind_scheduler.cpp` | 93–109 | `.def("add_sequence"` / `.def("schedule"` |
| D-QUEUE | `vendor/deepinfer/csrc/engine/scheduler.hpp` | 135–142 | `void add_sequence` |
| D-TYPE | 同上 | 275–343 | `py::gil_scoped_release` / `switch (engine_type)` |
| D-PREFILL | 同上 | 347–372；450–549 | `_schedule_prefill` / `确定调度该 seq` |
| D-DECODE | 同上 | 556–659 | `_schedule_decode` / `a->get_output_len() >` |
| D-METADATA | 同上 | 667–734 | `sequence_metadata_pool->allocate` / `if (log_stats)` |
| D-STEP | `vendor/deepinfer/deepinfer/engine/llm_engine.py` | 860–887；947–1002 | `self.scheduler.schedule` / `_execute_model_async_step` / `post_process` |
| D-WORKER | `vendor/deepinfer/deepinfer/worker/worker.py` | 89–188 | `execute_model_step` / `post process prev step` |
| D-MOE | `vendor/deepinfer/deepinfer/model_executor/models/v4_modules/mlp.py` | 379–401；461–511；565–616 | `to_physical_map=` / `def dispatch` / `def combine` / `def forward` |
| D-POLICY | `vendor/deepinfer/deepinfer/model_executor/eplb/policy.py` | 12–70；73–140 | `_batch_greedy_indices` / `rebalance_experts_global` |
| D-TRIGGER | `vendor/deepinfer/deepinfer/model_executor/eplb/__init__.py` | 146–165 | `_update_rearrangement` |
| D-REBALANCE | 同上 | 195–209；222–278 | `_rearrange_experts` / `logical_expert_token_counts.scatter_add_` |
| D-CONFIDENCE | `vendor/deepinfer/deepinfer/model_executor/models/mtp/dflash.py` | 46–79；279–343 | `class ConfidenceHead` / `def sample_drafts` |
| D-RATES | `vendor/deepinfer/csrc/engine/scheduler.hpp` | 1239–1250 | `accept_rate *= confidence_data` |
| D-MTP | `vendor/deepinfer/csrc/engine/mtp_varlen.hpp` | 139–147；150–266 | `void schedule` / `cur_throughput` / `best_bsz` |
| D-GPU-RATES | `vendor/deepinfer/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py` | 14–83 | `compute_seqlens_q_by_confidence_rates_tl` / `extra_budget` |
| D-SAMPLER | `vendor/deepinfer/deepinfer/model_executor/layers/sampler.py` | 65–84；427–478 | `Decode 阶段` / `get_accept_result` |
| D-COMMIT | `vendor/deepinfer/csrc/engine/scheduler.hpp` | 899–1003 | `advance_optimistically` / `commit_processed_tokens` |
| D-CACHE-COUNTER | `vendor/deepinfer/csrc/engine/sequence.hpp` | 210–221 | `advance_optimistically` / `commit_processed_tokens` |
| D-SWA | `vendor/deepinfer/doc/inference_deployment_qa.md` | 235–245 | `approximate SWA KV cache` / `Decoder SWA bounded replay` |
| D-SWA-CONFIG | `vendor/deepinfer/deepinfer/environ.py` | 136–139；194–204 | `DEEPINFER_MAX_BACKTRACKING_TOKENS` / `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` |
| D-SWA-ASSERT | `vendor/deepinfer/deepinfer/worker/model_runner.py` | 101–113 | `self.window_size + envs.DEEPINFER_MAX_BACKTRACKING_TOKENS` |
| D-SWA-RING | `vendor/deepinfer/deepinfer/worker/cache_engine.py` | 91–106 | `window_size + envs.DEEPINFER_MAX_BACKTRACKING_TOKENS` |
| D-SEED | `vendor/deepinfer/deepinfer/sampling_params.py` | 85–91 | `def _make_seed` |

## 固定提交来源链接

[V-LOOP]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/engine/core.py#L710-L740
[V-REQUEST]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/request.py#L165-L335
[V-BUDGET]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/sched/scheduler.py#L706-L864
[V-ORDER]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/sched/scheduler.py#L800-L1384
[V-ALLOC]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/sched/scheduler.py#L922-L1028
[V-ADVANCE]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/sched/scheduler.py#L1888-L1907
[V-OUTPUT]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/sched/scheduler.py#L2349-L2455
[V-CONTRACT]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/sched/output.py#L245-L264
[V-HASH]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/kv_cache_utils.py#L640-L740
[V-GROUP]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/block_pool.py#L175-L201
[V-CACHE]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/kv_cache_manager.py#L266-L328
[V-FIT]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/kv_cache_manager.py#L530-L587
[V-POOL]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/core/block_pool.py#L681-L830
[V-PACK]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/worker/gpu_model_runner.py#L1923-L2064
[V-META]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/worker/gpu_model_runner.py#L2102-L2372
[V-SLOT]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/worker/block_table.py#L348-L376
[V-FORWARD]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/v1/worker/gpu_model_runner.py#L4136-L4190
[V-GRAPH]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/docs/design/cuda_graphs.md#L40-L56
[V-GRAPH-ENUM]: https://github.com/vllm-project/vllm/blob/ef9339e64d458b10ab49b2c9af35645eff6388fd/vllm/config/compilation.py#L130-L146
[D-README]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md#L3-L52
[D-ADD]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L591-L631
[D-BIND]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/pybind_scheduler.cpp#L93-L109
[D-QUEUE]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L135-L142
[D-TYPE]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L275-L343
[D-PREFILL]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L347-L549
[D-DECODE]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L556-L659
[D-METADATA]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L667-L734
[D-STEP]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L860-L1002
[D-WORKER]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/worker.py#L89-L188
[D-MOE]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/mlp.py#L379-L616
[D-POLICY]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/eplb/policy.py#L12-L140
[D-TRIGGER]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/eplb/__init__.py#L146-L165
[D-REBALANCE]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/eplb/__init__.py#L195-L278
[D-CONFIDENCE]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/mtp/dflash.py#L46-L343
[D-RATES]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L1239-L1250
[D-MTP]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/mtp_varlen.hpp#L139-L266
[D-GPU-RATES]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py#L14-L83
[D-SAMPLER]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler.py#L65-L84
[D-COMMIT]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L899-L1003
[D-CACHE-COUNTER]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/sequence.hpp#L210-L221
[D-SWA]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md#L235-L245
[D-SWA-CONFIG]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L136-L204
[D-SWA-ASSERT]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/model_runner.py#L101-L113
[D-SWA-RING]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L91-L106
[D-SEED]: https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/sampling_params.py#L85-L91
