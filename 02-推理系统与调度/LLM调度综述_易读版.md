# 基于长度预测的 LLM 推理调度：方向综述

> 面向 vLLM 推理引擎层的请求调度研究。本文分三块：长度预测主线、不依赖预测的强基线系统、Agent 场景的延伸；并讨论在 chunked prefill 出现之后，这个方向还剩哪些可做的空间。

---

## 一、这个方向叫什么

学界还没有统一名字，常见有三种叫法，各有侧重：

- **Length-aware / Length-prediction-based LLM Scheduling**：最贴近本研究方向的叫法。
- **Output-length-aware Serving** / **Generation-length-aware Batching**：偏 batching 视角。
- **SRTF / SJF for LLM Inference**：强调它是从操作系统调度算法（最短剩余时间优先 / 最短作业优先）迁移过来的。

更上一层的研究伞是 **LLM Serving / Inference Scheduling**，与 chunked prefill、prefill-decode disaggregation（把 prefill 和 decode 分到不同实例上跑）、KV-aware migration 等方向并列。写论文 related work 时，把自己定位成 *"length-aware scheduling for LLM serving, with applications to agent workloads"* 是比较稳的表述。

---

## 二、问题脉络：chunked prefill 出现之后，长度预测调度还做不做？

很多 length-aware 调度论文回避 chunked prefill 这件事，但它绕不开。先把这层关系讲清楚，整篇综述的立论才稳。

简单回顾一下当前几条主流路线已经做到哪里：

- **Sarathi-Serve (OSDI'24)** 用 chunked prefill（把长 prompt 切成定长 chunk，逐步喂进 batch）+ stall-free batching，把 prefill 和 decode 在同一 step 内的 head-of-line blocking 干掉了。LLaMA-2-13B 上相对 vLLM 在 TBT SLO 下吞吐约 2.6×，70B TP 场景约 5.6×。
- **DistServe (OSDI'24)** / **Splitwise (ISCA'24)** 进一步把 prefill 和 decode 物理拆到不同 GPU 实例上，DistServe 在严格 SLO 下 goodput 最多 7.4×。
- **FastServe (NSDI'26)** 在不预测长度的前提下，用 skip-join MLFQ（多级反馈队列，按 prefill 大小直接跳到对应优先级）已经能逼近 SRPT（最短剩余处理时间优先）的效果。
- **Llumnix (OSDI'24)** 也不预测，靠跨实例的 live KV migration 事后纠错——哪个实例堵了就把请求迁走。

这四条路线吃掉了长度预测调度的部分传统卖点，比如"短请求被长 prefill / 长 decode 拖延"这种最经典的 motivation。所以现在做长度预测调度，**剩余收益不在 mean JCT（平均完成时间），而在四个二阶指标上**：

1. **decode pool 内部的 SRTF 排序**：chunked prefill 和 disagg 都不管同一个 decode 池里谁先做；
2. **KV 内存规划与 admission control**：提前预测能避免请求生成到一半 OOM，从而触发抢占级联；
3. **migration / disagg 的路由质量**：迁移代价正比于剩余生成长度，预测越准迁得越聪明；
4. **多租户 SLO + 公平性联合**：VTC 那种事后 token 计数没法做前瞻，预测可以。

写 evaluation 时建议把 Sarathi-Serve 和 FastServe 同时作为 baseline 打这四个二阶指标，否则审稿人很容易直接质疑"chunked prefill 不是已经解决了？"

---

## 三、长度预测调度主线（9 篇）

按"预测器形态"和"调度策略"两条轴整理。

| 论文 | 会议/年 | 预测器 | 调度算法 | 是否基于 vLLM | 主要指标 / 收益 |
|---|---|---|---|---|---|
| **Response Length Perception** ([2305.13144](https://arxiv.org/abs/2305.13144)) | NeurIPS'23 | LLM 自身（指令微调出 perception 能力，准确率约 81-86%） | length-balanced micro-batch / bin-packing（离线场景） | 否（FastChat） | throughput 约 1.5–1.86× |
| **S³** ([2306.06000](https://arxiv.org/abs/2306.06000)) | NeurIPS'23 | DistilBERT 回归 / 分桶 | 激进装箱 + 预测错时 preempt-recompute | 否（FasterTransformer） | throughput 最高约 6.49×（OPT-13B/30B，不过 baseline 是 max-length 静态 padding，PagedAttention 出来后这个收益会被摊薄） |
| **SSJF / Proxy-model** ([2404.08509](https://arxiv.org/abs/2404.08509)) | AIOps@ASPLOS'24 | BERT 级 proxy 分桶 | skip-join MLFQ + 长度先验 + starvation 守卫 | **是** | mean JCT 约 2-3×，HOL blocking 显著缓解 |
| **Magnus** ([2406.04785](https://arxiv.org/abs/2406.04785)) | NeurIPS'24 | 随机森林 + 业务语义特征 | HRRN（最高响应比优先）+ length-aware 装箱（LMaaS 多租户） | **是** | throughput / mean JCT |
| **Don't Stop Me Now** ([2410.01035](https://arxiv.org/abs/2410.01035)) | ICLR'25 | **直接复用 prefill 的 hidden state**，不需要额外前向 | SRPT + continuous batching | **是** | mean JCT、TTFT |
| **ELIS** ([2505.09142](https://arxiv.org/abs/2505.09142)) | 2025 | DistilBERT 回归 | **ISRTF**：iteration 级 SRTF，每生成一个 token 都更新一次剩余预测 | **是** | mean JCT 下降约 20-40%，接近 oracle |
| **TetriInfer** ([2401.11181](https://arxiv.org/abs/2401.11181)) | 2024 | 轻量分类器（short/mid/long 三档） | chunked prefill + P/D disagg + decode 实例间 length-aware 路由 | 独立系统 | TBT、interference |
| **SageSched** ([2603.07917](https://arxiv.org/abs/2603.07917)) | 2026 | **概率预测器（输出区间或分位数，不是点估计）** | SLO/goodput-aware 混合批，按预测尾部分位决定 KV margin；和 chunked prefill 协同 | **是** | goodput、SLO 达成率 |
| **Uncertainty-aware RLP** ([SCIS'25](https://link.springer.com/article/10.1007/s11432-024-4550-8)) | SCIS 2025 | LLM perception + 区间/方差 | risk-aware bin-packing | 否（仿真） | padding ratio、throughput |

把这张表读出方向感，可以看到**两条演进轴**：

- **预测器轴**：LLM 自我感知（贵但准）→ BERT/DistilBERT proxy（便宜但精度有限）→ 随机森林 + 业务特征（LMaaS 友好）→ **复用 prefill hidden state**（几乎零开销，工程上最优雅）→ **概率/区间预测**（让调度器能直接消费不确定性）。
- **调度收益轴**：batch 装箱（throughput）→ SRTF / 抢占式 MLFQ（mean JCT）→ P/D disagg 路由（系统级）→ SLO/goodput + 不确定性（最前沿）。

工程上离 vLLM 最近的三条路线是 **ELIS、Don't Stop Me Now、SSJF (Qiu)**；研究上最有空间的方向是 **SageSched 这类把概率预测内化进调度器**的工作。

---

## 四、不依赖长度预测的强基线（必须打的对照组）

写论文时，下面这些系统是必须放进 baseline 的对手。

| 系统 | 会议 | 核心机制 | 用长度预测吗 | 与本方向的关系 |
|---|---|---|---|---|
| **FastServe** ([2305.05920](https://arxiv.org/abs/2305.05920)) | NSDI'26 | iteration 级 skip-join MLFQ，用 prefill size 跳到对应优先级 | 否 | **直接竞争**，是不预测长度的最强 SRPT 近似 |
| **Sarathi-Serve** ([2403.02310](https://arxiv.org/abs/2403.02310)) | OSDI'24 | chunked prefill + stall-free batching | 否 | **正交**，可以叠加 |
| **DistServe** ([2401.09670](https://arxiv.org/abs/2401.09670)) | OSDI'24 | prefill 与 decode 物理 disagg，提出 **goodput** 指标 | 否（用离线分布做 placement） | **互补**，decode pool 内部仍可上 SRTF |
| **Splitwise** ([2311.18677](https://arxiv.org/abs/2311.18677)) | ISCA'24 | P/D 双 pool + KV via InfiniBand；强调能耗 / 成本 | 否 | 互补 |
| **Llumnix** ([2406.03243](https://arxiv.org/abs/2406.03243)) | OSDI'24 | 跨实例 live KV migration，OS 风格 | 否，反应式 | **互补，是研究空白点之一** |
| **VTC** ([2401.00588](https://arxiv.org/abs/2401.00588)) | OSDI'24 | per-tenant virtual token counter（类 WFQ 公平队列） | 否 | 正交，公平性层 |
| **Mooncake** | FAST'25 | KV-centric disagg + prefix-aware routing | 否 | 正交 |

---

## 五、Agent 场景的延伸

Agent workload 的特征是：多轮 + 工具调用 + 控制流 + 长链。**目前几乎所有 agent serving 工作都没有显式用 LLM 输出长度预测**，而是用各种代理量绕开它。

- **Parrot (OSDI'24)** ([2405.19888](https://arxiv.org/abs/2405.19888))：把 agent 抽象成 Semantic Variable + DAG，应用层调度做 prefix sharing 和端到端 SLO 反向分配；token 上限靠应用层声明，不预测。
- **Autellix → Agentix (NSDI'26)** ([2502.13965](https://arxiv.org/abs/2502.13965))：把 agent 抽象成 program，调度核心 PLAS 是 program 级 MLFQ——**用"已服务 token 累计量"近似剩余服务时间**，本质是回避了显式长度预测。声称 4-15× 提升。
- **InferCept (ICML'24)** ([2402.01869](https://arxiv.org/abs/2402.01869))：tool intercept 期间的 KV 决策（preserve / swap / recompute）；它预测的是**外部工具等待时间**，不是 LLM 输出长度。
- **Conveyor** ([2406.00059](https://arxiv.org/abs/2406.00059))：工具部分执行流水线，让 tool latency 与 LLM decode 重叠。
- **Teola** ([2407.00326](https://arxiv.org/abs/2407.00326))：primitive 粒度的 DAG，端到端 RAG 流水线优化。

由此可见**三个明确的研究空白点**：

1. **Agent 级输出长度预测器本身没人做**。现有预测器（Zheng、SSJF、Don't Stop Me Now）都在 single-turn chat 数据上验证，多轮 / 工具 / 分支控制流场景下，预测难度和价值都没被系统量化。
2. **预测 + migration 双层策略没人做**。把长度预测器塞进 Llumnix 这种 migration-capable runtime，前瞻调度 + 兜底迁移结合，这个组合实验上是空白。
3. **Program 级剩余长度预测**。Autellix/Agentix 已经把 program 当作调度单位，但仍用累计服务量近似剩余量。引入 per-program 剩余 token / 剩余步数预测器（甚至预测 ReAct agent 还会调几次工具），是 NSDI/OSDI 级别的可投点。

---

## 六、指标地图

写综述或论文时，建议按以下四个维度组织 evaluation：

- **TTFT（Time To First Token）**：单 request 视角，从请求到达至生成出第一个 token 的时间。Sarathi、DistServe、Llumnix 都直接打这个。length-aware 调度只能间接受益（让短请求排到前面）。
- **TPOT（Time Per Output Token）/ TBT（Time Between Tokens）**：decode 阶段每个 token 的间隔，体现流式输出的流畅度。Sarathi-Serve 把它定义为首要 SLO。length-aware 一般不直接优化它，靠 chunked prefill 维持。
- **Goodput**：DistServe 提出，定义为"TTFT 和 TPOT 都达标前提下的 RPS"，是当前最被认可的 SLO 综合指标。SageSched 直接打这个。
- **SLO Attainment**：达成率百分比，Llumnix / SageSched 都用。Agent 场景下要扩展为 end-to-end SLO（Parrot 强调）。
- **Fairness**：VTC 单独做，多租户场景必须打。
- **Preemption count / KV churn**：抢占次数和 KV 缓存换入换出量。Llumnix 报告它是 P99 长尾的主要污染源，length-aware 调度的间接优势体现在这里。

**建议**：主打 **goodput + 长尾 + preemption count** 这三个组合，避开和 Sarathi 在 mean throughput 上正面竞争。

---

## 七、可投切入点

按可投价值递减排：

1. **Agent program 上的剩余服务时间预测器**（program 级，不是 request 级），叠加在 Autellix/Agentix 的 PLAS 之上，证明显式预测优于"已服务量近似"。出口：NSDI / OSDI。
2. **Length prediction × Live migration**（Llumnix + 预测器）。研究"前瞻调度 + 事后迁移"什么时候各自占优、组合策略如何设计。出口：OSDI / SOSP。
3. **概率长度预测的 SLO-aware 调度**（SageSched 是起点）。把不确定性塞进 admission control 与 KV 预算，主打 P99 goodput。出口：NSDI。
4. **复用 prefill hidden state 的 agent 场景适配**（Don't Stop Me Now 是 single-turn 验证）。看 hidden state 在多轮 / 工具调用之后是否仍然是好用的预测特征。

---

## 八、阅读路线图

按"读完一篇能写半个 related work"的密度排：

1. **Sarathi-Serve、DistServe、Llumnix、FastServe、VTC** —— OSDI'24 / NSDI'26 的五件套，把"非长度预测的强基线"建立起来。
2. **Don't Stop Me Now、ELIS、SSJF、SageSched** —— 长度预测主线的四篇代表，覆盖了从 hidden-state 预测到概率预测的演进。
3. **Parrot、Autellix/Agentix** —— Agent serving 的两个代表抽象层（DAG、program）。
4. **InferCept、Teola、Conveyor** —— 补全 agent + tool 的非 LLM 部分。
5. （选读）Mooncake、Splitwise —— 工业落地参考。

---

## 九、九篇主线的多维分类

第三节的表格按"预测器 + 调度算法"两列并排，但九篇论文之间真正的关系是多维的。下面从四个轴重新切一遍，再合成一张二维总览图，能看清这个方向的演化路径而不是九个并列点。

### 1. 按预测器形态（怎么预测）

| 类别 | 代表论文 | 特点 |
|---|---|---|
| **A. LLM 自身预测** | Zheng (RLP)、Uncertainty-aware RLP | 准但贵，多一次前向 |
| **B. 外接小模型 proxy** | S³、SSJF、ELIS | BERT/DistilBERT 级，便宜，精度有限 |
| **C. 经典 ML + 业务特征** | Magnus | 随机森林,LMaaS 多租户场景友好 |
| **D. 轻量分类(粗分档)** | TetriInfer | short/mid/long 三档,最便宜 |
| **E. 零开销(复用主模型)** | Don't Stop Me Now | 直接吃 prefill 的 hidden state,工程最优雅 |
| **F. 概率/区间预测** | SageSched、Uncertainty-aware RLP | 给调度器消费不确定性,最前沿 |

注意 **Uncertainty-aware RLP 同时落在 A 和 F**——它是在 LLM 自感知基础上加方差。

### 2. 按预测输出形式（输出什么）

- **点估计**：Zheng、Magnus、Don't Stop Me Now、ELIS
- **分桶 / 分类**：S³、SSJF、TetriInfer
- **区间 / 分布 / 分位数**：SageSched、Uncertainty-aware RLP

### 3. 按调度算法家族（怎么用预测）

| 家族 | 论文 | 主打什么 |
|---|---|---|
| **装箱 / 批组合** | Zheng、S³、Uncertainty-aware RLP | throughput |
| **SRTF / SRPT(短剩余优先)** | Don't Stop Me Now、ELIS | mean JCT |
| **MLFQ / 优先级队列** | SSJF | mean JCT、防饿死 |
| **多租户公平(HRRN)** | Magnus | LMaaS 公平 + 吞吐 |
| **系统级路由 / P/D disagg** | TetriInfer | TBT、interference |
| **SLO / goodput-aware + 风险感知** | SageSched | P99 goodput |

### 4. 按调度粒度

- **离线 / 批级**：Zheng(offline serving)
- **请求级**（到来时一次决策）：Magnus、Uncertainty-aware RLP
- **Iteration 级**（每步重决策、可抢占）：S³、SSJF、Don't Stop Me Now、ELIS、SageSched
- **集群级**（跨实例路由）：TetriInfer

可以看出 **iteration 级是主流**——iteration-level scheduling 本身就是 vLLM 的范式。

### 5. 二维总览（最有信息量的一张图）

横轴 = 调度算法复杂度，纵轴 = 预测器复杂度，能直接读出研究路径：

```
                 装箱       SRTF        MLFQ      公平      系统路由     SLO/概率
LLM 自感知       Zheng
小模型 proxy     S³        ELIS         SSJF
ML+业务特征                                        Magnus
轻量分类                                                    TetriInfer
hidden state              Don't Stop
                          Me Now
概率预测                                                                 SageSched
                                                                         Uncertainty-aware RLP
```

读这张图能看出**两个对角演进方向**：

1. **左上 → 右下**：预测器从"贵 + 准"往"轻 + 耦合主模型"走(Zheng → ELIS → Don't Stop Me Now),调度从"装箱"往"SRTF / SLO"走。
2. **右下角是当前最前沿**：概率预测 + SLO-aware(SageSched),既不和 chunked prefill 抢饭吃,又把"预测必然有误差"这件事内化进调度器。

### 6. 仍是空格的位置（可投点）

从这张分类图能直接读出三个**还没人占的格子**：

- **hidden state 预测器 + 概率输出**：Don't Stop Me Now 工程优雅但只给点估计；如果能让 hidden state 直接输出分布或方差，就能拿掉 SageSched 的额外预测器开销。
- **概率预测 + 多租户公平**：Magnus 那一格是点估计 + HRRN，没人把不确定性塞进多租户调度。
- **概率预测 + 系统级路由**：TetriInfer 那一格用的是 3 分类，迁移 / disagg 路由的质量正比于剩余长度方差，这里有空间。

---

## Sources

**长度预测主线**
- [Response Length Perception (NeurIPS'23)](https://arxiv.org/abs/2305.13144)
- [S³ (NeurIPS'23)](https://arxiv.org/abs/2306.06000)
- [SSJF / Proxy-model](https://arxiv.org/abs/2404.08509)
- [Magnus (NeurIPS'24)](https://arxiv.org/abs/2406.04785)
- [Don't Stop Me Now (ICLR'25)](https://arxiv.org/abs/2410.01035)
- [ELIS](https://arxiv.org/abs/2505.09142)
- [TetriInfer](https://arxiv.org/abs/2401.11181)
- [SageSched](https://arxiv.org/abs/2603.07917)
- [Uncertainty-aware RLP (SCIS'25)](https://link.springer.com/article/10.1007/s11432-024-4550-8)

**非预测强基线**
- [Sarathi-Serve (OSDI'24)](https://arxiv.org/abs/2403.02310)
- [DistServe (OSDI'24)](https://arxiv.org/abs/2401.09670)
- [Splitwise (ISCA'24)](https://arxiv.org/abs/2311.18677)
- [FastServe (NSDI'26)](https://arxiv.org/abs/2305.05920)
- [Llumnix (OSDI'24)](https://arxiv.org/abs/2406.03243)
- [VTC (OSDI'24)](https://arxiv.org/abs/2401.00588)

**Agent serving**
- [Parrot (OSDI'24)](https://arxiv.org/abs/2405.19888)
- [Autellix / Agentix (NSDI'26)](https://arxiv.org/abs/2502.13965)
- [InferCept (ICML'24)](https://arxiv.org/abs/2402.01869)
- [Conveyor](https://arxiv.org/abs/2406.00059)
- [Teola](https://arxiv.org/abs/2407.00326)
