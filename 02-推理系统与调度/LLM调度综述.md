# 基于长度预测的 LLM 推理调度：方向综述

> 面向 vLLM 推理引擎层的请求调度研究。覆盖长度预测主线、非预测强基线、Agent 场景延伸三大块，并给出 chunked prefill 出现之后该方向的剩余研究空间与具体切入点。

---

## 一、方向命名

学界没统一称呼，常见叫法有三种，含义略有侧重：

- **Length-aware / Length-prediction-based LLM Scheduling**（最贴本研究方向）
- **Output-length-aware Serving** / **Generation-length-aware Batching**（偏 batching 视角）
- **SRTF / SJF for LLM Inference**（强调它从 OS 调度算法迁移而来）

更上一层的研究伞是 **LLM Serving / Inference Scheduling**，与 chunked prefill、prefill-decode disaggregation、KV-aware migration 并列。写论文 related work 时，把自己定位成 *"length-aware scheduling for LLM serving, with applications to agent workloads"* 是最稳的。

---

## 二、问题脉络：chunked prefill 出现之后，长度预测调度还做不做？

很多 length-aware 调度论文回避 chunked prefill。把这个问题先讲清，整篇综述才立得住。

- **Sarathi-Serve (OSDI'24)** 用 chunked prefill + stall-free batching 把 prefill→decode 的 step 内 HOL 干掉了，LLaMA-2-13B 上相对 vLLM 在 TBT SLO 下吞吐 ~2.6×，70B TP 场景 ~5.6×。
- **DistServe (OSDI'24)** / **Splitwise (ISCA'24)** 进一步把 prefill 与 decode 物理 disagg，DistServe 在严格 SLO 下 goodput 最多 7.4×。
- **FastServe (NSDI'26)** 在 length-oblivious 假设下用 skip-join MLFQ 已经能逼近 SRPT。
- **Llumnix (OSDI'24)** 不预测、靠 live KV migration 事后纠错。

这四条路线吃掉了 length-aware 调度的部分传统卖点（"短请求被长 prefill / 长 decode 拖延"）。**长度预测调度的剩余收益不在 mean JCT，而在四个二阶指标上**：

1. **decode pool 内部的 SRTF 排序**（chunked prefill / disagg 都不管这个）
2. **KV 内存规划与 admission control**（避免 mid-generation OOM 触发抢占级联）
3. **migration / disagg 路由质量**（迁移代价正比于剩余长度）
4. **多租户 SLO + 公平联合**（VTC 那种 token 计数是事后的）

写 evaluation 时**必须**把 Sarathi-Serve 和 FastServe 同时作为 baseline 打这四个二阶指标，否则审稿人会直接质疑"chunked prefill 不是已经解决了？"

---

## 三、长度预测调度主线（9 篇）

按"预测器形态"和"调度策略"两条轴排。

| 论文 | 会议/年 | 预测器 | 调度算法 | 是否基于 vLLM | 主要指标 / 收益 |
|---|---|---|---|---|---|
| **Response Length Perception** ([2305.13144](https://arxiv.org/abs/2305.13144)) | NeurIPS'23 | LLM 自身（指令微调出 perception 能力，准确率 ~81-86%） | length-balanced micro-batch / bin-packing（离线） | 否（FastChat） | throughput ~1.5–1.86× |
| **S³** ([2306.06000](https://arxiv.org/abs/2306.06000)) | NeurIPS'23 | DistilBERT 回归/分桶 | 激进装箱 + mispredict 时 preempt-recompute | 否（FasterTransformer） | throughput 最高 ~6.49×（OPT-13B/30B，但 baseline 是 max-length 静态 padding，被 PagedAttention 摊薄） |
| **SSJF / Proxy-model** ([2404.08509](https://arxiv.org/abs/2404.08509)) | AIOps@ASPLOS'24 | BERT 级 proxy 分桶 | skip-join MLFQ + 长度先验 + starvation 守卫 | **是** | mean JCT ~2-3×，HOL blocking 显著缓解 |
| **Magnus** ([2406.04785](https://arxiv.org/abs/2406.04785)) | NeurIPS'24 | 随机森林 + 业务语义特征 | HRRN + length-aware 装箱（LMaaS 多租户） | **是** | throughput / mean JCT |
| **Don't Stop Me Now** ([2410.01035](https://arxiv.org/abs/2410.01035)) | ICLR'25 | **复用 prefill 的 hidden state**，零额外前向 | SRPT + continuous batching | **是** | mean JCT、TTFT |
| **ELIS** ([2505.09142](https://arxiv.org/abs/2505.09142)) | 2025 | DistilBERT 回归 | **ISRTF**：iteration 级 SRTF，已生成 tokens 持续修正剩余预测 | **是** | mean JCT 下降 ~20-40%，接近 oracle |
| **TetriInfer** ([2401.11181](https://arxiv.org/abs/2401.11181)) | 2024 | 轻量分类器（short/mid/long） | chunked prefill + P/D disagg + decode 实例间 length-aware 路由 | 独立系统 | TBT、interference |
| **SageSched** ([2603.07917](https://arxiv.org/abs/2603.07917)) | 2026 | **概率预测器（区间/分位数）** | SLO/goodput-aware 混合批，按预测尾部分位决定 KV margin；和 chunked prefill 协同 | **是** | goodput、SLO 达成率 |
| **Uncertainty-aware RLP** ([SCIS'25](https://link.springer.com/article/10.1007/s11432-024-4550-8)) | SCIS 2025 | LLM perception + 区间/方差 | risk-aware bin-packing | 否（仿真） | padding ratio、throughput |

把这张表读出方向感，**两条演进轴**：

- **预测器轴**：LLM 自我感知（贵但准）→ BERT/DistilBERT proxy（便宜但精度有限）→ 随机森林 + 业务特征（LMaaS 友好）→ **复用 prefill hidden state**（几乎零开销，工程上最优雅）→ **概率/区间预测**（喂给调度器消费不确定性）。
- **调度收益轴**：batch 装箱（throughput） → SRTF / 抢占式 MLFQ（mean JCT） → P/D disagg 路由（系统级） → SLO/goodput + 不确定性（最前沿）。

工程上离 vLLM 最近的三条路线是 **ELIS、Don't Stop Me Now、SSJF (Qiu)**；研究上最有空间的方向是 **SageSched 这类把概率预测内化进调度器**。

---

## 四、不依赖长度预测的强基线（必须打的对照组）

写论文时它们是必须放进 baseline 的对手。

| 系统 | 会议 | 核心机制 | 用长度预测吗 | 与本方向的关系 |
|---|---|---|---|---|
| **FastServe** ([2305.05920](https://arxiv.org/abs/2305.05920)) | NSDI'26 | iteration 级 skip-join MLFQ，用 prefill size 跳到对应优先级 | 否 | **直接竞争**，是 length-oblivious 的最强 SRPT 近似 |
| **Sarathi-Serve** ([2403.02310](https://arxiv.org/abs/2403.02310)) | OSDI'24 | chunked prefill + stall-free batching | 否 | **正交**，可叠加 |
| **DistServe** ([2401.09670](https://arxiv.org/abs/2401.09670)) | OSDI'24 | prefill/decode 物理 disagg，定义 **goodput** | 否（用离线分布做 placement） | **互补**，decode pool 内部仍可上 SRTF |
| **Splitwise** ([2311.18677](https://arxiv.org/abs/2311.18677)) | ISCA'24 | P/D 双 pool + KV via IB；强调能耗/成本 | 否 | 互补 |
| **Llumnix** ([2406.03243](https://arxiv.org/abs/2406.03243)) | OSDI'24 | 跨实例 live KV migration，OS-style | 否，反应式 | **互补，研究空白点之一** |
| **VTC** ([2401.00588](https://arxiv.org/abs/2401.00588)) | OSDI'24 | per-tenant virtual token counter（类 WFQ） | 否 | 正交，公平层 |
| **Mooncake** | FAST'25 | KV-centric disagg + prefix-aware routing | 否 | 正交 |

---

## 五、Agent 场景的延伸

Agent workload 的特征是：多轮 + 工具调用 + 控制流 + 长链。**几乎所有 agent serving 工作目前都没显式用 LLM 输出长度预测**，而是用各种代理量。

- **Parrot (OSDI'24)** ([2405.19888](https://arxiv.org/abs/2405.19888))：把 agent 抽成 Semantic Variable + DAG，应用级调度做 prefix sharing 与端到端 SLO 反向分配；token 上限靠应用层声明，不预测。
- **Autellix → Agentix (NSDI'26)** ([2502.13965](https://arxiv.org/abs/2502.13965))：把 agent 抽成 program，调度核心 PLAS 是 program-level MLFQ——**用"已服务 token 累计量"近似剩余服务时间**，本质是回避了显式长度预测。声称 4-15× 提升。
- **InferCept (ICML'24)** ([2402.01869](https://arxiv.org/abs/2402.01869))：tool intercept 期间 KV 决策（preserve/swap/recompute）；预测的是**外部工具等待时间**，不是 LLM 输出长度。
- **Conveyor** ([2406.00059](https://arxiv.org/abs/2406.00059))：工具部分执行流水线，让 tool latency 与 LLM decode 重叠。
- **Teola** ([2407.00326](https://arxiv.org/abs/2407.00326))：primitive-granularity DAG，端到端 RAG 流水线优化。

**三个明确的研究空白点**：

1. **Agent 级输出长度预测器本身没人做**。现有预测器（Zheng、SSJF、Don't Stop Me Now）都在 single-turn chat 数据上验证，多轮/工具/分支控制流场景下预测难度和价值都没被系统量化。
2. **预测 + migration 双层策略没人做**。把长度预测器塞进 Llumnix 这种 migration-capable runtime，前瞻调度 + 兜底迁移结合，实验空白。
3. **Program-level 剩余长度预测**。Autellix/Agentix 已经把 program 作为调度单位，但仍用累计服务量近似剩余量。引入 per-program 剩余 token / 剩余步数预测器（甚至预测 ReAct 还会调几次工具），是 NSDI/OSDI 级别的可投点。

---

## 六、指标地图

写综述/论文时按这四个维度组织 evaluation：

- **TTFT**：单 request 视角，Sarathi、DistServe、Llumnix 都直接打这个。length-aware 在这里只能间接受益（短请求排到前面）。
- **TPOT / TBT**：Sarathi-Serve 把它定义成首要 SLO。length-aware 一般不直接优化它，靠 chunked prefill 维持。
- **Goodput**：DistServe 提出，定义为 "TTFT 和 TPOT 都达标前提下的 RPS"，是当前最被认可的 SLO 综合指标。SageSched 直接打这个。
- **SLO Attainment**：百分比，Llumnix / SageSched 都用。Agent 场景下要扩展为 end-to-end SLO（Parrot 强调）。
- **Fairness**：VTC 单独做，多租户场景必须打。
- **Preemption count / KV churn**：Llumnix 报告它是 P99 主要污染源，length-aware 调度的间接优势体现在这里。

**建议**：主打 **goodput + 长尾 + preemption count** 三个组合，避开和 Sarathi 在 mean throughput 上正面竞争。

---

## 七、可投切入点

按可投价值递减排：

1. **Agent program 上的剩余服务时间预测器**（program 级，不是 request 级），叠加在 Autellix/Agentix 的 PLAS 之上，证明显式预测优于"已服务量近似"。出口：NSDI / OSDI。
2. **Length prediction × Live migration**（Llumnix + 预测器）。研究"前瞻调度 + 事后迁移"什么时候各自占优、组合策略如何设计。出口：OSDI / SOSP。
3. **概率长度预测的 SLO-aware 调度**（SageSched 是起点）。把不确定性塞进 admission control 与 KV 预算，主打 P99 goodput。出口：NSDI。
4. **复用 prefill hidden state 的 agent 场景适配**（Don't Stop Me Now 是 single-turn）。看 hidden state 在多轮/工具调用后是否仍是好预测特征。

---

## 八、阅读路线图

按读完一篇能写半个 related work 来排：

1. **Sarathi-Serve、DistServe、Llumnix、FastServe、VTC** —— OSDI'24/NSDI'26 的五件套，把"非长度预测的强基线"建立起来。
2. **Don't Stop Me Now、ELIS、SSJF、SageSched** —— 长度预测主线的四篇代表，覆盖了从 hidden-state 预测到概率预测的演进。
3. **Parrot、Autellix/Agentix** —— Agent serving 的两个代表抽象层（DAG、program）。
4. **InferCept、Teola、Conveyor** —— 补全 agent + tool 的非 LLM 部分。
5. （选）Mooncake、Splitwise —— 工业落地参考。

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

