# 第二部分：Agent 服务场景中的调度问题

## 2.1 Agent 应用改变了传统推理范式

传统 LLM 推理服务建立在一个隐含假设之上：**请求是独立的、短的、无状态的**。一次调用就是「发一段 prompt → 模型 decode 到 EOS → 返回 → 释放资源」，请求之间互不相关，vLLM、TGI 这类引擎据此把 continuous batching + FCFS / 简单优先级做到了极致。

Agent 应用的出现打破了这个假设。一次用户任务不再等于一次 LLM 调用，而是一条 **「思考 → 行动 → 观察」反复循环、直到任务完成**的执行链。可以从两个视角刻画这种转变：

- **Trajectory（轨迹）视角**：一个任务是单条但很长的执行轨迹 —— 几次到几十次 LLM call，中间穿插任意数量的 tool call，端到端可达数十秒到数分钟。请求从此带上了**跨调用状态**（历史轨迹）和**多步依赖**。
- **Workflow（编排）视角**：多 Agent 协作、DAG-style 规划把任务组织成一张有向无环图 —— 有的节点能并发，有的必须等上游完成，节点之间还共享大段 prompt。请求从此带上了**依赖结构**和**共享关系**。

核心矛盾由此产生：**调度器的调度单位是「请求」，但用户体验和系统真正要优化的单位是「trajectory / workflow」** —— 粒度错配。传统系统对一次请求的全部认知是「进来 → prefill → decode → 出 EOS → 走人」，它既看不出哪些请求属于同一任务的多步，也看不出哪些请求彼此并发、彼此共享。旧假设全部失效，这正是 Agent 场景需要新调度抽象的根本原因。

## 2.2 Agent 引入的新问题

可以把 Agent 给推理系统带来的新问题归为四个层面，从上到下依次是「调度编排 → 工具阻塞 → KV 缓存 → 语义信息」。

### A. 调度与编排层

1. **Trajectory 级队头阻塞（HoL blocking）**：FCFS 以请求为单位排队，一个跑了 10 分钟的 Coding Agent 会一直占着 decode slot，后到的 200ms 简单请求被迫排在它后面，端到端 P99 被拉到分钟级。本质是「以请求排队，却要按 trajectory 衡量体验」。

2. **输出长度方差极大（≈10×）**：Agent 的单步输出从几十 token（tool 参数）到数万 token（长 CoT / 报告生成）不等，传统调度假设「输出长度方差不大」失效，长短任务混跑时短任务被长任务严重拖累。

3. **DAG 依赖与并发表达不了**：多 Agent / DAG-style Agent 本可并发的节点（如 Planner 之下的多个 Worker），发到后端后和别人的请求混在一起 FCFS 排队，被串行化；本该并发的没并发，端到端延迟翻倍，GPU 利用率反而下降。

4. **关键路径错排**：Reviewer 这类节点需要等多个上游 Worker 完成才能发，这段等待发生在前端，后端完全不可见，无法做关键路径分析，分不清哪个节点卡住了所有人、该优先跑。

5. **多 SLO 混跑互相干扰**：同一集群里聊天（要求 ~100ms）、Agent、推理流（可接受秒级）等异构负载混在同一 batch，慢任务拖累快任务，谁的 SLO 都难保证；缺少**准入控制**则会让注定无法在 SLO 内完成的 Agent 白占 KV 与算力。

### B. 工具阻塞层

6. **显存占用与算力占用解耦**：LLM 输出 tool call 后要等外部 API / 数据库 / 代码执行返回，这段时间从几百毫秒到几十秒，往往比 LLM 自己 decode 还长。这条请求的 **KV Cache 还占着显存**（下一步还要用历史），但 **GPU 算力对它完全闲置**。原本紧耦合的「显存占用 = 算力占用」被打破。

7. **「重新 prefill vs 死等」两难**：tool 等待期间，要么当 finish 释放 KV（返回后重新 prefill 几万 token 历史，几秒算力白烧），要么保留 KV 死等（几百 MB ~ 几 GB 显存闲置）。一台 H100 若一半显存被「在等 tool」的 trajectory 占着，有效吞吐直接腰斩。

### C. KV 缓存层

在 chatbot 里 KV 是「用完即弃」的瞬时中间产物，LRU 管够用。Agent 场景下 KV 变成了**带生命周期、有归属、有共享关系的资源**：

8. **跨轮复用**：同一 trajectory 的 step 5 要用 step 1–4 的 KV，但中间隔着几秒 tool 时间，显存压力下前缀 cache 可能已被 LRU 踢掉，下次只能重新 prefill（prefix cache 抖动）。

9. **跨 Agent 共享**：多个 Agent 共用 system prompt + tool 定义 + few-shot，这部分 KV 完全相同，但请求可能落到不同 GPU / 不同时间，复用机会被随机化。

10. **跨时间存活（TTL）**：用户在某条会话里思考 30 秒，这条会话的 KV 该保留还是踢掉？保留浪费显存，踢掉下次重新 prefill 几万 token。LRU「最近没用就踢」会大量误判（踢掉马上要用的、留着永不再用的）。

11. **非前缀复用**：RAG 拼上来的多个独立 chunk 不是前缀关系、顺序一变就匹配不上，简单的 prefix-only 复用抓不住这类机会。Agent 后期一次 call 的 prompt 可达 3 万 token、新增只有 200 token，命中率从 95% 掉到 60%，集群 prefill 算力要多花数倍 —— 这是资源浪费问题。

### D. 语义信息丢失层

12. **API 层信息丢失，后端盲调度**：前端框架（LangChain/AutoGPT）本已把 Agent 拼成有结构的程序（哪些顺序依赖、哪些并发、哪些共享 prompt），但 HTTP 请求发到 vLLM 那一刻这些结构信息全丢了，后端只看到 N 个独立的 `/v1/completions`。结果是共享 prefix 看不出、能并发的被串行、整条任务的端到端 deadline 不可见 —— 后端只能优化单请求 TTFT，做不了跨请求 / 全局优化。

> **一句话总结**：传统系统假设「请求独立、短、无状态」，而 Agent 场景的请求是「有依赖、长、有跨调用状态」的。上面 12 个问题都是这一根本错配在不同层面的投影。

## 2.3 现有 Agent 应用的特点与典型 Workflow

现有 Agent 应用大致可归为五种形态，它们在 LLM call 次数、tool 依赖、输出长度三个维度上分布很广，对调度的压力点也各不相同：

| 形态 | 典型应用 | Workflow 特点 | 主要调度压力点 |
| --- | --- | --- | --- |
| **ReAct（单 Agent 循环）** | LangChain、AutoGPT | 单链：思考→工具→观察反复迭代，前一步出完才能启动后一步 | 多轮交互 → trajectory 队头阻塞；tool 阻塞 |
| **多 Agent 协作** | MetaGPT、AutoGen、CrewAI | DAG：Planner→多 Worker 并发→Reviewer 汇聚，角色间互相对话 | DAG 并发表达、关键路径、跨 Agent KV 共享 |
| **DAG-style 规划** | LLMCompiler | 先一次性规划成函数调用图，再并发执行 10+ 个 tool | tool↔tool 并行、依赖图调度 |
| **Reasoning Agent** | OpenAI o1、DeepSeek R1、QwQ | 内部 CoT 可达数万 token，不一定调外部工具 | 输出长度方差大、长序列 KV 占用 |
| **RAG Agent** | Perplexity、企业知识库 | 思考 + 检索 + 生成，本质是带检索工具的 Agent | 非前缀 KV 复用（CacheBlend 类）、检索阻塞 |

**以 Deep Research 类 Agent 为例**，其 workflow 把上述压力点集中体现出来：

```
用户提问（复杂研究问题）
  → Planner 拆解成多个子问题（1 次长 LLM call，输出规划 DAG）
  → 多个子问题并发检索（多路 tool call：搜索引擎 / 网页爬取，秒级阻塞）
  → 每个子问题各自「读资料 → 反思 → 再检索」多轮迭代（多条子 trajectory）
  → 汇总各分支结果（依赖所有子分支完成才能发，关键路径汇聚点）
  → 生成长篇研究报告（1 次超长输出 LLM call，数千~上万 token）
```

它同时踩中：①Planner→子任务的 **DAG 并发**；②大量**检索 tool 阻塞**；③子任务间共享 system prompt + 工具定义的**跨 Agent KV 共享**；④汇总节点的**关键路径**；⑤报告生成的**长输出 + 大方差**。一个 Deep Research 任务端到端可达数分钟、几十次 LLM call，是 Agent 调度问题的「集大成者」。

**共性特征总结**：(1) 长 trajectory、多步依赖；(2) 输出长度方差大（10×）且事先未知；(3) 高度结构化、重复的 prompt（前缀 90% 共享）；(4) 频繁 tool 阻塞导致显存/算力解耦；(5) 前端有结构信息但后端不可见。

## 2.4 现有相关工作：针对哪些问题做了什么

按 2.2 划分的四个层面，把现有工作归为四类。下表汇总各工作针对的问题与核心贡献。

### 类别 1：Trajectory / Application-aware 调度 —— Agent 是「程序」不是「请求」

针对问题 1–5（队头阻塞、长度方差、DAG、关键路径、多 SLO）。核心思路：把调度单位从「单次请求」升级为「一条完整 trajectory / workflow」，从而允许短轨迹插队、为高 SLO 轨迹预留时间片、tool 等待期主动换出 KV。

| 论文 | 视角 | 核心贡献 |
| --- | --- | --- |
| **Parrot** [1] | DAG-aware 调度 | 用 Semantic Variable 把多请求还原成数据流 DAG，跨请求共享 prefix，优化目标是整条任务端到端延迟而非单请求 TTFT |
| **Autellix** [2] | Trajectory 调度 | 把 Agent 视作「通用程序」，感知多轮依赖与 thinking-tool 切换，以 trajectory 为单位做 SRTF 类调度，实测 4–15× 优于 vLLM |
| **Conveyor** [3] | tool 中断调度 | 检测到输出 tool call 时主动决定是否换出 KV、是否调度其他请求，不再被动等 tool 返回 |
| **Tempo** [4] | 应用感知调度 | 同集群混跑聊天/Agent/推理流时，按各自 SLO 做 Goodput 优化，避免互相干扰 |
| **Concur** [5] | 多 Agent 准入控制 | 入口判断 Agent 能否在 SLO 内完成，提前拒掉无效 Agent，避免占住 KV/算力 |
| **Hive** [6] | 多 Agent 横向扩展 | 算法层（Agent 数量）+ 任务层（每 Agent 请求）双层弹性扩缩 |
| **MoA serving** [7] | MoA 架构 | 针对 Mixture-of-Agents 的 fan-out 拓扑做 tree-routing + 依赖感知的 prefill-decode overlap |

> 演进路线：FCFS / 简单优先级 → 长度感知 → 应用感知 → trajectory 感知 → 多 Agent 准入。

### 类别 2：Tool Call 阻塞 —— GPU 不能空等

针对问题 6–7（显存/算力解耦、重新 prefill vs 死等）。两条互补思路：**抢时间**（让 tool 与 LLM 在时间轴上重叠）、**省资源**（tool 等待期把 KV 换出 GPU）。

| 论文 | 思路 | 重叠方向 |
| --- | --- | --- |
| **LLMCompiler** [8] | Planner 一次产出 N 个 tool call 的 DAG，无依赖的 tool 直接并发 | tool ↔ tool 并行 |
| **Conveyor** [3] | tool 参数 JSON 还没出完整，前缀够用即让 tool 做 partial execution，边输出边跑 | decode ↔ tool 重叠 |
| **APIServe / InferCept** [9] | tool 等待期把 KV 从 GPU 换出到 CPU/SSD，显存让给在 decode 的请求，返回时换回 | KV ↔ GPU 解耦 |
| **Parallel function calling** [10] | 协议层支持一次输出多个 tool call，客户端 SDK 并发执行（工业大规模落地） | tool ↔ tool 并行 |

> **与输出长度预测调度的接口**：tool 边界是长度预测的天然子问题 —— 预测「何时输出 tool 参数」「tool 多久回来」可指导 prefill/decode 调度与 KV 换入换出时机（换出开销不能比 tool 等待还长）。

### 类别 3：KV Cache × Agent 生命周期 —— 跨轮、跨 Agent、跨时间管理

针对问题 8–11（跨轮复用、跨 Agent 共享、TTL、非前缀复用）。核心思路：把 KV 从「缓存」提升为「调度变量」，带 TTL、共享关系、复用模式。

| 论文 | 视角 | 关键贡献 |
| --- | --- | --- |
| **RadixAttention (SGLang)** [11] | 前缀共享 | 用 Radix Tree 索引活跃请求 prompt，自动找最长公共前缀复用 KV；vLLM `--enable-prefix-caching` 思想来源 |
| **ChunkAttention** [12] | 前缀感知 kernel | attention kernel 层识别共享前缀 token，高共享率场景下 attention 提速 3.2–4.8× |
| **CacheBlend** [13] | 突破「只能前缀复用」 | RAG 中独立 chunk 非前缀关系也复用 KV，拼接后只对受影响关键 token 选择性 recompute |
| **Tokencake** [14] | 多 Agent 中心化 KV serving | KV 抽象为独立服务，识别 Agent 间共享段，按「还会被几次/多久后访问」分层管理而非 LRU |
| **Continuum** [15] | KV TTL × 调度 | 估计 KV TTL = 用户思考间隔 + 下一轮预计输出长度，据此排序，把「快被命中」的 KV 优先留 GPU |

> **与输出长度预测调度最近**：Continuum 的 TTL 估计本质是「输出长度 + 用户行为间隔」的组合预测，**输出长度预测就是其核心子组件**。在 vLLM 上把 LRU 换成「按预测剩余命中时间排序」的 KV 驱逐，Goodput 收益最直观。

### 类别 4：语义暴露 —— 让后端「看懂」Agent 的依赖与共享

针对问题 12（API 层信息丢失、盲调度）。核心思路：在前端/中间层把 Agent 的语义结构（DAG / Semantic Variable / primitive 节点）显式暴露给后端。

| 论文 | 暴露什么 | 后端利用什么 |
| --- | --- | --- |
| **SGLang** [11] | DSL：前端代码里写出共享 prefix / 分支 / 循环 | Radix Tree 自动 KV 复用，共享 prefix 只 prefill 一次 |
| **Parrot** [1] | Semantic Variable：标记输入/输出变量形成数据流 | 还原 DAG，做 DAG-aware batching、跨请求 prefix 共享、按整条任务延迟优化 |
| **Teola** [16] | 把 RAG/Agent 拆到 primitive 粒度（embedding/retrieval/prefill/decode） | 算子粒度并行 / 流水（retrieval 还在跑时下一段 prefill 已备 KV） |
| **ALTO** [17] | stage 间用 token 流式触发 | 跨 stage 流水，不必等上一 stage 完全结束 |
| **LLMCompiler** [8] | Planner 把任务编译成函数调用 DAG | 独立 tool call 自动并发，无依赖 LLM call 自动并行 |

> **当前局限**：DAG 暴露了，但**调度策略本身仍是经验规则**（谁先跑、谁抢占、KV 谁留）。这正是基于长度预测的调度可切入处 —— 把基于长度预测的 SRPT 调度嵌进 Parrot 的 DAG 调度器，让节点执行顺序由「剩余工作量预测」驱动。

## 2.5 小结

现有工作分别从调度、tool 阻塞、KV 生命周期、语义暴露四个层面缓解了 Agent 场景的问题，但呈现两个明显缺口：(1) **预测驱动不足** —— 多数调度仍依赖经验规则或事后观测，缺少对 trajectory 剩余长度 / tool 返回时机 / KV 命中时间的**前瞻性预测**；(2) **跨层割裂** —— 语义暴露（类别 4）给出了 DAG，但调度（类别 1）、KV 管理（类别 3）尚未充分消费这些信息。**基于输出长度预测的调度**恰好是贯穿这四类的共同子组件，是后续工作可统一切入的方向。

---

### 参考文献

[1] Parrot: Efficient Serving of LLM-based Applications with Semantic Variable. OSDI 2024.
[2] Autellix: An Efficient Serving Engine for LLM Agents as General Programs. 2025.
[3] Conveyor: Efficient Tool-aware LLM Serving with Tool Partial Execution. 2024.
[4] Tempo: Application-aware LLM Serving with SLO-driven Goodput Optimization. 2025.
[5] Concur: Admission Control for Multi-Agent LLM Serving. 2025.
[6] Hive: Elastic Scaling for Multi-Agent LLM Systems. 2025.
[7] Mixture-of-Agents Serving with Dependency-aware Scheduling. 2025.
[8] LLMCompiler: An LLM Compiler for Parallel Function Calling. ICML 2024.
[9] InferCept / APIServe: Efficient Intercept Support for Augmented LLM Inference. ICML 2024.
[10] OpenAI / Anthropic Parallel Function Calling (industry).
[11] SGLang: Efficient Execution of Structured Language Model Programs (RadixAttention). NeurIPS 2024.
[12] ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache. ACL 2024.
[13] CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. EuroSys 2025.
[14] Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications. 2025.
[15] Continuum: KV Cache TTL-aware Scheduling for LLM Serving. 2025.
[16] Teola: Towards End-to-End Optimization of LLM-based Applications. ASPLOS 2025.
[17] ALTO: An Efficient Network Orchestrator for Compound AI Systems. EuroMLSys 2024.

> 注：文献的会议/年份为综述整理时的标注，正式提交前请核对各条目的确切出处与作者信息。
