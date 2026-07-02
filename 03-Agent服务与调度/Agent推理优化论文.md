你这个方向可以搜，但要先提醒一个现实点：**“25 年之后 + 顶会 + 引用量高”这三个条件同时满足会比较苛刻**。因为 2025/2026 的论文刚发表不久，引用量天然还没完全积累起来；所以更合理的筛选方式是：

> **优先选 2025 及以后顶会 / 顶刊 / 顶级系统会议论文，同时把“引用量高”放宽为：已有较高关注度、被相关工作频繁提及、代码或系统影响力较大。**

下面我按 **Agent 场景下的推理优化** 给你整理一个论文清单，重点覆盖：**workflow-aware serving、agent scheduling、KV cache 管理、tool-aware inference、program-level inference abstraction**。

---

## 1. 推荐重点关注的论文清单

下面这些是最贴近你要找的“Agent 场景下推理优化”的工作。表里我把它们按相关性和可信度做了分层。

| **论文 / 系统**                                              |   **年份 / 会议状态** | **核心方向**                         | **为什么和 Agent 推理优化相关**                              |
| ------------------------------------------------------------ | --------------------: | ------------------------------------ | ------------------------------------------------------------ |
| **KVFlow: Efficient Prefix Caching for Accelerating LLM Agentic Workflows** |          NeurIPS 2025 | Agent workflow KV cache              | 面向 agentic workflows 的 prefix caching，优化多步骤 Agent 中重复 prompt / shared prefix 带来的 KV 复用问题 |
| **Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live** |      2025，OpenReview | Multi-turn Agent scheduling + KV TTL | 针对多轮 Agent workload，提出 KV cache TTL 机制，减少跨轮调用中的显存浪费和重复计算 |
| **Serve Programs, Not Prompts**                              | 2025，ACM Proceedings | Program-level inference abstraction  | 将推理请求从 prompt 提升到 program 级，使系统能显式管理 token prediction 和 KV cache |
| **Efficient LLM Serving for Agentic Workflows / Helium**     |           2026，arXiv | Workflow-aware Agent serving         | 从数据系统角度重构 Agent serving，强调 workflow-aware 调度，而不是把每个 LLM call 当独立请求 |
| **SAGA: Workflow-Atomic Scheduling for AI Agent Inference**  |      2025，OpenReview | Workflow-atomic scheduling           | 将 Agent 的一串 LLM calls 作为整体调度对象，减少把 workflow 拆散导致的缓存丢失和端到端延迟问题 |
| **Semantic Scheduling for LLM Inference**                    |      2025，OpenReview | Semantic-aware scheduling            | 通过语义信息辅助 LLM inference scheduling，和 Agent 场景中的任务结构暴露有关 |

这些论文里，**KVFlow、Continuum、Helium / Efficient LLM Serving for Agentic Workflows、SAGA** 最符合你要的 Agent 推理优化主题。KVFlow 已出现在 NeurIPS 2025 页面，Helium 是 2026 年 agentic workflow serving 方向的代表性新工作；Continuum 和 SAGA 则更偏 Agent 多轮调度与 workflow 调度。[2], [1], [2], [3]

---

## 2. 按研究问题分类整理

Agent 推理优化和普通 LLM serving 最大的区别是：**Agent 不是一次请求就结束，而是一个多步骤 workflow**。它可能经历：

- LLM planning；
- tool calling；
- tool waiting；
- observation parsing；
- 多轮 reasoning；
- 多次 LLM inference；
- 多次 KV cache 复用或失效。

所以相关论文大概可以分成四类。

---

### 2.1 Workflow-aware Serving：把 Agent 当 workflow 调度

传统 LLM serving 通常把每个请求看成独立 request。  
但 Agent 场景里，一个用户任务可能包含几十次甚至上百次 LLM call。如果系统仍然按单个 request 调度，就会出现：

- 端到端 latency 不可控；
- workflow 中间状态丢失；
- KV cache 复用机会被浪费；
- tool 等待期间 GPU 空转；
- 多个 Agent workflow 之间互相阻塞。

这类工作试图把调度对象从：

> **single LLM request**

提升到：

> **agent workflow / trajectory / program**

代表论文：

| **论文**                                                    | **核心思想**                                                 | **适合你关注的点**                             |
| ----------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| **Efficient LLM Serving for Agentic Workflows / Helium**    | 从数据系统视角重新设计 Agent serving，显式建模 workflow 结构 | Agent serving 架构、workflow-aware scheduling  |
| **SAGA: Workflow-Atomic Scheduling for AI Agent Inference** | 将 Agent workflow 作为 atomic scheduling unit                | 避免单个 LLM call 级调度破坏整体优化           |
| **Serve Programs, Not Prompts**                             | 用 program 抽象替代 prompt 抽象，暴露推理过程结构            | program-level inference、KV 管理、系统接口设计 |

这类论文适合放在你 PPT 的 **“From request-level serving to workflow-aware serving”** 这一页。[1], [3], [4]

---

### 2.2 Agent KV Cache Management：多轮 Agent 的 KV 复用和驱逐

Agent workload 里，KV cache 管理尤其重要。

原因是 Agent 会反复携带：

- system prompt；
- tool instruction；
- conversation history；
- planning context；
- intermediate observation；
- shared prefix；
- retrieved documents。

如果每一步都重新 prefill，成本和延迟会很高。  
但如果全部缓存，又会造成显存爆炸。

所以这类论文关心：

- 哪些 KV cache 应该保留；
- 哪些可以驱逐；
- 多轮 Agent 任务中 KV 的生命周期；
- prefix cache 如何跨步骤复用；
- cache eviction 是否应该结合 workflow dependency。

代表论文：

| **论文**                        | **核心机制**                   | **与 Agent 的关系**                             |
| ------------------------------- | ------------------------------ | ----------------------------------------------- |
| **KVFlow**                      | Workflow-aware prefix caching  | 利用 Agent workflow 中的共享 prefix 加速推理    |
| **Continuum**                   | KV cache Time-to-Live          | 根据多轮 Agent 的未来使用可能性管理 KV 生命周期 |
| **Serve Programs, Not Prompts** | Program-level KV cache control | 允许用户 / runtime 更细粒度控制 KV cache        |

这类工作非常适合你关注，因为它们直接对应 Agent 场景下最核心的推理优化问题：**如何减少重复 prefill 和显存浪费**。[2], [2], [4]

---

### 2.3 Scheduling for Agent Inference：调度多轮、多阶段、多依赖任务

Agent 推理调度比普通 LLM 调度复杂很多。普通请求通常只需要考虑：

- prefill；
- decode；
- batching；
- GPU utilization；
- latency / throughput trade-off。

但 Agent 还要考虑：

- 后续步骤是否依赖当前输出；
- tool call 什么时候返回；
- 当前 workflow 是否接近完成；
- 某个 Agent 是否占用了大量 KV cache；
- 是否应该优先调度短 trajectory；
- 是否应该保留即将复用的 KV。

代表论文：

| **论文**                                                     | **关注问题**                        | **关键词**                           |
| ------------------------------------------------------------ | ----------------------------------- | ------------------------------------ |
| **Continuum**                                                | 多轮 Agent job completion time 优化 | multi-turn scheduling, KV TTL        |
| **SAGA**                                                     | workflow-atomic scheduling          | agent inference, workflow scheduling |
| **Online Scheduling for LLM Inference with KV Cache Constraints** | KV cache 约束下的在线调度理论       | scheduling theory, KV capacity       |
| **Semantic Scheduling for LLM Inference**                    | 利用语义信息改善调度                | semantic-aware scheduling            |

这类论文可以帮助你总结 Agent serving 的一个关键转变：

> **调度目标不再只是单次请求 latency，而是整个 Agent trajectory 的 job completion time。** [2], [3], [1], [2]

---

### 2.4 Tool-aware / Program-aware Inference：处理 tool call 带来的空转

Agent 很多时候不是一直在 decode，而是：

1. LLM 生成 tool call；
2. 等工具执行；
3. 工具返回 observation；
4. LLM 继续推理；
5. 再调用工具。

这会造成一个系统问题：

> **GPU 可能在等待 tool 返回时空转，KV cache 却还占着显存。**

因此，Agent serving 系统需要考虑：

- tool call 与 decode 是否能 overlap；
- tool waiting 期间是否 offload KV；
- tool 返回后如何快速恢复上下文；
- workflow 是否可以并行执行多个 tool；
- program / DAG 信息能否暴露给 serving backend。

代表方向：

| **方向**            | **优化目标**                      | **相关论文 / 系统**                                  |
| ------------------- | --------------------------------- | ---------------------------------------------------- |
| Tool-call overlap   | 减少 tool waiting 造成的 GPU idle | Helium / Efficient LLM Serving for Agentic Workflows |
| Program abstraction | 暴露推理结构，方便系统优化        | Serve Programs, Not Prompts                          |
| Workflow scheduling | 根据 Agent DAG 做端到端优化       | SAGA, KVFlow                                         |

这部分适合你在综述里作为 **Agent-specific inference challenges** 来写。[1], [4], [3]

---

## 3. 最推荐你优先精读的 5 篇

如果你时间有限，我建议按这个顺序读。

---

### 1. KVFlow: Efficient Prefix Caching for Accelerating LLM Agentic Workflows

**推荐指数：★★★★★**

这是最贴合 Agent 推理优化的论文之一。  
它关注的是 Agent workflow 中大量重复 prefix 和共享上下文带来的缓存优化机会。

你可以重点看：

- 它如何抽象 agentic workflow；
- prefix caching 怎么和 workflow 结构结合；
- 相比普通 prefix cache，它多考虑了什么；
- 它的 evaluation workload 是不是 Agent benchmark；
- 它如何衡量 latency / throughput / cache hit rate。

适合放在你综述的 **KV cache reuse in agentic workflows** 部分。[2]

---

### 2. Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live

**推荐指数：★★★★★**

Continuum 的核心点是 **multi-turn Agent scheduling + KV cache TTL**。

它适合解释 Agent 场景下的一个关键问题：

> 当前这一轮 LLM call 结束后，KV cache 要不要保留？保留多久？什么时候驱逐？

这和普通 LLM serving 不一样。普通请求结束后，KV 通常就可以释放；但 Agent 多轮交互中，后面可能还会继续用到。

你可以重点看：

- TTL 如何决定；
- 是否预测未来复用；
- 如何优化 job completion time；
- 如何处理 KV memory pressure；
- 对多轮 Agent workload 的收益。

适合放在 **Agent KV lifecycle management** 部分。[2]

---

### 3. Efficient LLM Serving for Agentic Workflows / Helium

**推荐指数：★★★★☆**

这篇是 2026 年较新的 Agent serving 工作。  
它的问题意识很清楚：传统 LLM serving 系统没有为 Agent workflow 设计。

它的重点不是单个 kernel 优化，而是：

- workflow-aware serving；
- agent task graph；
- data-system style execution；
- scheduling across LLM calls and tool calls；
- end-to-end latency optimization。

如果你想写“Agent 推理优化为什么和普通 LLM serving 不同”，这篇很适合作为背景和系统架构参考。[1]

---

### 4. Serve Programs, Not Prompts

**推荐指数：★★★★☆**

这篇的价值在于它提出了一个更高层的系统抽象：

> 不要只 serve prompt，而要 serve program。

这对 Agent 很关键。因为 Agent 本质上不是一个 prompt，而是一个带控制流、工具调用、状态管理和缓存复用的程序。

你可以重点看：

- LLM Inference Programs / LIPs；
- runtime 如何管理 token prediction；
- KV cache 如何被程序级控制；
- 它和传统 prompt-level serving 的区别；
- 对 Agent serving backend 的启发。

适合放在 **semantic / program abstraction for inference optimization** 部分。[4]

---

### 5. SAGA: Workflow-Atomic Scheduling for AI Agent Inference

**推荐指数：★★★★☆**

SAGA 的核心思想是：Agent inference 不应该被拆成互不相关的 LLM calls 来调度。

它强调：

- AI agents execute chained LLM calls；
- GPU scheduler 如果把每个 call 当独立请求，会丢失 workflow 信息；
- workflow-atomic scheduling 可以减少跨步骤开销；
- 对端到端任务完成时间更友好。

这篇适合和 Helium 一起看，用来理解 **workflow-level scheduling** 这一脉络。[3]

---

## 4. 你可以用的论文分类框架

如果你要写综述或 PPT，可以直接用下面这个结构。

| **类别**                     | **核心问题**                                 | **代表论文**                                                 |
| ---------------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| **Workflow-aware Serving**   | 如何把 Agent trajectory 作为整体调度对象     | Helium, SAGA                                                 |
| **KV Cache Reuse**           | 如何复用多轮 Agent 中重复的 prefix / context | KVFlow, Continuum                                            |
| **KV Lifecycle Management**  | Agent 中间状态和 KV cache 保留多久           | Continuum                                                    |
| **Program-level Inference**  | 如何暴露 Agent 的控制流、工具调用和缓存需求  | Serve Programs, Not Prompts                                  |
| **Semantic Scheduling**      | 如何利用任务语义和 workflow 信息优化调度     | Semantic Scheduling, SAGA                                    |
| **Online Scheduling Theory** | 在 KV 容量约束下如何做在线调度               | Online Scheduling for LLM Inference with KV Cache Constraints |

这个分类能很好地串起 Agent 推理优化的核心逻辑：  
**Agent 的推理优化不是单点优化，而是 workflow、KV cache、tool call、scheduler 和 runtime abstraction 的联合优化。**

---

## 5. 推荐搜索关键词

你后面继续搜论文时，可以用这些关键词组合。

### Agent serving 方向

```text
agentic workflow serving LLM inference optimization
LLM agent serving workflow-aware scheduling
AI agent inference scheduling KV cache
multi-turn LLM agent serving system
```

### KV cache 方向

```text
agentic workflow KV cache
multi-turn agent KV cache TTL
prefix caching LLM agents
KV cache management for LLM agent serving
```

### Tool / workflow 方向

```text
tool-aware LLM serving
LLM agent tool call scheduling
workflow-aware inference serving
DAG scheduling for LLM agents
```

### 顶会限定

```text
site:neurips.cc LLM agent serving KV cache 2025
site:openreview.net LLM agent inference scheduling 2025
site:dl.acm.org LLM inference program KV cache 2025
site:usenix.org LLM agent serving 2025
site:acm.org LLM serving agent workflow 2025
```

---

## 6. 一段可以放进综述的总结

你可以这样概括这个领域：

> Recent work on agentic LLM inference optimization moves beyond request-level serving and focuses on workflow-aware execution. Unlike single-turn LLM queries, agent workloads consist of chained LLM calls, tool invocations, and multi-turn state dependencies. This introduces new challenges in end-to-end latency optimization, KV cache lifecycle management, tool-call waiting, and workflow-level scheduling. Representative systems such as KVFlow, Continuum, Helium, SAGA, and Serve Programs, Not Prompts explore different points in this design space, including prefix cache reuse, TTL-based KV retention, workflow-atomic scheduling, and program-level inference abstractions.

中文版本：

> 近期 Agent 场景下的 LLM 推理优化正在从传统 request-level serving 转向 workflow-aware execution。与单轮 LLM 请求不同，Agent workload 通常由多次 LLM 调用、工具调用和跨轮状态依赖组成，因此带来了端到端延迟优化、KV cache 生命周期管理、工具等待、以及 workflow 级调度等新问题。KVFlow、Continuum、Helium、SAGA 和 Serve Programs, Not Prompts 等工作分别从 prefix cache 复用、TTL-based KV 保留、workflow-atomic scheduling 和 program-level inference abstraction 等角度探索了这一设计空间。

---

## 7. 我的筛选建议

如果你后面要做正式 literature review，我建议按这个优先级筛：

1. **必须读**
   - KVFlow
   - Continuum
   - Helium / Efficient LLM Serving for Agentic Workflows
   - Serve Programs, Not Prompts

2. **强相关补充**
   - SAGA
   - Semantic Scheduling for LLM Inference
   - Online Scheduling for LLM Inference with KV Cache Constraints

3. **背景类论文**
   - SGLang / RadixAttention
   - vLLM / PagedAttention
   - LLMCompiler
   - CacheBlend
   - Parrot

其中第 3 类有些不是 2025 之后，但它们是理解 2025+ Agent serving 工作的基础。正式写论文综述时，可以作为 background / prior work 引入。

# 调度策略相关论文

如果你想**重点考虑“调度相关”问题**，我建议不要只盯 Agent 论文，而是按下面这条线读：

> **先读 LLM serving 调度基础论文 → 再读 Prefill/Decode 分离与 chunked prefill → 再读 KV cache 约束调度 → 最后读 Agent workflow-aware scheduling。**

因为 Agent 调度其实是在普通 LLM serving 调度上叠加了：

- 多轮调用；
- 工具调用等待；
- workflow 依赖；
- KV cache 生命周期；
- 端到端 Job Completion Time；
- 跨步骤 cache 复用。

下面是我建议你**最重点读的论文清单**。

---

# 一、最优先读：LLM Serving 调度基础

这几篇是你理解后续 Agent 调度的基础。

---

## 1. Orca: A Distributed Serving System for Transformer-Based Generative Models

**方向：continuous batching / iteration-level scheduling**

这是 LLM serving 调度里的经典工作之一，提出了非常重要的思想：

> 不要按请求级 batch，而要按 decode iteration 动态 batch。

传统 batching 是一批请求一起进、一起出，但 LLM 生成长度不同，请求会陆续结束。如果等整批都结束，会浪费 GPU。Orca 使用 **iteration-level scheduling**，每一轮 decode 都重新组织 batch。

你要重点看：

- 为什么 static batching 不适合自回归生成；
- iteration-level scheduling 是什么；
- continuous batching 如何提升 GPU 利用率；
- 它怎么处理不同长度输出的请求；
- 它和后来的 vLLM、SGLang、Sarathi 的关系。

面试/综述里可以这样总结：

> Orca 是 LLM serving 调度的重要基础，它将调度粒度从 request-level 降到 iteration-level，使新请求可以动态加入，完成请求可以及时退出，从而提升在线推理吞吐。

---

## 2. vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention

**方向：KV cache 管理 + continuous batching**

vLLM 不只是 KV cache 论文，它也对 serving 调度很重要。它的核心是 **PagedAttention**，解决 KV cache 显存碎片和利用率问题。

为什么它和调度相关？

因为调度器能接收多少请求、能 batch 多大，很大程度取决于 KV cache 是否足够。

你要重点看：

- PagedAttention 如何管理 KV cache；
- block-based KV allocation；
- 为什么传统 KV cache 分配浪费显存；
- KV cache 容量如何影响 batch size 和并发数；
- vLLM 的 continuous batching 如何和 PagedAttention 配合。

你可以这样理解：

> Orca 解决的是“什么时候调度谁”，vLLM 进一步解决了“有没有足够 KV cache 支撑这些请求一起调度”。

---

# 二、强烈建议读：Prefill / Decode 调度与 Chunked Prefill

如果你的方向写了“推理调度”，这部分非常关键。

---

## 3. Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

**方向：chunked prefill / prefill-decode 混合调度**

这篇非常适合你现在准备的 Prefill / Decode 调度问题。

它的核心观点是：

> Prefill 和 decode 的资源特征不同，可以通过 chunked prefill 把长 prefill 切小，然后和 decode 混合执行。

你要重点看：

- 为什么 prefill 会阻塞 decode；
- chunked prefill 怎么切；
- decode 如何 piggyback 在 prefill batch 里；
- 它如何平衡 TTFT、TPOT 和 throughput；
- chunk size 对吞吐和尾延迟的影响。

这篇跟你之前准备的面试回答高度相关。

你可以重点记这句话：

> Sarathi-Serve 通过将长 prefill 拆成 chunk，并把 decode 请求混入 prefill 批次中，减少 prefill 对 decode 的阻塞，改善在线 LLM serving 的端到端效率。

---

## 4. Splitwise: Efficient Generative LLM Inference Using Phase Splitting

**方向：prefill/decode disaggregation**

这篇也是调度相关重点论文。

它的核心思想是：

> Prefill 和 decode 的资源需求不同，可以把它们放到不同硬件/不同 worker 上执行。

Prefill 更偏 compute-intensive，decode 更偏 memory-bandwidth-intensive。因此 Splitwise 主张把二者解耦，让系统根据阶段特点调度资源。

你要重点看：

- prefill 和 decode 为什么适合分离；
- 分离后如何传递 KV cache；
- phase splitting 带来的调度收益；
- 分离后的通信开销；
- 什么场景下 disaggregation 更有优势。

适合放在综述里的这一类：

> phase-aware scheduling / prefill-decode disaggregation。

---

## 5. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

**方向：prefill/decode 分离 + goodput 优化**

DistServe 和 Splitwise 方向接近，但更强调 **goodput**，也就是在满足 SLO/SLA 延迟约束下的有效吞吐。

你要重点看：

- 为什么单纯 tokens/s 不是最好的 serving 指标；
- goodput 是怎么定义的；
- prefill 和 decode 分离如何改善 SLO；
- 它怎么建模 TTFT 和 TPOT；
- 它如何决定 prefill/decode 资源分配。

如果你研究调度，这篇很值得读，因为它把调度目标从简单吞吐推进到了：

> **在延迟 SLO 约束下最大化有效吞吐。**

---

# 三、KV Cache 约束下的调度

这部分和 Agent 特别相关，因为 Agent 会长期占用 KV cache。

---

## 6. Online Scheduling for LLM Inference with KV Cache Constraints

**方向：KV cache 容量约束下的在线调度理论**

这类论文重点研究：

> 当 KV cache 是稀缺资源时，应该如何在线调度请求？

普通调度问题只考虑 GPU compute，但 LLM serving 还要考虑 KV cache 显存。一个请求一旦进入 decode，就会持续占用 KV cache，直到结束。

你要重点看：

- KV cache constraint 如何改变调度问题；
- 为什么不能只按 shortest-job-first 或 FIFO；
- admission control 怎么做；
- preemption 是否值得；
- 理论模型如何抽象 LLM serving；
- 目标是平均完成时间、deadline 还是 throughput。

这篇适合作为你研究 Agent 调度的理论背景。

---

## 7. Llumnix: Dynamic Scheduling for Large Language Model Serving

**方向：动态调度 / migration / KV cache 管理**

Llumnix 关注分布式 LLM serving 中的动态调度问题，包括请求迁移、负载均衡和 KV cache 管理。

你要重点看：

- 为什么 LLM serving 的负载会动态变化；
- 不同 worker 之间如何做负载均衡；
- 迁移请求时 KV cache 怎么办；
- 调度器如何根据当前状态动态决策；
- 对 tail latency 的影响。

如果你后面考虑多 GPU / 多节点场景，这篇很有价值。

---

# 四、Agent 场景下的调度重点论文

如果你最终想做 **Agent 场景下的推理调度**，下面这几篇应该重点读。

---

## 8. Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live

**方向：multi-turn Agent scheduling + KV Cache TTL**

这是 Agent 调度里非常相关的一篇。

Agent 和普通单轮请求不同：一次用户任务可能有多轮 LLM call。中间可能调用工具，等工具返回后还会继续用之前的上下文。

问题是：

> 某一轮 LLM call 结束后，KV cache 要不要保留？保留多久？

如果立刻释放，后面可能要重新 prefill，浪费计算。  
如果一直保留，会占用大量显存，影响其他请求。

Continuum 的核心是 **KV cache Time-to-Live**。

你要重点看：

- Agent multi-turn workload 和普通 LLM workload 的区别；
- KV cache TTL 如何设置；
- 如何判断某个 KV 未来是否还会被用到；
- TTL 对 job completion time 的影响；
- 显存压力大时如何驱逐 KV；
- 调度器如何结合 TTL 做决策。

这篇非常适合你研究：

> **Agent 场景下 KV-aware scheduling。**

---

## 9. SAGA: Workflow-Atomic Scheduling for AI Agent Inference

**方向：workflow-level scheduling / workflow-atomic scheduling**

SAGA 的核心问题是：

> Agent 是一串有依赖关系的 LLM calls，不应该把每个 call 当成独立请求调度。

如果调度器只看到一个个独立 prompt，就会丢失 workflow 信息，导致：

- 中间 KV cache 被错误驱逐；
- workflow 被拆散；
- 端到端完成时间变差；
- 工具调用等待和 LLM 调用无法协同；
- 短期局部最优导致整体任务变慢。

SAGA 强调 **workflow-atomic scheduling**，也就是把 Agent workflow 作为整体调度对象。

你要重点看：

- workflow-atomic 是什么意思；
- 为什么 request-level scheduling 不适合 Agent；
- 它优化的是单次 LLM call latency，还是整个 workflow completion time；
- 如何处理多个 Agent workflow 的竞争；
- 如何避免长 workflow 饿死短 workflow，或者短 workflow 饿死长 workflow。

这篇非常适合放在你综述的：

> **From request-level scheduling to workflow-level scheduling**

这一节。

---

## 10. Efficient LLM Serving for Agentic Workflows / Helium

**方向：workflow-aware serving / tool-aware scheduling**

Helium 这类工作更偏系统架构，它强调：

> Agent serving 系统应该显式理解 workflow，而不是把每次 LLM call 当成普通请求。

Agent workflow 中除了 LLM 推理，还有：

- tool call；
- tool waiting；
- observation；
- planning；
- branching；
- retry；
- multi-step execution。

所以调度器不能只看 GPU 上的 prefill/decode，还要看整个 workflow 的执行图。

你要重点看：

- 它如何建模 agentic workflow；
- 工具调用等待时如何处理 KV cache；
- 是否支持 tool execution 和 LLM inference overlap；
- 如何减少 GPU idle；
- 如何优化 end-to-end latency；
- workflow-aware 调度相比普通调度的优势。

这篇适合用来拓展视野：Agent 调度不是单纯 GPU 调度，而是 **LLM + tool + memory + workflow runtime** 的联合调度。

---

## 11. KVFlow: Efficient Prefix Caching for Accelerating LLM Agentic Workflows

**方向：Agent workflow prefix caching / cache-aware scheduling**

KVFlow 更偏 KV cache 复用，但它和调度强相关。

Agent workflow 经常有大量重复上下文，比如：

- system prompt；
- tool descriptions；
- instruction；
- history prefix；
- shared reasoning template；
- retrieved documents。

如果调度器知道这些 prefix 可以复用，就可以减少重复 prefill。

你要重点看：

- Agent workflow 里有哪些 prefix 复用机会；
- 它怎么识别 shared prefix；
- prefix cache 如何和 workflow 结构结合；
- cache hit 如何影响调度；
- 是否应该优先调度 cache hit 的请求；
- 显存不足时如何做 cache eviction。

你可以把它归类到：

> **cache-aware agent scheduling。**

---

# 五、Program / Semantic 层调度

这部分更前沿，适合你想做创新点时读。

---

## 12. Serve Programs, Not Prompts

**方向：program-level inference abstraction**

这篇的核心观点是：

> LLM serving 不应该只接收 prompt，而应该接收 program。

为什么这和调度有关？

因为 prompt 只是一段文本，系统不知道：

- 哪些部分会复用；
- 哪些 token 是固定 prefix；
- 后面是否会调用工具；
- 哪些分支可能执行；
- 哪些 KV cache 未来还会用；
- 哪些步骤可以并行。

如果用 program abstraction，系统可以看到更完整的执行结构，从而做更好的调度和 cache 管理。

你要重点看：

- program abstraction 暴露了哪些信息；
- 它如何管理 token prediction；
- 它如何控制 KV cache；
- 和 Agent runtime 的关系；
- 对 workflow-aware scheduling 有什么启发。

这篇适合放在你的研究动机里：

> 要优化 Agent serving，必须把更多语义和结构暴露给底层 serving system。

---

## 13. Semantic Scheduling for LLM Inference

**方向：semantic-aware scheduling**

传统调度器通常只看：

- prompt length；
- output length；
- KV cache size；
- arrival time；
- priority。

Semantic Scheduling 这类工作想进一步利用任务语义，比如：

- 请求是否相似；
- 是否共享 prefix；
- 是否属于同一类任务；
- 是否可能访问相同工具；
- 是否有相似上下文；
- 是否可以一起 batch 或复用 cache。

你要重点看：

- semantic 信息如何获得；
- semantic 信息如何进入调度决策；
- 它优化的是 throughput、latency 还是 cache hit rate；
- 是否有额外 overhead；
- 和 Agent workflow 的关系。

这类论文适合启发你的创新方向，但不一定是最基础必读。

---

# 六、我建议你的阅读优先级

如果你的目标是**面试/实习项目准备**，我建议按这个顺序读。

---

## 第一优先级：必须读

这几篇和调度最直接。

| 优先级 | 论文              | 你要掌握的关键词                                |
| ------ | ----------------- | ----------------------------------------------- |
| 1      | **Orca**          | continuous batching, iteration-level scheduling |
| 2      | **vLLM**          | PagedAttention, KV cache, batching              |
| 3      | **Sarathi-Serve** | chunked prefill, prefill-decode mixing          |
| 4      | **DistServe**     | prefill/decode disaggregation, goodput          |
| 5      | **Continuum**     | Agent multi-turn scheduling, KV TTL             |

---

## 第二优先级：Agent 调度重点

如果你确定做 Agent inference scheduling，读这些。

| 优先级 | 论文                                      | 你要掌握的关键词              |
| ------ | ----------------------------------------- | ----------------------------- |
| 6      | **SAGA**                                  | workflow-atomic scheduling    |
| 7      | **Helium**                                | agentic workflow serving      |
| 8      | **KVFlow**                                | workflow-aware prefix caching |
| 9      | **Serve Programs, Not Prompts**           | program-level inference       |
| 10     | **Semantic Scheduling for LLM Inference** | semantic-aware scheduling     |

---

## 第三优先级：扩展系统背景

这些适合你深入系统实现时读。

| 论文 / 系统                                    | 价值                                         |
| ---------------------------------------------- | -------------------------------------------- |
| **SGLang / RadixAttention**                    | prefix cache、structured generation、runtime |
| **LightLLM**                                   | 高性能 serving 实现细节                      |
| **TensorRT-LLM**                               | 工业级推理优化                               |
| **Ray Serve / SkyPilot 相关 LLM serving 工作** | 分布式 serving 和部署                        |
| **Llumnix**                                    | 多实例动态调度、迁移、负载均衡               |

---

# 七、如果只能读 5 篇，我建议读这 5 篇

如果你时间有限，我建议你优先读：

## 1. Orca

**解决问题：** LLM decode 如何动态 batching。  
**你学到：** iteration-level scheduling 是现代 LLM serving 调度基础。

---

## 2. vLLM

**解决问题：** KV cache 显存管理和高并发 serving。  
**你学到：** 为什么 KV cache 是调度器必须考虑的核心资源。

---

## 3. Sarathi-Serve

**解决问题：** 长 prefill 阻塞 decode。  
**你学到：** chunked prefill 和 prefill/decode 混合调度。

---

## 4. DistServe

**解决问题：** Prefill 和 decode 资源特征不同。  
**你学到：** phase disaggregation 和 goodput-oriented scheduling。

---

## 5. Continuum

**解决问题：** Agent 多轮调用中的 KV cache 生命周期和调度。  
**你学到：** Agent scheduling 不只是一次请求调度，而是多轮 workflow 调度。

---

# 八、阅读时重点关注的问题

你读每篇论文时，不要只看方法，要带着这些问题看。

---

## 1. 它的调度对象是什么？

是：

- request？
- token？
- decode iteration？
- prefill chunk？
- KV block？
- workflow？
- program？
- agent trajectory？

这个决定了论文的核心抽象。

---

## 2. 它优化的目标是什么？

可能是：

- throughput；
- latency；
- TTFT；
- TPOT；
- goodput；
- job completion time；
- P95/P99 tail latency；
- cache hit rate；
- GPU utilization；
- KV cache utilization。

不同论文目标不一样，不能混着比较。

---

## 3. 它考虑了哪些资源约束？

常见资源约束：

- GPU compute；
- GPU memory；
- KV cache capacity；
- memory bandwidth；
- network bandwidth；
- CPU offload；
- tool execution time；
- batch token budget；
- max number of sequences。

---

## 4. 它用了什么调度策略？

比如：

- FIFO；
- SJF；
- shortest remaining processing time；
- deadline-aware；
- decode-prioritized；
- prefill/decode separation；
- chunked prefill；
- cache-aware scheduling；
- workflow-aware scheduling；
- priority scheduling；
- aging；
- admission control；
- preemption；
- migration。

---

## 5. 它的核心 trade-off 是什么？

常见 trade-off：

- throughput vs latency；
- TTFT vs TPOT；
- prefill efficiency vs decode smoothness；
- cache hit rate vs memory pressure；
- fairness vs average latency；
- workflow locality vs global batching；
- GPU utilization vs tail latency。

---

# 九、适合你综述/PPT的主线

你可以把调度相关论文组织成这条线：

```text
1. Static batching 不适合 LLM
   ↓
2. Orca 提出 iteration-level scheduling / continuous batching
   ↓
3. vLLM 发现 KV cache 是核心瓶颈，用 PagedAttention 支撑高并发
   ↓
4. Sarathi-Serve 发现长 prefill 会阻塞 decode，用 chunked prefill 混合调度
   ↓
5. DistServe / Splitwise 进一步把 prefill 和 decode 解耦，做 phase-aware scheduling
   ↓
6. Agent workload 出现后，单个 request 不再是合理调度单位
   ↓
7. Continuum / SAGA / Helium / KVFlow 转向 workflow-aware、cache-aware、agent-aware scheduling
```

这条主线非常适合面试或组会讲。

---

# 十、你可以形成的研究问题

读完这些论文后，你可以往这些问题上想：

## 1. Agent 场景下，调度对象应该是什么？

传统是：

```text
request / token / batch
```

Agent 场景可能应该是：

```text
workflow / stage / tool-call boundary / KV state
```

---

## 2. Agent 的 KV cache 是否应该跨轮保留？

关键问题：

```text
保留：减少重复 prefill，但占显存
释放：节省显存，但未来可能重算
```

这就是 Continuum / KVFlow 关注的问题。

---

## 3. Tool call 等待期间应该怎么处理？

可能策略：

- 保留 KV；
- offload KV；
- 设置 TTL；
- 预测工具返回时间；
- 工具快返回就保留；
- 工具慢返回就换出；
- 根据 workflow priority 决定。

---

## 4. 应该优化单次 LLM call，还是整个 Agent job？

普通 serving 优化：

```text
single request latency
```

Agent serving 应该优化：

```text
end-to-end job completion time
```

这是 SAGA / Helium 的核心动机。

---

## 5. Cache-aware scheduling 能不能和 workflow-aware scheduling 结合？

例如：

- 优先调度即将复用 KV 的 workflow；
- 避免驱逐即将使用的 KV；
- 把共享 prefix 的 Agent calls 放在一起；
- 根据 cache hit probability 做 admission control；
- 结合 TTL 和 workflow DAG 做驱逐。

这个方向很有研究潜力。

---

# 十一、最终推荐清单

如果你要一个最干净的调度方向阅读清单，我建议是：

| 类别                      | 必读论文                                                     |
| ------------------------- | ------------------------------------------------------------ |
| LLM serving 调度基础      | **Orca**                                                     |
| KV cache + batching       | **vLLM**                                                     |
| Chunked prefill           | **Sarathi-Serve**                                            |
| Prefill/decode 分离       | **DistServe**, **Splitwise**                                 |
| 动态调度/迁移             | **Llumnix**                                                  |
| KV 约束调度理论           | **Online Scheduling for LLM Inference with KV Cache Constraints** |
| Agent 多轮调度            | **Continuum**                                                |
| Agent workflow 调度       | **SAGA**, **Helium**                                         |
| Agent cache-aware serving | **KVFlow**                                                   |
| Program-level 调度抽象    | **Serve Programs, Not Prompts**                              |

---

# 十二、一句话建议

如果你想做**推理调度方向**，最应该先读：

> **Orca → vLLM → Sarathi-Serve → DistServe/Splitwise → Continuum → SAGA/Helium/KVFlow**

这条线能帮你从普通 LLM serving 调度，顺利过渡到 Agent 场景下的 workflow-aware scheduling。