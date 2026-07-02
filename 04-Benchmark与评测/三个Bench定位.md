这三个 benchmark 可以理解成 **Agent 能力评测的三个不同切面**：  
**PinchBench 更像日常 mixed workload 压测**，看模型在短小、多类型请求中的成功率、速度和成本；**WildClawBench 更像 hard long-horizon benchmark**，专门考复杂、长链路、多工具/多模态任务；**LiveClawBench 则更偏通用数字助理场景**，覆盖邮件、文档、日程、研究、开发、购物等真实助手型 workflow。它们不是简单的“谁更难”，而是分别强调 **workload 分布、任务复杂度、真实运行环境和应用泛化能力**。

---

## 1. 三个 Bench 的定位对比

下面这张表可以直接放到你的材料里，用来快速说明三者差异。

| **Benchmark**     | **面向对象**                                       | **任务特点**                                                 | **主要评测目标**                                 | **更像哪类负载**                                |
| ----------------- | -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------ | ----------------------------------------------- |
| **PinchBench**    | OpenClaw coding agent / 日常 assistant-coding 请求 | 任务异质性高，但单任务通常不长；覆盖日历、股票研究、博客写作、天气脚本等 | 比较不同模型在日常请求中的**成功率、速度、成本** | **Mixed workload / 日常短任务集合**             |
| **WildClawBench** | 真实 OpenClaw runtime 下的 hard agent benchmark    | 复杂、长链路、多工具、多模态；更接近困难真实任务             | 测 Agent 是否能在真实环境中完成复杂任务          | **Long-horizon hard tasks / 极端复杂任务**      |
| **LiveClawBench** | 通用助手型 LLM Agent                               | 复杂、真实世界、多步骤助手任务；覆盖邮件、文档、日程、研究、开发、购物 | 评测通用数字助理的多场景执行能力                 | **General assistant workflow / 真实助手任务流** |

一句话区分：

- **PinchBench**：日常请求多而杂，重点是 **效率 + 成本 + 成功率**。  
- **WildClawBench**：任务少但难，重点是 **长链路鲁棒性 + 工具协作 + 真实环境执行**。  
- **LiveClawBench**：场景广且真实，重点是 **通用助手能力 + 多步骤任务完成度**。

---

## 2. PinchBench：更适合评估日常 Mixed Workload

PinchBench 的核心价值在于它不像传统 benchmark 只挑少量复杂任务，而是更接近日常使用中高频出现的 **assistant / coding 混合请求**。

### 主要特点

- **任务类型多样**：日历、股票研究、博客写作、天气脚本等。
- **单任务通常不长**：不像 deep research 或复杂 coding agent 那样跑很久。
- **异质性高**：不同任务对模型能力、工具调用、输出长度、上下文需求都不同。
- **评测维度实用**：成功率、速度、成本都和真实部署强相关。

### 对 Agent Serving 的意义

PinchBench 很适合拿来研究 **mixed workload 下的调度问题**：

- 短任务与中等任务混跑；
- 不同任务输出长度不同；
- 不同任务 tool 使用频率不同；
- 用户关心的不只是准确率，还包括 latency 和 cost。

### 它更适合回答的问题

- 哪个模型在日常 assistant/coding 请求中性价比最高？
- 在大量异构短任务混跑时，系统 goodput 如何？
- 小模型是否能在常见任务上替代大模型？
- 调度策略是否会让短任务被中等复杂任务拖慢？

### 局限

PinchBench 的任务通常不长，所以它对以下问题的压力不够强：

- 超长 trajectory；
- 多十轮 tool call；
- 复杂 DAG 依赖；
- 长时间 KV 生命周期；
- 极端 tail latency。

所以它更像 **daily workload benchmark**，不是专门的 **long-horizon stress test**。

---

## 3. WildClawBench：更适合评估复杂长链路 Agent 能力

WildClawBench 的关键词是 **hard benchmark** 和 **真实 OpenClaw runtime**。它主要考察 Agent 在真实环境中面对复杂任务时，能不能持续推进、正确调用工具、处理多模态信息，并最终完成目标。

### 主要特点

- **复杂任务**：不是一步问答，而是需要规划、执行、观察、修正。
- **长链路执行**：任务可能包含多轮 reasoning 和多次 tool call。
- **多工具**：需要调用不同工具，并正确处理工具返回。
- **多模态**：可能涉及网页、图片、文件、表格、代码等。
- **真实 runtime**：不是离线静态问答，而是在真实 agent 环境中执行。

### 对 Agent Serving 的意义

WildClawBench 更适合暴露 Agent Serving 的系统瓶颈：

- 长 trajectory 导致队头阻塞；
- tool call 阻塞导致 GPU 空转；
- KV cache 需要跨轮保留；
- 任务失败可能发生在任意中间步骤；
- 输出长度和执行时间高度不可预测；
- 关键路径节点会影响整个任务完成时间。

### 它更适合回答的问题

- Agent 能否完成真实复杂任务？
- 长链路执行中，模型是否会中途偏离目标？
- 工具调用是否稳定、准确、可恢复？
- 多模态信息是否能被正确整合？
- 系统是否能承受长时间、多轮、多工具运行？

### 局限

WildClawBench 因为任务更难、更长，通常会带来几个问题：

- 评测成本高；
- 单次运行时间长；
- 任务结果更难自动判定；
- 统计样本可能不如短任务 benchmark 大；
- 更偏 hard case，不一定代表日常平均负载。

所以它更像 **stress benchmark / hard benchmark**，适合看系统和模型的上限与鲁棒性。

---

## 4. LiveClawBench：更适合评估通用数字助理能力

LiveClawBench 更强调 **通用助手型 LLM Agent**。它不只关注 coding，也不只关注极端复杂任务，而是模拟一个数字助理在真实工作和生活场景中的多步骤任务执行。

### 主要特点

- **通用助手场景**：邮件、文档、日程、研究、开发、购物等。
- **真实世界任务**：任务不是抽象问答，而是模拟真实用户需求。
- **多步骤执行**：需要规划、检索、工具调用、整合结果。
- **覆盖面广**：更像评估 agent 是否能成为“数字助理”。

### 对 Agent Serving 的意义

LiveClawBench 可以作为介于 PinchBench 和 WildClawBench 之间的 benchmark：

- 比 PinchBench 更强调真实多步骤工作流；
- 比 WildClawBench 更偏通用助手场景，而不一定全是 hard case；
- 能覆盖多种 workflow pattern；
- 适合评估通用 Agent 系统的实用性。

### 它更适合回答的问题

- Agent 是否能处理真实用户的复杂助手任务？
- 在邮件、日程、文档、研究等场景中表现是否稳定？
- 多步骤任务中是否能保持目标一致性？
- 工具使用和结果整合是否可靠？
- 不同模型作为数字助理时的综合表现如何？

### 局限

LiveClawBench 覆盖面广，但也因此可能存在：

- 任务分布较复杂，难以定位具体瓶颈；
- 不同场景的难度不完全可比；
- 自动评测可能需要更多规则或人工校验；
- 如果任务长度分布没有明确分层，系统调度分析会比较难做。

所以它更像 **general-purpose assistant benchmark**，适合评估真实助手能力的广度和综合性。

---

## 5. 从 Agent Serving 角度看三者差异

如果你的重点是 **Agent 服务场景中的调度问题**，这三个 benchmark 的价值不一样。

| **维度**   | **PinchBench**             | **WildClawBench**                  | **LiveClawBench**                |
| ---------- | -------------------------- | ---------------------------------- | -------------------------------- |
| 任务长度   | 短到中等                   | 长链路为主                         | 中到长，视场景而定               |
| 任务异质性 | 高                         | 高，但偏 hard case                 | 很高，覆盖助手场景               |
| Tool 使用  | 有，但不一定很深           | 强，多工具、多轮                   | 强，贴近真实助手                 |
| 多模态     | 可能有，但不是核心         | 核心特征之一                       | 可能覆盖多模态                   |
| 调度压力   | mixed workload、成本、吞吐 | 长任务阻塞、KV 生命周期、tool 阻塞 | 多场景 workflow、SLO、多工具协作 |
| 更适合评估 | 日常平均性能               | 复杂任务上限                       | 通用助手实用性                   |

从 serving 角度可以这样总结：

- **PinchBench** 更适合研究 **异构短任务混跑下的调度与 cost-latency trade-off**。
- **WildClawBench** 更适合研究 **长 trajectory、tool 阻塞、KV 生命周期和 tail latency**。
- **LiveClawBench** 更适合研究 **通用助手 workflow 下的端到端成功率和多场景泛化能力**。

---

## 6. 三者和你 PPT 主题的关系

你的 PPT 主题是 **Agent 服务场景中的调度问题**。这三个 benchmark 可以用来支撑“Agent workload 不再是传统单请求”的论点。

### PinchBench 对应的问题

PinchBench 说明：

> Agent serving 不只是少量超长任务，也包含大量日常 assistant/coding mixed workload。

对应调度问题：

- 多 SLO 混跑；
- 输出长度方差；
- 短任务被中任务拖慢；
- 成本敏感调度；
- model routing / model selection。

### WildClawBench 对应的问题

WildClawBench 说明：

> 真实 hard agent 任务会形成复杂长链路 workflow，对 serving 系统产生持续压力。

对应调度问题：

- trajectory-level HoL blocking；
- tool call 阻塞；
- KV 长生命周期；
- DAG / critical path scheduling；
- tail latency 和 timeout 控制。

### LiveClawBench 对应的问题

LiveClawBench 说明：

> 通用助手 Agent 的真实工作负载跨越多个应用场景，具有多步骤、多工具、多目标特征。

对应调度问题：

- workflow-aware scheduling；
- heterogeneous application serving；
- end-to-end latency；
- multi-domain SLO；
- 语义信息暴露与后端调度联动。

---

## 7. 可以放进 PPT 的精简表述

如果你要在 PPT 里加一页 benchmark 分析，可以直接用下面这版。

### Benchmark Landscape for Agent Workloads

| **Benchmark**     | **Workload 特征**                                    | **主要压力点**                                            | **适合评估**                        |
| ----------------- | ---------------------------------------------------- | --------------------------------------------------------- | ----------------------------------- |
| **PinchBench**    | 日常 assistant/coding 请求；任务短但异质性高         | mixed workload、成本、速度、短任务延迟                    | 日常负载下的成功率 / latency / cost |
| **WildClawBench** | 真实 runtime 中的复杂、长链路、多工具、多模态任务    | long trajectory、tool blocking、KV 生命周期、tail latency | hard agent 任务完成能力与系统鲁棒性 |
| **LiveClawBench** | 通用数字助理任务；邮件、文档、日程、研究、开发、购物 | 多步骤 workflow、多场景泛化、端到端成功率                 | 通用助手型 Agent 的综合执行能力     |

### 一句话总结

> **PinchBench 代表日常 mixed workload，WildClawBench 代表复杂 hard workload，LiveClawBench 代表通用助手 workflow。三者共同说明 Agent Serving 需要从 request-level 调度转向 workflow-aware / prediction-aware 调度。**

---

## 8. 最关键的结论

这三个 benchmark 的互补性很强：

- **PinchBench** 看“日常使用是否快、省、稳”；
- **WildClawBench** 看“复杂任务是否扛得住、跑得完”；
- **LiveClawBench** 看“通用助手是否能覆盖真实多场景 workflow”。

如果用于你的 Agent Serving 调度综述，它们可以作为 workload 证据说明：

> Agent workload 不是单一分布，而是由短任务 mixed workload、长链路 hard workload、通用多场景助手 workflow 共同组成。  
> 因此 serving 系统需要同时考虑 **异质性、依赖结构、tool 阻塞、KV 生命周期和端到端 SLO**。