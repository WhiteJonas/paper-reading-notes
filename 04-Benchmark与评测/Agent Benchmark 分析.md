下面给你整理一份**Agent 领域 benchmark 清单**，重点筛选满足你两个要求的：

1. **能测试多 Agent 并行 / 协作 / 竞争**
2. **能构造复杂负载**，适合做 **Agent serving / scheduling / workflow-aware inference** 研究

我会按适合程度分成：

- **A 类：强相关，多 Agent 原生 benchmark**
- **B 类：复杂 Agent 任务 benchmark，可改造成多 Agent 并行负载**
- **C 类：Web / Tool / OS / Code 类复杂环境 benchmark**
- **D 类：适合系统调度研究的组合负载方案**

---

# 一、最推荐：多 Agent 原生 Benchmark

---

## 1. MultiAgentBench

**关键词：multi-agent、collaboration、competition、interactive scenarios**

这是目前最贴近你需求的 benchmark 之一。它专门用于评估 **LLM-based multi-agent systems** 在多样化交互场景中的协作和竞争能力。

它适合测试：

- 多 Agent 协作；
- 多 Agent 竞争；
- 多轮交互；
- 任务分解；
- 角色分工；
- Agent 间通信；
- 多 Agent 并发执行；
- 多智能体系统整体任务完成率。

### 为什么适合你的方向？

如果你关注 **Agent 推理调度 / 多 Agent 并行 / 复杂负载**，MultiAgentBench 很适合作为核心 benchmark。

原因是它天然包含多个 Agent 之间的交互，不像一些单 Agent benchmark 需要你强行改造成多 Agent。

它可以帮助你构造类似下面的系统负载：

```text
一个用户任务
   ↓
Planner Agent
   ↓
多个 Worker Agents 并行执行子任务
   ↓
Critic / Verifier Agent 汇总结果
   ↓
最终输出
```

这种负载非常适合研究：

- workflow-aware scheduling；
- 多 Agent 并发推理；
- Agent 间依赖；
- 多轮 LLM call；
- KV cache 生命周期；
- GPU serving 队列压力；
- end-to-end job completion time。

### 你应该重点看什么？

读 MultiAgentBench 时建议关注：

- 它的任务是否是 collaborative 还是 competitive；
- 每个任务中 Agent 数量；
- Agent 之间是否有通信；
- 是否有多轮交互；
- 是否有工具调用；
- 是否容易统计每个任务的 LLM call 数量；
- 是否可以提取 workflow DAG；
- 是否支持并行执行多个 Agent。

### 适合作为论文/项目中的什么？

它可以作为你项目里的：

> **multi-agent workload benchmark**

尤其适合用来证明你的调度器不仅能处理普通 LLM requests，还能处理复杂 Agent workflows。

参考来源：MultiAgentBench 论文介绍其为用于评估 LLM-based multi-agent systems 在 diverse interactive scenarios 中协作和竞争能力的综合 benchmark。[3]

---

## 2. AgentBench

**关键词：LLM-as-Agent、多环境、tool use、web、OS、database、games**

AgentBench 是比较经典的 LLM Agent benchmark。它不是严格意义上的多 Agent benchmark，但它覆盖了多个复杂环境，可以很方便地扩展成多 Agent 并发 workload。

AgentBench 包含多种环境，例如：

- OS 操作；
- database；
- knowledge graph；
- digital card game；
- web shopping；
- web browsing；
- house-holding；
- coding / reasoning 类任务。

### 为什么适合你的方向？

虽然 AgentBench 原始设计更偏单 Agent，但它非常适合做系统调度负载，因为它有几个特点：

1. **任务类型复杂**
   - 不只是问答，而是环境交互；
   - 包含多步决策；
   - 包含工具使用；
   - 包含状态转移。

2. **每个任务会产生多次 LLM 调用**
   - 很适合模拟 Agent workflow；
   - 能产生长短不一的 trajectory；
   - 有利于测试调度器对复杂负载的处理能力。

3. **可并发运行多个任务**
   - 可以同时启动多个 AgentBench 环境；
   - 每个环境对应一个 Agent workflow；
   - 构造多 Agent 或多 workflow 并行负载。

### 如何改造成多 Agent benchmark？

你可以把一个 AgentBench 任务拆成多个角色：

```text
Planner Agent：制定计划
Executor Agent：执行环境操作
Verifier Agent：检查结果
Memory Agent：维护状态
```

或者并发运行多个 AgentBench task：

```text
Workflow 1: WebShop Agent
Workflow 2: OS Agent
Workflow 3: DB Agent
Workflow 4: Game Agent
...
```

这样可以测试：

- 多 Agent / 多 workflow 并行；
- 工具调用等待；
- 不同任务长度混合；
- 长短请求混合；
- KV cache 压力；
- 调度公平性；
- P95/P99 job completion time。

### 适合作为论文/项目中的什么？

它适合作为：

> **complex agent workload benchmark**

尤其适合你测试调度器面对不同环境、不同任务长度、不同工具调用模式时的稳定性。

参考来源：AgentBench 被介绍为用于评估 LLM-as-Agent 的多维 benchmark，包含 8 个不同环境，用于评估推理和决策能力。[4]

---

# 二、强相关：Web / Tool / Software Agent Benchmark

这些 benchmark 不一定是多 Agent 原生，但非常适合构造**复杂 Agent workload**。

---

## 3. WebArena

**关键词：web agent、browser interaction、multi-step tasks**

WebArena 是 Web Agent 方向非常重要的 benchmark，用真实或仿真的网站环境测试 Agent 完成复杂网页任务的能力。

它通常包括：

- 电商网站；
- Reddit-like 社区；
- GitLab-like 开发平台；
- map / information seeking；
- content management；
- 多步骤网页操作。

### 为什么适合复杂负载？

WebArena 的任务不是一次 LLM 调用就能完成，而是：

```text
observe webpage
→ reason
→ choose action
→ interact with browser
→ observe new state
→ continue
```

每个任务可能产生几十轮 LLM 调用。

这非常适合系统研究：

- tool call latency；
- browser action waiting；
- multi-turn Agent inference；
- long trajectory；
- workflow scheduling；
- KV cache 是否保留；
- 多 Agent 并发浏览任务。

### 怎么改造成多 Agent？

可以设计成：

```text
Navigator Agent：负责网页导航
Reader Agent：负责读取页面信息
Planner Agent：负责制定步骤
Verifier Agent：负责检查任务是否完成
```

或者多个 WebArena 任务并发：

```text
Agent 1 处理购物任务
Agent 2 处理 GitLab issue
Agent 3 处理 forum post
Agent 4 处理 map search
```

这会形成非常真实的复杂负载。

---

## 4. WebShop

**关键词：shopping agent、web navigation、decision making**

WebShop 是购物场景的 Agent benchmark。任务通常是根据用户需求，在网页环境里搜索、比较、选择商品。

它适合测试：

- 多步搜索；
- 工具使用；
- 环境交互；
- 长上下文；
- 决策能力；
- observation-action loop。

### 和调度的关系

WebShop 任务通常会生成较长的 Agent trajectory：

```text
query
→ search
→ inspect item
→ compare
→ go back
→ inspect another item
→ buy
```

这会导致：

- 多次短 LLM call；
- 中间有环境等待；
- prompt 中有大量历史 observation；
- KV cache 可复用但也占显存。

适合研究 Agent serving 里的：

- 多轮 KV cache 保留；
- tool waiting 期间 KV offload；
- 短 decode + 多 prefill 混合；
- 多 workflow 并发调度。

---

## 5. MiniWoB++

**关键词：web UI control、browser tasks、interactive agent**

MiniWoB++ 是网页 UI 操作任务 benchmark，任务包括点击按钮、填写表单、选择项目等。

相比 WebArena，它的任务更轻量、更可控，适合做系统实验。

### 为什么适合你？

如果你想构造一个可控的多 Agent 并发 benchmark，MiniWoB++ 很合适：

- 任务短；
- 环境轻；
- 可并发启动大量实例；
- 每个任务有明确成功率；
- 适合压测 serving backend。

可以用它测试：

- 高并发短 Agent workload；
- 多 Agent 并行执行；
- scheduler 对短任务的公平性；
- TTFT / TPOT / job completion time；
- throughput under many small workflows。

---

# 三、软件工程 / 代码 Agent Benchmark

如果你想测试复杂负载，代码 Agent benchmark 很重要，因为它们通常任务长、步骤多、上下文大。

---

## 6. SWE-bench

**关键词：software engineering agent、GitHub issue、code editing**

SWE-bench 是评估 LLM 解决真实 GitHub issue 的 benchmark。Agent 需要理解代码库、定位 bug、修改代码、运行测试。

### 为什么适合复杂负载？

SWE-bench 的任务通常非常复杂：

- 长上下文；
- 多文件检索；
- 多轮 reasoning；
- 工具调用；
- shell 命令；
- test execution；
- 代码编辑；
- 失败重试。

这类任务很适合构造重负载 Agent workflow：

```text
Issue understanding
→ repo search
→ file inspection
→ patch generation
→ test execution
→ debugging
→ final patch
```

### 和多 Agent 的结合方式

你可以把 SWE-bench 改造成多 Agent：

```text
Repo Search Agent：检索相关文件
Bug Localization Agent：定位问题
Patch Agent：生成修改
Test Agent：运行测试
Review Agent：检查 patch
```

或者并行运行多个 SWE-bench issue，模拟复杂系统负载。

### 适合测试什么？

- 长上下文 prompt；
- 长 workflow；
- 工具调用等待；
- CPU/GPU 混合资源；
- 多 Agent collaboration；
- cache retention；
- end-to-end latency；
- workflow-level scheduling。

---

## 7. SWE-bench Multimodal / SWE-bench Verified

如果你想要更接近真实复杂任务，可以看 SWE-bench 的衍生版本，例如 Verified 或 Multimodal。

它们通常比基础版本更适合做高质量评估，但系统压测可能成本更高。

### 适合用途

- 高质量复杂 Agent workload；
- 长任务调度；
- tool-heavy workload；
- multi-agent software development simulation。

---

## 8. HumanEval / MBPP Agent 化版本

HumanEval 和 MBPP 本身是代码生成 benchmark，不是 Agent benchmark。  
但如果你把它们做成 Agent loop，也可以用于调度压力测试：

```text
generate code
→ run tests
→ observe error
→ repair code
→ run tests again
```

### 优点

- 成本低；
- 可控；
- 易并发；
- 易统计成功率；
- 适合做大量并发实验。

### 缺点

- 任务复杂度不如 SWE-bench；
- 多 Agent 协作特征较弱；
- 更偏 coding loop，不够通用。

---

# 四、工具调用 / API Agent Benchmark

---

## 9. ToolBench

**关键词：tool use、API calling、tool selection**

ToolBench 是工具调用方向的重要 benchmark，用来测试 LLM Agent 使用工具/API 的能力。

它适合测试：

- 工具选择；
- 参数生成；
- API 调用；
- 多步 tool chain；
- tool response parsing；
- planning。

### 为什么适合系统调度？

ToolBench 的任务天然包含：

```text
LLM inference
→ tool call
→ waiting
→ observation
→ next LLM inference
```

这非常接近真实 Agent workload。

适合研究：

- tool waiting 期间 KV cache 怎么处理；
- LLM 和工具执行如何 overlap；
- 多 Agent tool call 并发；
- 工具响应时间不确定时如何调度；
- workflow-aware serving。

### 多 Agent 改造方式

可以设计：

```text
Planner Agent：决定工具链
Tool Selector Agent：选择 API
Executor Agent：调用工具
Verifier Agent：检查结果
```

或者并发运行多个 tool-use workflows。

---

## 10. API-Bank

**关键词：API calling、tool-augmented LLM、multi-turn interaction**

API-Bank 也是工具使用类 benchmark。它构造了需要调用 API 完成任务的场景。

### 适合测试

- 多轮 API 调用；
- 工具调用正确性；
- 参数构造；
- 状态跟踪；
- Agent planning；
- tool-LLM alternating workload。

### 调度价值

它和 ToolBench 类似，很适合测试：

- LLM call 和 tool call 的交替；
- 工具等待造成的 GPU idle；
- KV cache 生命周期；
- workflow-level JCT；
- 多 workflow 并发。

---

# 五、多 Agent 协作任务 / 游戏 / 仿真环境

---

## 11. Overcooked-AI / Overcooked Multi-Agent

**关键词：cooperative multi-agent、coordination、planning**

Overcooked 是经典多智能体协作环境。虽然不一定是 LLM Agent 专用 benchmark，但可以用 LLM Agent 进行协作规划。

### 适合测试

- 多 Agent 协作；
- 实时决策；
- shared environment；
- coordination；
- task allocation；
- communication。

### 和 LLM serving 调度的关系

如果多个 LLM Agent 在每个 timestep 都需要推理，系统负载会呈现明显的并发和周期性：

```text
timestep 1: Agent A/B/C 同时请求 LLM
timestep 2: Agent A/B/C 同时请求 LLM
...
```

适合测试：

- 同步多 Agent 推理；
- step-level deadline；
- 多 Agent 并行调度；
- latency 对任务成功率的影响。

---

## 12. Hanabi / Diplomacy / Avalon 类 benchmark

这些是多 Agent 协作/竞争/博弈环境，经常用于测试沟通、欺骗、策略和团队协作。

### 适合测试

- 多 Agent communication；
- partial observability；
- cooperation / competition；
- long-horizon planning；
- role-based reasoning；
- negotiation。

### 对调度研究的价值

这类任务的每一轮可能有多个 Agent 同时发言/思考，适合模拟：

- 多 Agent 同步推理；
- 多 Agent 竞争资源；
- turn-based workload；
- 不同 Agent priority；
- deadline-aware scheduling。

---

## 13. PettingZoo 多智能体环境

PettingZoo 是多智能体强化学习环境集合，不是 LLM benchmark，但可以封装 LLM Agent。

### 优点

- 多 Agent 原生；
- 环境多；
- 可控；
- 可并发；
- 容易构造 synthetic workload；
- 可以控制 Agent 数量、step 数、环境复杂度。

### 适合你的用途

如果你不是只想评估 Agent 能力，而是想评估 **serving / scheduling 系统**，PettingZoo 很有价值，因为你可以灵活控制负载形态。

---

# 六、任务型 / Embodied Agent Benchmark

---

## 14. ALFWorld

**关键词：embodied agent、text environment、long-horizon task**

ALFWorld 是文本化 embodied agent 环境。Agent 需要在家庭场景中完成任务，比如拿取、移动、清洁、加热物体等。

### 适合复杂负载

它具有：

- 多步操作；
- 状态跟踪；
- 长程规划；
- 环境交互；
- observation-action loop。

### 多 Agent 改造方式

可以把任务拆成：

```text
Planner Agent
Navigator Agent
Object Search Agent
Action Executor Agent
Verifier Agent
```

或者并发运行多个 household tasks。

---

## 15. ScienceWorld

**关键词：scientific reasoning、interactive environment**

ScienceWorld 是科学实验类交互环境，要求 Agent 在环境中进行探索和实验。

### 适合测试

- 长程推理；
- 环境交互；
- 规划；
- observation parsing；
- 多步任务；
- 复杂状态。

### 系统调度价值

它适合构造：

- 长 trajectory；
- 多次 LLM call；
- 任务长度差异明显；
- 长短任务混合负载；
- Agent reasoning + environment interaction 负载。

---

# 七、如果你的目标是“测试多 Agent 并行 + 复杂负载”，最推荐这 8 个

我建议你优先看下面这几个：

| 优先级 | Benchmark                  | 是否多 Agent 原生   | 复杂负载 | 推荐用途                         |
| -----: | -------------------------- | ------------------- | -------- | -------------------------------- |
|      1 | **MultiAgentBench**        | 是                  | 高       | 多 Agent 协作/竞争核心 benchmark |
|      2 | **AgentBench**             | 否，但易扩展        | 高       | 多环境复杂 Agent workload        |
|      3 | **WebArena**               | 否，但易扩展        | 高       | Web Agent 长 workflow            |
|      4 | **ToolBench**              | 否，但工具链复杂    | 高       | Tool-call heavy workload         |
|      5 | **SWE-bench**              | 否，但易多 Agent 化 | 很高     | 软件工程复杂长任务               |
|      6 | **MiniWoB++**              | 否，但可高并发      | 中       | 大量短 Agent workflow 压测       |
|      7 | **ALFWorld**               | 否，但可角色化      | 中高     | 长程规划与环境交互               |
|      8 | **PettingZoo + LLM Agent** | 是，多智能体环境    | 可控     | 自定义多 Agent 并行负载          |

---

# 八、按你的研究方向分类推荐

你如果是做 **Agent serving / scheduling**，benchmark 选择要看你到底想测什么。

---

## 1. 想测多 Agent 协作 / 竞争

优先：

- **MultiAgentBench**
- PettingZoo + LLM Agents
- Overcooked-AI
- Hanabi / Avalon / Diplomacy 类环境

适合研究：

- 多 Agent 同步推理；
- 多 Agent communication；
- 角色分工；
- collaborative planning；
- competitive scheduling；
- per-agent latency 对整体任务的影响。

---

## 2. 想测复杂 workflow / 长链路 Agent

优先：

- **WebArena**
- **SWE-bench**
- AgentBench
- ALFWorld
- ScienceWorld

适合研究：

- workflow-level scheduling；
- end-to-end job completion time；
- long trajectory；
- 多阶段任务；
- prompt 长度增长；
- KV cache 生命周期。

---

## 3. 想测工具调用和 GPU 空转

优先：

- **ToolBench**
- API-Bank
- WebArena
- AgentBench

适合研究：

- tool waiting；
- LLM/tool overlap；
- KV offload；
- tool-aware scheduling；
- workflow runtime。

---

## 4. 想测高并发 serving 压力

优先：

- MiniWoB++
- HumanEval/MBPP Agent loop
- AgentBench 多实例
- ToolBench 多实例
- WebShop 多实例

适合研究：

- 高并发短 workflow；
- TTFT / TPOT；
- P95 / P99 latency；
- throughput；
- KV cache pressure；
- admission control；
- fairness。

---

# 九、适合系统论文的负载构造方式

如果你的目标不是单纯评测 Agent 能力，而是要做 **Agent 推理调度系统**，我建议不要只原样跑 benchmark，而是构造 workload mix。

---

## Workload 1：多 Agent 协作型

来源：

- MultiAgentBench
- Overcooked
- PettingZoo

负载形态：

```text
每个 job 包含 3–6 个 Agent
每个 Agent 有多轮 LLM call
Agent 之间存在通信依赖
部分 step 可以并行
部分 step 必须同步 barrier
```

测试指标：

- job completion time；
- per-agent waiting time；
- communication round latency；
- GPU utilization；
- scheduling fairness；
- P95/P99 workflow latency。

---

## Workload 2：Tool-heavy 型

来源：

- ToolBench
- API-Bank
- WebArena

负载形态：

```text
LLM call → tool call → waiting → observation → LLM call
```

测试重点：

- tool waiting 期间 KV 是否保留；
- offload 策略；
- tool 与 LLM overlap；
- workflow-aware scheduling；
- GPU idle time。

---

## Workload 3：Long-context / Long-workflow 型

来源：

- SWE-bench
- WebArena
- AgentBench
- ALFWorld

负载形态：

```text
长 prompt
多轮 LLM call
上下文逐渐增长
工具调用穿插
部分任务极长
```

测试重点：

- KV cache 占用；
- prefix cache 命中率；
- 长短任务公平性；
- tail latency；
- starvation；
- admission control。

---

## Workload 4：High-concurrency Mixed 型

来源：

- MiniWoB++
- WebShop
- HumanEval Agent loop
- ToolBench

负载形态：

```text
大量短任务 + 少量长任务混合到达
```

测试重点：

- 多工作流并发；
- P95/P99；
- throughput；
- goodput；
- TTFT；
- TPOT；
- scheduler robustness。

---

# 十、建议你最终选用的 Benchmark 组合

如果你做的是 **Agent 调度 / Agent serving**，我建议组合如下：

## 核心组合

```text
MultiAgentBench + AgentBench + WebArena + ToolBench + SWE-bench
```

原因：

| Benchmark       | 负责覆盖                         |
| --------------- | -------------------------------- |
| MultiAgentBench | 多 Agent 协作/竞争               |
| AgentBench      | 多环境 Agent 复杂任务            |
| WebArena        | Web workflow 和 tool interaction |
| ToolBench       | 工具调用密集型负载               |
| SWE-bench       | 长上下文、长任务、代码 Agent     |

---

## 如果你需要高并发压测

再加：

```text
MiniWoB++ + HumanEval/MBPP Agent Loop
```

它们任务更轻，可以跑大量并发，适合测系统吞吐和尾延迟。

---

## 如果你需要可控多 Agent 环境

再加：

```text
PettingZoo + Overcooked-AI
```

它们适合控制：

- Agent 数量；
- step 数；
- 同步/异步模式；
- 协作/竞争关系；
- workload intensity。

---

# 十一、你应该记录哪些系统指标？

如果你的目标是调度研究，不要只记录 benchmark success rate。建议同时记录：

## Agent 层指标

| 指标                 | 含义                    |
| -------------------- | ----------------------- |
| Success Rate         | 任务完成率              |
| Average Reward       | 环境奖励                |
| Turns / Steps        | 完成任务所需轮数        |
| Tool Calls           | 工具调用次数            |
| LLM Calls            | LLM 推理次数            |
| Communication Rounds | Agent 通信轮数          |
| End-to-End JCT       | 整个 Agent job 完成时间 |

---

## Serving 层指标

| 指标                 | 含义                     |
| -------------------- | ------------------------ |
| TTFT                 | 首 token 延迟            |
| TPOT                 | 每 token 延迟            |
| P95/P99 Latency      | 尾延迟                   |
| Throughput           | tokens/s 或 jobs/s       |
| Goodput              | 满足 SLO 的有效吞吐      |
| GPU Utilization      | GPU 利用率               |
| KV Cache Utilization | KV cache 占用            |
| Cache Hit Rate       | prefix / KV cache 命中率 |
| Queueing Time        | 排队等待时间             |
| Preemption Count     | 抢占次数                 |
| Offload Traffic      | KV offload 流量          |
| Tool Waiting Time    | 工具等待时间             |

---

# 十二、最适合你论文/项目的 Benchmark 表述

你可以在开题或综述里这样写：

> To evaluate scheduling under agentic workloads, we consider both native multi-agent benchmarks and complex single-agent benchmarks that can be transformed into multi-workflow or role-based multi-agent workloads. MultiAgentBench is used to evaluate collaborative and competitive multi-agent scenarios. AgentBench, WebArena, ToolBench, and SWE-bench are used to generate complex long-horizon workloads involving environment interaction, tool calls, long contexts, and heterogeneous workflow lengths. For high-concurrency serving evaluation, lightweight environments such as MiniWoB++ and HumanEval-style agent loops can be used to stress-test scheduling policies under mixed short/long workflows.

中文版本：

> 为了评估 Agent 场景下的推理调度，我们可以同时使用原生多 Agent benchmark 和可改造成多 Agent / 多 workflow 的复杂单 Agent benchmark。MultiAgentBench 用于评估多 Agent 协作和竞争；AgentBench、WebArena、ToolBench 和 SWE-bench 用于构造包含环境交互、工具调用、长上下文和异构任务长度的复杂负载；MiniWoB++ 和 HumanEval-style agent loop 则适合构造高并发短任务，用于压测调度器的吞吐和尾延迟表现。

---

# 十三、最终推荐清单

如果你只想要一个最精简但够用的列表：

| 类型                   | 推荐 Benchmark                               |
| ---------------------- | -------------------------------------------- |
| 多 Agent 原生          | **MultiAgentBench**                          |
| 多环境 Agent           | **AgentBench**                               |
| Web Agent              | **WebArena**, **WebShop**, **MiniWoB++**     |
| 工具调用 Agent         | **ToolBench**, **API-Bank**                  |
| 软件工程 Agent         | **SWE-bench**                                |
| Embodied / Interactive | **ALFWorld**, **ScienceWorld**               |
| 可控多智能体环境       | **PettingZoo**, **Overcooked-AI**            |
| 高并发轻量压测         | **MiniWoB++**, **HumanEval/MBPP Agent Loop** |

---

# 十四、我最建议你重点看这 5 个

如果你时间有限，优先看：

1. **MultiAgentBench**：最贴近多 Agent 协作/竞争；
2. **AgentBench**：多环境复杂 Agent 能力评估；
3. **WebArena**：复杂 Web Agent workflow；
4. **ToolBench**：工具调用密集型 Agent 负载；
5. **SWE-bench**：长上下文、长任务、复杂代码 Agent。

对于你的“多 Agent 并行 + 复杂负载”需求，最推荐的组合是：

> **MultiAgentBench + AgentBench + WebArena + ToolBench + SWE-bench**

这套组合基本能覆盖：

- 多 Agent 协作；
- 多 Agent 竞争；
- 工具调用；
- Web 交互；
- 长任务；
- 长上下文；
- 高异构性负载；
- workflow-aware scheduling；
- KV cache 生命周期管理。