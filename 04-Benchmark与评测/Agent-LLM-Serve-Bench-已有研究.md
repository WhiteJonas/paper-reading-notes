# 已有研究（Agent LLM Serve Bench 补充）

> 配合提纲三栏中的「已有研究」一栏。按「评测/模拟工具 → Agent serving 系统优化 → Agent 能力 benchmark」三类组织，
> 每条标出它做了什么、与本研究的关系（互补 or 待解决的 gap）。
>
> ⚠️ 说明：论文正文抓取受网络限制，以下方法描述为**标题+摘要级别**的概括，
> 引用前请按链接核对原文的具体机制与数字（尤其是 Continuum 的 TTL、Tokencake 的调度算法细节）。

---

## 一、LLM serving 评测 / 模拟工具（和本研究最直接对标）

这类工作回答的是「**怎么评测一个 serving 系统**」，正是本研究要解决的问题。

### 1. Vidur（MLSys 2024，Microsoft）
- **做什么**：大规模 LLM **推理仿真**框架。不真实跑模型，而是先**离线 profile 各算子的延迟**建出预测模型，再用**离散事件仿真**模拟请求调度与批处理，预测延迟 / 吞吐 / TTFT。配套 Vidur-Search 在满足 SLO 下搜最省成本的部署配置。
- **效果**：延迟预测误差约 9%。
- **与本研究关系**：开创了「用仿真代替真实跑 GPU 来评测 serving」的路线，但 **面向的是 chat / 单轮请求**，没有 agent 的多轮、工具调用、动态工作流。本研究正是把这条路线**推进到 agent 场景**。
- 链接：https://github.com/microsoft/vidur ｜ https://proceedings.mlsys.org/paper_files/paper/2024/file/b74a8de47d2b3c928360e0a011f48351-Paper-Conference.pdf

### 2. AgentServeSim（你提纲里提到的那个）
- **做什么**：**面向多轮 LLM agent serving 的硬件感知仿真器**。这是与本研究**最接近的工作**——同样想用模拟方式评测 agent serving，同样要处理「真实跑 agent 受网络/计算/存储影响、难以精准复现」的问题。
- **与本研究关系**：直接对标。需要在文中明确**本研究相对它的差异/改进**（例如：本研究强调「记录-重放」把不确定性降到最低、对 LLM 规划路径做约束、对工具调用做统计+方差建模——看 AgentServeSim 在这几点上是怎么做的、有没有留空间）。
- 链接：https://arxiv.org/abs/2606.09613 （注：该 arXiv id 日期偏未来，注意核对版本）

### 3. LLMServingSim
- **做什么**：serving 仿真工具，已支持 **agentic sessions** 工作负载，并定义了 **JSONL 的 trace 格式**来描述 agent 多轮会话——这与本研究「记录 tool call 过程 / llm call 序列」的思路高度一致。
- **与本研究关系**：可作为 trace 格式和重放机制的参考/对比基线。
- 链接：https://llmservingsim.ai/docs/workloads/agentic-sessions ｜ https://llmservingsim.ai/docs/workloads/jsonl-format

---

## 二、Agent serving 系统优化（被评测的"对象"，说明评测为什么重要）

这类是**真实的 agent serving 优化系统**——它们正是本研究的评测工具想要客观对比的对象。它们也都印证了提纲里指出的痛点（KV Cache 管理、Prefill/Decode 负载差异）。

### 4. Continuum（ICLR 2026）
- **做什么**：多轮 LLM agent 调度。提出给 **KV Cache 设"存活时间"（Time-to-Live, TTL）**——根据 agent 下一轮何时回来，决定缓存留多久，平衡「保留缓存省重算」与「释放显存给别人」。针对的正是提纲里「工具调用不定 → LLM 调用间隔不定 → KV Cache 管理问题」。
- **与本研究关系**：典型的「被评测系统」。它的优化效果如何客观衡量，正是本研究要解决的。
- 链接：https://arxiv.org/abs/2511.02230

### 5. Tokencake（你同周读的另一篇）
- **做什么**：**以 KV-Cache 为中心**的多 agent serving 框架，用**空间-时间（space-time）调度**管理工具调用停顿（tool-call stall）期间的 KV cache——agent 等工具返回时 GPU 空着，把这段时间/显存调度给别的请求。
- **与本研究关系**：被评测对象 + 和你 memory 里的 tokencake.pdf 是同一篇，可与本研究的 serving 痛点分析互相印证。
- 链接：https://arxiv.org/abs/2510.18586

### 6. Cortex（SAA'25，Columbia DAPLab）
- **做什么**：**工作流感知（workflow-aware）的资源池化与调度**。利用 agent 工作流的结构信息来池化资源、调度请求。直接呼应提纲「agent 从静态工作流→动态工作流」的转变。
- 链接：https://arxiv.org/abs/2510.14126 ｜ https://daplab.cs.columbia.edu/projects/cortex/

### 7. Astraea
- **做什么**：**状态感知（state-aware）的 LLM agent 调度引擎**，根据 agent 的执行状态做调度决策。
- 链接：https://arxiv.org/html/2512.14142

### 8. KVFlow
- **做什么**：面向**多 agent 工作流**的高效**前缀缓存（prefix caching）**，加速重复前缀（如共享 system prompt / 工具描述）的复用。
- 链接：https://openreview.net/pdf/2c47adb29432f99879fceb1371b72f6e97e1f3ac.pdf

> 小结：第二类系统百花齐放，但它们各自在不同硬件、不同 workload、不同 agent 框架上评测，
> **难以横向客观对比**——这恰好是提纲指出的核心问题，也是本研究（一个标准化、可复现的 agent serving benchmark）的价值所在。

---

## 三、Agent 能力 benchmark（区分：评的是 agent 好不好，不是 serving 系统好不好）

需要在文中**划清边界**：已有大量 agent benchmark 评测的是「agent 完成任务的能力/正确率」，而**不是 serving 系统的性能（延迟/吞吐/SLO）**。本研究填的是后者的空白。

### 9. τ-bench（你同周读的）
- **做什么**：评测 agent 在「工具+用户+领域规则」真实交互中的**任务完成可靠性**（pass^k 指标），关注 agent 决策对不对、守不守规则。
- **与本研究关系**：**正交**。τ-bench 评 agent 能力，本研究评 serving 系统性能。可在文中作为「能力侧 benchmark 已成熟，但系统侧 serving benchmark 缺失」的对比，凸显本研究定位。
- （另有 AgentBench、WebArena、SWE-bench 等同属"能力评测"，均不解决 serving 性能评测问题。）

---

## 总结：本研究在已有研究中的位置（可直接用作"已有研究"段落收尾）

| 类别 | 代表工作 | 它们解决了什么 | 留下的 gap（本研究切入点） |
|------|---------|--------------|------------------------|
| serving 仿真评测 | Vidur | chat/单轮的仿真评测 | 不支持 agent 多轮/工具调用/动态工作流 |
| agent serving 仿真 | AgentServeSim、LLMServingSim | 开始做 agent serving 模拟 | 仿真受网络/计算/存储影响难精准；本研究用**记录-重放+约束规划+统计建模**进一步降低不确定性 |
| agent serving 系统 | Continuum、Tokencake、Cortex、Astraea、KVFlow | 各自优化 KV Cache/调度 | 缺**统一可复现的评测基准**来客观对比这些系统 |
| agent 能力 benchmark | τ-bench、AgentBench… | 评 agent 任务能力 | 不评 serving 系统性能（TTFT/TPOT/吞吐/SLO） |

**一句话定位**：已有研究要么停在 chat 场景的 serving 评测（Vidur），要么是各说各话、难以横向对比的 agent serving 系统（Continuum/Tokencake/…），要么只评 agent 能力而非系统性能（τ-bench）。本研究通过**记录-重放、约束 LLM 规划路径、对工具调用做统计+方差建模**，构建一个能**控制变量、客观对比不同 agent serving 系统**的评测基准，补上这块空白。
