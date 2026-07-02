# Agent Benchmark 调研：工具模拟器 + 工具时间可控

> 调研日期：2026-06-23
> 起点 idea：做一个针对 Agent 的 benchmark，用工具模拟器模拟每次工具调用、返回预设好的
> 工具调用结果；通过限制/固定模型调用工具的过程，降低"工具调用时间不同"对 Agent 方法
> 性能评估的影响。

---

## 一、结论先行

**这个 idea 没有人原样实现，但它由两块各自成熟的工作拼成，新意在"把两块拼起来的那个点"。**

- "工具模拟器返回预设结果 + 可复现评测" → 已被 τ-bench 等占据，**纯做这个没有新意**。
- "工具执行时间对 Agent serving 的影响" → 最近一年一批调度论文在做，是**正热**的方向。
- **真正的空白点**：把"工具延迟"作为一个**显式可控变量**（固定 / 按分布注入），用来
  ①公平对比 Agent 方法，或 ②隔离评估调度策略——这个具体的点没搜到有人系统做过。

---

## 二、idea 拆成两个已有方向

### 方向 A：工具模拟器 / mock 工具结果（已经很成熟）

"用模拟器返回预设的工具调用结果"这件事，已有不少现成做法：

- **τ-bench / τ²-bench (Sierra Research)** —— Agent 评测的事实标准。工具是**确定性模拟**的
  （数据库 + 预设 API 行为），不打真实后端，保证可复现。几乎就是"工具模拟器返回设置好的
  结果"。但它控制的是工具的**返回内容**，**没有把工具延迟作为可控变量**。
  - https://github.com/sierra-research/tau-bench
  - https://github.com/sierra-research/tau2-bench
- **REAL: Deterministic Simulations of Real Websites** —— 把真实网站做成确定性模拟，
  去掉打活网站的波动。
  - https://openreview.net/pdf?id=Un1sWxmZuI
- **AWS ToolSimulator** —— 工程上 scalable 地 mock 工具返回值测试 Agent。
  - https://aws.amazon.com/blogs/machine-learning/toolsimulator-scalable-tool-testing-for-ai-agents/
- **agentest（mock 工具结果框架）**
  - https://r-prem.github.io/agentest/guide/mocks.html

> 小结：A 这一块"模拟器返回预设结果"**本身不新**，τ-bench 已是标配。

### 方向 B：工具执行时间对 Agent serving 的影响（正在热的研究）

"工具调用耗时不同会影响性能"这个角度，最近一年有一批调度论文：

- **Parallelizing Tool Execution and LLM Generation for Low-Latency Agent Serving**
  —— 直接处理工具执行时间与生成的重叠。
  - https://arxiv.org/html/2603.18897v3
- **Justitia: Fair and Efficient Scheduling of Task-parallel LLM Agents** —— Agent 公平调度。
  - https://arxiv.org/pdf/2510.17015
- **Astraea: State-Aware Scheduling for LLM-Powered Agents**
  - https://arxiv.org/html/2512.14142
- **SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters**
  - https://arxiv.org/html/2605.00528
- **Kairos: Low-latency Multi-Agent Serving with Shared LLMs**
  - https://arxiv.org/pdf/2508.06948
- **Agentix: An Efficient Serving Engine for LLM Agents as General Programs (NSDI'26)**
  - https://www.usenix.org/system/files/nsdi26-luo.pdf

> 这些把工具耗时当成**核心变量去建模/优化**，但目的不是"公平对比 Agent 方法"。

---

## 三、关键概念分歧（必须先想清楚自己要哪个）

idea 原话："通过限制模型调用工具的操作，把工具时间固定下来，以降低工具调用时间不同对
Agent 方法性能的影响"。

这里有一个**方向 A 和方向 B 相反的态度**：

| 目标 | 对"工具时间"的态度 | 做法 | 已有工作覆盖? |
|---|---|---|---|
| **评测 Agent 方法本身**（算法/prompt/规划策略） | 工具耗时是**噪声**，要**消除** | 模拟器对每个工具返回**固定延迟（甚至零延迟）**，对比不被"今天 API 慢了"干扰 | **τ-bench 没明确做**（控了内容，没控延迟）→ 空白点 |
| **研究调度**（memory 里的方向） | 工具耗时是**核心信号**，要**精确建模/注入** | 模拟器按设定分布产生不同延迟，看调度器表现 | 方向 B 那批论文在做 |

**这两个目标矛盾**：一个要抹掉工具时间，一个要放大工具时间。
idea 的描述偏向第一个（消除），但研究方向是调度（第二个）。**这是动手前必须定的事。**

---

## 四、判断与建议

1. "工具模拟器返回预设结果" + "可复现评测" → 已被 τ-bench 占了，纯做没新意。
2. "把工具延迟作为显式可控变量（固定/按分布注入），用来公平对比 Agent 方法或评估调度
   策略" → **这个具体的点没搜到有人系统做过**，有空间。已有工作要么固定内容没固定时间
   （τ-bench），要么研究时间但不为"公平对比方法"（调度论文）。
3. **与调度研究结合的最有价值版本**：做一个能精确注入"可控工具延迟分布"的 Agent 模拟
   环境，用它隔离评估"输出长度预测调度"在不同工具耗时下的 **Goodput / SLO** 表现——
   把 benchmark idea 和调度研究主线焊在一起。

### 待确认的关键问题
做这个 benchmark，主目的是 **评测 Agent 方法（消除工具时间噪声）**，还是
**评测/驱动调度研究（建模工具时间）**？这决定整个设计方向和对标的已有工作。

---

## 五、参考链接汇总

**Agent 评测 / 工具模拟**
- τ-bench: https://github.com/sierra-research/tau-bench
- τ²-bench: https://github.com/sierra-research/tau2-bench
- REAL (deterministic website sim): https://openreview.net/pdf?id=Un1sWxmZuI
- AWS ToolSimulator: https://aws.amazon.com/blogs/machine-learning/toolsimulator-scalable-tool-testing-for-ai-agents/
- agentest mocks: https://r-prem.github.io/agentest/guide/mocks.html

**Agent 调度 / 工具时间**
- Parallelizing Tool Execution and LLM Generation: https://arxiv.org/html/2603.18897v3
- Justitia (fair scheduling): https://arxiv.org/pdf/2510.17015
- Astraea (state-aware scheduling): https://arxiv.org/html/2512.14142
- SAGA (workflow-atomic scheduling): https://arxiv.org/html/2605.00528
- Kairos (multi-agent serving): https://arxiv.org/pdf/2508.06948
- Agentix (NSDI'26): https://www.usenix.org/system/files/nsdi26-luo.pdf
- Stop Comparing LLM Agents Without Disclosing the Harness: https://arxiv.org/html/2605.23950
