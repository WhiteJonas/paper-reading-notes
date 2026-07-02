# Agent 推理加速综述

## 一、Agent定义

### 1.定义

**Agent = 让 LLM 不只"回答一次",而是反复"思考-行动-观察",直到把一个任务做完的程序。**

传统 LLM 调用:

```
用户输入 prompt → 模型输出一段文字 → 结束
```

Agent 调用:

```
用户给任务 → 模型思考(LLM call 1)
           → 决定调工具(tool call:搜索/查数据库/执行代码)
           → 拿到工具返回结果
           → 再思考(LLM call 2)
           → 再调工具(tool call)
           → ...
           → 直到判断任务完成 → 输出最终结果
```

中间可能有几次到几十次 LLM call,以及任意数量的 tool call。

### 2. Agent 的几种典型形态

| 形态                     | 典型例子                    | 特点                                                         |
| ------------------------ | --------------------------- | ------------------------------------------------------------ |
| **ReAct(Reason+Act,推理+行动,单 Agent 循环)** | LangChain、AutoGPT          | 一个 LLM反复"思考→工具→观察",最经典                         |
| **多 Agent 协作**        | MetaGPT、AutoGen、CrewAI    | 多个 LLM 扮演不同角色互相对话(产品经理 / 程序员 / 测试)      |
| **DAG-style** | LLMCompiler                 | 先规划成函数调用图,再并发执行多个 tool                       |
| **Reasoning Agent(推理型 Agent)**      | OpenAI o1、DeepSeek R1、QwQ | "思考过程"本身就很长,不一定调外部工具,但内部 CoT(Chain-of-Thought,思维链) 可达数万 token |
| **RAG Agent(检索增强型 Agent)**            | Perplexity、企业知识库      | 思考 + 检索 + 生成,本质是带工具(检索)的 Agent                |

### 3. 和普通LLM调用的区别

| 维度              | 普通 LLM 调用     | Agent 调用                                                   |
| ----------------- | ----------------- | ------------------------------------------------------------ |
| **Prompt 谁写的** | 用户手写一句      | 框架自动拼:目标 + 工具列表 + 历史思考 + 工具返回 + "请继续"  |
| **输出怎么用**    | 直接展示给用户    | **程序解析**输出,可能是 `{"action": "search", ...}` → 触发外部行为 |
| **何时停止**      | 模型输出 EOS(End of Sequence,序列结束符) 就完 | **程序判断**:tool call就继续,final answer 才停,超过最大轮数强制终止 |

### 4. Agent 给推理系统带来的 4 个新问题

1.**多轮交互**:一次用户任务 = 多次 LLM 调用,trajectory 很长。FCFS 调度按"请求"排队会让短任务被长任务卡死(HoL blocking)。

2.**工具阻塞**:LLM 输出 tool call 后,要等外部 API/数据库返回,这中间几秒到几十秒,KV Cache 占着 GPU 不能释放,GPU 算力闲置。

3.**大量共享前缀**:同一 Agent 的所有调用都共享 system prompt + tool 定义 + 历史轨迹;多个 Agent 之间也共享。重复 prefill 浪费巨大。

4.**依赖图复杂**:多 Agent 协作不是单链,而是 DAG(有的节点能并发,有的必须等);单请求批处理调度无法表达这种依赖。

#### 详细解释

##### 问题 1:多轮交互 → trajectory 级队头阻塞

**现象**

一次用户任务 ≠ 一次 LLM 调用。一个 ReAct Agent 任务("帮我订下周三去上海的机票,预算 2000 以内")的真实生命周期是:

```
LLM call 1 (思考: 我要先查航班) → tool: 调机票 API
LLM call 2 (看结果: 这趟太贵) → tool: 换日期重查
LLM call 3 (看结果: 这趟可以) → tool: 查酒店
LLM call 4 (整合信息) → tool: 下单
LLM call 5 (确认结果输出给用户)
```

5 次 LLM 调用 + 4 次 tool 调用,**端到端可能 30-60 秒**。复杂任务(写代码、做研究)能到几十次 LLM call。

**为什么传统系统应付不了**

vLLM 的 continuous batching 调度器看到的是"一堆独立的 HTTP 请求",**不知道哪些请求属于"同一个任务的多步"**。它对一次请求的认知是:进来 → prefill → decode → 出 EOS → 走人。

调度器**无法区分**:

- 这是一个刚来的 1-step 简单任务(用户问天气,下一步就是 final answer)
- 还是一个已经跑到第 8 步、还要再跑 5 步的长任务

在 FCFS(先来先服务)下,**先到的长任务会一直排在前面**,后到的短任务只能等。

**实际后果:trajectory 级队头阻塞**

类比:高速收费站 ETC 通道里,前面一辆车要数 100 张零钱,后面 50 辆只要刷一下卡的车全堵在后面。

具体场景:

- 一个 Coding Agent 跑了 10 分钟还没完,占着 KV / decode slot
- 这期间一个简单的"翻译这句话"请求进来,本来 200ms 能搞定
- 结果它被排在那个 Coding Agent 后面,**端到端 P99 延迟被拉到分钟级**

> **本质**:调度器以"请求"为单位,但用户体验以"trajectory"为单位 — 粒度错配。

##### 问题 2:工具阻塞 → GPU 显存被占,算力闲置

**现象**

LLM 输出 `{"action": "search", "query": "上海天气"}` 后,需要:

- 客户端解析这个 tool call
- 真的去调外部 API(搜索引擎、数据库、HTTP 接口、Python 解释器)
- 拿到返回结果
- 再发回给 LLM 让它继续推理

中间这段 **tool 执行时间** 是 LLM 完全不参与的,但又**不能直接结束这条请求** — 因为下一步 LLM 还要看历史。

时间尺度对比:

- LLM decode 一段 tool call 参数:**几十毫秒**
- 调一个搜索 API:**500ms - 3s**
- 跑一段 Python 代码:**几秒到几十秒**
- 调慢的外部服务(OCR、深度爬虫):**几十秒**

**为什么传统系统应付不了**

vLLM 设计假设是"请求来了一直 decode 直到 EOS"。tool call 这种"中间停下来等外部"的语义在 vLLM 模型里**根本不存在**。两种处理方式都不好:

方式 A:把这次 LLM call 当 finish,KV 释放

- tool 返回后下次 call 重新 prefill 整段历史
- Agent 后期 prompt 动辄几万 token,**重新 prefill 就是几秒算力白烧**

方式 B:保留 KV 等 tool 返回

- KV 占着几百 MB ~ 几 GB 显存
- 但这条请求不在 batch 里,**GPU 算力对它而言完全闲置**

**实际后果:GPU 利用率断崖**

一台 H100 80GB 在 chatbot 场景能并发跑 64 个请求,**算力和显存同步打满**。换成 Agent 场景:

- 一半的"活跃"trajectory 其实在等 tool 返回
- 这些 trajectory 占着 40GB 显存,**一个 token 都没在生成**
- 真正在 decode 的只有另外一半 → **有效吞吐直接腰斩**

> **本质**:原本紧耦合的"显存占用 = 算力占用"在 Agent 场景下解耦了 — KV 占着资源,CUDA core 却没事干。

##### 问题 3:大量共享前缀 → 重复 prefill 浪费

**现象**

Agent 的 prompt 不是用户手写的一句话,而是框架自动拼出来的一大段:

```
[System Prompt: 你是一个 helpful agent...]              ← 500 token
[Tool 定义: search(query), calculator(expr), ...]        ← 1500 token
[Few-shot 示例: 比如用户问 X,你应该这样思考...]         ← 2000 token
[历史轨迹: step 1 思考 + tool 结果 + step 2 思考 + ...]  ← 5000-50000 token
[当前用户输入]                                            ← 50 token
```

**前面 90% 都是共享的**,只有最后一段是新的。

共享层级:

- **同一个 Agent 内多步**:每一步只在末尾追加一点新内容
- **同一个用户的多个 Agent**:共用同一套 system prompt + tool 定义
- **不同用户的同类 Agent**:共用 system prompt + tool 定义 + few-shot

**为什么传统系统应付不了**

vLLM 的 `--enable-prefix-caching`(基于 RadixAttention)确实做了一定复用,但**两个限制**让它在 Agent 场景效果打折:

a) 只能复用前缀完全相同的部分
跨 Agent 共享 system prompt 没问题,但**RAG 拼上来的多个 chunk 顺序变一变**就匹配不上。CacheBlend 这类工作正是为此而生。

b) prefix cache 抖动严重
同一 trajectory 的 step 5 想复用 step 1-4 的 KV,但中间隔着几秒 tool 时间。这期间显存压力大,**前缀 cache 可能已经被 LRU 踢掉**,下次只能重新 prefill。

**实际后果:算力浪费**

举个数字:

- Agent 后期一次 LLM call 的 prompt 是 30000 token,新增部分只有 200 token
- 不复用:**prefill 30000 token → ~600ms 算力**
- 完美复用:**prefill 200 token → ~5ms 算力**
- **差 100 倍**

如果 prefix cache 命中率从 95% 掉到 60%,**整个集群的 prefill 算力就要多花 8 倍** — 这不是优化问题,是**资源浪费问题**。

> **本质**:Agent 的 prompt 高度结构化重复,KV 复用应该是默认行为,但简单的 prefix-only LRU 抓不住所有复用机会。

##### 问题 4:依赖图复杂 → 单请求批处理表达不了 DAG

**现象**

单 Agent(ReAct)是**单链**:step 1 → step 2 → step 3,前一步必须出完才能启动后一步。

## 多Agent调用怎么优化、怎么做实验、系统怎么设计

但多 Agent 系统(MetaGPT、AutoGen、CrewAI)是 **DAG**(有向无环图):

```
        ┌→ Worker A (写后端代码)  ┐
Planner ┼→ Worker B (写前端代码)  ┼→ Reviewer (代码审查)
        └→ Worker C (写测试)      ┘
```

- Planner 必须先跑完
- 三个 Worker 可以**并发**
- Reviewer 必须等三个 Worker 都完成

LLMCompiler 这类 DAG-style Agent 更夸张,可能一次产出 **10+ 个独立的 tool call**,本来都能并发。

**为什么传统系统应付不了**

vLLM / TGI 这类引擎只接收 **HTTP 请求队列**,每个请求是独立的。所以:

a) 看不出"这三个请求是并发的"
用户的 SDK 把 3 个 worker 请求并发发出去了,但**到了 vLLM 这边它们和别人的请求混在一起 FCFS 排队**。如果 GPU 有空,本来 3 个能同时跑,结果被插到不同 batch 里,串行化掉。

b) 看不出"Reviewer 在等谁"
Reviewer 这条请求其实需要等三个 Worker 的输出拼好才能发,这段等待时间在前端发生,vLLM 完全不知道。**关键路径分析做不了** — 本该优先跑 Planner(卡住所有人),但调度器分不清哪个是关键节点。

c) 看不出"哪些请求共享同一段 KV"
3 个 Worker 共用 Planner 的输出当 prompt 前缀,本来 prefill 一次就够。但 3 个请求可能落到不同的 GPU / 不同时间,**前缀复用机会随机化**。

**实际后果:并发度损失 + 关键路径错排**

- 本来 DAG 的最长路径是"Planner + 1 个 Worker + Reviewer" ≈ 30s
- 实际跑成串行 → "Planner + 3 个 Worker 串行 + Reviewer" ≈ 70s
- **端到端延迟翻倍以上,但 GPU 利用率反而下降**(本来能并发的没并发)

> **本质**:Agent 是个**程序**(有依赖、有并发、有共享),但传统推理引擎只看到**一堆独立请求** — 这是 API 层的信息丢失,不是引擎本身的能力问题。

##### 五个问题之间的关系

这五个问题不是独立的,它们其实指向**同一个根本矛盾**:

> **传统推理系统假设"请求是独立的、短的、无状态的",但 Agent 场景下请求是"有依赖的、长的、有跨调用状态的"。**

| 问题 | 暴露的假设错误 |
| ---- | -------------- |
| 1. 多轮交互 | 假设"请求 = 任务" → 错,任务是 trajectory |
| 2. 工具阻塞 | 假设"请求一直在 decode" → 错,会停下来等 |
| 3. 思考很长 | 假设"输出长度方差不大" → 错,10× 方差 |
| 4. 共享前缀 | 假设"请求 prompt 各不相同" → 错,90% 重复 |
| 5. 依赖图复杂 | 假设"请求互相独立" → 错,是 DAG |

旧假设全部失效,所以需要新的调度抽象 — 这就是后面"二、针对 Agent 场景的优化"五大类工作的共同出发点。


## 二、针对 Agent 场景的优化

### 类别 1:Trajectory / Application-aware 调度 — Agent 是"程序",不是"请求"

**它解的痛点**:传统的 LLM 推理系统（比如 vLLM、TGI）把每一次“发 prompt 收回答”当成一个独立的、没有关联的请求，用先来先服务（FCFS）或简单的优先级去排队。但在 Agent 场景下，一个用户任务会拆成几十甚至上百次 LLM 调用（多轮 reasoning + 多次 tool call），如果还当独立请求处理，就会出三个问题：

1. **短任务被长任务卡死（Head‑of‑Line blocking）**
   想象一个长 Agent 任务（比如要生成 2000 token）先排进队列，它后面跟着 100 个只需生成 20 token 的短请求。FCFS 会让长任务一直霸占 GPU，短请求虽然很快就能跑完，却只能干等。这叫“车队头阻塞”。
2. **Tool 切换时 KV 失序**
   Agent 在调用工具前后，可能会多次进入 LLM（比如 tool call 前生成参数，拿到工具结果后再继续推理）。如果这些 LLM 调用被不同批次随机调度，它们共享的 KV Cache 可能还没准备好或者已被踢出，导致重复计算甚至错误。
3. **不同 SLO（服务等级目标）混跑互相干扰**
   有的 Agent 任务要求低延迟（比如 100ms），有的可以接受高延迟（比如 10s）。传统调度不区分，同一个 batch 里混在一起，慢任务拖累快任务，谁的 SLO 都难保证。

**核心思路**：不再把调度单位看作“单次 LLM 请求”，而是升级成 **一条完整的 Agent 执行轨迹（trajectory）**。调度器可以做出更聪明的决策，比如：

- **允许短轨迹“插队”**：如果后面来的轨迹很短，可以绕过正在排队的长轨迹先执行（类似 Shortest Remaining Processing Time 调度）。
- **为重要轨迹预留 GPU 时间片**：确保高 SLO 任务不被干扰。
- **在 tool 等待期间主动换出 KV**：把 GPU 腾给其他轨迹。

| 论文                          | 视角              | 核心贡献                                                     |
| ----------------------------- | ----------------- | ------------------------------------------------------------ |
| **Parrot**                    | DAG-aware(有向无环图感知) 调度    | 用 Semantic Variable(语义变量) 把多个 LLM 请求还原成有数据流的 DAG(有向无环图),跨请求共享 prefix(前缀),优化的目标是"整条任务的端到端延迟"而不是单次请求的 TTFT |
| **Conveyor**                  | tool中断时调度   | 检测到 LLM 输出 tool call时主动决定是否换出 KV、是否调度其他请求,不再被动等 tool 返回 |
| **Autellix**                  | Trajectory 调度   | 把 Agent 视作"通用程序",感知多轮依赖与 thinking-tool(思考-工具) 切换,以 trajectory为单位做 SRTF(Shortest Remaining Time First,剩余最短时间优先) 类调度,实测 4-15× 优于 vLLM |
| **Tempo**                     | 应用感知调度      | 同一集群同时跑聊天/Agent/推理流多种负载时,按各自 SLO(服务等级目标) 做 Goodput(有效吞吐) 优化,避免互相干扰 |
| **Concur**                    | 多 Agent 准入控制 | 在入口判断 Agent 是否值得跑(能否在 SLO 内完成),提前拒掉无效 Agent,避免占住 KV / 算力 |
| **Hive**                      | 多 Agent 横向扩展 | 算法层(Agent 数量)和任务层(每个 Agent 的请求)双层弹性扩缩 |
| **Mixture-of-Agents serving(Agent 混合架构服务)** | MoA(Mixture-of-Agents,Agent 混合) 架构          | MoA 天然有 fan-out 拓扑,做 tree-routing(树形路由) + 依赖感知的 prefill-decode overlap(预填充-解码阶段重叠) |

**演进路线**:FCFS / 简单优先级 → 长度感知 → 应用感知 → trajectory 感知 → 多 Agent 准入。



### 类别 2:Tool Call 阻塞 — GPU 不能空等

**它解的痛点**:Agent 输出 tool call 后,LLM 在等外部 API/数据库返回,这段时间 KV 占着 GPU、算力闲置 — Agent 场景最大的"GPU 浪费源"。

#### 详细解释

**为什么 tool call 是个大问题**

一次 Agent step 的真实生命周期是:

```
[LLM prefill] → [LLM decode 出 tool 参数] → [tool 执行(外部)] → [LLM 看 tool 结果再 decode]
   毫秒级           几十毫秒到几秒              几百毫秒到几十秒        毫秒到几秒
```

中间那个 tool 执行段(查数据库、调搜索 API、跑 Python),时间可能比 LLM 自己 decode 还长。在这段时间里:

- 这条请求的 **KV Cache 还占着 GPU 显存**(因为下一步 LLM call 还要用这段历史)
- 但 **GPU 算力是闲的**(LLM 没在算这条请求)
- vLLM 默认行为是要么把它当 finish 释放(下次重新 prefill,白干),要么死等(算力浪费)

一台 H100 如果一半显存被"在等 tool"的 trajectory 占着,有效吞吐直接腰斩。

**两条互补思路**

- **抢时间**:不要等 tool 完全跑完再启动下一步,让 tool 和 LLM 在时间轴上重叠
- **省资源**:tool 等的这段时间,把 KV 从 GPU 挪走,让出显存给别的请求

| 论文                                            | 思路                                                                                                                          | 重叠方向                   |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| **LLMCompiler**                                 | 让 planner(规划器) 一次性产出 N 个 tool call的 DAG，看清哪些 tool 之间无依赖,这些 tool 直接并发执行,不串行排队                      | tool ↔ tool(工具间) 并行           |
| **Conveyor**                                    | LLM 还在 decode 输出 tool 参数(JSON 还没出完整),只要参数前缀已经够用,就让 tool 做 **partial execution(部分执行)**,边输出边跑              | decode ↔ tool(解码与工具) 重叠         |
| **APIServe / InferCept**                        | tool 等待期间把这条请求的 KV(Key-Value Cache,键值缓存) 从 GPU 换出到 CPU/SSD,显存让给其他正在 decode 的请求,tool 返回时再换回来                          | KV ↔ GPU(显存与算力) 解耦              |
| **Parallel function calling(并行函数调用)**(OpenAI/Anthropic) | 在模型训练/推理协议层支持一次输出多个 tool call,客户端 SDK(软件开发工具包) 直接并发执行(已大规模落地)                                          | tool ↔ tool 并行(工业落地) |

**与基于输出长度预测的调度的接口**:tool 边界是输出长度预测的天然子问题 — 预测"这次 decode 大概什么时候输出 tool-call 参数"、"tool 大概多久回来",可以指导 prefill/decode 调度与 KV 换入换出的时机(KV 换出的开销不能比 tool 等待时间还长)。

### 类别 3:KV Cache × Agent 生命周期 — 跨轮、跨 Agent、跨时间的 KV 管理

**它解的痛点**:Agent 场景下 KV Cache 不再是"用完即弃",而是"可能下一轮还要用"、"另一个 Agent 也要用"、"Tool 期间得让位"。传统 LRU/简单驱逐不够。

#### 详细解释

**普通 chatbot 里 KV 是什么**

一次 chat 请求来了 → 算 KV → 边 decode 边用 → 出 EOS → KV 直接丢。生命周期就是单次请求,KV 是纯粹的"中间产物",用 LRU 管够用。

**Agent 场景下 KV 变成了什么**

- **跨轮复用**:同一 trajectory 的 step 5 要用 step 1-4 的 KV(同一历史的扩展)
- **跨 Agent 共享**:多个 Agent 共用 system prompt + tool 定义 + 少样本示例,这部分 KV 完全相同
- **跨时间存活**:用户在辅助工具里思考 30 秒,这条会话的 KV 是该保留还是踢掉?保留浪费显存,踢掉下次重新 prefill 几万 token

也就是说,KV 不再是"瞬时中间产物",而是**带生命周期、有归属、有共享关系的资源**。LRU 这种"最近没用就踢"的策略,在 Agent 场景下会做出大量错误决策(踢掉马上要用的、留着永远不再用的)。

**核心思路**:把 KV 从"缓存"提升为"调度变量",带 TTL(还能活多久)、共享关系(哪些请求共享同一段)、复用模式(只能前缀复用还是任意位置)。

| 论文                       | 视角                       | 关键贡献                                                                                                                                                |
| -------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RadixAttention(基数树注意力,SGLang)** | 前缀共享                   | 用 Radix Tree(基数树,一种压缩前缀树) 把所有活跃请求的 prompt 索引起来,自动找出最长公共前缀并复用 KV(键值缓存);这就是 vLLM `--enable-prefix-caching(启用前缀缓存)` 的思想来源                            |
| **ChunkAttention(分块注意力)** 🟡       | 前缀感知 kernel(算子内核)            | 在 attention(注意力) kernel 层就识别"哪些 token是共享前缀",在高共享率(多 Agent 共用 system prompt,系统提示词)场景下 attention 提速 3.2-4.8×                              |
| **CacheBlend** 🟡           | 突破"只能前缀复用"限制     | RAG(Retrieval-Augmented Generation,检索增强生成) 里多个独立 chunk(分片) 不是前缀关系也想复用 KV;CacheBlend 直接拼接独立 chunk 的 KV,只对受拼接影响的关键 token 选择性 recompute(重计算),避免全量重算                |
| **Tokencake**              | 多 Agent 中心化 KV serving(服务) | 把 KV 抽象成一个独立服务,识别多个 Agent 实例间的共享段,按"还会被几次访问 / 多久后访问"分层管理,而不是按 LRU(Least Recently Used,最近最少使用)                                              |
| **Continuum**              | KV TTL(Time To Live,存活时间) × 调度              | 估计 KV 的 TTL = 用户思考间隔 + 下一轮预计输出长度,据此决定保留还是换出;调度器据 TTL 排序,把"快被命中"的 KV 优先留 GPU                                  |

**演进路线**:前缀完全相同才能共享 → 非前缀位置选择性复用 → 跨 Agent 跨时间生命周期管理。

**与基于输出长度预测的调度最近**:Continuum 的 TTL 估计本质是"输出长度 + 用户行为间隔"的组合预测,**输出长度预测就是它的核心子组件**。在 vLLM 上做 KV TTL 感知的驱逐策略(把 LRU 换成"按预测剩余命中时间排序"),Goodput 收益最直观。

### 类别 4:语义暴露 — 让后端"看懂" Agent 的依赖与共享

**它解的痛点**:Agent 是个程序(有依赖、并发、共享),但传统推理引擎只看到一堆独立请求,没法做跨请求优化。

#### 详细解释

**信息不对称的代价**

LangChain/AutoGPT 这类框架在前端把 Agent 拼成一个有结构的程序:哪些 LLM 调用是顺序依赖的、哪些可以并发、哪几个请求共用一段 prompt。但在 HTTP 请求发到 vLLM 的那一刻,**所有这些结构信息都丢了**,后端只看到 N 个独立的 `/v1/completions` 请求。后果:

- 共享 prefix 看不出来(明明三个请求 system prompt 完全相同),只能靠后端自己做前缀匹配,匹配失败就重复 prefill
- 能并发的请求被串行(planner → 3 个 worker 本来 worker 可以并发,但前端按顺序发,后端不知道它们独立)
- 整条任务的"端到端 deadline"完全不可见,后端只能优化单请求 TTFT

**核心思路**:在前端 / 中间层把 Agent 的语义结构(DAG / Semantic Variable / primitive 节点)显式暴露给后端,让后端不再做"盲调度"。

| 论文            | 暴露什么                                                                                                                | 后端利用什么                                                                                                                |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **SGLang**      | 提供 DSL(Domain-Specific Language,领域特定语言),在前端代码里就把"哪段是共享 prefix(前缀)、哪里是分支、哪里是循环"写出来                                              | 后端 Radix Tree(基数树) 据此自动 KV复用,共享 prefix 只 prefill(预填充,首次计算 KV) 一次                                                                |
| **Parrot**      | 引入 Semantic Variable(语义变量):把 prompt中"输入变量""输出变量"标记出来,变量之间形成数据流                                       | 后端能还原出 DAG(有向无环图),做 DAG-aware batching(DAG 感知的批处理)、跨请求 prefix 共享、按整条任务延迟优化                                              |
| **Teola**       | 把 RAG(检索增强生成)/Agent 流程拆到 primitive(原语) 粒度(embedding/retrieval/prefill/decode 即嵌入/检索/预填充/解码 都是 DAG 的节点)                                | 算子粒度并行 / 流水(retrieval 还在跑时下一段 prefill 已经准备 KV)                                                           |
| **ALTO**        | 在 stage之间用 token流式触发(上一 stage 一边出 token,下一 stage 一边消费)                                             | 跨 Stage 流水,不必等上一 stage 完全结束才启动下一 stage                                                                     |
| **LLMCompiler(LLM 编译器)** | Planner(规划器) 直接把任务编译成函数调用 DAG                                                                     | 独立 tool call自动并发执行,无依赖的 LLM call 自动并行                                                                      |

**共同手法**:把"黑盒请求"打开成"可分析的程序",再做编译器/调度器式的优化。

**当前局限**:DAG 暴露了,但**调度策略本身仍是经验规则**(谁先跑、谁抢占、KV 谁留)。这就是基于长度预测的调度能切入的地方 — 把基于长度预测的 SRPT 调度嵌进 Parrot 的 DAG 调度器,让 DAG 节点的执行顺序由"剩余工作量预测"驱动。