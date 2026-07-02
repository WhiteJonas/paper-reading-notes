# Agent 推理加速综述 — 论文是否针对 Agent 场景的分类讨论

> 配套文档:`Agent 推理加速综述.md`
> 整理时间:2026-05-26
> 目的:先用大白话讲清"Agent 是什么",再把综述里的论文按"是否专门针对 Agent 场景"分类,帮助快速定位哪些工作真的在解决 Agent 的痛点,哪些只是被借用过来的通用优化。

---

## 一、先搞懂 Agent 是什么(给系统/调度视角的人看)

### 1. 一句话定义

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

| 形态 | 典型例子 | 特点 |
|------|---------|------|
| **ReAct(单 Agent 循环)** | LangChain、AutoGPT | 一个 LLM 反复"思考→工具→观察",最经典 |
| **多 Agent 协作** | MetaGPT、AutoGen、CrewAI | 多个 LLM 扮演不同角色互相对话(产品经理 / 程序员 / 测试) |
| **DAG-style(规划+执行)** | LLMCompiler | 先规划成函数调用图,再并发执行多个 tool |
| **Reasoning Agent** | OpenAI o1、DeepSeek R1、QwQ | "思考过程"本身就很长,不一定调外部工具,但内部 CoT 可达数万 token |
| **RAG Agent** | Perplexity、企业知识库 | 思考 + 检索 + 生成,本质是带工具(检索)的 Agent |

### 3. 一个常见困惑:Agent 本质上不就是大模型反复回复 prompt 吗?

**对,本质上是这样**,但这个"反复"带来了三件全新的事,是把 Agent 推上独立研究方向的根源。

**本质相同的部分**:每一次 LLM 调用,底层确实就是

```
prompt(包含历史)→ Transformer forward → 输出 token 序列
```
模型本身没有任何变化,Agent 不是新模型,就是同一个 LLM 被反复调用。

**但有三件不一样的事**:

| 维度 | 普通 LLM 调用 | Agent 调用 |
|------|--------------|-----------|
| **Prompt 谁写的** | 用户手写一句 | 框架自动拼:目标 + 工具列表 + 历史思考 + 工具返回 + "请继续" |
| **输出怎么用** | 直接展示给用户 | **程序解析**输出,可能是 `{"action": "search", ...}` → 触发外部行为 |
| **何时停止** | 模型输出 EOS 就完 | **程序判断**:tool call 就继续,final answer 才停,超过最大轮数强制终止 |

一个类比:把 LLM 想成一颗 CPU
- **普通调用** = 跑一条指令
- **Agent** = 写了循环 + 条件分支 + 系统调用的程序,这颗 CPU 在程序里被反复调用

**LLM 是 CPU 没变,变的是外面套的"控制流"**。

→ 所以 Agent 推理加速 = 加速这两层
1. **每次单独的 LLM call**:还是 prefill + decode,通用优化(KV Cache / SD / PagedAttention)都能用
2. **整个 trajectory(多次 call 串起来)**:Agent 独有的加速空间 — 跨 call 共享 KV、tool 期间换出 KV、多 tool 并发、trajectory 级调度

下面 5 个新问题,都是因为"prompt 自动拼 / 输出被解析 / 循环由程序控"这三件事衍生出来的。

### 4. Agent 给推理系统带来的 5 个新问题(为什么单请求优化不够)

这是理解"Agent 推理加速为什么是独立研究方向"的核心:

1. ### **多轮交互**:一次用户任务 = 多次 LLM 调用,trajectory 很长。FCFS 调度按"请求"排队会让短任务被长任务卡死(HoL blocking)。
   
   - 根源:循环由程序控,一个用户任务被拆成几次到几十次 LLM call
   
   **澄清:一次用户任务 = 一个请求,还是多个请求?**
   
   答案分两层:**底层被拆成多个独立请求,但语义上是一个 trajectory** — 这正是 HoL 的根源。
   
   - **vLLM 视角**:Agent 框架(LangChain / AutoGen)每次 LLM call 都通过 HTTP 重发一个新请求,vLLM 收到的就是一个全新的 `request_id`,跟其他用户请求混在一起 batching。引擎**不知道** request_1 和 request_2 来自同一个用户任务、它们之间有时间间隔(等 tool 返回)、request_2 的 prompt 包含 request_1 的输出。
   
   - **用户视角**:这 N 个请求逻辑上是一个任务,有一个端到端 latency、一个总 SLO、一份共享上下文。
   
   **HoL 的形成场景**:用户 A 是简单任务(1 次 call),用户 B 是复杂 Agent(10 次 call)。vLLM 用 FCFS 看到的请求流是 `B_1, A_1, B_2, B_3, ..., B_10` 交错排队,A 不只是等 B 一次 call,而是被夹在 B 的整个轨迹里反复排队 ——— 端到端体验崩塌。
   
   **Agent-aware 调度的解法**:Autellix / Parrot / Tempo 让引擎知道哪些请求属于同一个 trajectory(框架带 `trajectory_id`,或调度器直接接管 Agent 循环),然后做:
   
   - **trajectory 级排队**:把"剩余总长度"作为优先级(SRPT 扩展到 trajectory)
   - **trajectory 级抢占**:短 trajectory 可抢长 trajectory 资源
   - **trajectory 级 KV 复用**:同 trajectory 的多次 call 之间 KV 不释放
   - **trajectory 级 SLO**:按端到端任务延迟算,而不是单次 call 延迟
   
   → 这也是你方向"基于输出长度预测的调度"扩展到 trajectory 级的真正含义:**预测对象从"单次请求剩余 token 数"变成"整个 trajectory 剩余 token 数"**(剩余轮数 × 每轮 token)。
   
2. ### **工具阻塞**:LLM 输出 tool call 后,要等外部 API/数据库返回,这中间几秒到几十秒,**KV Cache 占着 GPU 不能释放**,GPU 算力闲置。
   
   - 根源:输出被程序解析成"指令",触发了 GPU 之外的外部行为
   
   **澄清:外部 API / 数据库具体是干啥的?**
   
   LLM 本身只会"预测下一个 token",有几个根本短板必须靠外部工具补:知识过期(查实时信息)、不会真做事(订机票/下订单)、算数不可靠(数值计算)、看不到私有数据(企业数据库)、不能执行代码(跑 Python)。**Agent 让 LLM 输出"工具调用指令",框架去执行,把结果喂回 LLM** — 这就是 tool call。
   
   **常见的"外部工具"按调用频率排**:
   
   | 工具类型 | 干啥的 | 典型例子 | 耗时 |
   |---------|-------|---------|------|
   | **搜索引擎 API** | 实时信息查询 | Google / Bing / Perplexity | 0.5-3 秒 |
   | **数据库查询** | 查私有数据 | MySQL、Postgres、向量库(Milvus / Pinecone)| 50ms-2 秒 |
   | **代码执行器** | 跑 Python 算东西 | Jupyter sandbox、Python REPL | 0.1-30 秒 |
   | **业务 API** | 真实交易/操作 | 订机票、下订单、发邮件、调内部系统 | 1-10 秒 |
   | **文件/文档读取** | 读 PDF/Word/Excel | unstructured、PDF parser | 1-10 秒 |
   | **第三方服务** | 地图/天气/汇率/翻译 | 高德、OpenWeather、DeepL | 0.3-2 秒 |
   | **多模态生成** | 画图/TTS/视频 | DALL-E、Midjourney、ElevenLabs | 5-60 秒 |
   | **另一个 LLM** | 子 Agent / 小模型 | 调 GPT-4 做 review、调小模型分类 | 1-30 秒 |
   
   **一个具体例子**:用户问"明天北京天气怎么样,适合穿什么?"
   ```
   Round 1:LLM 输出 {"action": "get_weather", "args": {"city": "北京"}}
            ↓ 框架解析 → 调天气 API
            ↓ 【1 秒,GPU 在等,KV 占着】← 工具阻塞
            ↓ API 返回 {"temp": "5-12℃", "weather": "小雨"}
   Round 2:prompt 拼上 API 结果 → LLM 输出最终答复 → 任务结束
   ```
   
   **为什么对调度器是大问题**:
   
   | 单请求场景 | Agent 工具阻塞场景 |
   |-----------|------------------|
   | 一直在 decode,GPU 持续算 | tool call 后**挂起**,GPU 0 利用率 |
   | KV 一直被用 | KV 占着没用,**纯浪费** |
   | 调度器知道何时结束 | 调度器**不知 tool 何时回来**,不敢释放 KV |
   
   一个 Agent 任务常有 5-10 次 tool call,累计下来**一个 trajectory 一半时间在等外部工具** — GPU 在它身上干等。
   
   **综述第 3 类(Tool Call 阻塞)三篇论文怎么解**:
   - **LLMCompiler**:让 LLM 一次输出多个独立 tool call,**并发执行**(5 次串行 × 1s → 5 次并行 1s)
   - **Conveyor**:LLM 还在输出 tool 参数时,外部工具就开始 **partial execution**,decode 和 tool 重叠
   - **APIServe / InferCept**:tool 等待期间把 KV **换出 CPU**,GPU 让给其他请求,回来再换回
   
3. ### **思考很长**:reasoning 模型一道题能输出 5000-30000 token,decode 阶段被严重拉长,KV Cache 持续增长。
   
   - 根源:Agent / Reasoning 把"想清楚"也做成了输出 token,而不是一步出答案
   
4. ### **大量共享前缀**:同一 Agent 的所有调用都共享 system prompt + tool 定义 + 历史轨迹;多个 Agent 之间也共享。重复 prefill 浪费巨大。
   
   - 根源:prompt 是程序拼出来的,模板高度重复(framework 每轮都拼一份相同的 system prompt + tool def)
   
5. ### **依赖图复杂**:多 Agent 协作不是单链,而是 DAG(有的节点能并发,有的必须等);单请求批处理调度无法表达这种依赖。
   
   - 根源:多 Agent / 多 tool call 之间存在程序级的依赖关系,不是简单的"一前一后"
   
   **澄清:DAG 是什么?**
   
   **DAG = Directed Acyclic Graph,有向无环图** — 描述"任务之间先后依赖关系"的图。
   - **Directed(有向)**:边有方向,A → B 表示"A 完成后才能做 B"
   - **Acyclic(无环)**:不能转圈回到自己(否则死锁)
   - **Graph(图)**:节点 + 边
   
   **生活类比 — 番茄炒蛋**:
   ```
   洗番茄 ─→ 切番茄 ─┐
                      ├→ 下锅炒 ─→ 出锅
   打鸡蛋 ─→ 炒蛋 ───┘
   ```
   "洗番茄"和"打鸡蛋"无依赖可并发,"下锅炒"必须等"切番茄"和"炒蛋"都完。这就是 DAG。
   
   **计算机里 DAG 用在哪**:Makefile 构建系统、Git 提交历史、Spark / Flink 数据流、Airflow 工作流、CPU 指令流水、神经网络计算图 — 凡是要描述"依赖关系"的地方都是 DAG。
   
   **Agent 为什么是 DAG**:
   - **单 Agent ReAct**(链状,DAG 的特例):`LLM_1 → tool_1 → LLM_2 → tool_2 → 输出`
   - **多 tool 并发**(典型 DAG,LLMCompiler 的目标场景):
     ```
     用户问"对比 iPhone 17 和 Mate 70 的价格 + 评分"
                   │
              LLM(规划)
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
     搜价格 1   搜评分 1   搜价格 2   搜评分 2  ← 4 个独立分支可并发
       │           │           │           │
       └───────────┴───────────┴───────────┘
                   ▼
              LLM(汇总)
     ```
   - **多 Agent 协作**(产品经理 → 前后端测试并行 → 集成 Agent 汇总):每个 Agent 是节点,Agent 间的输入输出是边
   
   **为什么 DAG 对推理调度重要 — vLLM 视角的对比**:
   ```
   传统 vLLM:[req1, req2, req3, req4, req5]   ← 一个平铺数组,不知谁依赖谁
   Agent 实际:req1 → req2 ─┐
                            ├→ req5            ← 有依赖 + 有并发
              req3 → req4 ─┘
   ```
   
   **调度器知道 DAG 后能做的优化**(对应综述里的论文):
   - **并发独立分支**:不在一条依赖链上的节点可并发(LLMCompiler)
   - **流水线触发**:上游节点 decode 中,下游节点的 prefill 可预热(Teola、ALTO)
   - **关键路径优先**:DAG 中最长路径决定端到端 latency,优先调度它
   - **跨节点 KV 共享**:同一 DAG 的节点共享 system prompt → 复用 KV(Parrot、SGLang)
   - **整体抢占**:整条 DAG 当一个调度单位,而不是一个个节点排队
   
   → 综述第 1 类"语义暴露"(SGLang / Parrot / Teola / ALTO / LLMCompiler)的核心动作就是**把 Agent 的 DAG 暴露给后端**,让推理引擎能做基于结构的优化,而不是把每个节点当独立请求黑盒处理。
   
   → 这也回答了之前的问题:**"一次用户任务 = 一个请求还是多个请求?"** 底层是 DAG 的多个节点(多个请求),语义上是一个 DAG(整体)。Agent-aware 调度的核心改动是**把调度对象从"单个请求"升级到"整个 DAG / trajectory"**。

→ 所以 Agent 推理加速 ≠ 单请求加速。它需要从**前端语义、KV 生命周期、tool 边界、trajectory 调度、多 Agent 准入**等多个维度重新设计系统。

---

## 二、论文是否针对 Agent 场景:速查表

按"原综述出现的论文 + 我的判断"打标签。三档:

- 🟢 **Agent-native**:论文动机/实验/系统设计明确围绕 Agent / 多轮 / tool / 多 Agent
- 🟡 **Agent-friendly**:不是为 Agent 设计,但对 Agent 场景有强适配性,综述把它纳入是合理的延伸应用
- ⚪ **通用 / 单请求优化**:论文本身不针对 Agent,只是 Agent 场景也能用

(单独分一档:🟣 **Reasoning-native**,即 reasoning 模型的"长思考",是 Agent 的近亲但不完全等同 — Reasoning 通常是单 LLM 的内部 CoT,不一定有 tool 循环)

| # | 论文 | 标签 | 判断依据 |
|---|------|------|---------|
| 1 | **SGLang / RadixAttention** (NeurIPS'24) | 🟢 | 论文动机就是多轮对话 / few-shot / Tree-of-Thought 的前缀共享,Agent 是核心 use case |
| 2 | **Parrot** (OSDI'24) | 🟢 | 标题就是"Efficient Serving of LLM-based Applications",Semantic Variable + DAG-aware 调度专为 Agent/Compound AI 设计 |
| 3 | **Teola** (arXiv 2407.00326) | 🟢 | 把 LLM 应用建模成 primitive-级 DAG,典型 Compound AI / Agent serving |
| 4 | **ALTO** (arXiv 2403.04311) | 🟢 | "Compound AI System 的网络编排器",目标就是 Agent / Pipeline |
| 5 | **ServerlessLLM** (OSDI'24) | ⚪ | 解决冷启动模型加载,通用 LLM serving,不是 Agent 专属(Agent 多模型场景下能用) |
| 6 | **LLMCompiler** (ICML'24) | 🟢 | ReAct 的替代品,直接面向 Agent 函数调用 DAG |
| 7 | **Conveyor** (arXiv 2406.00059) | 🟢 | "Tool-aware serving",目标就是 Agent 的 tool call 场景 |
| 8 | **APIServe / InferCept** (arXiv 2402.01869) | 🟢 | 把 API/tool 调用当作 interception,Agent 专属 |
| 9 | **Parallel function calling** (OpenAI/Anthropic) | 🟢 | 工业接口,直接服务 Agent |
| 10 | **ChunkAttention** (ACL'24) | 🟡 | 论文做的是 prefix-aware attention kernel,通用前缀加速;Agent 是高共享率的典型场景 |
| 11 | **CacheBlend** (EuroSys'25) | 🟡 | 论文动机是 RAG,非 Agent;但 RAG 是 Agent 的子模式,适配性强 |
| 12 | **Tokencake** (arXiv 2510.18586) | 🟢 | 标题/动机都是"多 Agent 应用的 KV serving",Agent-native |
| 13 | **Continuum** (arXiv 2511.02230) | 🟢 | KV TTL 概念直接来自"Agent 多轮交互间隔",Agent-native |
| 14 | **Medusa** (ICML'24) | ⚪ | 通用 decode 加速,不针对 Agent |
| 15 | **EAGLE / EAGLE-2 / EAGLE-3** | ⚪ | 通用推测解码,不针对 Agent |
| 16 | **Lookahead Decoding** (ICML'24) | ⚪ | 通用 SD,不针对 Agent |
| 17 | **SpecInfer / Sequoia** | ⚪ | 通用 tree-based SD,不针对 Agent |
| 18 | **Goodput-driven SD scheduling** (arXiv 2406.14066) | ⚪ | 通用 SD × 调度,可迁移到 Agent 但论文本身不针对 |
| 19 | **Autellix** (NSDI'26) | 🟢 | 标题就是"把 Agent 视作通用程序",Agent-native 的 SOTA 调度工作 |
| 20 | **Tempo** (arXiv 2504.20068) | 🟢 | 混合 SLO 显式包含 Agent / 推理流,Application-aware 调度 |
| 21 | **S-LoRA / Punica** (MLSys'24) | 🟡 | 论文做的是多 LoRA serving 的 batching kernel,通用;Agent 场景"每 Agent 一个 LoRA"是典型应用 |
| 22 | **Learning-to-Rank Scheduling** (arXiv 2408.15792) | ⚪ | 通用调度策略,不针对 Agent |
| 23 | **Proxy-model length prediction** (arXiv 2404.08509) | ⚪ | 通用单请求长度预测,不针对 Agent(但你方向最直接对标) |
| 24 | **SpecReason** (arXiv 2504.07891) | 🟣 | Reasoning 专属,long-CoT 推测解码 |
| 25 | **SSR** (arXiv 2505.15340) | 🟣 | Reasoning 专属,self-consistency 加速 |
| 26 | **LightThinker** (arXiv 2502.15589) | 🟣 | Reasoning 专属,思考过程压缩 |
| 27 | **O1-Pruner** (arXiv 2501.12570) | 🟣 | Reasoning 专属,o1-like 模型思考长度优化 |
| 28 | **ThinKV** (arXiv 2510.01290) | 🟣 | Reasoning 专属,thought-adaptive KV |
| 29 | **Reasoning Path Compression** (arXiv 2505.13866) | 🟣 | Reasoning 专属 |
| 30 | **Mixture-of-Agents serving** (arXiv 2512.18126) | 🟢 | MoA 多 Agent 架构,Agent-native |
| 31 | **Concur** (arXiv 2601.22705) | 🟢 | Agent-level admission control,Agent-native |
| 32 | **Hive** (arXiv 2604.17353) | 🟢 | 多 Agent 双层 scaling 基础设施 |

**统计**:
- 🟢 Agent-native:**16 篇**(SGLang、Parrot、Teola、ALTO、LLMCompiler、Conveyor、APIServe、Parallel FC、Tokencake、Continuum、Autellix、Tempo、MoA serving、Concur、Hive,以及 Parrot 在第 5 节重复)
- 🟡 Agent-friendly(被借用):**3 篇**(ChunkAttention、CacheBlend、S-LoRA/Punica)
- 🟣 Reasoning-native(Agent 近亲):**6 篇**(SpecReason、SSR、LightThinker、O1-Pruner、ThinKV、RPC)
- ⚪ 通用 / 单请求优化:**8 篇**(ServerlessLLM、Medusa、EAGLE、Lookahead、SpecInfer/Sequoia、Goodput-SD、L2R Scheduling、Proxy length prediction)

→ 综述里**约一半的论文是 Agent-native**,另一半是被纳入"Agent 视角下能用 / 必须改造才能用"的通用工作。这也呼应了综述结论:"Agent 推理加速是系统层 × 算法层 × 调度层的三维优化"。

---

## 三、针对 Agent 场景的优化:分类讨论

只看 🟢 + 🟡 这一批"真正在解 Agent 问题"的论文,可以按**"它解决 Agent 的哪个具体痛点"**分成 5 类。这个分类比原综述的 7 章更贴近"Agent 痛点视角"。

### 类别 1:语义暴露 — 让后端"看懂" Agent 的依赖与共享

**它解的痛点**:Agent 是个程序(有依赖、并发、共享),但传统推理引擎只看到一堆独立请求,没法做跨请求优化。

**核心思路**:在前端 / 中间层暴露 Agent 的语义结构(DAG / Semantic Variable / primitive 节点),让调度器据此做共享与并发。

| 论文 | 暴露什么 | 后端利用什么 |
|------|---------|------------|
| **SGLang** | 前端 DSL 标注共享 prefix 与控制流 | Radix Tree 自动 KV 复用 |
| **Parrot** | Semantic Variable(prompt 的输入/输出依赖)| DAG-aware batching、跨请求 prefix 共享 |
| **Teola** | Primitive-级 DAG(embedding/retrieval/prefill/decode 都是节点)| 算子粒度并行 / 流水 |
| **ALTO** | Stage 间 token 流式触发 | 跨 Stage 流水,减少等待 |
| **LLMCompiler** | Planner 编译成函数调用 DAG | 独立 tool call 并发执行 |

**共同手法**:把"黑盒请求"打开成"可分析的程序",再做编译器/调度器式的优化。

**当前局限**:DAG 暴露了,但**调度策略本身仍是经验规则**。这就是你方向能切入的地方 — 把基于长度预测的 SRPT 调度嵌进 Parrot 的 DAG 调度器。

---

### 类别 2:KV Cache × Agent 生命周期 — 跨轮、跨 Agent、跨时间的 KV 管理

**它解的痛点**:Agent 场景下 KV Cache 不再是"用完即弃",而是"可能下一轮还要用"、"另一个 Agent 也要用"、"Tool 期间得让位"。传统 LRU/简单驱逐不够。

**核心思路**:把 KV 从"缓存"提升为"调度变量",带生命周期、TTL、共享关系。

| 论文 | 视角 | 关键贡献 |
|------|-----|---------|
| **RadixAttention(SGLang)** | 前缀共享 | Radix Tree 自动复用,vLLM `--enable-prefix-caching` 由此启发 |
| **ChunkAttention** 🟡 | 前缀感知 kernel | 高共享率下 attention 提速 3.2-4.8× |
| **CacheBlend** 🟡 | 突破"只能前缀复用"限制 | RAG 拼多个独立 chunk 的 KV,选择性 recompute 重要 token |
| **Tokencake** | 多 Agent 中心化 KV serving | 识别 Agent 间共享 KV,按生命周期分层管理 |
| **Continuum** | KV TTL × 调度 | 根据 agent 多轮交互间隔预测 KV 何时再被命中 |

**演进路线**:前缀完全相同才能共享 → 非前缀位置选择性复用 → 跨 Agent 跨时间生命周期管理。

**与你方向最近**:Continuum 的 TTL 估计 = 用户思考间隔 + 输出长度,**输出长度预测就是它的核心子组件**。在 vLLM 上做 KV TTL 感知的驱逐策略,Goodput 收益最直观。

---

### 类别 3:Tool Call 阻塞 — GPU 不能空等

**它解的痛点**:Agent 输出 tool call 后,LLM 在等外部 API/数据库返回,这段时间 KV 占着 GPU、算力闲置 — Agent 场景最大的"GPU 浪费源"。

**核心思路**:让 tool 执行与 LLM decode 重叠,或让 KV 在 tool 等待期间让出 GPU。

| 论文 | 思路 | 重叠方向 |
|------|-----|---------|
| **LLMCompiler** | Planner 一次产出多 tool call,独立的并发执行 | tool ↔ tool 并行 |
| **Conveyor** | LLM 还在生成 tool 参数时,外部工具就开始 partial execution | decode ↔ tool 重叠 |
| **APIServe / InferCept** | Tool 等待时把 KV 换出 CPU,LLM 让出 GPU | KV ↔ GPU 解耦 |
| **Parallel function calling**(OpenAI/Anthropic) | 模型一次输出多 tool call,客户端并发执行 | tool ↔ tool 并行(工业落地) |

**两条互补路径**:
- **抢时间**:Conveyor 让 tool 提前跑,缩短关键路径
- **省资源**:APIServe 让 KV 让出 GPU,提高 GPU 在 tool 等待期间的利用率

**与你方向**:tool 边界是输出长度预测的天然子问题 — 预测"何时输出 tool-call 参数 / 何时调用结束",可指导 prefill/decode 调度与 KV 换入换出时机。

---

### 类别 4:Trajectory / Application-aware 调度 — Agent 是"程序",不是"请求"

**它解的痛点**:把 Agent 多轮 LLM 调用当独立请求扔进 FCFS batch → 短任务被长任务卡死 (HoL blocking)、tool 切换时 KV 失序、不同 SLO 混跑互相干扰。

**核心思路**:调度对象从"单次 LLM 请求"升级到"trajectory / 应用 / 多 Agent 系统",感知依赖、SLO、思考-工具切换。

| 论文 | 视角 | 核心贡献 |
|------|-----|---------|
| **Parrot** | DAG-aware 调度 | 跨请求共享 prefix + 端到端延迟优化 |
| **Conveyor** | tool 中断时调度 | tool 触发 KV swap 决策 |
| **Autellix** | Trajectory 调度 | 把 Agent 视作"通用程序",感知多轮依赖与 thinking-tool 切换,4-15× 优于 vLLM |
| **Tempo** | 应用感知调度 | 混合 SLO(聊天/Agent/推理流)的 Goodput 优化 |
| **S-LoRA / Punica** 🟡 | 多 LoRA 共存 | 每 Agent 一个 adapter 的统一 batching kernel |
| **Concur** | 多 Agent 准入控制 | 避免无效 Agent 占住 KV / 算力 |
| **Hive** | 多 Agent 横向扩展 | 算法-任务双层 scaling |
| **Mixture-of-Agents serving** | MoA 架构 | Tree-routing + 依赖感知 prefill-decode overlap |

**演进路线**:FCFS / 简单优先级 → 长度感知 → 应用感知 → trajectory 感知 → 多 Agent 准入。

**与你方向最强**:Autellix 是 2025 年 Agent 调度的 SOTA,但它的调度信号还没用上"剩余 trajectory 长度预测"。你的研究方向(基于输出长度预测的调度)从"单请求"扩到"trajectory 级",正好填上这个空白。

---

### 类别 5:Reasoning Agent 的长输出加速(Agent 的近亲)🟣

**它解的痛点**:o1 / R1 / QwQ 让"输出长度暴涨"成为新常态,一道题的 reasoning 轨迹 5000-30000 token,decode 完全主导,KV 持续增长。

**核心思路**:从"思考压缩(算法)"和"长 KV 承载(系统)"两条路下手。

| 论文 | 路径 | 做什么 |
|------|-----|-------|
| **O1-Pruner** | 算法 | Length-harmonizing fine-tune,准确率不降但缩短思考 |
| **LightThinker** | 算法 | 训练模型在思考过程中动态压缩中间 token |
| **Reasoning Path Compression** | 算法 | 压缩生成轨迹本身 |
| **ThinKV** | 系统 | Thought-adaptive KV 压缩 |
| **SpecReason** | SD | 小模型推测整段思考步骤,大模型只验证关键步 |
| **SSR** | SD | self-consistency 多采样的投机式并行剪枝 |

**严格来说 Reasoning ≠ Agent**(Reasoning 是单 LLM 的内部 CoT,不一定有 tool 循环),但综述把它放进来是合理的:**reasoning 是 Agent 思考阶段的常见形态**,且"输出长度暴涨"这一痛点完全相同。

**与你方向**:reasoning 模型的输出长度方差比 chatbot 大一个数量级(同一题不同推理路径长度差 10×+),是输出长度预测**最值得突破的场景**。当前所有 reasoning 工作都默认"先生成完整轨迹再压缩",**实时预测剩余思考长度并据此调度是空白**。

---

## 四、被借用的"通用优化":在 Agent 上要怎么改造

这部分的 ⚪ 论文(单请求 / 通用)不是为 Agent 设计的,但 Agent 场景能用 — 关键看怎么"改造"才能在 Agent 上发挥价值。

| 论文(原本目标) | Agent 场景下的改造方向 |
|-----------------|----------------------|
| **Medusa / EAGLE / Lookahead / SpecInfer**(单请求 SD)| 在 Agent 长输出 / reasoning 阶段开 SD,**draft 长度按"剩余输出长度预测"动态调** |
| **Goodput-driven SD scheduling**(通用 SD × 调度)| 与 Agent trajectory 结合:不同轮 / 不同任务用不同 draft 策略 |
| **Learning-to-Rank Scheduling**(通用调度)| 把"trajectory 剩余长度"作为 ranking 特征 |
| **Proxy-model length prediction**(单请求长度预测)| **直接对标你方向** — 扩展到 trajectory 级:剩余轮数 + 每轮 token |
| **ServerlessLLM**(模型冷启动)| 多 Agent 跨模型场景下,Agent 切换触发模型加载,locality-aware 加载有用 |

**结论**:通用优化在 Agent 上能用,但**几乎都需要叠加"trajectory 级信号"**才能发挥最大价值。这正是输出长度预测从单请求扩到 trajectory 的研究价值所在 — 它是多个上层决策(SD draft 大小、调度排序、KV TTL、admission control)的**共同子组件**。

---

## 五、一张图:论文分类总览

```
                       Agent 推理加速论文池
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
   🟢 Agent-native        🟡 Agent-friendly      ⚪ 通用 / 单请求
   (16 篇)                (3 篇)                 (8 篇)
       │                      │                      │
   ┌───┴────┐             ┌───┴───┐              ┌───┴────┐
   语义暴露   KV 生命周期   通用 KV  通用 LoRA     通用 SD   通用调度  长度预测
   SGLang    Tokencake    Chunk   S-LoRA        Medusa    L2R       Proxy LP
   Parrot    Continuum    Att.    Punica        EAGLE     Goodput   ServerlessLLM
   Teola                  Cache                  Lookahead SD
   ALTO     Tool 调度      Blend                 SpecInfer
            Conveyor                             Sequoia
   Trajectory APIServe
   调度       LLMCompiler                  ─────────────────────
   Autellix                                🟣 Reasoning-native (6 篇)
   Tempo                                   O1-Pruner / LightThinker
   多 Agent                                Reasoning Path Compression
   Concur / Hive / MoA serving             ThinKV / SpecReason / SSR
   Tokencake
```

---

## 六、给你方向的关键判断

把上面的分类映射到"基于输出长度预测的调度"研究方向,几个直接结论:

1. **Agent 场景的调度工作(Autellix / Tempo / Parrot)是你的对标论文**,但它们的调度策略**还没用上 trajectory 级长度预测** — 这是空白
2. **KV TTL × 长度预测(Continuum)是工程上最容易落地的方向**,因为 vLLM 的 prefix caching 已经在做相关基础设施
3. **Tool boundary 预测(Conveyor 之上)** 是输出长度预测的子问题,有现成的系统可以叠加
4. **Reasoning 场景(ThinKV / O1-Pruner)输出长度方差最大**,长度预测的红利也最大,适合做"算法 + 系统"双结合的 paper
5. **通用 SD 工作(EAGLE / Medusa)+ Agent trajectory 信号** = 一类新的研究范式,Goodput-driven SD 已经开了头,可继续推

→ 一句话:**输出长度预测从单请求扩到 trajectory 级,在 Agent 场景下能同时撬动 KV、SD、调度、admission 四个决策维度,这是当前最缺、最值得做的一件事。**



Agent本质上，还是大模型不断完成对prompt的回复？

● 对,本质上是的,但有几个关键区别让它不只是"反复回复 prompt":

  一、本质相同的部分

  每一次 LLM 调用,底层确实就是:

  prompt(包含历史)→ Transformer forward → 输出 token 序列

  模型本身没有任何变化,Agent 不是"新模型",就是同一个 LLM 被反复调用。

  二、不一样的三件事

  1. Prompt 是"自动拼出来的",不是用户写的

  ┌───────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │         普通调用          │                                     Agent 调用                                      │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 用户写一句 prompt →       │ 用户给个目标,框架自动拼 prompt:目标 + 工具列表 + 历史思考 + 工具返回结果 +          │
  │ 模型答                    │ "请继续思考"                                                                        │
  └───────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

  每一轮 LLM 调用的 prompt 都是程序拼出来的,而不是人写的。

  2. 输出会被"程序解析",触发外部行为

  普通调用:模型输出文字 → 直接显示给用户。

  Agent 调用:模型输出可能是
  {"action": "search", "query": "iPhone 17 价格"}
  → 框架解析这段输出 → 真的去调搜索 API → 把结果塞回下一轮 prompt。

  模型输出的不再只是"答案",而是"指令"。

  3. 何时停止由"程序判断",不是模型说停就停

  普通调用:模型输出 EOS,结束。

  Agent 调用:框架检查输出
  - 如果是 tool call → 执行工具,继续下一轮
  - 如果是 final answer → 结束
  - 如果超过最大轮数 → 强制结束

  模型只负责"想 + 说","循环控制"是外面的程序干的。

  三、一个类比

  把 LLM 想成一颗 CPU:
  - 普通调用 = 跑一条指令,出结果
  - Agent = 写了一个循环 + 条件分支 + 系统调用的程序,这颗 CPU 在程序里被反复调用

  LLM 是 CPU,Agent 框架(LangChain / AutoGen / 你自己写的循环)是程序。模型本身没变,变的是外面套的"控制流"。

  四、所以"Agent 推理加速"在加速什么

  既然每次 LLM 调用本质还是普通推理,那加速的对象就分两层:

  1. 每次单独的 LLM call:还是 prefill + decode,KV Cache、SD、PagedAttention 这些通用优化都能用
  2. 整个 trajectory(多次 call 串起来):这才是 Agent 独有的加速空间 — 跨 call 共享 KV、tool 期间换出 KV、多 tool

    并发、trajectory 级调度

  → 所以你之前看的综述里:
  - ⚪ 通用单请求优化 解决的是第 1 层
  - 🟢 Agent-native 工作 解决的是第 2 层

  一句话总结

  Agent = LLM(没变)+ 一个不停拼 prompt、解析输出、触发工具、循环调用的程序壳子。

  底层确实是"反复回复 prompt",但"prompt 怎么来 / 输出怎么用 / 何时停"全被程序接管了 — 这就是 Agent
  给推理系统带来新问题的根源。

---

## 七、每篇论文方法详解(按"五大类别"顺序)

每篇论文用"**问题 → 方法 → 亮点 → 与你方向连接**"四段式讲。顺序严格按第三章的五大类别:语义暴露 → KV 生命周期 → Tool 阻塞 → Trajectory 调度 → Reasoning 长输出。

> 通用单请求优化(Medusa / EAGLE / Lookahead / SpecInfer / Goodput-SD / L2R Scheduling / Proxy length prediction / ServerlessLLM)放在最后单列一节,因为它们不在五大类别里,但 Agent 场景能借用。

---

### 类别 1:语义暴露 — 让后端"看懂" Agent 的依赖与共享

这一类共 5 篇,都在解决同一个核心问题:**怎么把 Agent 的内部结构(DAG / 共享 prefix / 算子拓扑)告诉推理引擎**。每篇切入角度不同。

#### 1-① SGLang / RadixAttention 🟢(NeurIPS'24)

- **问题**:多轮对话 / few-shot / Tree-of-Thought 这些场景下,大量请求**共享前缀**(system prompt + 历史轮),vLLM 的 prefix cache 用 hash 比对,只能识别完全相同的 prefix,共享率低。
- **方法**:
  - **前端**:提供 SGLang DSL,程序员用 `fork`、`gen`、`select` 原语写 Agent 逻辑,前端自动暴露共享 prefix
  - **后端 RadixAttention**:用 **Radix Tree(基数树)** 组织所有请求的 KV Cache。共享前缀的 KV 自动落到同一个树节点,新请求来了沿树查找最长公共前缀,直接复用
  - **驱逐策略**:树节点带引用计数,LRU 驱逐叶子节点的 KV
- **亮点**:把"前缀共享"从"完全一样才能复用"提升到"任意位置任意结构都能自动检测"。vLLM 的 `--enable-prefix-caching` 直接受其启发。**目前是工业主流推理引擎之一**(xAI、字节内部部署)。
- **与你方向**:Radix Tree 是后续所有 Agent KV 工作(Tokencake、Continuum)的基础设施。你的长度预测可喂给 Radix Tree 的驱逐策略,做"剩余命中价值"感知的 LRU。

#### 1-② Parrot 🟢(OSDI'24)

- **问题**:LLM-based 应用(Agent / Compound AI)被拆成很多次独立 LLM call,后端把每次 call 当独立请求,**完全看不到 call 之间的依赖**(plan 是 LLM 1 输出又是 LLM 2 输入,这种关系丢失)。
- **方法**:
  - 引入 **Semantic Variable**(语义变量):程序员声明 `plan = SemanticVariable()`,把 LLM call 的输入输出标注成命名变量
  - 后端拿到所有请求 + 变量映射,自动构建出 **DAG**:谁是谁的输入输出
  - 调度器据此做三件事:
    - **DAG-aware batching**:同 DAG 节点优先编排,避免下游空等
    - **跨请求 prefix 共享**:同一变量被多请求引用 → KV 复用
    - **流式触发**:上游 decode 中,下游就开始 prefill
- **亮点**:端到端延迟降 **11.7×**(vs 把每个 call 当独立请求)。开创"语义可见的 LLM serving"范式,启发了 SGLang 后续工作和 vLLM v1 的接口设计。
- **与你方向**:Semantic Variable 暴露了依赖,但调度策略仍是规则式。**你的"trajectory 级长度预测"直接可以喂进 Parrot 的调度器,做 SRPT 排序** — 这是直接的延伸方向。

#### 1-③ Teola 🟢(arXiv 2407.00326)

- **问题**:LLM 应用不只 LLM call,还有 embedding、retrieval、rerank、tool call、output parsing 等多种算子。vLLM 视角只能看到 prefill / decode,前后的 CPU/IO 算子完全是黑盒,各种算子被串行执行。
- **方法**:
  - 把整个应用建模成 **primitive-级 DAG**:每种算子(embedding / retrieval / prefill / decode / tool)都是一个节点
  - 调度器在 primitive 粒度做**算子级并行 / 流水**:CPU 算子放 CPU 节点,GPU 算子放 GPU 节点,异构调度
  - 同一时刻可以:用户 A 的 embedding 在 CPU 跑,用户 B 的 prefill 在 GPU 跑,用户 C 的 retrieval 在向量库查
- **亮点**:把 LLM 应用从"黑盒请求"打开成"算子图",类似 Spark / Flink 数据流引擎的思路。比 Parrot 的粒度更细 — Parrot 看 LLM call 之间,Teola 看 LLM call 内部 + 外部所有算子。
- **与你方向**:你的长度预测可扩展为"primitive 完成时间预测",指导整个算子图的关键路径调度。

#### 1-④ ALTO 🟢(arXiv 2403.04311)

- **问题**:Compound AI 应用的 pipeline 中,下游 stage 必须等上游 stage **完整结束**才能启动。但 LLM 输出是流式的,这种等待是浪费 — 上游边 decode 下游就能边处理。
- **方法**:
  - 设计 stage 间的**流式触发协议**:上游 LLM 边 decode 边把 token 流向下游 stage
  - 下游 stage 拿到 partial 输出就开始处理(比如下游是 retrieval,可以提前用前几个 token 启动检索)
  - 跨 stage 的部分计算重叠,降低总 pipeline 延迟
- **亮点**:从"stage 串行"演进到"stage 流水",pipeline 等待时间显著缩短。和 Conveyor 思想互通(都是流式触发),只是 Conveyor 针对 tool call,ALTO 针对通用 stage。
- **与你方向**:流式触发的关键是预测"还要多久才能流完",这又回到长度预测。

#### 1-⑤ LLMCompiler 🟢(ICML'24,跨类引用)

- **问题**:ReAct(LangChain 默认)是**串行循环**,模型每次只想一步、调一个 tool、看结果、再想下一步。即使 5 个 tool call 完全独立,也排队 5 秒。
- **方法**:
  - **Planner**:让 LLM 一次性把整个任务**编译成函数调用 DAG**,一次规划替代多次 ReAct 思考
  - **Executor**:DAG 里的独立节点**并发执行**(类似传统编译器的指令调度)
  - **Joiner**:所有节点完成后,LLM 汇总结果
- **亮点**:相比 ReAct 平均提速 **~2×** 且省 token。启发了 OpenAI / Anthropic 的 parallel function calling 工业接口。
- **与你方向**:DAG 形状和节点长度都是调度信号,可叠加长度预测做更细粒度的 DAG 调度。

> 注:LLMCompiler 在原综述里同时出现在第 1 类(系统级)和第 2 类(tool 并行),按"五大类别"归类应放在**类别 3:Tool 阻塞**,这里只是补充其语义暴露视角。

---

### 类别 2:KV Cache × Agent 生命周期 — 跨轮、跨 Agent、跨时间的 KV 管理

这一类共 5 篇,演进路线非常清晰:**前缀完全相同才能共享 → 非前缀位置选择性复用 → 跨 Agent 跨时间生命周期管理**。

#### 2-① RadixAttention(SGLang)🟢

(已在类别 1 详述,这里只补"KV 视角"的核心:用 Radix Tree 统一组织所有请求的 KV 块,共享前缀的 KV 物理上只存一份,引用计数 + LRU 驱逐叶子。是后续所有工作的基础设施。)

#### 2-② ChunkAttention 🟡(ACL'24)

- **问题**:RadixAttention 解决了 KV **存储层**的共享(用 Radix Tree),但 attention **kernel 计算层**没改,还是按非共享方式跑 — 共享带来的内存收益没传到计算环节。
- **方法**:
  - **Prefix-aware attention kernel**:把共享 prefix 的 attention 计算单独提取出来,**只算一次**,然后把结果分发给所有共享该 prefix 的请求
  - **两阶段 partition**:① 共享 prefix 的 KV 集中算 ② 各自独立部分再分别算
- **亮点**:高共享率下 attention kernel 提速 **3.2-4.8×**。把"存储共享"延伸到"计算共享" — 这两层都共享后,prefix caching 收益才完整。
- **与你方向**:你的长度预测可指导何时启用 prefix-aware kernel(prefix 越长 + 共享越多,收益越大)。

#### 2-③ CacheBlend 🟡(EuroSys'25)

- **问题**:RadixAttention / SGLang 只能复用**完全相同前缀**的 KV。但 RAG 场景下,一个 prompt 经常这么拼:`<system> + <chunk_A> + <chunk_B> + <question>` — A 和 B **都被单独缓存过**,但**它们的拼接组合从未出现**,按 prefix-only 规则就要重新算 prefill,极浪费。**能不能直接拼这两块缓存?**
- **方法**:
  - **直接拼接**多个独立 chunk 的预算 KV(打破 prefix-only 限制)
  - 拼接后的 KV 不完全正确(因为 A 没看过 B 的 attention 上下文),所以对**少量"重要 token"做选择性 recompute** 修正
  - 重要 token 用一个轻量打分器选出(基于 attention 权重显著性)
- **亮点**:RAG 场景 TTFT 大幅下降,质量几乎无损。把"前缀复用"扩展到"任意位置复用",是 KV 复用的**第二阶段**。
- **与你方向**:重要 token 选择本身可建模成预测问题,与长度预测的方法论相通。

#### 2-④ Tokencake 🟢(arXiv 2510.18586,2025)

- **问题**:多 Agent 系统里,每个推理实例独立管理自己的 KV,但很多 Agent **共享同一个底座模型 + 同一个 system prompt + 同一份工具定义**,KV 在不同实例间高度重复存储。
- **方法**:
  - **中心化 KV serving**:多 Agent 共用一个 KV 存储层(类似分布式缓存)
  - **生命周期分层**:KV 块按热度分为短期 / 中期 / 长期三层(GPU / CPU / SSD),冷的逐层下沉
  - 自动识别 Agent 间共享 KV,只存一份,所有 Agent 引用同一份物理存储
  - 按 Agent 活跃度做分层驱逐
- **亮点**:把 KV 从"每个推理实例独立"提升到"集群级共享资源"。多 Agent 场景显存利用率显著提升,是 KV 复用的**第三阶段**(跨 Agent 跨时间)。
- **与你方向**:跨 Agent KV 生命周期管理依赖"下次命中时间预测",和你的长度预测高度互补。

#### 2-⑤ Continuum 🟢(arXiv 2511.02230,2025)

- **问题**:vLLM 的 prefix cache 用 LRU 驱逐,但 LRU 不知道**这个 KV 是否还会被命中、什么时候命中**。Agent 多轮交互间隔不确定(用户思考几秒到几分钟),粗糙的 LRU 可能踢掉马上要用的 KV、保留永远不会再用的 KV。
- **方法**:
  - 引入 **KV Cache TTL(Time-To-Live)** 概念:每个 KV 块预测一个"剩余有效时间"
  - **TTL = 用户思考间隔预测**(基于历史统计)**+ 下一轮输出长度预测**(LLM 还要 decode 多久才用到这个 KV)
  - 调度器据此做"调度感知的 KV 驻留 / 抢占":TTL 短的优先保留,TTL 长的可以驱逐;接近过期的可以提前预热下一次命中所需的 prefill
- **亮点**:首次把 KV 驱逐从"LRU 启发式"提升到"基于预测的最优化决策"。
- **与你方向最近**:**TTL 估计的核心子组件就是输出长度预测**。在 vLLM 的 prefix caching 之上做 TTL 感知驱逐策略,是工程上**最容易落地、Goodput 收益最直观**的方向。

---

### 类别 3:Tool Call 阻塞 — GPU 不能空等

这一类共 4 篇,都在解决"LLM 输出 tool call 后,等外部 API 期间 GPU + KV 浪费"。两条互补思路:**抢时间**(让 tool 早点跑 / 多个并发)和**省资源**(让 KV 在等待期间让出 GPU)。

#### 3-① LLMCompiler 🟢(ICML'24)

- **问题**:ReAct(LangChain 默认)是**串行循环**,模型每次只想一步、调一个 tool、看结果、再想下一步。即使任务里 5 个 tool call 完全独立(比如同时查 4 个商品的价格),也排队 5 秒。
- **方法**:
  - **Planner**:让 LLM 一次性把整个任务**编译成函数调用 DAG**,而不是一步一步想
  - **Executor**:DAG 里的独立节点**并发执行**(类似传统编译器的指令调度)
  - **Joiner**:所有节点完成后,LLM 汇总结果
  - 关键洞察:很多 Agent 任务的依赖度比想象低,大量 tool call 其实可以并发
- **亮点**:相比 ReAct **平均提速 2×** 且省 token(因为一次规划替代多次 ReAct 思考)。直接启发了 OpenAI / Anthropic 的 parallel function calling 工业接口,目前是 GPT-4 / Claude 标配。
- **与你方向**:DAG 形状和节点长度都是调度信号。可以叠加长度预测做更细粒度的 DAG 调度(优先调度关键路径节点)。

#### 3-② Conveyor 🟢(arXiv 2406.00059)

- **问题**:LLM 输出 tool 参数是流式的(比如 `{"action": "search", "query": "iPhone 17 价格..."}`),框架默认要等参数**完整输出**才能调 tool。但很多 tool **不需要完整参数**就能开工 — 搜索引擎只要前几个关键词就能开始检索;SQL 只要前面的 SELECT/FROM 就能开始扫表。
- **方法**:
  - **Tool-aware serving**:LLM 还在 decode tool 参数时,框架就把 partial 参数喂给 tool
  - Tool 开始 partial execution(部分搜索 / 部分 SQL),等 LLM 输出完整参数时,tool 已经做完一半甚至全部
  - **decode 和 tool 重叠执行**,关键路径上减一段
  - 系统层面定义了"哪些 tool 支持 partial、partial 程度怎么判断"的接口
- **亮点**:首次系统化提出"LLM 边生成边触发外部行为",重叠 decode 和 tool 等待。和 ALTO 思想互通(都是流式触发),但 Conveyor 专注 tool call 场景。
- **与你方向**:tool boundary token(`{"action":` 这一段在第几步出现)本质是**输出长度预测的子问题**。预测 boundary 后可指导 KV 换入换出时机。

#### 3-③ APIServe / InferCept 🟢(arXiv 2402.01869)

- **问题**:Tool call 期间 LLM 啥也没做,但 KV Cache 占着 GPU 显存(几 GB),把这块显存让给其他请求能显著提升整体吞吐 — 但什么时候让、怎么换回,需要专门的机制。
- **方法**:
  - 把 tool call 看作 **interception**(中断):LLM 进入"挂起"状态,类比操作系统的进程中断
  - 挂起时把这个请求的 KV Cache **swap 到 CPU 内存**,GPU 显存让给其他请求
  - tool 返回时 swap 回 GPU,继续 decode
  - 如果 KV 已经被 swap 太久 / CPU 也吃紧,直接 **drop + recompute** 也是策略(预测哪种成本低就选哪种)
  - 系统级管理"哪些请求处于 tool 等待"、"swap-in / swap-out 何时触发"
- **亮点**:首次系统性处理"tool call 期间的 KV 调度",让 GPU 在 tool 等待期间被其他请求充分利用。是 Conveyor 的"对偶"工作 — Conveyor 让 tool 早跑,APIServe 让 KV 让位,二者可以同时用。
- **与你方向**:swap 决策依赖"tool 多久回来"和"剩余输出多久" — 两者都是预测问题,长度预测可直接用。

#### 3-④ Parallel function calling 🟢(OpenAI / Anthropic 工业接口)

- **问题**:工业 API 默认串行调 tool,延迟堆叠(N 个 tool 就要 N 倍时间)。
- **方法**:OpenAI 和 Anthropic 在 API 协议层支持"模型一次输出多个独立 tool call",客户端 SDK 自动并发调用,结果回来后一次性塞回下一轮 prompt。
- **亮点**:LLMCompiler 思想的工业落地。目前 GPT-4 / Claude / Gemini 都是默认行为,Function Calling JSON Schema 里的 `parallel_tool_calls: true` 就是这个特性。
- **与你方向**:工业版 LLMCompiler,本身没有研究价值,但作为**实验场景**很重要 — 你的 Agent benchmark 必须支持 parallel function calling。

---

### 类别 4:Trajectory / Application-aware 调度 — Agent 是"程序",不是"请求"

这一类是**与你方向最强相关**的一组,共 8 篇。核心改变:**调度对象从"单次 LLM 请求"升级到"整个 trajectory / 应用 / 多 Agent 系统"**。

#### 4-① Parrot 🟢(OSDI'24,跨类引用)

(已在类别 1 详述。从调度视角:Semantic Variable + DAG-aware 调度,跨请求 prefix 共享 + 端到端延迟优化。是 trajectory 调度的开山级工作。)

#### 4-② Conveyor 🟢(arXiv 2406.00059,跨类引用)

(已在类别 3 详述。从调度视角:tool 中断时调度,tool 触发 KV swap 决策。)

#### 4-③ Autellix 🟢(NSDI'26)— **2025 最重要的 Agent 调度工作**

- **问题**:vLLM 把每次 LLM call 当独立请求 FCFS 调度,完全无视它们属于同一个 trajectory。结果:短任务被长 Agent 的多次 call 反复夹击,**端到端延迟严重 HoL blocking**。
- **方法**:
  - **核心抽象**:把 Agent 看作"通用程序"(trajectory),整个 trajectory 是调度单位,不是单次 call
  - **Trajectory-level scheduler**:感知 trajectory 内的依赖关系,知道哪些 call 属于"thinking"、哪些属于"tool 等待"、哪些是"汇总"
  - **公平性保证**:用户级公平,而不是请求级公平 — 否则发了 100 次 call 的复杂 Agent 用户会"霸占"队列
  - **抢占策略**:短 trajectory 可以抢占长 trajectory 的资源(SRPT 扩展到 trajectory 级)
  - **HoL 缓解**:通过 trajectory-level priority,避免单次 call 排队中被夹击
- **亮点**:**端到端吞吐 4-15× 优于 vLLM**(Agent workload),重新定义了 Agent serving 的基线。
- **与你方向最强**:Autellix 把场景从单请求扩到 trajectory,但**调度信号还没用上"剩余 trajectory 长度预测"**。你的研究方向直接填这个空白:**预测剩余轮数 + 每轮 token,喂给 Autellix 的调度器,做 trajectory 级 SRPT**。

#### 4-④ Tempo 🟢(arXiv 2504.20068,2025)

- **问题**:实际系统里同时跑多种应用 — 聊天机器人(SLO 严:TTFT < 500ms)、Agent(SLO 中:端到端 < 30s)、reasoning(SLO 松:端到端 < 5min,但 token 极多)。**用同一调度策略服务所有应用,要么聊天卡顿、要么 reasoning 饿死、要么 Agent 体验差**。
- **方法**:
  - **Application-aware scheduling**:调度器识别请求来自哪种应用,采用不同 SLO 策略
  - **混 SLO Goodput 优化**:目标不是单一吞吐或延迟,而是"满足各自 SLO 的请求数 / 总请求数"
  - 不同应用类型有不同的优先级、不同的抢占规则、不同的 batch 编排策略
- **亮点**:首次系统化处理"混合 SLO 场景",这正是真实云服务面对的工况。
- **与你方向**:Tempo 的策略仍是规则式。可以把"长度预测 + RL / learning-to-rank"用于混 SLO 调度决策,与 vLLM 的 Goodput metric PR(#9338)直接对接。

#### 4-⑤ S-LoRA / Punica 🟡(MLSys'24)

- **问题**:多 Agent 场景下,每个 Agent 经常用一个不同的 LoRA adapter(微调过的版本)。如果每个请求都换 adapter,GPU 利用率极低 — adapter 切换开销大。能不能让多个不同 LoRA 的请求在**同一个 batch** 里跑?
- **方法**:
  - **统一 batching kernel**:S-LoRA 提出 SGMV(Scatter-Gather Matrix-Vector),Punica 提出 BGMV(Batched GEMV)
  - 在 GPU kernel 层面同时处理多个不同 LoRA adapter 的请求,只多一次"按 adapter 分发"的开销
  - 主模型权重共享,只有 adapter 部分按请求切换
- **亮点**:让"每 Agent 一个 LoRA"在生产环境可行,同一 batch 可跑数百个不同 adapter。
- **与你方向**:S-LoRA 解决了多 LoRA 共存的 kernel 层问题,但**调度策略仍是简单 FCFS**。可以做"按 LoRA + 按预测剩余长度"双维度 batching,Goodput 改善明显。

#### 4-⑥ Concur 🟢(arXiv 2601.22705,2025)

- **问题**:多 Agent 系统里,有些 Agent 拿到任务但**根本无法完成**(比如目标太模糊、tool 不可用),会在 loop 里浪费 KV 和算力直到 max_iter。如果一开始就拒绝它,整体 Goodput 更高。
- **方法**:
  - **Agent-level admission control**:在 Agent 进入系统前评估"可完成概率"
  - 评估信号:任务复杂度、当前系统负载、Agent 历史成功率、可用 tool 列表
  - **拒绝预测会失败的 Agent**(返回 graceful fallback),把资源留给可完成的 Agent
- **亮点**:从"已经在跑的 Agent 怎么调度"扩展到"哪些 Agent 该被接受",首次把准入控制引入 Agent serving。
- **与你方向**:准入控制核心是"放进哪些、踢哪些",和"基于长度预测的调度"是同一决策的不同尺度。可以把"剩余 trajectory 长度"作为多 Agent 准入的关键信号。

#### 4-⑦ Hive 🟢(arXiv 2604.17353,2025)

- **问题**:多 Agent 系统的扩展性有两层 — 算法层面(Agent 数量)和任务层面(任务并发量),传统 serving 系统只看一层。
- **方法**:
  - **双层 scaling**:算法层(Agent 实例数 / 副本数)+ 任务层(每个 Agent 处理的并发任务数)同时弹性
  - 跨层调度:任务来时既考虑"该分配给哪个 Agent",也考虑"该 Agent 是否需要扩容"
  - 拓扑感知的资源放置
- **亮点**:把多 Agent 系统当成"分布式系统"来设计 serving 基础设施,而不是把它当成 LLM 推理的延伸。
- **与你方向**:Hive 的资源放置可以叠加"DAG 形状感知 + 长度预测"做更精细的扩缩容决策。

#### 4-⑧ Mixture-of-Agents serving 🟢(arXiv 2512.18126,2025)

- **问题**:Mixture-of-Agents(MoA)架构下,多个 Agent 形成"router → 专家 Agent → aggregator"的层级,且每个层级 Agent 数量不同。怎么编排 prefill / decode / 路由,让整体延迟最低?
- **方法**:
  - **Tree-routing**:Agent 间的 router 决策按树结构组织
  - **Dependency-aware prefill-decode overlap**:上层 Agent 还在 decode 时,根据已有 token 预测路由方向,下层 Agent 提前 prefill
  - 针对 MoA 这种"层级 + 路由"结构特化
- **亮点**:把 MoA 架构作为独立 serving 对象优化,2025 年下半年新方向。
- **与你方向**:路由决策依赖"上层 Agent 输出何时稳定",这又是输出长度 / boundary 预测问题。

---

### 类别 5:Reasoning Agent 的长输出加速(Agent 的近亲)🟣

这一类共 6 篇,严格说不是 Agent 而是 **Reasoning**(o1 / R1 / QwQ),但"输出长度暴涨"这一痛点跟 Agent 完全相同。两条路:**算法侧让模型少想 / 系统侧让长思考可承受**。

#### 5-① O1-Pruner 🟣(arXiv 2501.12570,2025)— 算法侧

- **问题**:o1-like 推理模型为追求精度,会"过度思考" — 简单题也用极长 CoT,平均输出长度比标准模型大 5-10×,大量 token 是冗余推理。
- **方法**:
  - **Length-harmonizing fine-tune**:在 SFT / RL 阶段加一个"长度奖励项",鼓励模型用最短的思考路径达到相同精度
  - 训练数据:对同一问题采样多条推理路径,选最短且正确的作为目标
  - 不修改模型结构,仅 fine-tune 阶段干预
- **亮点**:准确率几乎不降,**思考长度显著缩短**(部分任务降 50%+)。
- **与你方向**:O1-Pruner 是离线方法,你的"在线长度预测"可在推理时做"自适应早停",和它互补。

#### 5-② LightThinker 🟣(arXiv 2502.15589,2025)— 算法侧

- **问题**:reasoning 思考过程中,中间 token 大量是"探索性"的,推理完成后这些中间 token 占着 KV 没贡献。
- **方法**:
  - 训练模型在思考过程中**主动输出"压缩 token"**,把前面一段思考浓缩为一个总结 token,后续生成只看总结不看原文
  - 类似"边思考边写读书笔记",原文可以丢
  - **KV 与生成长度同时降低**(原文 token 的 KV 可释放)
- **亮点**:从"事后压缩"变成"边生成边压缩",更激进。
- **与你方向**:压缩时机本身可以是预测问题 — 何时压缩、压缩多少,都和"剩余还要思考多久"相关。

#### 5-③ Reasoning Path Compression 🟣(arXiv 2505.13866,2025)— 算法侧

- **问题**:reasoning 轨迹里大量重复或冗余的"再确认"步骤(模型常常推完又"让我再想想"),这些是质量保证但不是精度必需。
- **方法**:对生成的轨迹做后处理压缩 — 识别冗余步骤、合并重复推理、保留关键步骤。可作为 fine-tune 数据训练"更紧凑"的模型。
- **亮点**:轨迹本身可以缩短 30-50%,精度无损。
- **与你方向**:轨迹冗余度本身可预测 — 能预测就能在线提示模型"不需要再想了"。

#### 5-④ ThinKV 🟣(arXiv 2510.01290,2025)— 系统侧

- **问题**:reasoning 模型 KV Cache 持续增长(几万 token 的 CoT),显存压力大,decode 速度被 KV 访存拖慢。
- **方法**:
  - **Thought-adaptive KV 压缩**:在思考过程中识别"已经被推理过"的部分(模型 attention 权重显示已不再关注),对其 KV 做 **激进压缩 / 量化**
  - 关键 token(如最近的推理结论、数学公式)保留高精度,探索性中间 token 量化或丢弃
  - 专门面向 reasoning 模型的超长 KV 设计
- **亮点**:把通用 KV 压缩(KIVI / H2O 等)针对性优化到 reasoning 场景。
- **与你方向**:压缩决策依赖"还要思考多久",直接需要长度预测。

#### 5-⑤ SpecReason 🟣(arXiv 2504.07891,2025)— SD 侧

- **问题**:reasoning 模型每个思考步骤(`step 1...`、`step 2...`)语义比较固定,但 token-level SD 不会利用这种结构。
- **方法**:
  - **小模型推测整段"思考步骤"**(不是逐 token,而是逐 step)
  - 大模型只**验证关键步骤**(数学计算、关键结论),非关键步骤跳过验证
  - 结构感知的推测解码,粒度从 token 升到 reasoning step
- **亮点**:针对 long-CoT 场景,把 SD 从 token 级提升到 step 级,接受率显著提升。
- **与你方向**:step 边界预测 ≈ 长度预测的子问题。

#### 5-⑥ SSR(Speculative Parallel Scaling Reasoning)🟣(arXiv 2505.15340,2025)— SD 侧

- **问题**:Self-consistency(同一题采样多条推理路径,投票选答案)能提精度但成本高 — N 条路径就要 N 倍计算。
- **方法**:
  - **投机式并行剪枝**:N 条路径并发生成,用小模型监控各条路径的"前缀质量"
  - 一旦发现某条路径明显走错(前缀和其他路径分歧太大),提前终止,把资源让给更有希望的路径
  - 类似 SD 的 "early exit",但应用在多路采样场景
- **亮点**:self-consistency 的成本下降 40-60%,精度几乎不降。
- **与你方向**:剪枝决策本质是"哪条路径剩余产出价值最高",和长度预测 + 质量预测耦合。

---

### 类别 6(附):通用 / 单请求优化 — Agent 场景能借用但需改造 ⚪

这 8 篇不在五大 Agent 类别里,但 Agent 场景能借用,**改造关键是叠加 trajectory 信号**。

#### 6-① Medusa ⚪(ICML'24)

- **问题**:推测解码常规做法需要单独训一个 draft 小模型,部署麻烦。
- **方法**:在主模型上加几个**额外的 decoding head**(MLP 头),每个 head 预测后几个位置的 token。**单模型即可推测**。
- **亮点**:2-3× decode 提速,部署简单(只多了几个 head)。
- **Agent 场景如何改造**:Agent 长输出场景 SD 收益更大,但要根据 trajectory 阶段动态决定开不开 SD(thinking 阶段开,tool 等待时关)。

#### 6-② EAGLE / EAGLE-2 / EAGLE-3 ⚪(ICML'24 / EMNLP'24)

- **问题**:Medusa 在 token 层做 draft,信息损失大;独立 draft model 又部署重。
- **方法**:在 **feature 层(隐藏层 hidden states)** 做 draft,更准确。EAGLE-2 动态构建 draft tree(根据 confidence 决定每层探多宽);EAGLE-3 更激进的 multi-token 预测。
- **亮点**:**单机推测解码 SoTA**,decode 提速 3-5×。
- **Agent 场景如何改造**:draft tree 形状可以根据"剩余输出长度预测"调 — 长输出深 tree 收益大。

#### 6-③ Lookahead Decoding ⚪(ICML'24)

- **问题**:推测解码都需要某种 draft model / draft head。能不能完全不需要?
- **方法**:用 **Jacobi 迭代**(数值方法)并行猜测多个位置的 n-gram,主模型并行验证。**零额外模型**。
- **亮点**:零成本部署,2× 提速。
- **Agent 场景如何改造**:Agent 重复模板多(system prompt / tool 格式),Jacobi 收敛快,Agent 上比通用场景收益更大。

#### 6-④ SpecInfer / Sequoia ⚪(ASPLOS'24)

- **问题**:线性 draft 序列接受率低,猜错就全部废掉。
- **方法**:**Tree-based speculative**(并发探多条 draft 路径)+ **Sequoia 研究最优 draft 树形状**。
- **亮点**:接受率提升,长上下文 verification 更高效。
- **Agent 场景如何改造**:不同 trajectory 阶段最优 tree 形状不同,可结合长度预测自适应。

#### 6-⑤ Goodput-driven SD scheduling ⚪(arXiv 2406.14066)

- **问题**:SD 不是无脑加速 — 高负载下 draft verification 占用 batch slot,反而拖慢系统。
- **方法**:**在线根据负载动态调 draft 长度**,负载低时开大,高时关小。
- **亮点**:首次把 SD 决策"调度化",从"开 / 不开"二元变成"开多大"连续决策。
- **与你方向最相关 SD 工作**:"开多大 draft"和"剩余输出长度"高度相关 — 长输出请求开大 draft 收益大。**这是输出长度预测在 SD 决策上的直接应用**。

#### 6-⑥ Learning-to-Rank Scheduling ⚪(arXiv 2408.15792,NeurIPS'24)

- **问题**:基于精确长度预测的 SRPT 调度,长度预测误差大时性能不稳定。能不能不预测精确长度,只学习"排序"?
- **方法**:用 **learning-to-rank** 训练一个排序器,直接学"哪条请求该排前面",而不是"它有多长"。比 SJF/SRPT 类长度预测更鲁棒。
- **亮点**:绕开长度预测的精度瓶颈,排序问题比回归问题简单。
- **与你方向**:**长度预测的另一条路** — 你可以拿它当 baseline 对比,或思考"两者融合"(用排序器粗筛,长度预测精排)。

#### 6-⑦ Proxy-model length prediction ⚪(arXiv 2404.08509)

- **问题**:vLLM 默认 FCFS,有 HoL blocking。需要长度感知调度,但又不想训练大模型本身。
- **方法**:用一个**轻量代理模型**(BERT 级别)预测每个请求的输出长度,据此做 SRPT 调度。
- **亮点**:vLLM 上 **5.9× 缓解 HoL blocking**,代理模型成本可忽略。
- **与你方向最直接对标**:**这是你方向最直接的 baseline,必须读透**。它的 baseline 设计、metric 选择、误差分析是你工作的起点。Agent 场景下要扩展到 trajectory 级,但单请求级别的方法论可以借鉴。

#### 6-⑧ ServerlessLLM ⚪(OSDI'24)

- **问题**:云上 LLM 推理,模型加载冷启动慢(几十秒到几分钟),Serverless 场景不可接受。
- **方法**:Locality-aware 模型加载、checkpoint 重排、快速恢复。
- **亮点**:cold-start 降到秒级。
- **Agent 场景如何借用**:多 Agent 用不同 LoRA / 微调版本时,冷启动是真实痛点。

---

## 八、读论文的优先级建议

如果时间紧张,按这个顺序读:

1. **必读 5 篇**(直接对标你方向):
   - Proxy-model length prediction(6-⑦)— 你的直接 baseline
   - Autellix(4-③)— Agent 调度 SOTA
   - Continuum(2-⑤)— KV TTL × 长度预测
   - Goodput-driven SD(6-⑤)— SD × 调度
   - vLLM 原论文(SOSP'23)— 底座

2. **强烈建议**(理解 Agent serving 范式):
   - Parrot(1-②)+ SGLang(1-①)— 语义暴露开山之作
   - Tempo(4-④)— 混 SLO 调度
   - Conveyor(3-②)— Tool 中断处理

3. **方向延伸**(Reasoning / SD 方向想做的话):
   - ThinKV + O1-Pruner + SpecReason — Reasoning 加速三个角度
   - Learning-to-Rank Scheduling — 长度预测的"另一条路"

4. **了解即可**(知道存在,不一定细读):
   - Medusa / EAGLE / Lookahead / SpecInfer — 通用 SD,挑一篇细读即可
   - Tokencake / Hive / MoA serving / Concur — 多 Agent 方向,跟踪而非深读



明白了,目标变了 — 1 天做综述,核心是形成对这个方向的完整认知,不是面试准备。重新规划。

  一个关键提醒:你已经完成 70% 了

  你已经写的两个文档 — Agent 推理加速综述.md + Agent 场景论文分类.md — 本身已经是一份相当完整的综述:覆盖了 32 篇论文、5
  大类别、研究空白判断、与你方向的连接。

  → 1 天的目标不是"从零做综述",而是把现有文档从"二手整理"升级为"一手理解" —
  即:对几篇代表作有自己的判断,而不是只信综述博客的转述。

  一天 8 小时的具体分配

  第一段(1 小时)— 重读自己的文档,定位空白

  打开你的两个文档完整读一遍,边读边标记:
  - 哪些论文你只知道一句话总结(高风险,可能理解错)
  - 哪些类别你讲不出核心 idea 的"why"(只知道做了什么,不知道为什么这么做)
  - 哪些"与你方向连接"是你自己想清楚的,哪些只是抄过来的

  → 产出:一张"待补强清单",大概 5-8 个具体问题。

  第二段(5 小时)— 精读 5 篇代表作,每类 1 篇

  关键原则:不追求看完 32 篇,而是每类选 1 篇深度看,代表整个类别。这 5 篇看透后,综述 95% 的内容你都能讲清楚。

  ┌────────────────────┬───────────────────┬─────────────────────────────────────────────────────┬────────┐
  │        类别        │     必读 1 篇     │                     为什么选它                      │  用时  │
  ├────────────────────┼───────────────────┼─────────────────────────────────────────────────────┼────────┤
  │ 1. 语义暴露        │ Parrot(OSDI'24)   │ 开山级,Semantic Variable 概念覆盖 SGLang/Teola/ALTO │ 60 min │
  ├────────────────────┼───────────────────┼─────────────────────────────────────────────────────┼────────┤
  │ 2. KV 生命周期     │ Continuum(2025)   │ 最新、和你方向最近、KV TTL 概念干净                 │ 60 min │
  ├────────────────────┼───────────────────┼─────────────────────────────────────────────────────┼────────┤
  │ 3. Tool 阻塞       │ Conveyor(2024)    │ 思路最巧、能秒懂"流式 tool 触发"                    │ 50 min │
  ├────────────────────┼───────────────────┼─────────────────────────────────────────────────────┼────────┤
  │ 4. Trajectory 调度 │ Autellix(NSDI'26) │ 2025 SOTA,定义了 trajectory-级调度范式              │ 70 min │
  ├────────────────────┼───────────────────┼─────────────────────────────────────────────────────┼────────┤
  │ 5. Reasoning       │ ThinKV(2025)      │ 长输出 + KV 双痛点,代表系统侧                       │ 60 min │
  └────────────────────┴───────────────────┴─────────────────────────────────────────────────────┴────────┘

  每篇怎么读(精读法):

  5 min  → Abstract + Intro 第 1 段(它要解决啥)
  10 min → 它定义的"新问题"是什么(对比已有工作)
  20 min → 核心 method:它的 idea 是什么(用一句话能讲出 why)
  10 min → 关键图表 1-2 张(metric + 收益数字)
  10 min → 写笔记:它和综述里其他工作的关系是什么

  绝对不要做的事:
  - ❌ 看实现细节、伪代码
  - ❌ 看附录、消融实验
  - ❌ 追相关工作的引用
  - ❌ 试图复现

  第三段(1.5 小时)— 输出"优化点地图"

  读完 5 篇,把综述从"按论文组织"重写成"按优化点组织" — 这才是综述的真正价值。

  格式建议(直接写在你现有文档的"九、优化点地图"新章节里):

  优化点 #1:跨请求 prefix 共享
  ├── 痛点:Agent 重复模板,vLLM 默认看不见
  ├── 核心思路:暴露 prefix 结构 + 自动检测复用
  ├── 代表方法:Radix Tree(SGLang)/ Semantic Variable(Parrot)
  ├── 收益量级:vLLM 上 prefix 命中率 20% → 80%+
  ├── 当前局限:跨 Agent / 非前缀位置共享还不成熟
  └── 与你方向:可叠加长度预测做"剩余命中价值"驱逐

  优化点 #2:Tool 等待期间 KV 让位
  ├── 痛点:tool call 几秒,KV 占着 GPU
  ├── 核心思路:KV swap 到 CPU,GPU 让给其他请求
  ├── 代表方法:APIServe(swap)/ Conveyor(让 tool 早跑)
  ├── 收益量级:Agent workload Goodput 提升数倍
  ├── 当前局限:swap 时机依赖经验规则
  └── 与你方向:swap 决策需要"剩余输出 + tool 时长"双预测

  ...(每个优化点这么写)

  → 这一步是你综述的最大增量 — 别人综述按论文列,你按优化点列,这是真正的"研究者视角"。

  第四段(0.5 小时)— 收尾产出 + 复盘

  - 把"优化点地图"写进文档
  - 列出3 个你判断最值得做的方向(从你文档的"6 个发力方向"里挑出最具体的 3 个)
  - 写一段 200 字的"我对这个方向的总体判断" — 这是组会汇报的开场

  → 产出:可以直接拿去组会汇报的综述材料。

  推荐顺序的逻辑

  为什么是 Parrot → Continuum → Conveyor → Autellix → ThinKV?

  - Parrot 先读:它定义了"DAG / Semantic Variable",是后面所有 Agent 工作的语言
  - Continuum 第二:它的 KV TTL 直接对接你方向,读完你能立刻看到"长度预测在哪能用"
  - Conveyor 第三:轻松的一篇,中场调节,但思路最巧妙
  - Autellix 第四:最重要的一篇 SOTA,前面铺垫够了再啃,不会迷失
  - ThinKV 最后:reasoning 是 Agent 的近亲,最后看能形成"Agent vs Reasoning"的对比

  不要做的事(1 天综述的禁区)

  - ❌ 不读 32 篇 — 边际收益极低,5 篇代表作 + 综述结构已经覆盖 95%
  - ❌ 不读通用 SD 工作(Medusa / EAGLE / Lookahead) — 跟 Agent 关系不大,综述里提一句就行
  - ❌ 不写代码、不复现 — 综述阶段不是工程阶段
  - ❌ 不纠结优先级排名 — 你的两个文档已经做了排序,信任自己的判断
  - ❌ 不要发散到 LLM 推理基础(KV Cache 原理、PagedAttention) — 这些你已经懂

  一句话

  1 天综述 = 5 篇精读 + 1 张优化点地图 + 1 段总体判断。其他都是噪声。

  如果今天结束你能自信地讲清楚"Agent 推理加速分几大类、每类核心思路、当前空白在哪、我能做什么",这一天就成功了。
