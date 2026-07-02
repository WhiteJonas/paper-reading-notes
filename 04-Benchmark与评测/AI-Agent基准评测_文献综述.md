# AI Agent 任务完成能力基准：文献综述

> 本综述整理自同目录《2026年人工智能智能体（AI Agent）性能与任务完成能力基准深度评测报告》中提及的全部基准与方法论文，并逐一核实了真实出处（机构、年份、发表 venue、arXiv 编号）。
>
> **可靠性说明**：原报告中部分性能数字属于面向未来的推测（如 "Claude Mythos 5"、"DeepSeek v3.2"、"Gemini-3-Flash" 等尚不存在或未公开核实的模型成绩），本综述只保留可追溯到论文/官方来源的事实，对推测性内容不予转述。原报告提及的 **ExpGraph** 在 Semantic Scholar / OpenReview / ACL Anthology 等权威索引中均查不到，疑似虚构，已剔除。

---

## 一、研究背景与综述范围

随着大语言模型从"单次问答"走向"长视距自主执行"，传统静态基准（MMLU 等选择题）已无法刻画 Agent 在真实、动态环境中的任务完成能力。现代 Agent 评测的核心转向两点：**任务完成度**（基于系统底层状态变更的功能性验证，而非动作序列或文本匹配）与**执行效率/成本**。

本综述覆盖五个方向的代表性工作：

1. 操作系统与 GUI 控制基准
2. Web 自动化与 API 交互基准
3. 垂直专业领域基准（软件工程、科学计算、网络安全）
4. 对话-工具-用户交互基准
5. 评估基础设施与方法论（含统计学重构）

---

## 二、操作系统与 GUI 控制基准

这一方向评测 Agent 通过人类标准的键鼠/触屏原语操控真实操作系统的能力，核心难点在于多模态视觉接地（visual grounding）、动作定位与跨应用规划。

### OSWorld

- **全称**：OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments
- **机构 / 年份 / venue**：XLANG Lab（香港大学主导，含 CMU、Salesforce、滑铁卢大学）/ 2024 / NeurIPS 2024 Datasets & Benchmarks Track
- **来源**：arXiv:2404.07972；OSWorld-Verified 公告 https://xlang.ai/blog/osworld-verified
- **评测内容**：在真实 Ubuntu/Windows/macOS 虚拟机上执行开放式、跨应用的长视距任务。Agent 只接收屏幕截图（部分设置附无障碍树），输出鼠标点击/拖拽、键盘输入等底层动作原语，像真人一样操作真实软件——涵盖浏览器（Chrome）、办公套件（LibreOffice 的文档/表格/幻灯片）、代码编辑器（VS Code）、图像工具（GIMP）、文件管理器与系统设置等。任务往往需要**多个应用协作 + 几十步操作**，例如"从某网页复制一张表格的数据、粘进电子表格、算出汇总列再导出为 PDF""在 GIMP 里把一批图片批量裁剪并另存""修改系统设置后在终端验证配置生效"。强调多模态视觉接地（在截图里定位控件）、精确动作定位与跨应用规划。
- **方法与规模**：369 个真实长视距任务；采用**基于执行结果**（execution-based）的验证——每个任务有自定义校验脚本，在 Agent 声明完成后检查文件系统/应用状态判定功能正确性。`OSWorld-Verified`（2025-07）为官方修订版，修复边界条件 bug 并引入云端并行，大幅缩短单次评测时间。
- **评测指标**：**任务成功率（Success Rate, %）**——由执行校验脚本判定最终状态是否达成目标，而非动作序列匹配。

### Windows Agent Arena（WAA）

- **全称**：Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale
- **机构 / 年份 / venue**：Microsoft（含 CMU、哥伦比亚大学）/ 2024 / arXiv 首发，后收录 PMLR (Bonatti et al., 2025)
- **来源**：arXiv:2409.08264；项目页 https://microsoft.github.io/WindowsAgentArena/
- **评测内容**：聚焦 **Windows 生态**的 OS Agent 测评，依托 Azure 虚拟机提供可并行、可复现的评测环境，约 154 个原生任务，覆盖文档编辑（Office / 记事本）、文件资源管理器操作、Edge 浏览器、系统设置、命令行等日常场景。相比 OSWorld，它特别强调 **Windows 特有的交互模式**——如注册表/控制面板、多级级联的右键上下文菜单、UIA（UI Automation）控件树，这些正是通用视觉模型容易失手的地方。
- **代表方法**：论文提出 baseline Agent **Navi**，融合 UI 自动化树（UIA tree）解析、DOM 解析，以及 Grounding DINO / OmniParser 的视觉元素检测。论文凸显现有视觉模型在 Windows 特有交互（如深层嵌套右键菜单）上的认知盲区。
- **评测指标**：**任务成功率**，基于虚拟机最终状态的程序化校验判定；强调云虚拟机上的可并行、可复现。

### AndroidWorld

- **全称**：AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents
- **机构 / 年份 / venue**：Google Research / Google DeepMind / 2024 / ICLR 2025
- **来源**：arXiv:2405.14573；项目页 https://google-research.github.io/android_world/
- **评测内容**：一个完整可运行的 **Android 手机环境**，跨 20 个真实 App（时钟、通讯录、短信、日历、Markor 笔记、文件管理、Simple 系列工具等）提供 116 个程序化任务。Agent 通过截图 + 无障碍树感知界面，用点击/滑动/输入等触屏原语操作手机，完成如"设置一个明早 7:30 的闹钟""给某联系人发指定内容的短信""在笔记 App 里新建文件并写入内容""按条件批量整理文件"等日常手机操作。
- **方法亮点**：放弃静态测试集，通过**任务参数化随机生成**数百万变体以抗过拟合；用 Android 原生状态管理（SQLite 查询、ADB 文件系统转储）做程序化奖励信号，而非脆弱的截图比对。随基准提供多模态 baseline Agent **M3A**（结合截图与无障碍树输入）。
- **评测指标**：**任务成功率**——用 SQLite 查询 + ADB 文件系统转储读取设备真实状态判定，抗截图比对的脆弱性。

---

## 三、Web 自动化与 API 交互基准

相比 OS 层的纯视觉控制，Web 方向更强调 DOM 解析、长跨度信息检索、跨页面流程与编程级 API 交互。

### WebArena

- **全称**：WebArena: A Realistic Web Environment for Building Autonomous Agents
- **机构 / 年份 / venue**：Carnegie Mellon University / 2023 / ICLR 2024
- **来源**：arXiv:2307.13854；项目页 https://webarena.dev/
- **评测内容**：自托管的高保真网站沙盒，把四类真实站点做成可本地部署的副本——电商购物网站（OneStopShop）、社交论坛（Reddit 副本）、代码协作平台（GitLab）、内容管理后台（CMS），外加地图等工具站，共 812 个长视距任务。Agent 通过浏览器（读取网页 DOM/截图，执行点击、填表、导航）完成真实网页操作，例如"在论坛发一个符合要求的帖子""在 GitLab 给某仓库创建 issue 并指派""按筛选条件下单某商品""跨多个页面检索并回答某统计问题"。强调 DOM 解析、长跨度信息检索与跨页面流程。
- **方法**：依据任务完成后的**功能正确性**自动打分（基于结果状态而非动作序列匹配）。
- **评测指标**：**任务成功率**——检查任务目标状态是否达成（如订单是否真的创建），信息检索类任务用答案匹配核对，均为结果导向。
- **变体说明**（需谨慎引用）：
  - **WebArena Verified**：真实存在，由 ServiceNow 于 2025 年发布，对原任务做可靠化修正（github.com/ServiceNow/webarena-verified）。原报告所称 "Verified Hard" 未见独立命名，可能仅为其难度子集的口语化叫法，引用前应核对原文。
  - **WebArena Infinity**：有项目页与代码/数据集（标注 2026），主题为规模化生成可验证浏览器任务环境，但**无正式论文**，应作为"工程项目/数据集"而非已发表论文定位。（注意与第三方 autoppia 的 "InfiniteWebArena" 区分。）

### AppWorld

- **全称**：AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents
- **机构 / 年份 / venue**：Stony Brook University 等 / 2024 / ACL 2024（**Best Resource Paper**）
- **来源**：arXiv:2407.18901；ACL Anthology 2024.acl-long.850；项目页 https://appworld.dev/
- **评测内容**：一个模拟日常数字生活的可控世界，含约 9 个 App（Gmail、Spotify、Amazon、Todoist、Venmo、简书式笔记、手机/联系人等）、457 个 API、750 个任务，还预置了"人物关系网"（好友、家人、账户）。与前面几个"点界面"的基准不同，AppWorld 要求 Agent **通过写代码调用 API** 来完成任务（交互式编程 Agent），任务常需跨多个 App 组合、含条件判断与循环，例如"把我上个月所有音乐订阅的花费用 Venmo 向室友各收一半""删除收件箱里所有来自某发件人的邮件""根据购物清单在 Amazon 下单并把待办标记完成"。难点在于要理解复杂的 API 文档、维护跨 App 状态、且不能误伤无关数据。
- **方法亮点**：苛刻的**基于状态的单元测试**——不仅验证核心业务逻辑，还通过全局内存/数据库快照对比检查"附带损害"（collateral damage，如误删他人邮件、错误支付）。
- **评测指标**：**状态单元测试通过率**，两层判定——① 核心业务逻辑是否达成；② 附带损害检查（有无副作用）。是最严格的状态验证之一。
- **相关方法**：
  - **LOOP**（Apple，arXiv:2502.01600, 2025）："Reinforcement Learning for Long-Horizon Interactive LLM Agents"，以 AppWorld 为在线训练环境的 RL 框架（是训练方法，非基准本身）。
  - **CUGA**（IBM Research，arXiv:2503.01861, 2025）：Configurable Generalist Agent，企业级通用 Agent 框架，曾在 AppWorld 等基准报告成绩。
  - ~~ExpGraph~~：原报告提及，但权威索引无法核实，疑似虚构，**不予收录**。

---

## 四、垂直专业领域基准

### 4.1 软件工程与机器学习工程

| 基准 | 全称 / 机构 / venue | 来源 | 评测内容与方法 |
|---|---|---|---|
| **SWE-bench** / **SWE-bench Verified** | Can Language Models Resolve Real-World GitHub Issues? / Princeton + UChicago（Verified 由 OpenAI 协作）/ ICLR 2024 | arXiv:2310.06770；Verified: openai.com/index/introducing-swe-bench-verified | 解决真实 GitHub issue，生成补丁。指标为 % Resolved（补丁能否通过仓库测试套件 FAIL_TO_PASS / PASS_TO_PASS）。原 2294 任务（12 个 Python 仓库）；Verified 为人工筛选的 500 任务子集 |
| **MLE-bench** | Evaluating Machine Learning Agents on ML Engineering / OpenAI / ICLR 2025 | arXiv:2410.07095；github.com/openai/mle-bench | 端到端 ML 工程（数据准备、训练、调试、提交）。基于 Kaggle 奖牌制评分，与人类排行榜对比。75 个精选 Kaggle 竞赛 |
| **DataSciBench** | An LLM Agent Benchmark for Data Science / 清华 THUDM / Findings of ACL 2026 | arXiv:2502.13897；datascibench.github.io | 数据分析与可视化。提出 **Task-Function-Code (TFC)** 半自动评估框架，用程序化测试用例 + 多个聚合函数评分。约 222 prompts / 519 test cases |

**评测内容展开**：
- **SWE-bench**：给 Agent 一个真实开源项目（如 Django、scikit-learn、matplotlib 等 12 个流行 Python 库）的完整代码库，加一个真实的 GitHub issue（bug 报告或功能请求），要求它像开发者一样定位问题、修改代码、生成一个 patch 补丁。任务全部来自这些仓库已合并的真实 PR，难点在于要读懂大型陌生代码库、跨文件改动、且不破坏已有功能。**Verified** 是 OpenAI 联合人工筛掉"描述含糊/测试有问题"后的 500 个高质量子集。
- **MLE-bench**：把 75 个 Kaggle 竞赛整体丢给 Agent，让它走完一名数据科学家的**全流程**——理解赛题、探索数据、清洗与特征工程、选模型、训练调参、调试、生成提交文件。考的不是单点代码，而是端到端的机器学习工程闭环能力。
- **DataSciBench**：更聚焦"数据分析 + 可视化"这一段——给数据和自然语言需求，要求 Agent 写代码做统计分析、生成图表等，产出既包含数值结果也包含图形要素。

### 4.2 科学计算、严谨推理与研究复现

| 基准 | 全称 / 机构 / venue | 来源 | 评测内容与方法 |
|---|---|---|---|
| **CORE-bench** | Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark / Princeton / TMLR | arXiv:2409.11363 | 计算可复现性：Agent 接管论文原始代码库与数据集，搭建环境、解决依赖冲突、复现图表与统计结果。270 任务 / 90 篇论文（CS、社科、医学），分 Easy/Medium/Hard |
| **SciCode** | A Research Coding Benchmark Curated by Scientists / UChicago、Princeton 等 / NeurIPS 2024 D&B | arXiv:2407.13168；scicode-bench.github.io | 为真实科研问题生成代码，将抽象科学定理分解为可计算子任务。80 主问题 / 338 子问题（数理化生与材料） |
| **LAB-Bench** | Measuring Capabilities of Language Models for Biology Research / FutureHouse / arXiv 预印本 | arXiv:2407.10362 | 生物学研究工作流（文献检索推理、序列分析、图表/数据库解读等）。2400+ 多选题，分 LitQA、SeqQA、FigQA 等子任务，支持评估"弃答"行为 |
| **Humanity's Last Exam (HLE)** | Humanity's Last Exam / CAIS + Scale AI / arXiv，后发 Nature | arXiv:2501.14249；lastexam.ai | 人类专家知识前沿的极难学术测试（针对 MMLU 饱和而设）。约 2500 题（含多模态），指标含 accuracy + calibration error（校准误差） |
| **AIME 2025** | American Invitational Mathematics Examination 2025 / MAA 主办（非论文式 benchmark）| 无官方论文；实现见 Inspect Evals、math-ai-org/aime25 | 多步数学推理。答案为 0–999 整数，精确匹配评分，常用 pass@1 / avg@k。每年 30 题，因新题降低数据污染。**应作为"被改造为评测的竞赛数据集"引用，而非虚构基准** |

**评测内容展开**：
- **CORE-bench**：给 Agent 一篇已发表论文的**原始代码仓库 + 数据**，让它从零把实验跑起来——搭环境、装依赖、解决版本冲突、执行脚本，最终**复现出论文里的那些图表和统计数字**。考的是"科研可复现"这件苦活，按 Easy/Medium/Hard 分级（难的连原作者环境说明都不全）。
- **SciCode**：由各领域科学家出题，把真实科研中的计算问题（量子力学、电磁学、生物信息、材料模拟等）拆成一连串**子问题**，要求 Agent 逐步写出可运行、数值正确的科学计算代码，最后拼成完整解。考"把抽象科学定理翻译成正确代码"的能力。
- **LAB-Bench**：模拟生物学研究者的日常，出 2400+ 道多选题，分几类子任务——LitQA（读文献做推理）、SeqQA（DNA/蛋白序列分析）、FigQA（看图表/数据库回答问题）等，还专门考"该不该弃答"（面对没把握的题选择"不确定"而非乱猜）。
- **HLE（人类最后的考试）**：为应对 MMLU 被刷爆而设的**极难**学术题库，约 2500 题横跨数学、物理、人文、生物等前沿，很多题需要领域博士才答得上，部分含图像。除答对率外还考"校准"——模型要如实报告自己有多少把握。
- **AIME 2025**：美国高中数学邀请赛真题，每年 30 道，答案是 0–999 的整数（便于精确判分）。考多步数学推理，每年换新题以降低"背答案"式的数据污染。

### 4.3 网络安全攻防

| 基准 | 全称 / 机构 / venue | 来源 | 评测内容与方法 |
|---|---|---|---|
| **Cybench** | A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models / Stanford / ICLR 2025 | arXiv:2408.08926；cybench.github.io | CTF 形式评测攻防能力。40 个专业级 CTF 任务（来自 4 个竞赛，17 个带子任务分解）。指标为 flag 捕获成功率 + first-solve-time 难度刻画 |
| **CyberGym** | Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale / UC Berkeley / arXiv（ICLR 2026 投稿轨）| arXiv:2506.02548；cybergym.io | 大规模真实漏洞复现。给定 codebase 与漏洞描述，要求生成可触发漏洞的 PoC。约 1507 个真实 OSS 漏洞任务 |
| **AgentThreatBench** | Evaluating LLM Agent Resilience to OWASP Agentic Threats / UK AISI 生态（Inspect Evals 套件）/ 2026 | inspect_evals 文档；OWASP Agentic Top 10 (2026) | 将 OWASP Agentic Top 10 威胁（记忆中毒、自主性劫持、数据外泄等）操作化为可执行任务。**为评测套件实现而非同行评审论文，且项目较新** |

**评测内容展开**：
- **Cybench**：以 CTF（夺旗赛）形式考攻防能力——给 Agent 一个含漏洞的靶机/程序环境，让它做逆向、Web 渗透、密码破解、内存取证等，最终目标是拿到藏在系统里的 flag 字符串。40 个专业级任务取自 4 场真实 CTF 竞赛，其中 17 个还拆了子任务便于细粒度诊断。
- **CyberGym**：偏"真实世界漏洞复现"而非竞赛——给 Agent 一个真实开源项目的代码库和某个漏洞的描述，要求它生成一段能**实际触发该漏洞的 PoC（概念验证）**（如让程序崩溃/越界）。约 1507 个来自真实 OSS 的漏洞，规模大、贴近安全研究实务。
- **AgentThreatBench**：不是考 Agent 去攻击别人，而是考它**自身的安全韧性**——把 OWASP 总结的十大 Agent 威胁（记忆被投毒、自主权被劫持、被诱导外泄数据等）做成可执行的攻击场景，看 Agent 能否顶住。属评测套件实现，项目较新。

---

## 五、对话-工具-用户交互基准：τ-bench 系列

- **τ-bench**：A Benchmark for Tool-Agent-User Interaction in Real-World Domains / Sierra / 2024 / arXiv:2406.12045（ICLR 2025 会议版）
- **τ²-bench**：Evaluating Conversational Agents in a Dual-Control Environment / Sierra / 2025 / arXiv:2506.07982

**评测内容**：τ-bench 在 **retail（零售客服）和 airline（航空客服）** 两个真实业务域中，评测 Agent 扮演客服、与一个由 LM 模拟的"用户"进行**多轮对话**，边聊边调用 API 工具（查订单、改地址、退换货、改签等）来解决用户诉求，同时必须**严格遵守领域策略文档**（如"退货须在 30 天内""改签要收差价"这类业务规则）。难点在于：要从模棱两可的口语里问清需求、按规则办事、还要在对话中正确地一步步操作后端。τ²-bench 进一步扩展到 **dual-control（双向控制）** 协作场景——用户和 Agent 都能操作环境（例如电信域里，客服和用户各自能改一部分设置，需要协同排障），新增 telecom 等域。

**方法亮点**：以最终数据库状态是否符合唯一目标状态计 pass（基于状态验证）；提出 **pass^k** 指标——同一任务 i.i.d. 重复 k 次**全部成功**的概率，用以衡量 Agent 的一致性/可靠性（区别于 code 领域"至少一次成功"的 pass@k）。该系列是后文"幻觉成功"研究的重要数据来源。

---

## 六、评估基础设施与方法论重构

### 6.1 标准化基础设施

**HAL (Holistic Agent Leaderboard)**
- Princeton（Kapoor、Narayanan 等）/ 2025 / arXiv:2510.11977（ICLR 2026 poster）；hal.cs.princeton.edu
- 提供跨基准统一、可复现的评测基础设施，引入**模型 × 脚手架 × 基准**三维分析与 **accuracy-cost Pareto** 成本感知机制。其大规模实证揭示反直觉现象：在相当比例的环境组合中，分配更多推理 token 反而降低准确率，挑战简单的"算力换智能"缩放假设。

**Inspect AI**
- UK AI Security Institute (AISI) / 2024 起持续维护 / 开源框架（非论文）；inspect.aisi.org.uk
- 事实标准级评测中枢：提供 Task/Solver/Scorer 抽象、工具调用与 Agent 编排，原生支持 Docker/K8s 等沙盒隔离，`inspect_evals` 收录大量社区基准实现（含 SWE-bench、OSWorld、Cybench、AgentThreatBench 等）。

### 6.2 "幻觉成功"（False Success）现象

- **From Confident Closing to Silent Failure: Characterizing False Success in LLM Agents** / arXiv:2606.09863（2026）
- 刻画并检测 Agent "自信声称成功、实际失败"的现象。研究在缺乏外部状态验证的纯 API 基准（如 AppWorld）中发现大量失败任务被 Agent 自信声明为"已完成"，并指出 **LLM-as-a-judge** 在此类场景近乎失效（AUROC 接近随机），因为裁判锚定于 Agent 的肯定性陈述而非底层 API 调用逻辑。
- **核实提示**：原报告所述"用 TF-IDF 序列探测器替代 LLM 裁判、AUROC 达 0.83/0.95、延迟低 3300 倍"等具体数字，论文标题与主题吻合，但该方法的精确技术细节在可检索摘要中未完全坐实，**引用前建议核对 arXiv:2606.09863 正文**。结论方向（唯有基于状态变更的验证才可靠）与 τ-bench / AppWorld 的设计理念一致。

### 6.3 IRT 与中等难度过滤：低成本可靠排行榜

- **Efficient Benchmarking of AI Agents** / F. S. Ndzomga / 2026 / arXiv:2603.23749（个人预印本，未见同行评审 venue）
- 将心理测量学的**项目反应理论（Item Response Theory, IRT）**引入 Agent 评测：用双参数逻辑模型（2PL）将任务固有难度 $b_i$ 与 Agent 潜在能力 $\theta_j$ 解耦，解释"简单平均法"在不同难度子集上失效的原因。
- 提出**中等难度过滤器（Mid-Range Difficulty Filter）**：剔除所有模型都易解（通过率 >70%）或都束手无策（<30%）的极端任务，只保留区分度最高的中等难度子集。利用"绝对分数易漂移、相对排名稳健"的实证不对称性，在保持排行榜保真度（Spearman $\rho > 0.90$）的同时大幅削减评测任务量与算力成本。

---

## 七、综合观察

1. **验证机制是分水岭**：从 OSWorld 的系统级断言、AppWorld 的状态单元测试、AndroidWorld 的 SQLite/ADB 校验，到 τ-bench 的数据库状态比对，"基于底层状态变更的功能性验证"已取代动作匹配和文本相似度，成为可信评测的共识。"幻觉成功"研究进一步证明：任何依赖 Agent 自述或 LLM 裁判语义判断的评分都不可靠。

2. **从单一准确率到多维评估**：HAL 的成本-性能 Pareto、τ-bench 的 pass^k 一致性、IRT 的难度-能力解耦，共同推动评估从"一个成功率数字"走向"准确率 × 成本 × 可靠性"的多维刻画。

3. **基础设施标准化**：Inspect AI 与 HAL 解决了异构基准代码不兼容、环境配置脆弱、不可信代码隔离等工程痛点，使大规模、可复现、跨基准的 Agent 对比成为现实。

4. **引用风险提醒**：本领域 2026 年的新工作多为 arXiv 预印本或工程项目，部分名称（ExpGraph）疑似虚构、部分变体（WebArena Verified Hard / Infinity）定位不清、部分性能数字属推测。撰写正式论文引用时，务必回到 arXiv/官方页核对一手来源。

---

## 附录：参考文献速查表

| 基准/方法 | arXiv / URL | venue | 年份 |
|---|---|---|---|
| OSWorld | arXiv:2404.07972 | NeurIPS 2024 D&B | 2024 |
| Windows Agent Arena | arXiv:2409.08264 | PMLR 2025 | 2024 |
| AndroidWorld | arXiv:2405.14573 | ICLR 2025 | 2024 |
| WebArena | arXiv:2307.13854 | ICLR 2024 | 2023 |
| AppWorld | arXiv:2407.18901 | ACL 2024 (Best Resource) | 2024 |
| LOOP (Apple) | arXiv:2502.01600 | arXiv | 2025 |
| CUGA (IBM) | arXiv:2503.01861 | arXiv / AAAI-IAAI 2026 | 2025 |
| SWE-bench | arXiv:2310.06770 | ICLR 2024 | 2023 |
| SWE-bench Verified | openai.com/index/introducing-swe-bench-verified | OpenAI blog | 2024 |
| MLE-bench | arXiv:2410.07095 | ICLR 2025 | 2024 |
| DataSciBench | arXiv:2502.13897 | Findings of ACL 2026 | 2025 |
| CORE-bench | arXiv:2409.11363 | TMLR | 2024 |
| SciCode | arXiv:2407.13168 | NeurIPS 2024 D&B | 2024 |
| LAB-Bench | arXiv:2407.10362 | arXiv | 2024 |
| Humanity's Last Exam | arXiv:2501.14249 | arXiv / Nature | 2025 |
| AIME 2025 | math-ai-org/aime25；Inspect Evals | 竞赛数据集 | 2025 |
| Cybench | arXiv:2408.08926 | ICLR 2025 | 2024 |
| CyberGym | arXiv:2506.02548 | arXiv (ICLR 2026 投稿) | 2025 |
| AgentThreatBench | inspect_evals / OWASP Agentic Top 10 | 评测套件 | 2026 |
| HAL | arXiv:2510.11977 | ICLR 2026 | 2025 |
| Inspect AI | inspect.aisi.org.uk | 开源框架 | 2024 |
| τ-bench | arXiv:2406.12045 | ICLR 2025 | 2024 |
| τ²-bench | arXiv:2506.07982 | arXiv | 2025 |
| False Success 检测 | arXiv:2606.09863 | arXiv | 2026 |
| Efficient Benchmarking (IRT/MR) | arXiv:2603.23749 | arXiv | 2026 |
