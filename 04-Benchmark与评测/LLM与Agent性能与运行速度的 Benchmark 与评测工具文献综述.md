# LLM与Agent性能与运行速度的 Benchmark 与评测工具文献综述

> 本节系统梳理正文提及的所有评测基准与性能测试工具，逐一核实真实出处（开发方/机构、年份、发表 venue、arXiv 编号或官方仓库）。所有条目均确认真实存在，无虚构。文末对正文中几处**未经一手来源证实的表述**给出核实提示，便于正式引用时回溯一手资料。

## 一、综述范围与分类

正文跨越了"运行速度/系统性能"与"Agent 任务能力"两条评测主线。据此将文献分为四类：

1. **性能/负载测试工具**：测 TTFT、TPS、延迟、吞吐等运行期指标。
2. **本地化/硬件基准工具**：面向消费级与边缘设备的推理测速。
3. **Agent 能力评测基准**：测任务完成度、工具调用、策略合规等。
4. **综合评测平台与 Agent 训练方法**：跨域评测中枢与能力获取算法。

## 二、性能 / 负载测试工具

| 工具                     | 全称 / 开发方                                               | 类型 / 年份                     | 来源                                                        | 功能与核心指标                                               |
| ------------------------ | ----------------------------------------------------------- | ------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| **LLMPerf**              | LLM Performance Benchmark / Anyscale（ray-project）         | 开源工具 / 2023（v2.0 2023-12） | github.com/ray-project/llmperf                              | 跨云 LLM API 性能基准，统一输入输出长度做横向对比。指标：TTFT、inter-token latency、throughput。核心脚本 `token_benchmark_ray.py` |
| **GenAI-Perf**           | GenAI Performance Analyzer / NVIDIA（Triton perf_analyzer） | 开源工具 / 2024                 | github.com/triton-inference-server/perf_analyzer            | 面向生成式 AI 端点（Triton/NIM、OpenAI 兼容 API）的负载测试。支持多模态/Embedding/LoRA；用滑动窗口剔除冷启动噪点。指标：TTFT、ITL、request/output throughput |
| **optimum-benchmark**    | Optimum-Benchmark / Hugging Face                            | 开源库 / 2023–2024              | github.com/huggingface/optimum-benchmark                    | Transformer 模型多后端（PyTorch/ONNX Runtime/OpenVINO/TensorRT）统一效能基准。指标：latency、throughput、memory |
| **k6 (+ xk6-sse)**       | Grafana k6 / 最初 Load Impact，后 Grafana Labs              | 开源工具 / 2016（1.0 2025-05）  | github.com/grafana/k6；xk6-sse: github.com/phymbert/xk6-sse | JS 脚本驱动的负载测试；社区扩展 `xk6-sse`（作者 phymbert）补足对 SSE 流式 LLM 端点的压测 |
| **Locust**               | Locust / 开源社区（locustio）                               | 开源工具 / 2011 起              | github.com/locustio/locust                                  | Python 编写、Master-Worker 分布式负载注入，模拟大量并发用户  |
| **FutureAGI Simulation** | Future AGI — Simulate / Future AGI（商业公司）              | 商业平台 / 2025–2026            | futureagi.com/platform/simulate                             | **persona 驱动**的 Agent 仿真测试：用模拟客户画像做多轮场景化测试，将测试与效果评估融合在同一追踪平面 |

## 三、本地化 / 硬件基准工具

| 工具             | 开发方                     | 类型 / 年份                       | 来源                                               | 功能与指标                                                   |
| ---------------- | -------------------------- | --------------------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| **llama-bench**  | llama.cpp 项目（ggml-org） | 内置 CLI / 2023 起                | github.com/ggml-org/llama.cpp（tools/llama-bench） | 直接对 C++ 推理引擎测速，规避网络/API 开销。指标：prompt processing (pp, prefill) 与 text generation (tg, decode)，tokens/s 含标准差，支持 markdown/CSV/JSON/SQL 输出 |
| **llama-benchy** | 社区开发者 eugr            | 开源工具 / 2026 初（PyPI v0.3.8） | github.com/eugr/llama-benchy                       | "llama-bench 风格"但面向任意后端（vLLM/SGLang/llama.cpp），经 OpenAI 兼容 API 测速；区分 prompt processing 与 context prefill。**注**：正文所述"长上下文衰减测试 + 古登堡文学文本 prompt"为很新的个人项目特性，引用前建议核对 README |

## 四、推理引擎（正文重点横向对比对象）

正文以 H100 上的吞吐/延迟数据对比三大引擎，其架构出处与机制如下：

- **vLLM**：UC Berkeley 提出，首创 **PagedAttention**（KV Cache 分页，借鉴 OS 虚拟内存）。论文 *Efficient Memory Management for LLM Serving with PagedAttention*，SOSP 2023，arXiv:2309.06180。
- **SGLang**：**RadixAttention**（KV Cache 组织为前缀树以做 token 级全局共享）。论文 *SGLang: Efficient Execution of Structured Language Model Programs*，NeurIPS 2024，arXiv:2312.07104。
- **TensorRT-LLM**：NVIDIA 官方推理库，预编译静态图 + 算子融合（开源工程库，非论文；github.com/NVIDIA/TensorRT-LLM）。

> 引用提示：正文表 3.2 的具体跑分（如 SGLang 16,215 tok/s、TensorRT-LLM P95 1,280ms、编译 28 分钟）出自第三方测评（Spheron / AIMultiple），属特定软硬件配置下的实测，引用时应注明来源与版本，不宜作为引擎的固有性能断言。

## 五、Agent 能力评测基准（正文"五大基准"等）

| 基准                     | 全称 / 机构 / venue                                          | 来源                                                         | 评测内容与方法 / 规模                                        |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **SWE-bench / Verified** | Can Language Models Resolve Real-World GitHub Issues? / Princeton+UChicago（Verified 由 OpenAI 协作）/ ICLR 2024 | arXiv:2310.06770；Verified: openai.com/index/introducing-swe-bench-verified | 解决真实 GitHub issue 生成补丁；% Resolved（能否通过仓库测试套件）。原 2294 任务，Verified 人工筛选 500 任务 |
| **GAIA**                 | a Benchmark for General AI Assistants / Meta AI、HuggingFace、AutoGPT 等 / ICLR 2024 | arXiv:2311.12983                                             | 通用 AI 助手真实问题，需多步推理+网页浏览+多模态+工具调用。准确率（quasi-exact-match），分 Level 1/2/3 三难度。466 题 |
| **τ-bench (TAU-bench)**  | A Benchmark for Tool-Agent-User Interaction / Sierra / ICLR 2025 | arXiv:2406.12045                                             | retail/airline 企业客服模拟，测策略合规与多轮对话+工具调用。基于数据库最终状态计 pass，提出 **pass^k**（k 次全部成功率）衡量一致性。τ²-bench (arXiv:2506.07982) 扩展到 dual-control |
| **WebArena**             | A Realistic Web Environment for Building Autonomous Agents / CMU / ICLR 2024 | arXiv:2307.13854                                             | 自托管真实网站副本（电商/论坛/GitLab/CMS），812 任务；基于功能正确性（结果状态）自动打分 |
| **AgentBench**           | Evaluating LLMs as Agents / 清华 THUDM 等 / ICLR 2024        | arXiv:2308.03688                                             | **8 个异构环境**（OS、DB、知识图谱 KG、卡牌 DCG、谜题 LTP、ALFWorld、WebShop、Mind2Web），约 1091 任务；各环境成功率汇总，量化多轮交互与运维成本 |

> 引用提示：正文 6.1 关于"AutoGen 单任务 $0.45 vs LangGraph $0.08""θ=0.87 混合架构""流式执行降延迟 89%/每千次 23 次竞态故障"等数字，未在本次核实范围内追溯到一手论文，应视为示意性数据，正式引用前需补一手来源。

## 六、综合评测平台与 Agent 训练方法

**OpenCompass（司南）** — 上海人工智能实验室 / 2023 / opencompass.org.cn、github.com/open-compass/opencompass

- 通用大模型评测平台，内嵌 MMLU、MathBench、DevBench 等上百数据集；含启发式任务切分以平衡载入与算力开销。
- 配套验证/评判模型：**CompassVerifier**（统一鲁棒的评测与奖励验证器，EMNLP 2025，aclanthology.org/2025.emnlp-main.1698）、**CompassJudger**（all-in-one LLM-as-judge）。
- 旗下数据集：**MathBench**（分层数学基准，arXiv:2405.12209）、**DevBench**（软件开发生命周期代码生成基准，arXiv:2403.08604）。

**SuperCLUE** — CLUE 团队 / 2023 / arXiv:2307.15020、superclueai.com

- 中文大模型综合基准，多维度评测（含基础能力、中文特性、专业知识、安全性），常做开源/闭源、国内/海外对比。
- 引用提示：正文"四象限雷达"具体表述未从官方材料直接证实，建议核对官方榜单页用词。

**Agent-FLAN 系列（Agent 能力获取方法）** — 上海 AI 实验室 / InternLM / ACL 2024 Findings / arXiv:2403.12881

- **Agent-FLAN**：通过能力解耦（分离格式遵循与通用推理）+ 负样本学习消除工具调用幻觉，Llama2-7B 上相对基线约 +3.5%。
- **T-Eval**（arXiv:2312.14033, ACL 2024）：将 tool-use 能力**逐步分解**为 Planning/Reasoning/Retrieval/Understanding/Instruct/Review 等维度分项评测。
- **Agent-H**：Agent-FLAN 论文内提出的**幻觉专项评测**（区分格式幻觉与动作幻觉），非独立发表工作，应表述为"Agent-FLAN 提出的 Agent-H"。

## 七、参考文献速查表

| 名称                    | arXiv / URL                                      | venue / 类型      | 年份        |
| ----------------------- | ------------------------------------------------ | ----------------- | ----------- |
| vLLM (PagedAttention)   | arXiv:2309.06180                                 | SOSP 2023         | 2023        |
| SGLang (RadixAttention) | arXiv:2312.07104                                 | NeurIPS 2024      | 2023        |
| TensorRT-LLM            | github.com/NVIDIA/TensorRT-LLM                   | 开源工程库        | 2023        |
| LLMPerf                 | github.com/ray-project/llmperf                   | 开源工具          | 2023        |
| GenAI-Perf              | github.com/triton-inference-server/perf_analyzer | 开源工具          | 2024        |
| optimum-benchmark       | github.com/huggingface/optimum-benchmark         | 开源库            | 2023        |
| k6 / xk6-sse            | github.com/grafana/k6                            | 开源工具          | 2016        |
| Locust                  | github.com/locustio/locust                       | 开源工具          | 2011        |
| FutureAGI Simulate      | futureagi.com/platform/simulate                  | 商业平台          | 2025+       |
| llama-bench             | github.com/ggml-org/llama.cpp                    | 内置 CLI          | 2023        |
| llama-benchy            | github.com/eugr/llama-benchy                     | 开源工具          | 2026        |
| SWE-bench / Verified    | arXiv:2310.06770                                 | ICLR 2024         | 2023        |
| GAIA                    | arXiv:2311.12983                                 | ICLR 2024         | 2023        |
| τ-bench / τ²-bench      | arXiv:2406.12045 / 2506.07982                    | ICLR 2025 / arXiv | 2024 / 2025 |
| WebArena                | arXiv:2307.13854                                 | ICLR 2024         | 2023        |
| AgentBench              | arXiv:2308.03688                                 | ICLR 2024         | 2023        |
| OpenCompass             | github.com/open-compass/opencompass              | 开源平台          | 2023        |
| CompassVerifier         | aclanthology.org/2025.emnlp-main.1698            | EMNLP 2025        | 2025        |
| MathBench               | arXiv:2405.12209                                 | 开源基准          | 2024        |
| DevBench                | arXiv:2403.08604                                 | 开源基准          | 2024        |
| SuperCLUE               | arXiv:2307.15020                                 | 开源基准          | 2023        |
| Agent-FLAN              | arXiv:2403.12881                                 | ACL 2024 Findings | 2024        |
| T-Eval                  | arXiv:2312.14033                                 | ACL 2024          | 2023        |

## 八、需二次核对的表述（正文 vs 一手来源）

1. **LLMPerf "已归档"**：仓库仍有活跃 release，"Archived"未证实，断言前请核对仓库顶部状态。
2. **llama-benchy 的古登堡长上下文测试**：工具真实，但该特性需核对 README。
3. **FutureAGI**：官方定位是"persona 驱动 Agent 仿真测试"，"负载/压力测试"表述更准确的说法是仿真测试。
4. **SuperCLUE "四象限雷达"**：需核对官方用词。
5. **引擎跑分、AutoGen/LangGraph 成本、混合架构 θ=0.87 等数字**：均为第三方测评或示意数据，引用须注明来源。