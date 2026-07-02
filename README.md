# 大模型推理优化 · 论文阅读笔记

自驱学习 LLM / Agent 推理优化方向的论文阅读笔记与讲解 PPT。围绕**大模型推理系统、请求调度、注意力算子、Agent 服务与评测基准**几个主题,持续精读近年顶会(ICML / NeurIPS / OSDI / MLSys / ATC / HPDC 等)论文,整理成带自己理解与分析的中文笔记,部分论文做了讲解 PPT。

> 这里收录的是自己动手写的分析、综述与方向调研笔记(逐段照搬的纯翻译稿未收录),以及讲论文时做的 PPT。图片统一放在 [`assets/`](./assets)。

## 目录结构

| 分类 | 内容 |
| --- | --- |
| [01-注意力机制与推测解码](./01-注意力机制与推测解码) | 注意力机制与 RNN 基础、FlashAttention-2/3、skipGPT 层剪枝;MInference / IAM / MEDUSA 讲解 PPT |
| [02-推理系统与调度](./02-推理系统与调度) | vLLM / PagedAttention 机制详解、vLLM 架构设计、基于长度预测的 LLM 调度综述、推理调度综述、GPU 组成 |
| [03-Agent服务与调度](./03-Agent服务与调度) | Agent 推理加速综述、Agent 场景论文分类、AgentServeSim 核心方法、Agent 服务场景调度问题;SGAG / continuum 等讲解 PPT |
| [04-Benchmark与评测](./04-Benchmark与评测) | τ-bench 工具沙盒机制分析、Agent benchmark 调研、LLM/Agent 性能评测基准综述与全景报告 |

## 关注的核心问题

- **推理服务性能**:TTFT / TPOT / Goodput 与 SLO 之间的权衡
- **请求调度**:基于输出长度预测的调度、抢占与资源分配
- **Agent 场景**:多轮工具调用下的 serving 优化与评测方法
- **算子与内存**:注意力稀疏化、KV Cache 管理

## 说明

笔记以 Markdown 为主、配图内联(集中在 `assets/`);`.pptx` 为对应论文的讲解材料。内容为个人学习整理,如涉及他人论文观点,版权归原作者所有。
