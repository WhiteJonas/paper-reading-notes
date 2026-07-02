# vLLM PagedAttention 机制详解

> 面向 vLLM 调度研究者的深度解读：从问题动机、核心设计、内存管理、内核实现，到对调度/SLO/Goodput 的影响。

---

## 1. 为什么需要 PagedAttention：传统 KV Cache 的痛点

LLM 自回归推理中，每生成一个 token 就要做一次 attention，计算时需要前面所有 token 的 Key/Value。为了避免重复计算，主流框架都会缓存这些 KV，称为 **KV Cache**。

### 1.1 传统连续分配的问题

在 PagedAttention 之前，主流做法是**为每个请求预分配一段连续显存**，长度等于 `max_seq_len`：

```
Request A: [■■■■■■■■░░░░░░░░░░░░░░░░░░░░░░░░]   max=2048, 实际用 512
Request B: [■■■░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   max=2048, 实际用 192
Request C: [■■■■■■■■■■■■■■■■■■■■■■■■■░░░░░░░]   max=2048, 实际用 1600
```

这带来三类碎片：

| 碎片类型     | 含义                                                 | 浪费比例（论文实测） |
| ------------ | ---------------------------------------------------- | -------------------- |
| **内部碎片** | 请求实际生成长度 < max_seq_len，剩余部分浪费         | 60–80%               |
| **外部碎片** | 不同请求的预留区之间无法拼接给新请求用               | 显著                 |
| **预留碎片** | 已分配但当前还没写入的部分（未来要用，但当下空着）   | 视生成进度而定       |

论文 *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP'23) 指出，传统方案 KV Cache 的**有效利用率仅 20–40%**。在 A100-80G 上跑 13B 模型，剩下的显存常常只够同时跑十几个请求，吞吐受限。

### 1.2 借鉴 OS 虚拟内存

操作系统遇到过类似问题：进程要的是逻辑上"一段连续的地址空间"，但物理内存按 4KB 页（page）分散管理，靠页表把虚拟地址映射到物理页。这样：

- 物理页可被任意进程使用，无外部碎片
- 进程按需分配页，无内部碎片
- 不同进程可共享物理页（COW、共享库）

PagedAttention 把同一思路搬到 KV Cache：**逻辑上一个序列的 KV 是连续的，物理上分散在固定大小的 KV Block 里，靠 block table 做映射**。

---

## 2. 核心抽象：KV Block 与 Block Table

### 2.1 KV Block（物理块）

GPU 上预分配一大块显存，整体是一个形如 `[num_blocks, block_size, num_heads, head_dim]` 的张量（K 和 V 各一份）。其中：

- `block_size`：每块容纳的 token 数，通常 16（也可设 8/32）
- `num_blocks`：在剩余显存里能切多少块，启动时算好

**关键特性**：所有块**大小相同、连续存放在一整块显存里**。"块"只是逻辑切分，`block_id` 就是这个大张量的下标。

### 2.2 Block Table（页表）

每个请求维护一个 `block_table: List[block_id]`，按 token 顺序排列。

```
请求 A 的逻辑视角:
  tokens:    [t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 ...]
  逻辑块号:  [    block 0           ][    block 1          ][...]
  block_table = [7, 2, 9, ...]    # 物理块 id

物理显存视角（block_size=4 示意）:
  block 0:  [  -  -  -  - ]
  block 1:  [  -  -  -  - ]
  block 2:  [ t4 t5 t6 t7 ]   ← A 的逻辑块 1 落在物理块 2
  ...
  block 7:  [ t0 t1 t2 t3 ]   ← A 的逻辑块 0 落在物理块 7
  block 9:  [ t8 t9 ?? ?? ]   ← A 的逻辑块 2 落在物理块 9，还没写满
```

定位第 `i` 个 token：
- 块内偏移：`i % block_size`
- 物理块：`block_table[i // block_size]`

block_table 是普通**数组**，不是链表——attention kernel 要随机访问任意 token 位置，链式遍历会是性能灾难。

### 2.3 Free Block Queue（空闲块池）

一个**双向链表**串联所有当前未被占用的 block：

- 新请求 / 新 token 要写入时：从队首 pop 一个 block_id
- 请求结束 / 块被驱逐：push 回队尾
- prefix caching 命中时：可能要把链表**中间**某个块直接摘出来给新请求 → 因此必须双向链表，O(1) 中删

vLLM v1 在 `vllm/v1/core/kv_cache_utils.py` 的 `FreeKVCacheBlockQueue` 里实现，每个 `KVCacheBlock` 自带 `prev_free_block` / `next_free_block` 指针。CPU 端指针操作微秒级，完全不会成为瓶颈。

---

## 3. PagedAttention Kernel：分块计算 attention

逻辑映射搞定后，attention kernel 必须能**直接在分散的物理块上工作**，不能要求显存连续。

### 3.1 单个 query 的 attention 公式

对解码阶段的某个 query token `q`，需要：

```
A_j = exp(q · k_j / √d) / Σ exp(q · k_i / √d)
out = Σ A_j · v_j
```

`k_j, v_j` 来自历史所有 token 的 KV。

### 3.2 PagedAttention 的分块迭代

kernel 按 block 为粒度遍历：

```python
for logical_block_idx in range(num_blocks_for_this_seq):
    physical_block_id = block_table[logical_block_idx]
    K_block = K_cache[physical_block_id]   # [block_size, num_heads, head_dim]
    V_block = V_cache[physical_block_id]

    # 在该块内做局部 softmax + 加权求和
    partial_logits = q @ K_block.T
    # 与之前块的结果做 online-softmax 合并
    out = merge(out, softmax(partial_logits) @ V_block)
```

每个 CUDA block / warp 负责若干 head × 若干物理块，按 **online softmax**（FlashAttention 同款技巧）逐块累积 max/sum/output 三个量，避免回头扫描。

### 3.3 实现位置

- **v0**：`csrc/attention/attention_kernels.cu`，自研的 paged attention v1/v2 kernel
- **v1**：默认走 FlashAttention/FlashInfer 后端，它们都已原生支持 paged KV layout
- 推理时 `block_table` 作为 tensor 一起传给 kernel，kernel 内做 gather

block_size=16 是经验最优值：太小时 block_table 间接寻址开销占比高，kernel 启动开销也变多；太大又退化成连续分配，碎片重新出现。

---

## 4. 内存管理流程（端到端）

### 4.1 启动阶段

1. 加载模型权重，扣掉 activation buffer，估出 KV Cache 可用显存 `M_kv`
2. 单块大小 `B = block_size × num_kv_heads × head_dim × 2(bytes for fp16) × num_layers × 2(K+V)`
3. `num_blocks = M_kv // B`
4. 分配 `[num_blocks, ...]` 大张量；构建 `FreeKVCacheBlockQueue`，把所有 block_id 串进去

### 4.2 请求到达（Prefill）

Prompt 长 `L`，需要 `⌈L / block_size⌉` 个块：

1. 从 free queue 头部 pop 出对应数量的 block_id
2. 写入请求的 block_table
3. Prefill 一次性算出所有 KV，按映射写入物理块
4. 最后一个块通常没写满，剩余槽位留给后续 decode 使用

### 4.3 Decode 阶段（每一步）

每生成一个 token：

1. 看当前最后一个 block 还有没有空槽
   - 有 → 直接写入，`block_table` 不变
   - 没有 → 从 free queue 再 pop 一个块 append 到 block_table 末尾
2. 调用 paged attention kernel 算下一步

### 4.4 请求结束 / 被抢占

- **结束**：释放该请求所有 block_id，push 回 free queue
- **被抢占**（资源不够）：vLLM 有两种策略
  - **Recompute**：直接释放，未来重新调度时重 prefill（适合短 prompt）
  - **Swap**：把这些 block 拷到 CPU 内存，恢复时再拷回（适合长 prompt）

---

## 5. PagedAttention 的两个高阶能力

### 5.1 Copy-on-Write 与 Beam Search / Parallel Sampling

同一个 prompt 用 beam search 生成 4 条不同序列时，前缀 KV 完全相同。PagedAttention 让多个 sequence 的 block_table 指向**同一批物理块**：

- 每块带一个 `ref_count`
- 读：自由共享
- 写：发现 `ref_count > 1` → 申请新块、复制内容、`ref_count -= 1`，新 sequence 改写新块

显存节省 = (并行采样数 - 1) × 共享前缀长度。

### 5.2 Prefix Caching（跨请求复用）

观察：很多请求 prompt 前缀相同（system prompt、few-shot 模板、Agent tool 定义、多轮对话历史）。这些前缀的 KV 算一次就行。

实现要点：

- 每个**满块**计算 hash：`hash(parent_block_hash, tuple(token_ids))`，链式依赖前一块
- 注册到 `cached_block_hash_to_block: Dict[hash, block]`
- 新请求来 → 按块切 token → 逐块查表
  - 命中：`ref_count += 1`，若该块还在 free queue 里则**中间摘出**
  - 未命中：从命中位置之后开始正常 prefill
- 块被释放时不立即清空，先进 free queue 末尾，`ref_count=0` 但 hash 仍可命中；只有真被复用才正式覆盖（**LRU 风格的隐式驱逐**）

代码位置：`vllm/v1/core/kv_cache_manager.py` 和 `kv_cache_utils.py`。

---

## 6. 对调度 / SLO / Goodput 的影响（研究视角）

这部分对调度方向特别相关，单独展开。

### 6.1 让 batch 大小变得"软性可调"

传统连续分配下，能并发的请求数 ≈ `显存 / max_seq_len`，是个硬上限。PagedAttention 下，并发数取决于**当前实际占用的块数总和**，而非每个请求的 max_seq_len：

- 短输出请求很快释放块 → 腾出空间给新请求
- 长输出请求逐块增长，按需占用
- 调度器可以**激进地放更多请求进 running**，靠抢占兜底

直接影响：vLLM 的 max_num_seqs 在同等显存下能比传统方案高 2–4 倍，吞吐随之提升。

### 6.2 影响 TTFT/TPOT 预测的关键变量

如果把 PagedAttention 的能力当作黑盒做调度建模，会在两个地方踩坑：

1. **prefill 长度 ≠ prompt 长度**（开了 prefix caching 后）
   - 实际 prefill FLOPs 只跟 cache miss 段成正比
   - Agent 场景里 tool def + 历史轨迹命中率常 >90%，按裸 prompt 长度估 TTFT 会严重高估

2. **decode 也可能因为块不足被抢占**
   - 一个表面在 running 的请求可能被 swap 出去几百毫秒
   - TPOT 不是稳定的，分布有长尾，需要预测器考虑抢占概率

### 6.3 与"输出长度预测调度"的耦合点

你做的方向（基于输出长度预测的调度）跟 PagedAttention 有几个直接的接口：

- **预测短输出 → 优先调度**：短请求很快还块，相当于增加全局空闲块速率，改善整体队列等待 → TTFT 收益
- **预测长输出 → 推迟或单独通道**：避免它长期占用大量块，挤占并发位
- **驱逐策略选择**：被预测为**输出长**的请求在被抢占时倾向 **swap**（避免重复 prefill 长 prompt），输出短的倾向 **recompute**（快、还省 PCIe 带宽）
- **Prefix block 保留权重**：被命中过的前缀块如果属于"高命中率模板"，应延迟驱逐；这又跟请求长度分布相关
- **Goodput 优化**：在 SLO 约束下最大化每秒完成请求数，块占用速率是天然的中间变量——长度预测 → 块占用预测 → 调度排序

### 6.4 Chunked Prefill 与 PagedAttention 的协同

vLLM 支持把长 prompt 的 prefill 切成多 chunk，与 decode 混合调度。PagedAttention 让这种混合成为可能：

- prefill 块和 decode 块可以同属一个 batch，attention kernel 都按 block_table 取 KV，不区分
- 调度器决定每步给 prefill 多少 token budget，剩下给 decode
- 长度预测可以反过来指导 chunk 大小选择：预计输出长 → 把 prefill 切小，避免一次性塞太多 KV 块

---

## 7. 一张图总结数据流

```
                ┌────────────────────────────────────────────┐
                │              GPU KV Cache 大张量            │
                │  shape = [num_blocks, block_size, H, D]    │
                └────────────────────────────────────────────┘
                              ▲          ▲           ▲
                  block_id=2  │ id=7     │ id=9      │ ...
                              │          │           │
       ┌──────────────────────┼──────────┼───────────┼────────┐
       │ Request A            │          │           │        │
       │ block_table = [ 7, 2, 9, ... ]                       │
       │   逻辑块0=7  逻辑块1=2  逻辑块2=9（未写满）          │
       └──────────────────────────────────────────────────────┘

       ┌──────────────────────────────────────────────────────┐
       │ Free Block Queue (双向链表)                          │
       │   head ⇄ block_5 ⇄ block_8 ⇄ block_3 ⇄ ... ⇄ tail  │
       │   pop: 分配  ;  push: 回收  ;  中删: prefix 命中      │
       └──────────────────────────────────────────────────────┘

       ┌──────────────────────────────────────────────────────┐
       │ Prefix Cache: hash → block                            │
       │   hash(prompt_prefix_block) → block_id                │
       └──────────────────────────────────────────────────────┘
```

---

## 8. 关键源码索引（vLLM v1，便于追读）

| 模块            | 文件                                          | 关注点                               |
| --------------- | --------------------------------------------- | ------------------------------------ |
| Block / 队列    | `vllm/v1/core/kv_cache_utils.py`              | `KVCacheBlock`, `FreeKVCacheBlockQueue`, hash 计算 |
| KV Cache Manager| `vllm/v1/core/kv_cache_manager.py`            | 分配/释放/前缀命中主逻辑             |
| 调度器          | `vllm/v1/core/sched/scheduler.py`             | 与 manager 的交互、抢占决策          |
| Attention 后端  | `vllm/v1/attention/backends/`                 | Flash/FlashInfer 的 paged 适配       |
| 物理张量分配    | `vllm/v1/worker/gpu_model_runner.py`          | KV Cache 张量的实际创建              |

读源码顺序建议：`kv_cache_utils.py` → `kv_cache_manager.py` → `scheduler.py` 与 manager 的接口 → 后端 attention kernel 调用层。

---

## 9. 一句话总结

PagedAttention = **把 OS 的虚拟内存分页思想搬到 GPU KV Cache**，用固定大小的物理块 + 每请求 block table，把"按需分配、灵活共享、零外部碎片"这三件事一次解决；它既是显存效率的基础设施，也是 vLLM 上层一切高级调度（连续批处理、抢占恢复、prefix caching、chunked prefill、beam 共享）能成立的前提。

对调度研究而言：PagedAttention 不仅是一项内存技术，它**重新定义了"调度的资源单位"**——从"请求 / token"细化到"block"，让基于长度预测的精细调度有了可操作的底层抓手。
