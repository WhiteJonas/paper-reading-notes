### 1.要解决的主要问题

​	flashattention-2在 新一代 GPU (H100) 上仍然存在问题：

| 方法             | GPU利用率 |
| ---------------- | --------- |
| FlashAttention-2 | ~35%      |
| GEMM (矩阵乘)    | 80%~90%   |

​	attention kernel 没有充分利用 GPU 的计算能力。

### 2.核心技术

#### 技术1：Producer-Consumer 异步执行（Warp Specialization）

##### 核心思想

利用 GPU 的 异步执行能力：

GPU 中有不同硬件单元：

| 单元        | 作用     |
| ----------- | -------- |
| Tensor Core | 矩阵乘   |
| TMA         | 内存加载 |
| CUDA Core   | 其他计算 |

FlashAttention-3：

把线程分成两类

```
Producer warp  → 负责数据加载
Consumer warp  → 负责计算
```

形成流水线：

```
HBM → Shared Memory → TensorCore
```

同时进行：

```
加载下一块数据
+
计算当前块
```

这样：

内存访问 latency 被隐藏。

------

##### 论文实现

使用：

- warp specialization
- TMA async copy
- circular shared memory buffer

执行流程：

```
Producer:
   load K,V blocks

Consumer:
   compute QK^T
   softmax
   multiply V
```

两者 并行执行。 flashattention3

------

#### 技术2：GEMM 与 Softmax 重叠执行

在传统 Attention：

```
1 QK^T
2 softmax
3 PV
```

这是严格 顺序依赖。

FlashAttention-3 的创新：

把 softmax 隐藏在 GEMM 中执行

形成流水线：

```
Block 1: softmax
Block 2: GEMM
Block 3: GEMM
```

示意：

```
时间轴

GEMM block1
        softmax block1
GEMM block2
        softmax block2
GEMM block3
```

通过：

- 异步 WGMMA
- 两阶段 pipeline

实现：

```
softmax 与 GEMM 重叠
```

效果：

隐藏 softmax latency。 flashattention3

------

#### 技术3：FP8 低精度 Attention

H100 支持 FP8 Tensor Core：

吞吐量：

| 精度 | 吞吐 |
| ---- | ---- |
| FP16 | 1x   |
| FP8  | 2x   |

但 FP8 会带来：

- 量化误差
- outlier 特征问题

论文提出两个关键技术：

------

##### 1 Block Quantization

不是整个 tensor 一个 scale：

```
Tensor quantization
```

而是：

```
Block quantization
```

示意：

```
Matrix
[ block ][ block ]
[ block ][ block ]
```

每个 block 一个 scale。

优点：

- 精度更高
- 减少溢出

------

##### 2 Incoherent Processing

将数据打乱 / 分块处理：

减少 outlier feature 的影响。

最终效果：

> FP8 attention 比普通 FP8 实现 误差降低 2.6×。 flashattention3

------

### 三、FlashAttention-3整体算法流程

总体流程（简化版）：

```
for query block Qi:

    load Qi

    for key block Kj:

        async load Kj, Vj

        compute Sij = Qi * Kj^T

        update online softmax

        compute O += Pij * Vj
```

关键优化：

1️⃣ block tiling
 2️⃣ online softmax
 3️⃣ async pipeline
 4️⃣ FP8 tensor core