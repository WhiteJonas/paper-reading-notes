# vllm 代码

### 1.v1/engine/core.py

​	实现 EngineCore 类。核心：step()函数

```
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """调度、执行并生成输出。

        返回输出以及一个标志，该标志指示模型是否被执行。
        """

        # 检查 scheduler 中是否还有剩余请求——未完成的，
        # 或已完成但尚未从 batch 中移除的。
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # 在处理模型输出之前，先处理模型执行期间发生的
        # 任何中止操作。
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

​	负责执行单步的推理。首先执行调度器的调度函数，确定当前批次进行哪些请求的执行。然后执行 llm 的推理/预填充过程，最后对模型输出的结果进行采样，返回这一步的输出内容。

### 2.v1/request.py

定义了一个 Request 类

```
class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) 
```

| 字段                      | 含义                                      |
| ------------------------- | ----------------------------------------- |
| `num_computed_tokens`     | 已算到第几个 token（prefill+decode 累计） |
| `max_tokens`              | 该请求最多生成多少 output token           |
| `priority`                | 优先级（PRIORITY 调度用，越小越优先）     |
| `arrival_time`            | 到达时间（同优先级时的 tiebreak）         |
| `status`                  | 状态机                                    |
| `num_preemptions`         | 被抢占次数（可作为饥饿/抖动信号）         |
| `spec_token_ids`          | 投机 token                                |
| `num_output_placeholders` | 异步调度用的输出占位                      |

##### 请求的status：WAITING、RUNNING、FINISHED_*、PREEMPTED

WAITING：正在等待调度

RUNNING：请求已经被调度进 running batch，持有 KV cache，并会在后续 step 中继续被调度。

FINISHED_*：已经完成的请求

PREEMPTED: 被强占的请求，等待恢复

状态机：WAITING / WAITING_FOR_*  →  RUNNING  ⇄  PREEMPTED  →  FINISHED_*

### 3.v1/core/sched/scheduler.py

vllm 里的核心调度器

##### 核心函数：schedule()

在调度器中没有显式的“解码阶段”或“预填充阶段”，每个请求只维护 ：

```
num_computed_tokens
num_tokens_with_spec
```

num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)

调度器在每一步都会尝试为请求分配 token，使其 num_computed_tokens追上 num_tokens_with_spec，这一机制足够通用，可以覆盖 chunked prefill、prefix caching、speculative decoding 以及未来的 jump decoding 优化。

步骤0.初始化本轮预算和临时记录结构

步骤1.先调度 running 请求

running 请求已经在 batch 中，通常已经占有 KV cache，所以优先推进。

步骤2.再调度 waiting /skipped_waiting 请求

当 running 请求调度完，还有 token budget，才拉新请求进来。

步骤3.构造 SchedulerOutput

关键字段有：

| 字段                           | 含义                            |
| ------------------------------ | ------------------------------- |
| `scheduled_new_reqs`           | 本轮新进入 batch 的请求         |
| `scheduled_cached_reqs`        | 已在 batch 中继续执行的请求     |
| `num_scheduled_tokens`         | 每个请求本轮调度多少 token      |
| `total_num_scheduled_tokens`   | 本轮总 token 数                 |
| `scheduled_spec_decode_tokens` | 本轮要验证的 speculative tokens |
| `scheduled_encoder_inputs`     | 本轮需要计算的 encoder 输入     |
| `preempted_req_ids`            | 本轮被抢占的请求                |
| `finished_req_ids`             | 上一轮到当前轮之间完成的请求    |

步骤4.调度后更新 scheduler 内部状态

### 4.v1/core/sched/request_queue.py

核心结构：

```
class SchedulingPolicy(Enum)
class RequestQueue(ABC)
class FCFSRequestQueue(deque[Request], RequestQueue)
class PriorityRequestQueue(RequestQueue)
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue
```

```
SchedulingPolicy
    ↓
决定用哪种队列

RequestQueue
    ↓
定义统一接口

FCFSRequestQueue
    ↓
先到先服务队列

PriorityRequestQueue
    ↓
优先级队列

create_request_queue
    ↓
根据策略创建具体队列
```

class FCFSRequestQueue(deque[Request], RequestQueue)

deque[Request] 是一个真正存储请求的数据结构，RequestQueue是vLLM 定义的请求队列抽象接口。新请求从队尾进入，老请求从队头出，按照先进先出的顺序。

class PriorityRequestQueue(RequestQueue)

最小堆 heapq，每一次 heappop() 出优先级最高的请求

### 5.v1/core/sched/output.py

```
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData

    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int

    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]

    finished_req_ids: set[str]
    free_encoder_mm_hashes: list[str]

    preempted_req_ids: set[str] | None = None

    has_structured_output_requests: bool = False
    pending_structured_output_tokens: bool = False

    num_invalid_spec_tokens: dict[str, int] | None = None

    kv_connector_metadata: KVConnectorMetadata | None = None
    ec_connector_metadata: ECConnectorMetadata | None = None

    new_block_ids_to_zero: list[int] | None = None
```

SchedulerOutput 是 Scheduler 对本轮模型执行做出的完整“执行计划”。 它告诉 worker / executor / model runner：这一轮有哪些请求要跑、每个请求跑多少 token、哪些请求是新来的、哪些请求已有缓存、哪些 KV/encoder/cache 状态需要更新、哪些请求完成或被抢占。

| 核心属性                       | 类型                         | 简单解释                                                  | 作用                                                         |
| ------------------------------ | ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| `scheduled_new_reqs`           | `list[NewRequestData]`       | 本轮首次被调度的新请求列表，包含请求的完整数据            | worker 还没有缓存这些请求，必须发送完整信息，例如 prompt、采样参数、多模态输入、KV block 信息等 |
| `scheduled_cached_reqs`        | `CachedRequestData`          | 本轮继续执行的老请求的增量数据                            | worker 已经缓存过这些请求，不需要重复发送完整请求，只发送变化部分，降低通信开销 |
| `num_scheduled_tokens`         | `dict[str, int]`             | 每个请求本轮要执行多少个 token                            | 决定本轮 forward 的实际计算内容，是 SchedulerOutput 最核心的调度决策之一 |
| `total_num_scheduled_tokens`   | `int`                        | 本轮所有请求调度 token 数总和                             | 用于快速分配 tensor、统计负载、判断本轮是否有模型计算任务    |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]`       | speculative decoding 中每个请求本轮需要验证的 draft token | 支持一次验证多个候选 token，提高 decode 吞吐                 |
| `scheduled_encoder_inputs`     | `dict[str, list[int]]`       | 每个请求本轮需要处理的 encoder 输入索引                   | 用于多模态或 encoder-decoder 模型，例如处理图片、音频等 encoder 输入 |
| `num_common_prefix_blocks`     | `list[int]`                  | 每个 KV cache group 中请求共享的公共前缀 block 数         | 用于 prefix cache / cascade attention 优化，减少重复计算或优化 attention 访问 |
| `finished_req_ids`             | `set[str]`                   | 已完成、需要 worker 清理本地状态的请求 ID                 | 防止 worker 继续保留无用请求状态，释放缓存资源               |
| `free_encoder_mm_hashes`       | `list[str]`                  | 需要释放的 encoder cache 的多模态 hash                    | 用于清理不再使用的图片/音频 encoder 输出缓存                 |
| `kv_connector_metadata`        | `KVConnectorMetadata | None` | KV cache connector 的元数据                               | 用于分布式 KV cache 传输、远程 KV 加载、prefill/decode 分离等场景 |
| `new_block_ids_to_zero`        | `list[int] | None`           | 本轮新分配且需要清零的 KV block ID                        | 防止旧数据、NaN 或未初始化内容污染 attention / SSM 计算      |

1. **`v1/engine/core.py`** 的 `EngineCore.step()` —— 看清 schedule/execute/update 三拍循环（约 402 行）。
2. **`v1/core/sched/scheduler.py`** 的 `Scheduler.schedule()` —— 90% 精力在这。看 prefill/decode 怎么混批、`max_num_batched_tokens` / `max_num_seqs` 怎么起作用、抢占怎么触发。
3. **`v1/core/sched/interface.py`** —— 看要 override 哪些钩子（自定义调度器的契约）。
4. **`v1/core/sched/request_queue.py`** —— 排队顺序，short-output-first 在这改。
5. **`v1/request.py`** —— `Request` 状态，预测输出长度字段挂这里。
6. **`v1/core/sched/output.py`** —— `SchedulerOutput`，理解调度决策怎么传给 executor。
7. **`v1/core/kv_cache_manager.py`** —— 涉及抢占/释放与 KV 占用时再看。



SPEC
