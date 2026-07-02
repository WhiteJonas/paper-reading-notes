# tau2-bench 工具调用模拟机制分析

> 分析对象：`src/tau2/environment/`、`src/tau2/orchestrator/`、`src/tau2/domains/`
> 结论：tau2-bench 实现了一个**有状态、确定性、可回放的工具沙盒**。工具不是真实外部 API，而是"操作内存数据库的 Python 函数"，通过装饰器自动暴露成 LLM 的 function-calling 接口。

---

## 总览

整体分四层：

| 层 | 文件 | 职责 |
|----|------|------|
| 工具定义层 | `environment/toolkit.py`、`environment/tool.py` | 装饰器 + docstring 自动生成 OpenAI function schema |
| 状态后端层 | `environment/db.py`、各 domain 的 `data_model.py` + `db.json` | 内存 pydantic 数据库 |
| 执行/模拟层 | `environment/environment.py` | 工具执行、错误模拟、序列化、回放 |
| 驱动层 | `orchestrator/orchestrator.py` | agent ↔ user ↔ env 回合循环 |

一句话总结：

```
LLM 看到的工具 = 装饰器扫描方法 + docstring 自动转 OpenAI schema
工具被调用    = Environment.get_response() 在内存 pydantic DB 上跑 Python 函数
出错模拟      = try/except → ToolMessage(error=True)
是否做对      = 跑完比对 DB 的 hash 是否等于参考终态 hash
可复现性      = 只重放写工具 + 确定性序列化，保证 trace 可精确回放
```

---

## 1. 工具定义层 —— 装饰器 + docstring 自动生成 schema

`toolkit.py`：用 `@is_tool(ToolType.READ/WRITE/...)` 把 toolkit 里的普通方法标记成工具。元类 `ToolKitType` 在类创建时扫描所有带 `__tool__` 标记的方法，注册进 `_func_tools`。

每个工具方法的 **docstring 就是工具定义**（`tool.py` 的 `Tool.parse_data`，依赖 `docstring_parser`）：

- 首段 → tool description
- `Args` 段 → 逐参数解析出 名字 / 类型 / 描述，`create_model` 动态构造一个 pydantic 参数模型
- 返回注解 → returns schema

最后 `Tool.openai_schema` 属性直接转成标准 OpenAI function schema：

```python
{
  "type": "function",
  "function": {
    "name": self.name,
    "description": self._get_description(),
    "parameters": self.params.model_json_schema(),
  },
}
```

**这就是 agent 看到的工具列表来源**，无需手写 JSON schema。

### 工具分类

`ToolType` 枚举对工具做概念分类：

- `READ`：查询/检索，无副作用
- `WRITE`：增删改，有副作用
- `THINK`：纯推理，无新信息、无副作用
- `GENERIC`：其它工具

`is_tool` 装饰器还有一个 `mutates_state` 参数（默认从 `tool_type` 推断：WRITE → True，其余 → False）。这个标记**只在评测回放时使用**：只有 `mutates_state=True` 的工具才会被重新执行。

内置 `GenericToolKit` 提供 `think`（纯推理，返回空串）和 `calculate`（受限 eval 的计算器）。

---

## 2. 状态后端层 —— 内存 pydantic 数据库

`db.py` 的 `DB` 基类（pydantic `BaseModelNoExtra`）。每个 domain 子类（airline 的 `FlightDB`、retail 的 DB 等）从 `data/tau2/domains/<domain>/db.json` 反序列化成一个内存对象。工具方法通过 `self.db` 读写它。

例如 airline 的 `AirlineTools._get_user / _get_reservation / _get_flight` 全部是在 `self.db.users[...]`、`self.db.reservations[...]` 这些 dict 上操作。

### 各 domain 后端数据库实例

后端数据全部是 JSON 文件，加载后反序列化成 pydantic 模型。

**airline（航空订票）— 3 张表：**

- `flights`：300 个航班。key 是航班号 `HAT001`。每个航班按日期 `dates` 存状态：`landed` / `cancelled` / `available`。available 的日期带 `available_seats`（三档舱位剩余座位）和 `prices`（动态票价）。
- `users`：500 个用户。含姓名、地址、`payment_methods`（信用卡/礼券，礼券带余额）、`saved_passengers`、会员等级 `membership`、关联的 `reservations` id 列表。
- `reservations`：2000 个订单。含 user_id、往返航段 `flights`、乘客、`payment_history`、行李数、保险等。

样例（`reservations`）：

```json
{
  "reservation_id": "4WQ150",
  "user_id": "chen_jackson_3290",
  "origin": "DFW",
  "destination": "LAX",
  "flight_type": "round_trip",
  "cabin": "business",
  "flights": [
    {"origin": "DFW", "destination": "LAX", "flight_number": "HAT170", "date": "2024-05-22", "price": 883},
    {"origin": "LAX", "destination": "DFW", "flight_number": "HAT022", "date": "2024-05-26", "price": 779}
  ],
  "passengers": [
    {"first_name": "Chen", "last_name": "Jackson", "dob": "1956-07-07"}
  ],
  "payment_history": [{"payment_id": "gift_card_3576581", "amount": 4986}],
  "created_at": "2024-05-02T03:10:19",
  "total_baggages": 5,
  "nonfree_baggages": 0,
  "insurance": "no"
}
```

**retail（零售/电商）— 3 张表：**

- `products`：50 个商品。key 是 product_id。每个商品下有多个 `variants`（SKU，key 是 item_id），变体带 `options`（颜色/尺码/材质/款式）、`available`、`price`。
- `users`：500 个用户。
- `orders`：1000 个订单。

**telecom：** 没有静态 `db.json`，DB 由代码里的 builder 程序化生成（涉及设备/网络状态模拟）。

**mock（测试用）：** 极简，`tasks` 1 条 + `users` 1 条，单元测试玩具数据。

### 哈希 —— 评测核心

`DB.get_hash()` 对整个 DB 做 pydantic 哈希。跑完之后比对 `db_hash == reference_hash`，判断 agent 是否把数据库改成了正确的最终状态。

---

## 3. 执行/模拟层 —— Environment

`environment.py` 是模拟器引擎，核心入口 `get_response(ToolCall) -> ToolMessage`：

1. `make_tool_call` 按 `requestor`（user / assistant）路由到对应 toolkit
2. 实际跑 domain 函数，在内存 DB 上读/写
3. **try/except 包裹**：异常时返回 `"Error: {e}"` 并标记 `error=True`（这就是"模拟"出错时的工具反馈，进程不会崩）
4. 结果用 `to_json_str` 递归序列化（处理 pydantic / datetime / 嵌套结构）成 JSON 字符串，包进 `ToolMessage` 返回给 agent

```python
def get_response(self, message: ToolCall) -> ToolMessage:
    error = False
    try:
        resp = self.make_tool_call(message.name, requestor=message.requestor, **message.arguments)
        self.sync_tools()
    except Exception as e:
        resp = f"Error: {e}"
        error = True
    resp = self.to_json_str(resp)
    return ToolMessage(id=message.id, content=resp, requestor=message.requestor, role="tool", error=error)
```

### 三个关键设计

- **READ/WRITE/THINK/GENERIC 分类 + `mutates_state` 标记**：只有写工具才算"改状态"。
- **回放 `set_state`**：重建一条历史轨迹时，只重新执行 `mutates_state=True` 的工具，只读/think 工具直接跳过（避免非确定性输出导致比对失败）；遇到 LLM 幻觉出来的不存在工具，按 live 行为当 no-op 跳过。
- **双向 + `solo_mode`**：assistant 和 user 各有一套 toolkit（user 也能调工具，模拟真实用户操作），`solo_mode` 下 agent 同时拿到两边工具，且校验工具名不重叠。

---

## 4. 驱动层 —— Orchestrator

`orchestrator.py` 管 agent ↔ user ↔ env 的回合循环（半双工 / turn-based）：

- agent 产出 `AssistantMessage`，若含 `tool_calls` → `_execute_tool_calls` 逐个调 `environment.get_response`
- 工具结果（多个时包成 `MultiToolMessage`）回灌给 agent
- `error=True` 时 `num_errors++`，超过 `max_errors`（默认 10）终止；超过 `max_steps`（默认 100）也终止
- 消息严格交替，且校验"n 个 tool_call 必须紧跟 n 个对应 tool_message"

通信流：

```
AGENT -> USER : agent 给用户发文本
AGENT -> ENV  : agent 调工具
USER  -> AGENT: 用户给 agent 发消息
USER  -> ENV  : 用户调工具
ENV   -> AGENT: 环境返回工具结果（在 agent 调用后）
ENV   -> USER : 环境返回工具结果（在用户调用后）
```

终止条件：agent / user 发出 stop 信号（`###STOP###`）、达到 max_steps、达到 max_errors、或检测到通信协议违规。

---

## 与 vLLM 调度研究的关联点

- **tool-result 体积差异大**：一个 `search_flight` 返回 30 天 × 3 舱位的明细可能上千 token，而 `book_reservation` 只返回一个确认、几十 token。agent 上下文里 tool-result 占比波动剧烈，直接影响后续 turn 的 prompt 长度 → TTFT，以及输出长度的可预测性。
- **READ/WRITE 标记现成可用**：若要按"是否改状态 / 返回体量"给 agent 请求分类做调度，这套 `ToolType` + 工具 schema 是现成的特征来源。

---

## 为什么这么设计 —— 设计意图

核心目的：**让"会用工具的对话 agent"的评测变得可靠、可量化、可复现。**

LLM agent 评测最大的痛点是**没有客观标准答案**——一段多轮对话 + 一串工具调用，很难判断 agent 做对了没。tau2-bench 的整套设计就是在解决这个问题。

### 1. 内存 DB + 哈希比对：把"对不对"变成确定性判断

靠 LLM judge 去评"回答得好不好"既不稳定（judge 有噪声）又不客观。tau2 的做法是把任务成败归结为**数据库的最终状态**。

例如"把订单 4WQ150 改签到 5 月 26 号"——做没做成不看 agent 说了什么，而是直接看 `reservations["4WQ150"]` 这条记录有没有被正确修改。跑完 `get_hash()` 跟参考终态比对，一致即对。这是**确定性、二值**的判断，绕开了"用模型评模型"的噪声。这也是为什么要区分 READ/WRITE——只有 WRITE 才改变那个被哈希的状态。

### 2. 模拟工具而非真实 API：钉死环境，保证可比

接真实订票 API 会导致：不可复现（票价天天变、座位被别人订走）、不安全（真去改真实订单）、不可控（无法设计"航班 5/14 被取消"这类边界场景）。

用固定的内存数据库，就把环境钉死了：同一个 `db.json`，任何人任何时候跑，初始状态完全一样。这样不同模型、不同时间的得分才可比——这是 benchmark 的生命线。

### 3. docstring 自动生成 schema：降低扩展成本

写一个带规范 docstring 的 Python 函数，工具定义就自动有了，不用同时维护"函数实现"和"给 LLM 看的 JSON schema"两份东西，也避免两者不同步。加新 domain / 新工具的成本因此很低。

### 4. 可回放（set_state）：复现与中途接续

τ²-bench 相比初代 τ-bench 的关键升级。评测常需从一条已有轨迹的中间状态接着跑，或复现某条 trace。回放时只重执行写工具、跳过只读工具，既快又避免只读工具的非确定性输出（如带时间戳的返回）干扰比对。

### 5. user simulator + 双向工具：考验多轮、信息不全下的工具使用

真实用户的信息是"挤牙膏式"给出的。tau2 用一个 LLM 扮演用户，agent 得主动追问、澄清。这考验 agent 在**信息不全 + 多轮交互**下用工具完成任务的能力，而非单轮 function calling。`user_tools` 则模拟"用户那边也能操作"的场景（如电信 domain 让用户自己重启路由器）。

### 一句话

> 它把一个开放的、主观的"agent 表现好不好"，转化成了一个**封闭的、确定性的状态机问题**：固定初始 DB → agent 在模拟环境里多轮交互调工具 → 比对最终 DB 哈希。这样才能得到稳定、可复现、模型间可比的分数。

### 对本研究的意义

tau2 是个**离线可复现的 agent workload 生成器**：可拿同一批任务反复跑，得到稳定的"多轮 + 工具调用"请求序列，用来研究 agent 场景下的 TTFT/TPOT/输出长度分布。环境的确定性正好排除了"每次请求负载都不一样"这个干扰变量。

---

## 工具模拟时，LLM 和 KV cache 各在干什么

关键：tau2-bench 的"工具模拟"和 LLM 推理（含 KV cache）是**两个完全分离的世界**，不在一个进程、一个设备上。

### 工具模拟那一步：LLM 和 KV cache 都不参与

`Environment.get_response()` 执行工具时，做的纯粹是 **CPU 上的 Python 字典操作**：查 `self.db.users[...]`、改 `reservations[...]`、`json.dumps` 序列化结果。

- **LLM 不参与**——工具的"执行"不需要模型推理，就是跑普通 Python 函数。
- **KV cache 不参与**——根本没碰 GPU。

工具调用本身在 tau2 里就是查表改表，微秒级。重的是它**前后**的 LLM 调用。

### 一个完整 turn 里 LLM / KV cache 的位置

tau2 里 LLM 只在三处被调用：agent 决策、user simulator（扮演用户）、可能的 judge。以 agent 主线看一个 turn：

```
① agent LLM 推理  →  产出 tool_call          [GPU，烧 KV cache]
② Environment 执行 tool_call → tool result     [CPU，查字典，与 LLM 无关]
③ tool result 塞回对话历史
④ agent LLM 再推理 → 下一步                    [GPU，又烧 KV cache]
```

- **LLM 的工作**：在 ① 和 ④。输入 prompt = `system(policy + 所有工具 schema)` + 完整对话历史（用户消息 + agent 历史回复 + 之前每次的 tool result）+ 最新 tool result。
- **KV cache 的工作**：缓存这个 prompt 里每个 token 的 K/V，避免重算。

### 为什么 agent 场景对 KV cache 特别敏感

1. **Prompt 单调增长、前缀高度重合。** 第 N 轮 prompt = 第 N-1 轮 prompt + 上轮 agent 输出 + 新 tool result，即每轮 prompt 把上一轮完整包含为前缀。这正是 **prefix caching** 的理想场景：
   - system prompt（policy + 工具 schema，airline 轻松上千 token）整个 session 不变，跨 turn 跨 task 都能命中。
   - 对话历史前缀逐轮累加，本轮 prefill 理论上只需算新增的一小段，前面全部命中 cache。
   - 没有 prefix cache 时，每轮都要把越来越长的历史从头 prefill → TTFT 随 turn 数线性恶化。

2. **tool result 体积直接决定 KV cache 增长速度。** `search_flight` 返回 30 天×3 舱位可能上千 token，`book_reservation` 只几十 token。每个 tool result 进历史后，会永久占用后续所有 turn 的 KV cache（直到 session 结束）。所以一条 agent 轨迹的 KV cache 占用是被工具返回体积**阶跃式**推高的，而非平滑增长。

### 对应表

| tau2 概念 | 落到推理 / 调度层 |
|-----------|------------------|
| agent 决定调工具 | LLM 一次 decode，产出 tool_call token，**KV cache 增量写入** |
| Environment 执行工具 | 纯 CPU，**LLM 空闲，KV cache 不变**（此期间 GPU 可调度别的请求） |
| tool result 回灌历史 | 下次 LLM 调用 **prefill 输入变长**，KV cache 阶跃增长 |
| 多轮 turn | prompt 前缀单调累加，**prefix cache 命中率是 TTFT 命门** |

---

## 这个数据集测的是什么，测不测时延

### 测的是什么：能力 / 正确性

tau2-bench 测的是 **agent 在真实客服场景下、按既定业务规则、通过多轮对话 + 工具调用完成任务的能力**：

- **任务完成度（主指标）**：DB 终态哈希比对，pass/fail。
- **遵守 policy**：每个 domain 有 `policy.md`（如"basic economy 不能改签""24 小时内取消免费"）。agent 不光要完成任务，还要不违规——这是 tau2 的核心考点：在约束下正确地用工具。
- **多轮交互能力**：user simulator 挤牙膏式给信息，考 agent 主动追问、澄清。
- **pass^k（可靠性）**：同一任务跑 k 次都对才算过，measure 稳定性而非碰运气。

一句话：**task success + rule compliance + 多轮鲁棒性**，是个能力 / 正确性 benchmark。

### 测不测查库 / 推理时延：完全不测

**tau2-bench 不测任何时延，不测 TTFT/TPOT，不测吞吐。** 原因：

1. **"查数据库"这步没有 LLM、也没有 LLM 时延。** 查 DB 是 CPU 字典操作，微秒级，与 LLM 推理无关——查库时 LLM 是空闲的，"查数据库时 LLM 的时延"这个组合在 tau2 里不存在。

2. **tau2 把"时间"整个抽象掉了。** 它是半双工、回合制（turn-based）模拟器：agent → user → env 严格交替。终止条件是 `max_steps`（回合数）、`max_errors`（错误数），**没有任何 wallclock 概念**，工具执行被当成瞬时。它度量的是**逻辑步数**，不是物理时间。

> 注：orchestrator 有个 `timeout` 参数，但那是防止 sim 卡死的兜底墙钟超时，不是性能指标，与查库 / 推理延迟无关。

### 对本研究的切入点

tau2 **不测**的恰恰是本研究**要测**的——它给了"正确性"维度，TTFT/TPOT/Goodput/SLO 维度整个留空：

- tau2 是现成的 agent workload 生成器：稳定产出"多轮 + 工具调用"轨迹（prompt 如何累积、tool result 多大、几个 turn），但自身不关心这些请求在 vLLM 上跑得快不快。
- 要做的是给它**补上时间维度**：把 agent/user 的 LLM 调用接到 vLLM patch，记录每次 ①④ 推理的 TTFT/TPOT、KV cache 占用、turn 间 GPU 空窗，研究"按输出长度预测调度 agent 请求"。
- **坑**：tau2 默认 turn-based 串行，单个 session 内的"等待"是逻辑等待，不产生真实并发压力。要研究调度，必须**并发跑多个 session**（许多 task 同时处在不同对话进度），才能在 vLLM 端形成有意义的请求队列——单 session 喂不饱调度器。
- **调度切入点**：上面 ② 步（工具执行，CPU）期间，该 session 在 GPU 上是空窗的（agent 在等工具返回）。多 session 并发下这些空窗如何错峰、要不要在空窗期把该 session 的 KV cache 换出，正是 agent 场景调度区别于普通 serving 之处。

---

## 数据库是怎么建出来的

> 更正：telecom **不是**程序化生成的。三个 domain 的后端 DB 都是**预先准备好的静态文件**，运行时只是 `load` + pydantic 校验，不在运行时建库。

### 库 = checked-in 的静态文件

| domain | 后端文件 | 规模 | 怎么造的 |
|--------|---------|------|---------|
| airline | `db.json` | 300 航班 / 500 用户 / 2000 订单 | 脚本随机采样合成（承自初代 τ-bench） |
| retail | `db.json` | 50 商品 / 500 用户 / 1000 订单 | 脚本随机采样合成（承自初代 τ-bench） |
| telecom | `db.toml` + `user_db.toml` | 5 套餐 / 几十设备 / 设备状态 | 手工精心设计（为排障场景） |

加载只有一行：`FlightDB.load(path)` → 读文件 → `model_validate(data)` 反序列化成 pydantic 对象。**schema 不是从数据推断的**，而是先有 `data_model.py` 里的 pydantic 类定义，数据去适配它。

### 数据本身两种造法

- **airline / retail：脚本随机采样合成。** 证据：航班 `available_seats`(1–20)、`prices`(50–500) 是均匀随机数；`actual_departure_time` 在 scheduled 附近抖几分钟；30 天随机插几天 `cancelled`；用户名 `mia_li_3668`、邮箱 `example.com`、id 带随机后缀——典型 faker 式合成。这套从初代 τ-bench(Sierra) 继承，tau2 直接搬产物 `db.json`。

- **telecom：手工/半手工小规模设计。** `db.toml` 仅 9.6KB，plans 5 个、devices 是 `Galaxy S23` 这种、IMEI 是 `123456789012345` 顺子号——明显人手编。因为 telecom 考点是**技术支持排障**（信号/APN/漫游/流量超限），需要精心构造特定故障场景，随机生成造不出有意义的排障 case。`user_db.toml` 里 `airplane_mode`、`network_signal_strength`、`apn_settings` 就是被排障流程检查的设备状态。

### 关键：库是"种子"，真正初始状态由 task 叠加

`db.json` 只是**共享基础种子**。每个任务跑时，初始状态由 `Environment.set_state()` 在种子上叠加（task 的 `initial_state` 字段，内含 `initialization_data` + `initialization_actions`）：

```
最终初始状态 = db 种子文件
            + task.initial_state.initialization_data   (patch 字段)
            + task.initial_state.initialization_actions (跑几个 env 函数把状态调到位)
```

实测证据（统计 task 文件）：

- **airline：0/50** 个 task 有非空 `initial_state` —— 全部直接用共享种子。
- **telecom：20/20** 个 task 都有非空 `initial_state` —— 库小（9.6KB），但每个 task 自带初始化。

telecom 一个真实 `initialization_actions` 例子（把场景调成"用户在国外、漫游关闭"）：

```json
[
  {"env_type": "user", "func_name": "set_user_location", "arguments": {"abroad": true}},
  {"env_type": "user", "func_name": "turn_roaming_off", "arguments": {}},
  {"env_type": "assistant", "func_name": "enable_roaming", "arguments": {"customer_id": "C1001", "line_id": "L1002"}}
]
```

这解释了为什么 telecom 是 **9.6KB 小库 + 13MB tasks.json**：库小，但每个 task 通过初始化动作把 DB 调成自己需要的故障场景。**"建库"真正的工作量在 task 设计，不在基础 db 文件。**

### 一句话

> 库不是运行时建的，是**预先合成好的静态种子文件**（大规模 domain 用脚本随机采样，排障类 domain 用手工设计），用 pydantic schema 卡结构；每个 task 再通过 `initial_state`（initialization_data / actions）在种子上叠加出自己需要的初始状态。

### 对本研究的意义

workload 的"参数化"入口就在 task 的 `initial_state`：同一个种子 DB，不同 task 叠加出不同初始状态 → 不同的对话长度、工具调用序列、tool-result 体积。要构造可控的 agent 请求负载，调 task 集（而非改 db）是正确的旋钮。

---

## 这个 bench 能不能测调度策略

**结论：原版不能直接测，但可改造成理想的 agent workload 发生器。**

tau2-bench 的输出是 task pass/fail（pass^k），衡量 **agent 做对了没、守不守规**，与"跑得快不快、调度好不好"完全无关——它不产生 TTFT/TPOT/吞吐/Goodput，回合制把时间抽象成 `max_steps`，工具执行算瞬时。**同一组 task 用两种调度策略跑，分数会一模一样**，因为调度只改快慢不改对错，而 tau2 对快慢"色盲"。

但它能当 workload 来源：产生的请求是真·agent 形态（多轮、prompt 单调增长、prefix 高度重合、tool-result 体积波动）。改造点：

| 要的 | 现状 | 改造 |
|------|------|------|
| 系统指标 | 完全没有 | LLM 调用接到 vLLM patch，在那层埋点 |
| 真实并发压力 | 默认单 session 串行（逻辑等待，GPU 空窗） | **并发跑大量 session** 才能形成请求队列 |
| 切换调度策略 | 不感知 | 在 vLLM 侧切，tau2 不用动 |

**"降低工具影响"正是 tau2 的强项**：工具执行是 CPU 查内存字典，微秒级、零方差、不碰 GPU、不调外部网络，且给定 task 完全可复现（tool-result 逐字节相同）。所以切换策略 A→B 时工具侧是干净对照组，性能差异能干净归因到调度。唯一的间接影响是 tool-result 进历史推高后续 prompt/KV（确定性、可预先算出）。

> 注意：跑实验时把 orchestrator 的 wallclock `timeout` 放大或关掉，避免性能维度污染正确性维度；若调度策略含 preemption/KV 换出重算，需确认重算输出 token 一致（定 seed/贪心下应一致）。

---

## 任务种类、toolcall 如何被"控制"、幻觉 toolcall 如何处理

### 1. 任务种类：按"考点"分，不按功能分

50 个 airline task 的 `purpose` 显示主要几类：

1. **守规拒绝**：用户提不合规要求（取消不该取消的订单、加不允许的保险），考 agent 顶住压力拒绝。
2. **抗欺骗/核实**：用户撒谎（谎称 Gold 会员、谎称航班取消骗补偿），考 agent 调工具核实而非轻信。
3. **多意图/话题切换**：对话中途换需求，考上下文跟踪。
4. **正常事务执行**：订票/改签/加乘客，考基本工具操作正确性。
5. **部分操作/边界**：只改往返中一段、改乘客数，考对业务约束的精确遵守。

telecom 任务全是**技术支持排障**（信号/漫游/流量/APN/MMS 故障），task id 自带场景标签如 `mobile_data_issue]user_abroad_roaming_enabled_off`。

评测维度由 `evaluation_criteria.reward_basis` 指定：`DB`（终态对不对）、`COMMUNICATE`（该告知用户的信息说了没）、`ACTION`（该调的工具调了没，对照 `actions` 列表）、`NL_ASSERTION`（自然语言断言，如"应当拒绝"）。

### 各 domain 工具清单（assistant 侧）

- **airline**(14)：book/cancel/update_reservation_*、search_direct/onestop_flight、get_*_details、get_flight_status、send_certificate、transfer_to_human_agents、calculate
- **retail**(16)：cancel/modify/return/exchange_order_*、get_product/item/order/user_details、find_user_id_by_*、modify_user_address、transfer_to_human_agents
- **telecom**(13 assistant + 30 user)：assistant 侧 get_customer_by_*、suspend/resume_line、enable/disable_roaming、refuel_data、send_payment_request 等；**user 侧 30 个**（check_network_status、toggle_airplane_mode、reseat_sim_card、set_apn_settings、reboot_device、make_payment...）模拟用户在自己手机上操作。

### 2. toolcall 怎么被"控制"——三层

1. **只给 schema（白名单）**：`generate()` 里 `tools_schema = [tool.openai_schema for tool in tools]`，把当前 domain 工具列表作为 OpenAI function schema 传给 LLM。LLM 只知道这些工具存在。
2. **`tool_choice="required"`**（`llm_agent.py:473`，solo 模式）：协议层强制模型这一步**必须输出一个 tool_call**，不准只说自然语言；普通模式用 `"auto"`。
3. **底层走 litellm**：`completion(model, messages, tools, tool_choice)`。litellm 支持 `model + base_url`，**把 base_url 指向 vLLM 即可接管所有 LLM 调用，改造量极小**。

### 3. 天马行空 / 没见过的 toolcall —— 不预测，事后兜底

tau2 **完全不预测、不限制** LLM 会发什么。处理是确定性兜底：

- **A. 幻觉工具名**：`Environment.get_response()` 的 try/except 兜住 → `make_tool_call` 找不到 → 返回 `ToolMessage("Error: ... not found", error=True)`，进程不崩，agent 可纠正；`num_errors++`，攒够 `max_errors`(默认10) 终止。
- **B. 工具名对、参数乱/查不存在的 id**：同样 try/except 兜住，domain 函数自己 `raise ValueError(...)` → 变成 `Error:` 的 ToolMessage 回灌。这常常是**故意的**——很多 task 就考 agent 遇错会不会重试/换策略。
- **C. 回放时遇幻觉工具**：`set_state` 当 **no-op 跳过**（live 时它本来也没改成功）。

> 哲学：LLM 想发啥发啥，环境只负责**确定性地执行或确定性地报错**，再看 agent 能否从错误恢复、最终 DB 改对没。toolcall 的"无限可能"不是靠预测限制，而是靠 **白名单 schema + 强制格式 + 异常兜底 + 错误计数终止** 四件套兜住。

### 对本研究的连接

- `tool_choice="required"` → agent 每步必产生一个 tool_call → turn 数 = LLM 调用数，负载结构规整。
- **litellm + base_url** 是接入 vLLM 的现成钩子，近零改造。
- 错误兜底不污染延迟测量：幻觉/错误 toolcall 只是多一轮 LLM 交互（多一个请求），工具侧仍是确定性 CPU 操作。
- 

| 测评指标       | 记号 / 公式                                                  | 衡量内容                                          | 判定方式                                                     | 说明                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 动作正确性     | $$r_{action} \in \{0,1\}$$                                   | Agent 是否执行了正确的数据库写操作                | 比较 episode 结束后的数据库状态与人工标注的目标数据库状态是否一致 | 主要检查工具调用是否产生了正确最终结果，例如是否正确退货、换票、改地址等 |
| 输出正确性     | $$r_{output} \in \{0,1\}$$                                   | Agent 是否向用户提供了必要信息                    | 检查 Agent 回复中是否包含任务要求的关键信息                  | 例如用户要求知道退款金额、节省金额、追踪号等，回复中必须包含对应内容 |
| 单次任务奖励   | $$r = r_{action} \times r_{output}$$                         | 一次完整对话任务是否成功                          | 只有动作正确且输出正确时，任务成功                           | 若数据库操作正确但没告诉用户必要信息，也算失败               |
| 平均单次成功率 | $$pass^1 = pass@1 = E[r] = E[c/n]$$                          | Agent 单次执行任务的平均成功率                    | 对所有任务、所有 trial 的奖励取平均                          | 论文默认主指标，用于比较不同模型整体能力                     |
| 一致性成功率   | $$pass^k = E_{task}\left[\frac{\binom{c}{k}}{\binom{n}{k}}\right]$$ | 同一任务重复运行 $$k$$ 次时，Agent 是否每次都成功 | 从 $$n$$ 次 trial 中抽 $$k$$ 次，要求这 $$k$$ 次全部成功     | 论文提出的新指标，强调真实客服场景中的可靠性和稳定性         |
| 至少一次成功率 | $$pass@k = 1 - E_{task}\left[\frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$ | 同一任务运行 $$k$$ 次时，是否至少有一次成功       | 从 $$n$$ 次 trial 中抽 $$k$$ 次，只要一次成功就算成功        | 常见于代码生成评测，但论文认为它不适合衡量真实客服 Agent 的可靠性 |

整理benchmark集合

搭环境，准备数据集

openclaw codex是啥环境

分析bench综述：测llm/Agent性能的bench、测Agent的bench(性能、表现)

帮我搜集目前全部的测llm或者Agent性能的bench，专注测评大模型或者Agent的性能

帮我搜集目前全部的测Agent性能或者表现能力的bench，专注于Agent的性能或者表现能力、任务完成度
