# 架构规格：三档调度策略与温控插件（input / method / output）

代码整体按 **输入（`input`）→ 调度与仿真（`method`）→ 指标输出（`output`）** 三段组织。`method` 层由一个**全局调度器**（负责请求初次放置 + 运行中途迁移）和一个**链路传输开销模型**组成。

## 论文三档调度策略 ↔ 代码标识

| 档位 | 中文名称（见 `GLOSSARY.md`） | `scheduler` 取值 | `link_cost` 取值 | 说明 |
|---|---|---|---|---|
| 第 1 档 | 首次分配 | `place_once` | `"off"` | 请求到达时放一次，后续绝不迁移 |
| 第 2 档 | 无消耗迁移 | `rebalance_free` | `"off"` | 按算力负载启发式迁移；假设迁移瞬间完成、不占星间带宽（理想上界） |
| 第 3 档 a | 随机迁移 | `rebalance_random` | `"full"` / `"net"` | 占带宽迁移的简单基线：随机选请求迁往算力更高的卫星 |
| 第 3 档 b | 自研迁移策略 | `rebalance_ours` | `"full"` / `"net"` | **TODO**：目前为空操作占位，待填入拥塞感知迁移算法 |

注：`nobg`（关闭背景流量）只是环境开关，不算第四种调度算法。

## 执行流程

```text
YAML 配置 (scheduler + link_cost)
  → registry 组装出 Scheduler（调度器）+ LinkCost（链路开销模型）
  → engine 事件循环：新请求到达时调用 place()；每个控制周期调用 maybe_rebalance()
  → claims 汇总对比：
      结论 C1：对比第 1 档（首次分配）vs 第 2 档（无消耗迁移），看重调度能提升多少吞吐
      结论 C2：对比第 2 档（无消耗迁移）vs 第 3 档 a（随机迁移），看抢带宽会把吞吐拖低多少
      ours：目前仅标记 status="todo"
```

## YAML 配置示例

```yaml
policies:
  place_once: { scheduler: place_once, link_cost: "off" }
  rebalance_free: { scheduler: rebalance_free, link_cost: "off", hysteresis: 0.15 }
  rebalance_random: { scheduler: rebalance_random, link_cost: "full" }
  rebalance_ours: { scheduler: rebalance_ours, link_cost: "full" }  # TODO
```

注意：`link_cost: "off"` 必须加引号，防止被 YAML 解析成布尔值 `False`。

## 核心评估指标

| 指标名 | 优劣方向 | 白话含义 |
|---|---|---|
| `good_long` | ↑ 越大越好 | 限功切换后，生成速度仍达标的长请求数量（多种子均值） |
| `goodput` | ↑ 越大越好 | 请求到达期内，生成速度达标的有效输出吞吐（token/s） |
| `link_traffic_GB` | ↓ 通常越小越好* | 星间链路上的迁移总流量（GB·跳；无消耗迁移档恒为 0） |
| `abort` | ↓ 越小越好 | 迁移过程中因链路拥塞导致传输超时中断的次数 |

\* 注：占带宽迁移必然会产生链路流量，关键看它能否换来更高的 `goodput` 以及能否避免超时 `abort`。

各实验的具体结果表见根目录 `README.md`。

## 吞吐率计算公式

设仿真总时长 `horizon` = `arrival_window_s`（请求到达期）+ `drain_s`（尾流排空期）：

- `request_throughput_rps` = 完成请求总数 / `horizon`
- `total_token_throughput_tok_s` = 已完成请求的（输入 prompt + 输出 output）token 总和 / `horizon`
- `goodput_tok_s` = 限功切换后每 token 生成耗时（TPOT）满足延迟要求的输出 token 总数 / `arrival_window_s`

## 温控与降频模型（`ThermalModel`）

引擎通过 `World.f_of(node, t)` 查询节点当前的时钟频率倍率（由 `thermal.clock_factor` 提供），并在每个控制周期调用 `world.thermal_tick()` 更新温度状态。

| `thermal.mode` | 行为说明 |
|---|---|
| `explicit`（默认） | **定时限功模式**：直接读取 `explicit_events` 里写死的时间段与功耗上限 `(node, t0, t1, P_cap)`，按 DVFS 公式折算频率倍率 |
| `dynamic` | **动态温控模式**：每颗星按一阶热阻热容（RC）模型随负载发热升温，带回差阈值触发两档降频；开启后会**忽略** `explicit_events` |

```yaml
thermal:
  mode: dynamic
  P_idle: 60         # 空闲功耗 (W)
  P_max: 300         # 满载功耗上限 (W)
  gamma: 2.4         # DVFS 功耗-频率指数
  f_min: 0.35        # 最低频率倍率下限
  tau_s: 30.0        # 热时间常数 (s)
  R_th: 0.4          # 热阻 (°C/W，示意值)
  T_amb_C: 20.0      # 环境基准温度 (°C)
  T0_C: 40.0         # 初始温度 (°C)
  T_throttle_C: 84.0 # 触发降频温度阈值 (°C)
  T_release_C: 78.0  # 解除降频恢复温度 (°C)
  f_throttle: 0.71   # 过热降频后的频率倍率
  P_busy: 300        # 有任务运行时的发热功率 (W)
```
