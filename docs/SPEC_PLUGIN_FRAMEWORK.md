# SPEC：论文三档调度（input / method / output）

顶层：**input → method → output**。方法内一个全局调度策略（初次分配 + 再分配）+ 链路代价模型。

## 论文三档 ↔ 实现 id

| 档 | 术语（GLOSSARY） | scheduler | link_cost | 备注 |
|---|---|---|---|---|
| 1 | 首次分配 | `place_once` | off | 再分配空 |
| 2 | 无消耗再分配 | `rebalance_free` | off | 启发式；通常最优 |
| 3a | 随机再分配 | `rebalance_random` | full/net | 有消耗基线 |
| 3b | ours 再分配 | `rebalance_ours` | full/net | **TODO**，空操作占位 |

`nobg`（关背景）是旋钮，不是第四档算法。

## 阅读路径

```text
YAML scheduler + link_cost
  → registry → Scheduler + LinkCost
  → engine：到达 place；周期 maybe_rebalance
  → claims：C1 档1 vs 档2；C2 档2 vs 档3a；ours 只报 todo
```

## YAML

```yaml
policies:
  place_once: { scheduler: place_once, link_cost: "off" }
  rebalance_free: { scheduler: rebalance_free, link_cost: "off", hysteresis: 0.15 }
  rebalance_random: { scheduler: rebalance_random, link_cost: "full" }
  rebalance_ours: { scheduler: rebalance_ours, link_cost: "full" }  # TODO
```

`link_cost: "off"` 必须加引号。

## 指标方向（主表）

| 指标 | 方向 | 含义 |
|---|---|---|
| good_long | ↑ 更好 | 换位后仍达标的长请求数（均值） |
| goodput | ↑ 更好 | 到达窗内有效 token/s |
| link_traffic_GB | ↓ 通常更好* | 链路上流量×跳（无消耗档应为 0） |
| abort | ↓ 更好 | 迁移超时放弃次数 |

\* 有消耗档为付费迁移所必需；与 goodput 权衡。

主结果数字见根目录 `README.md`。

## 吞吐

- `request_throughput_rps` = \#completed / (`arrival_window_s`+`drain_s`)
- `total_token_throughput_tok_s` = Σ(prompt+output of completed) / 同上 horizon
- `goodput_tok_s` 仍为 SLO 过滤的输出 token / `arrival_window_s`（claim 对照）

## 热模型（ThermalModel）

`World.f_of` → `thermal.clock_factor`；控制周期 `world.thermal_tick()` → `advance`。

| mode | 行为 |
|---|---|
| `explicit`（缺省） | `explicit_events` + DVFS，与旧 golden 同路径 |
| `dynamic` | 每节点一阶 RC + 滞回两档节流；**忽略** explicit_events |

```yaml
thermal:
  mode: dynamic
  P_idle: 60
  P_max: 300
  gamma: 2.4
  f_min: 0.35
  tau_s: 30.0
  R_th: 0.4          # °C/W，示意
  T_amb_C: 20.0
  T0_C: 40.0
  T_throttle_C: 84.0 # LAB-1178/1214 叙事，非本仓标定
  T_release_C: 78.0
  f_throttle: 0.71
  P_busy: 300
```

