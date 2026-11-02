# SPEC：全局调度器框架（input / method / output）

状态：已落地（Scheduler = 初次分配 + 再分配）  
范围：本地；C1/C2 语义不变；不推远程。

## 1. 目标

- 顶层仍是 **input → method → output**。
- **方法**内：`engine`（内核）+ `interfaces`（Scheduler / LinkCost）+ `policies`（实现）。
- **一个全局调度器**同时负责：
  - `place(world, req)`：初次分配（可看全图）
  - `maybe_rebalance(world)`：再分配 / 提前干预入口
- 引擎**不再**并列 placement / rebalance 两套钩子。
- YAML：**`scheduler` + `link_cost`**（+ `hysteresis` / `knobs`）。无 placement+rebalance 双轨。

## 2. 阅读路径

```text
configs/*.yaml  (或 input/scenarios/)
    │  scheduler / link_cost / knobs
    ▼
input/load.py + schema.py          # 准入、展开 jobs
    ▼
input/generate/*                   # 线下请求序列
    ▼
method/registry.py                 # → Scheduler + LinkCostModel
    ▼
method/engine/loop.py + world.py   # place @ 到达；maybe_rebalance @ 控制周期
    ▼
output/metrics.py + claims.py      # 按 scheduler×link_cost×nobg 出 C1/C2
```

## 3. 目录（方法侧）

```text
method/
  engine/           world, loop, timing
  interfaces/
    scheduler.py    # place + maybe_rebalance
    link_cost.py    # transfer
  policies/
    schedulers/     # place_once, rebalance_local, rebalance_over_link
    link_cost/      # off, full, net
  registry.py
```

## 4. 接口

```python
class Scheduler(Protocol):
    id: str
    def place(self, world, req) -> int: ...
    def maybe_rebalance(self, world) -> None: ...

class LinkCostModel(Protocol):
    mode: str  # free|full|net（由 link_cost 推导进 cfg）
    def transfer(self, world, req, src, dst, nbytes): ...
```

实现：

| scheduler id | 初次分配 | 再分配 | 准入 link_cost |
|---|---|---|---|
| `place_once` | 按迭代时间选节点 | 空操作 | off |
| `rebalance_local` | 同上 | 迟滞选源宿并迁移 | off |
| `rebalance_over_link` | 同上 | 同上（付费由 link_cost） | full, net |

`nobg` 仍是 knobs（`background.gbps_per_node: 0`），不是独立 scheduler。

## 5. YAML 例子

```yaml
policies:
  place_once:
    scheduler: place_once
    link_cost: "off"
  rebalance_local:
    scheduler: rebalance_local
    link_cost: "off"
    hysteresis: 0.15
  rebalance_over_link:
    scheduler: rebalance_over_link
    link_cost: "full"
    hysteresis: 0.15
  rebalance_over_link_net:
    scheduler: rebalance_over_link
    link_cost: "net"
    hysteresis: 0.15
  rebalance_local_nobg:
    scheduler: rebalance_local
    link_cost: "off"
    hysteresis: 0.15
    knobs:
      background.gbps_per_node: 0
```

注意：`link_cost: "off"` 必须加引号（否则 YAML 1.1 读成 bool）。

## 6. Claims 选臂

| Claim | 臂 A | 臂 B |
|---|---|---|
| C1 | `place_once` + off（有背景） | `rebalance_local` + off |
| C2 | `rebalance_local` + off | `rebalance_over_link` + full |
| C2 nobg | 同上但 background=0 | 同上 |
| C2 net | — | `rebalance_over_link` + net 应对齐 full |

## 7. 非目标

- 不改 C1/C2 达标口径与主实验数字期望。
- 不恢复 placement/rebalance 并列字段。
- 本轮不推远程。
