# SPEC：论文三档调度臂（input / method / output）

状态：已落地  
顶层：**input → method → output**。方法内一个全局 **Scheduler**（`place` + `maybe_rebalance`）+ **link_cost**。

## 论文三档

| 档 | scheduler | link_cost | 含义 |
|---|---|---|---|
| 1 初次分配 only | `place_once` | off | `maybe_rebalance` 空 |
| 2 实时再分配 · **无消耗** | `rebalance_free` | off | 换位后启发式迁到更好节点，**不付链路代价** |
| 3 有消耗再分配 | `rebalance_random` | full/net | 随机基线，迁移走链路 |
| 3 有消耗 · ours | `rebalance_ours` | full/net | **TODO**：待解决问题，当前空操作 |

`nobg` 仍是 knobs（关背景），不是独立算法。

## 阅读路径

```text
YAML: scheduler + link_cost (+ hysteresis/knobs)
  → registry → Scheduler + LinkCostModel
  → engine: 到达 place；控制周期 maybe_rebalance
  → claims: C1 place_once vs free；C2 free vs random；ours 仅标记 todo
```

## YAML 例子

```yaml
policies:
  place_once:
    scheduler: place_once
    link_cost: "off"
  rebalance_free:
    scheduler: rebalance_free
    link_cost: "off"
    hysteresis: 0.15
  rebalance_random:
    scheduler: rebalance_random
    link_cost: "full"
  rebalance_ours:
    scheduler: rebalance_ours
    link_cost: "full"   # 待解决：有消耗下的 ours
```

## Claims

- **C1**：初次分配不够 → 无消耗再分配有收益  
- **C2**：有消耗随机基线相对无消耗掉收益（背景争用）；ours 不进对比数字，只报 `status: todo`
