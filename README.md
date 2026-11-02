# satsim

四节点环上的轨面舱段迁移仿真（SimPy）。论文三档调度：**首次分配**｜**无消耗再分配**｜**有消耗再分配**（random 基线 + ours TODO）。

```bash
uv sync
.venv/bin/satsim configs/s1_cost_claims_v3.yaml
.venv/bin/satsim configs/poisson_smoke_v1.yaml   # 泊松冒烟
.venv/bin/pytest
```

## 主结果（`s1_cost_claims_v3`，多种子均值）

相对无消耗再分配（**粗体**为该列最优；箭头表示期望方向）。

| 臂 | 档 | good_long ↑ | goodput ↑ | vs free | link_traffic_GB ↓ | abort ↓ |
|---|---|---:|---:|---:|---:|---:|
| place_once | 1 首次 | 12.8 | 242.8 | −31.9% | 0 | 0 |
| **rebalance_free** | 2 无消耗 | **19.2** | **356.6** | — | **0** | **0** |
| rebalance_random | 3 有消耗·随机 | 14.4 | 271.3 | −23.9% | 37.8 | 9.2 |
| rebalance_ours | 3 有消耗·TODO | 12.8 | 242.8 | −31.9% | 0 | 0 |

- **C1**：首次 → 无消耗：good_long 12.8→19.2；goodput **+46.9%**。
- **C2**：无消耗 → 有消耗随机：goodput **−23.9%**；abort≈9.2（关背景后 abort→0，仍慢于 free）。
- **ours**：占位，`mig=0`，表现同首次分配；待实现。

## Claims（口径）

- **C1**：首次分配解决不了功率帽换位；无消耗再分配才把请求放到仍有功率的节点。
- **C2**：有消耗再分配走链路，与背景流争带宽时随机基线收益掉回去。

达标只看换位后的 TPOT。场景镜像见 `input/scenes/`（不能当入口跑）。

## 阅读路径

1. 术语：`GLOSSARY.md`
2. 三档决策：`docs/adr/0002-three-tier-schedulers.md`
3. 插件规格：`docs/SPEC_PLUGIN_FRAMEWORK.md`
4. YAML：`scheduler` + `link_cost`（见 `configs/`）

输入在 `input/`（或 `configs/` symlink）；方法在 `src/satsim/method/`（全局 Scheduler）；输出在 `src/satsim/output/`。
