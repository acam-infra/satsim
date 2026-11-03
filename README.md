# satsim

四节点环上的轨面舱段迁移仿真（SimPy）。论文三档调度：**首次分配**｜**无消耗再分配**｜**有消耗再分配**（random 基线 + ours TODO）。

```bash
uv sync
.venv/bin/satsim configs/exp_ring_s1_cost_claims_v3.yaml
.venv/bin/satsim configs/exp_poisson_smoke_v1.yaml   # 泊松冒烟
.venv/bin/satsim configs/exp_gamma_smoke_v1.yaml     # Gamma 到达冒烟（cv 控突发）
.venv/bin/satsim configs/exp_orbit_planes_smoke_v1.yaml  # 轨道连边快照冒烟
.venv/bin/satsim configs/exp_orbit_planes_s1_cost_claims_v1.yaml  # 轨道拓扑三档 e2e
.venv/bin/satsim configs/exp_sharegpt_s1_smoke_v1.yaml  # ShareGPT 长度 + 定间隔冒烟
.venv/bin/python scripts/plot_gamma_cv_latency.py  # CV∈{1,2,4,8} latency CDF / p99
.venv/bin/pytest
```

## 命名（base / exp / scene）

| 前缀 | 目录 | 作用 |
|------|------|------|
| `base_*` | `input/scenarios/` | 世界底稿（拓扑、GPU、热、网）；**不能**当 CLI 入口 |
| `exp_*` | `input/scenarios/` + `configs/` symlink | 可跑实验（trace / policies / seeds）；CLI 用 `configs/exp_*.yaml` |
| `scene_*` | `input/scenes/` | 叙事镜像（`satsim.scene/v1`）；**不能**当入口跑 |

文件名 stem 与 YAML 顶层 `name:` 一致。`exp_*` 里的 `base:` / `metadata.scene:` 分别指向 `base_*.yaml` 与 `../scenes/scene_*.yaml`。

## 主结果（环 `exp_ring_s1_cost_claims_v3`，多种子均值）

表格式约束见 `AGENTS.md`（基线 place_once，% 写在数值旁）。

基线 **place_once**；括号内为相对基线；**粗体**为该列最优。

| 臂 | good_long ↑ | goodput ↑ | link_GB ↓ | abort ↓ |
|---|---|---|---|---|
| place_once | 12.8 | 242.8 | 0 | 0 |
| **free** | **19.2 (+50%)** | **356.6 (+46.9%)** | **0** | **0** |
| random | 14.4 (+12.5%) | 271.3 (+11.7%) | 37.8 | 9.2 |
| ours (TODO) | 12.8 (0%) | 242.8 (0%) | 0 | 0 |

- **C1**：free 相对首次明显更好（goodput **+46.9%**）。
- **C2**：random 相对 free 掉回（271 vs 357，约 **−23.9%**），abort≈9.2。
- **ours**：占位，同首次；待实现。

## 轨道连边快照（`exp_orbit_planes_s1_cost_claims_v1`）

`topology: orbit_planes`，sensing×2 + compute×2 类 grid；seeds 1–5；基线同样是 **place_once**。

| 臂 | good_long ↑ | goodput ↑ | link_GB ↓ | abort ↓ |
|---|---|---|---|---|
| place_once | 12.8 | 242.8 | 0 | 0 |
| **free** | **19.2 (+50%)** | **356.6 (+46.9%)** | **0** | **0** |
| random | 14.0 (+9.4%) | 264.2 (+8.8%) | 31.6 | 0 |
| random_nobg | 15.0 (+17%) | 282.0 (+16%) | 30.4 | 0 |

- 跨轨 grid 上 C1 与环同量级；random 只小幅好于首次，相对 free 约 **−25.9%**（nobg 仍约 **−20.9%**）——链路迁移本身在拖，abort 尚未逼出。
- 跑：`.venv/bin/satsim configs/exp_orbit_planes_s1_cost_claims_v1.yaml`；ADR：`docs/adr/0003-orbit-planes-topology-snapshot.md`。


## 自热 dynamic 试跑（`exp_ring_s1_dynamic_thermal_v1`）

`thermal.mode=dynamic` 对齐 s1 负载；**尚未**造出 C1（free ≫ place_once）。seeds 1–3：good_long 首次 12.0 / free 8.67（−28%）/ random 3.0；req/s·tok/s 全臂持平；四节点均进节流。细节与下一旋钮：`docs/notes/dynamic_thermal_cost_claims.md`。

## 跨轨 E★ 加严争用（`exp_orbit_planes_estar_contention_v1`）

构造：换位后暗 0+1、亮 2+3；有代价臂 **sync_burst 迁向 node2**，路径共享 E★=`(0,2)`。主轴 `link_gbps=0.5`，辅 timeout=30 / control=0.25 / 高并发；bg=0。每请求只尝试一次（避免 abort 重试刷爆）。

| 臂 | good_long ↑ | goodput ↑ | link_GB ↓ | abort ↓ | E★峰值流 |
|---|---|---|---|---|---|
| 首次分配 | 6.4 | 129.1 | 0.0 | 0.0 | 0.0 |
| **无消耗再分配** | **19.2 (+200%)** | **356.6 (+176%)** | **0.0** | **0.0** | **0.0** |
| 有消耗·E★汇聚 | 6.4 (+0%) | 129.1 (+0%) | 2.8 | 12.8 | 12.8 |

- 无消耗仍大好；有代价在 E★ 上互抢后 **全部超时 abort（mig=0）**，成绩被压回与首次持平，相对 free 约 **−66.7%** good_long。
- 跑：`.venv/bin/python scripts/exp_orbit_estar_contention.py`（配置 `configs/exp_orbit_planes_estar_contention_v1.yaml`；叙事见 `input/scenes/scene_orbit_planes_estar_contention_v1.yaml`）。

## 吞吐定义

分母 **horizon** = `arrival_window_s + drain_s`（仿真使役时间）。完成 = 有 `t_done`。

- **Request Throughput** `request_throughput_rps` = 完成请求数 / horizon（req/s）
- **Total Token Throughput** `total_token_throughput_tok_s` = 完成请求的 (prompt+output) token 之和 / horizon
- 旧 `goodput_tok_s`：换位后 TPOT 过 SLO 的 *输出* token / `arrival_window_s`（C1/C2 仍用它）

## Claims（口径）

- **C1**：首次分配解决不了功率帽换位；无消耗再分配才把请求放到仍有功率的节点。
- **C2**：有消耗再分配走链路，与背景流争带宽时随机基线收益掉回去。

达标只看换位后的 TPOT。场景镜像见 `input/scenes/scene_*.yaml`（不能当入口跑）。

## 阅读路径

1. 术语：`GLOSSARY.md`
2. 三档决策：`docs/adr/0002-three-tier-schedulers.md`
3. 轨道连边快照：`docs/adr/0003-orbit-planes-topology-snapshot.md`
4. 插件规格：`docs/SPEC_PLUGIN_FRAMEWORK.md`
5. YAML：`scheduler` + `link_cost`（见 `configs/`）

输入在 `input/`（CLI 入口 `configs/exp_*.yaml` symlink）；方法在 `src/satsim/method/`（全局 Scheduler）；输出在 `src/satsim/output/`。
