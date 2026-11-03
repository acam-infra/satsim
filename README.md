# satsim

四节点环上的轨面舱段迁移仿真（SimPy）。论文三档调度：**首次分配**｜**无消耗再分配**｜**有消耗再分配**（random 基线 + ours TODO）。

```bash
uv sync
.venv/bin/satsim configs/s1_cost_claims_v3.yaml
.venv/bin/satsim configs/poisson_smoke_v1.yaml   # 泊松冒烟
.venv/bin/satsim configs/gamma_smoke_v1.yaml     # Gamma 到达冒烟（cv 控突发）
.venv/bin/satsim configs/orbit_planes_smoke_v1.yaml  # 轨道连边快照冒烟
.venv/bin/satsim configs/orbit_planes_s1_cost_claims_v1.yaml  # 轨道拓扑三档 e2e
.venv/bin/python scripts/plot_gamma_cv_latency.py  # CV∈{1,2,4,8} latency CDF / p99
.venv/bin/pytest
```

## 主结果（环 `s1_cost_claims_v3`，多种子均值）

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

## 轨道连边快照（`orbit_planes_s1_cost_claims_v1`）

`topology: orbit_planes`，sensing×2 + compute×2 类 grid；seeds 1–5；基线同样是 **place_once**。

| 臂 | good_long ↑ | goodput ↑ | link_GB ↓ | abort ↓ |
|---|---|---|---|---|
| place_once | 12.8 | 242.8 | 0 | 0 |
| **free** | **19.2 (+50%)** | **356.6 (+46.9%)** | **0** | **0** |
| random | 14.0 (+9.4%) | 264.2 (+8.8%) | 31.6 | 0 |
| random_nobg | 15.0 (+17%) | 282.0 (+16%) | 30.4 | 0 |

- 跨轨 grid 上 C1 与环同量级；random 只小幅好于首次，相对 free 约 **−25.9%**（nobg 仍约 **−20.9%**）——链路迁移本身在拖，abort 尚未逼出。
- 跑：`.venv/bin/satsim configs/orbit_planes_s1_cost_claims_v1.yaml`；ADR：`docs/adr/0003-orbit-planes-topology-snapshot.md`。

## 吞吐定义

分母 **horizon** = `arrival_window_s + drain_s`（仿真使役时间）。完成 = 有 `t_done`。

- **Request Throughput** `request_throughput_rps` = 完成请求数 / horizon（req/s）
- **Total Token Throughput** `total_token_throughput_tok_s` = 完成请求的 (prompt+output) token 之和 / horizon
- 旧 `goodput_tok_s`：换位后 TPOT 过 SLO 的 *输出* token / `arrival_window_s`（C1/C2 仍用它）

## Claims（口径）

- **C1**：首次分配解决不了功率帽换位；无消耗再分配才把请求放到仍有功率的节点。
- **C2**：有消耗再分配走链路，与背景流争带宽时随机基线收益掉回去。

达标只看换位后的 TPOT。场景镜像见 `input/scenes/`（不能当入口跑）。

## 阅读路径

1. 术语：`GLOSSARY.md`
2. 三档决策：`docs/adr/0002-three-tier-schedulers.md`
3. 轨道连边快照：`docs/adr/0003-orbit-planes-topology-snapshot.md`
4. 插件规格：`docs/SPEC_PLUGIN_FRAMEWORK.md`
5. YAML：`scheduler` + `link_cost`（见 `configs/`）

输入在 `input/`（或 `configs/` symlink）；方法在 `src/satsim/method/`（全局 Scheduler）；输出在 `src/satsim/output/`。
