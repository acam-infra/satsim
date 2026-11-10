# configs（可运行实验入口）

本目录汇总两套仿真引擎的全部命令行入口配置文件：
1. **L1 机理验证实验（`exp_*.yaml`）**：全部是指向 `input/scenarios/exp_*.yaml` 的软链接，传给 `satsim` 运行。
2. **L2 轨迹协同仿真实验（`cosim_*.yaml`）**：读取 `input/traces/*.json` 真实卫星接触与地影迹线，传给 `satsim-cosim` 或 `scripts/cosim_*.py` 运行。

## 配置文件分层分工

| 前缀 | 存放目录 | 作用 |
|------|----------|------|
| `base_*` | 仅 `input/scenarios/` | **基础环境模板**（硬件、拓扑、温控参数）；不能直接运行 |
| `exp_*` | `input/scenarios/` + 本目录软链 | **L1 可运行实验**（请求到达方式、对比策略、随机种子） |
| `cosim_*` | 本目录 (`configs/cosim_*.yaml`) | **L2 轨迹协同仿真实验**（星历接触轨迹、热力学参数、Agentic 会话配置） |
| `scene_*` | `input/scenes/` | **场景背景说明**（解释参数背后的卫星业务场景）；仅供阅读，不能直接运行 |

---

## 一、L1 机理验证实验（`satsim`）

```bash
.venv/bin/satsim configs/exp_ring_s1_cost_claims_v3.yaml
.venv/bin/satsim configs/exp_orbit_planes_s1_cost_claims_v1.yaml
.venv/bin/python scripts/exp_orbit_estar_contention.py
```

- `exp_ring_s1_cost_claims_v3.yaml`：四星环拓扑主实验（验证结论 C1 重调度收益、结论 C2 带宽争用代价与 `CCE-Net`）
- `exp_orbit_planes_smoke_v1.yaml` / `exp_orbit_planes_s1_cost_claims_v1.yaml`：双轨四星静态跨轨拓扑实验
- `exp_orbit_planes_estar_contention_v1.yaml`：跨轨瓶颈边（E★）多流拥塞实验（配合 `scripts/exp_orbit_estar_contention.py` 运行）
- `exp_ring_s1_dynamic_thermal_v1.yaml`：四星环动态温控（发热降频）实验；分析见 `docs/notes/dynamic_thermal_cost_claims.md`
- `exp_first_placement_v1.yaml`：地面式初次放置策略（`round_robin` / `random` / `least_loaded`）对比实验
- `exp_poisson_smoke_v1.yaml` / `exp_gamma_smoke_v1.yaml` / `exp_sharegpt_s1_smoke_v1.yaml`：泊松、Gamma 突发与 ShareGPT 真实对话长度冒烟测试

L1 调度策略代号：`place_once`（首次分配）｜`rebalance_free`（无消耗迁移）｜`rebalance_random`（随机迁移）｜`rebalance_ours`（自研迁移 `CCE-Net`）。

---

## 二、L2 轨迹协同仿真实验（`satsim-cosim`）

```bash
uv run satsim-cosim configs/cosim_coop_w53_v1.yaml
uv run satsim-cosim configs/cosim_typhoon_agentic_v1.yaml
uv run python scripts/cosim_agentic_tether.py --out-dir runs/torch_paper_eval --seeds 15
```

- `cosim_coop_w53_v1.yaml` / `cosim_coop_sso2_v1.yaml`：Walker 倾角轨 / 太阳同步轨（SSO）多星接触迹线协同仿真
- `cosim_typhoon_agentic_v1.yaml`：台风应急响应多轮 Agentic Session 轨迹协同仿真
- `cosim_placement_thermal.yaml`：热感知初次放置与突发压力测试基准配置（配合 `scripts/cosim_burst_stress.py`、`scripts/cosim_agentic_tether.py` 等运行）

L2 调度策略与 `TORCH` 完整评测报告见 [`../docs/TORCH_PAPER_REPORT.md`](../docs/TORCH_PAPER_REPORT.md)，领域术语见根目录 [`../GLOSSARY.md`](../GLOSSARY.md)。
