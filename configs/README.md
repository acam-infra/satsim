# configs（可运行实验入口）

本目录下的文件全部是指向 `input/scenarios/exp_*.yaml` 的软链接，可直接传给 `satsim` 命令行运行。每个实验里的对比策略统一通过 `scheduler`（调度器）+ `link_cost`（链路传输开销）两个字段声明。

## 配置文件三层分工

| 前缀 | 存放目录 | 作用 |
|------|----------|------|
| `base_*` | 仅 `input/scenarios/` | **基础环境模板**（硬件、拓扑、温控参数）；不能直接运行 |
| `exp_*` | `input/scenarios/` + 本目录软链 | **可运行实验**（请求到达方式、对比策略、随机种子） |
| `scene_*` | `input/scenes/` | **场景背景说明**（解释参数背后的卫星业务场景）；仅供阅读，不能直接运行 |

## 常用运行命令

```bash
.venv/bin/satsim configs/exp_ring_s1_cost_claims_v3.yaml
.venv/bin/satsim configs/exp_poisson_smoke_v1.yaml
.venv/bin/satsim configs/exp_gamma_smoke_v1.yaml
.venv/bin/satsim configs/exp_orbit_planes_smoke_v1.yaml
.venv/bin/satsim configs/exp_orbit_planes_s1_cost_claims_v1.yaml
.venv/bin/satsim configs/exp_sharegpt_s1_smoke_v1.yaml
# 跨轨瓶颈边（E★）拥塞实验（使用独立脚本加载）：
.venv/bin/python scripts/exp_orbit_estar_contention.py
```

## 当前实验列表

- `exp_ring_s1_cost_claims_v3.yaml`：四星环拓扑主实验（验证结论 C1 重调度收益与结论 C2 带宽争用代价）
- `exp_poisson_smoke_v1.yaml` / `exp_gamma_smoke_v1.yaml`：泊松到达 / Gamma 突发到达冒烟测试
- `exp_orbit_planes_smoke_v1.yaml` / `exp_orbit_planes_s1_cost_claims_v1.yaml`：双轨四星静态轨道拓扑实验
- `exp_sharegpt_s1_smoke_v1.yaml`：ShareGPT 真实对话长度冒烟测试
- `exp_ring_s1_dynamic_thermal_v1.yaml`：四星环动态温控（发热降频）实验；分析见 `docs/notes/dynamic_thermal_cost_claims.md`
- `exp_orbit_planes_estar_contention_v1.yaml`：跨轨瓶颈边（E★）多流拥塞实验（配合脚本运行）

调度策略代号：`place_once`（首次分配）｜`rebalance_free`（无消耗迁移）｜`rebalance_random`（随机迁移）｜`rebalance_ours`（自研迁移，待实现）。完整术语表见根目录 `GLOSSARY.md`，实验结果汇总见 `README.md`。
