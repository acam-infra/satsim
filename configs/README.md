# configs

可跑 CLI 入口：全部是指向 `input/scenarios/exp_*.yaml` 的 symlink。策略只写 `scheduler` + `link_cost`。

## 三层命名

| 前缀 | 放哪 | 干什么 |
|------|------|--------|
| `base_*` | 仅 `input/scenarios/` | 世界底稿；**无** configs 入口 |
| `exp_*` | `input/scenarios/` + 本目录 symlink | 可跑实验 |
| `scene_*` | `input/scenes/` | 叙事镜像；不可 `satsim` 直接跑 |

旧名 `construct_*` / `stac_*` / 无前缀 smoke、以及 `s1_cost_claims_v3` 别名已去掉；只保留一套 `exp_*` 规范名。

```bash
.venv/bin/satsim configs/exp_ring_s1_cost_claims_v3.yaml
.venv/bin/satsim configs/exp_poisson_smoke_v1.yaml
.venv/bin/satsim configs/exp_gamma_smoke_v1.yaml
.venv/bin/satsim configs/exp_orbit_planes_smoke_v1.yaml
.venv/bin/satsim configs/exp_orbit_planes_s1_cost_claims_v1.yaml
.venv/bin/satsim configs/exp_sharegpt_s1_smoke_v1.yaml
# E★ 争用（自定义 loader，不走普通 policies）:
.venv/bin/python scripts/exp_orbit_estar_contention.py
```

当前 symlink：

- `exp_ring_s1_cost_claims_v3.yaml` → 环上主 C1/C2
- `exp_poisson_smoke_v1.yaml` / `exp_gamma_smoke_v1.yaml` → 到达冒烟
- `exp_orbit_planes_smoke_v1.yaml` / `exp_orbit_planes_s1_cost_claims_v1.yaml` → 轨道拓扑
- `exp_sharegpt_s1_smoke_v1.yaml` → ShareGPT 长度冒烟
- `exp_ring_s1_dynamic_thermal_v1.yaml` → 环上自热 dynamic；见 `docs/notes/dynamic_thermal_cost_claims.md`
- `exp_orbit_planes_estar_contention_v1.yaml` → E★ 加压（脚本入口）

三档 id：`place_once`｜`rebalance_free`｜`rebalance_random`｜`rebalance_ours`（TODO）。术语见根目录 `GLOSSARY.md`；主表见 `README.md`。
