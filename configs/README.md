# configs

可跑入口（symlink → `input/scenarios/`）。策略只写 `scheduler` + `link_cost`。

```bash
.venv/bin/satsim configs/s1_cost_claims_v3.yaml
.venv/bin/satsim configs/orbit_planes_smoke_v1.yaml  # 可选：轨道连边快照冒烟
.venv/bin/satsim configs/orbit_planes_s1_cost_claims_v1.yaml  # 轨道拓扑三档 e2e
```

三档 id：`place_once`｜`rebalance_free`｜`rebalance_random`｜`rebalance_ours`（TODO）。术语见根目录 `GLOSSARY.md`；主表见 `README.md`。
