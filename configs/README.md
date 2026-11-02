# configs

可跑 YAML 入口（symlink 到 `input/scenarios/`）。

策略只写 `placement` / `rebalance` / `link_cost`（+ `hysteresis` / `knobs`）。

```bash
.venv/bin/satsim configs/s1_cost_claims_v3.yaml
.venv/bin/satsim configs/poisson_smoke_v1.yaml
```
