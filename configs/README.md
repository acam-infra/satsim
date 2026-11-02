# configs

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
```

```bash
.venv/bin/satsim configs/s1_cost_claims_v3.yaml
```
