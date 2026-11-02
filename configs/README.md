# configs

论文三档 YAML 写法：

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
    link_cost: "full"  # TODO
```

```bash
.venv/bin/satsim configs/s1_cost_claims_v3.yaml
```
