# satsim

四节点环上的迁移事件循环，用 SimPy 跑。依赖从清华镜像解析，见 `uv.toml`。

```bash
uv sync
uv run satsim scenarios/stac_s1_cost_claims_v3.yaml
uv run pytest
```

`scenarios/stac_s1_cost_claims_v3.yaml` 指向底稿，用点路径覆盖已准入旋钮，再把六条已准入策略乘上种子。换位锚在压缩过境的最密时隙之后。汇总只算两句：

- **C1**：迁移有收益。换位后立刻改放置，换位之后的 TPOT 好于留在被压功率的节点上。
- **C2**：随意迁移的链路代价大。不设字节预算，把当时的整段 KV 送上 ISL，byte·hop 远大于免费改放置（免费不占链路）。

事件进程是到达后的解码、功率帽换位、周期控制器、背景流、KV 搬移。链路速率在流开始和结束时重算。换位后的 TPOT 才进达标判断，换位前节点仍是满功率。

`scenes/inferred_orbit_pod_v3.yaml` 把已经生效的旋钮说成紧凑轨面舱段。它镜像 spec，不提供第二套物理。拿它当入口会直接拒绝。

新策略名、新的覆盖路径、`grid`、非 `ring`、非严格优先级，加载时拒绝。准入表在 `src/satsim/spec.py`。
