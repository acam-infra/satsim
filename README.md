# satsim

四节点环上的迁移事件循环，用 SimPy 跑。依赖从清华镜像解析，见 `uv.toml`。

```bash
uv sync
# 主实验（显式 placement/rebalance/link_cost）
.venv/bin/satsim input/scenarios/stac_s1_cost_claims_v3.yaml
# 同入口：configs/
.venv/bin/satsim configs/s1_cost_claims_v3.yaml
# 泊松到达冒烟
.venv/bin/satsim configs/poisson_smoke_v1.yaml
.venv/bin/pytest
```

输入在 `input/`：`input/scenarios/stac_s1_cost_claims_v3.yaml` 指向底稿，用点路径覆盖已准入旋钮，再把六条已准入策略乘上种子。方法在 `src/satsim/method/`，换位锚在压缩过境的最密时隙之后。输出定义在 `src/satsim/output/`，汇总只算两句：

- **C1**：首次分配解决不了换位。运行时重分配把请求再放到仍有功率的节点，换位后的 TPOT 才回来。
- **C2**：每条链路有带宽上限。背景流和重分配流抢同一条上限时，上链路的重分配搬不完，收益掉回去。背景关掉后上限够用，两边重新重合。

事件进程是到达后的解码、功率帽换位、周期控制器、背景流、KV 搬移。链路速率在流开始和结束时重算。换位后的 TPOT 才进达标判断，换位前节点仍是满功率。

`input/scenes/inferred_orbit_pod_v3.yaml` 把已经生效的旋钮说成紧凑轨面舱段。它镜像 spec，不提供第二套物理。拿它当入口会直接拒绝。

新覆盖路径、`grid`、非 `ring`、非严格优先级，加载时拒绝。策略只写显式 `placement` / `rebalance` / `link_cost`（关背景用 `knobs.background.gbps_per_node: 0`）。准入表在 `src/satsim/input/schema.py`。C1/C2 按接口组合选臂（place_once / rebalance_local / rebalance_over_link…）。`trace.kind` 准入 `synthetic_pass` 与 `poisson`。

## 插件框架（进行中）

顶层仍是 **input / method / output**。方法内再拆 `engine` + `interfaces` + `policies`；请求生成在 `input/generate`（默认线下）。规格见 [`docs/SPEC_PLUGIN_FRAMEWORK.md`](docs/SPEC_PLUGIN_FRAMEWORK.md)。可跑入口：`input/scenarios/` 与 `configs/`。

