# satsim

低轨卫星组的在线推理与跨星迁移仿真器（基于 SimPy）。对比论文三档调度策略：
**首次分配（`place_once`）**｜**无消耗迁移（`rebalance_free`）**｜**占带宽迁移（随机基线 `rebalance_random` + 自研策略 `rebalance_ours` / `CCE-Net`）**。

```bash
uv sync
.venv/bin/satsim configs/exp_ring_s1_cost_claims_v3.yaml          # 主实验：四星环拓扑三档对比
.venv/bin/satsim configs/exp_poisson_smoke_v1.yaml                # 泊松到达冒烟测试
.venv/bin/satsim configs/exp_gamma_smoke_v1.yaml                  # Gamma 突发到达冒烟测试（cv 控制突发度）
.venv/bin/satsim configs/exp_orbit_planes_smoke_v1.yaml           # 静态轨道拓扑冒烟测试
.venv/bin/satsim configs/exp_orbit_planes_s1_cost_claims_v1.yaml  # 静态轨道拓扑三档完整对比
.venv/bin/satsim configs/exp_sharegpt_s1_smoke_v1.yaml            # ShareGPT 真实请求长度冒烟测试
.venv/bin/python scripts/plot_gamma_cv_latency.py                 # 绘制 CV∈{1,2,4,8} 延迟分布图
.venv/bin/pytest
```

## 协同仿真与多轨任务分配 (`satsim.cosim`)

支持读取 orbitsim 导出的星间接触与地影轨迹（`input/traces/*.json`），或通过 `satsim.cosim.synth.synth_multi_sso()` **在内存即时合成多轨道面异构 SSO 星座**（如 `LTAN 10:30/13:30` 成像伴随影轨 + `LTAN 06:00` 晨昏全亮轨），将调度解耦为正交的 **任务分配器（`Placer`）$\times$ 运行中迁移器（`Migrator`）**：

```bash
# 1. 前置问题确认（Motivation）：多轨道面异构 SSO 下传统分配算法 (nearest/rr/random) 的计算负载错配诊断
uv run python scripts/exp_motivation_multi_sso.py --seeds 1,2,3 --rate 10 --sweep-rates 6,8,10,12

# 2. 时变轨迹协同仿真并导出 3-D 回放 JSON
uv run satsim-cosim configs/cosim_coop_w53_v1.yaml --viz ../orbitsim/viz/public/playback
uv run satsim-cosim configs/cosim_typhoon_agentic_v1.yaml --viz ../orbitsim/viz/public/playback  # 台风应急 · 长时会话
```

| 策略 (`policy` / `place`) | 分配与迁移机制（术语见 `GLOSSARY.md`） |
|---|---|
| `nearest` | **就近分配（不迁移）**：按遥感星到算力星的最短星间链路跳数卸载（同跳数轮询），易造成成像同面影轨拥塞而晨昏亮轨闲置 |
| `rr` | **轮询分配（不迁移）**：在可达算力星间循环轮询分配，不感知队列负载与地影降频（引发暗星 Little's Law 队列积压） |
| `random` | **随机分配（不迁移）**：在可达算力星间均匀随机分配，不感知队列负载与地影降频 |
| `least_loaded` (`jsq`) | **最短队列分配（不迁移）**：选当前并发请求数最少的算力星，感知队列长度但不感知地影降频（$0.35\times$） |
| `greedy` | **贪心分配（不迁移）**：新请求发往当前单步耗时（`iter_of`）最快的连通算力星，感知当前明暗与队列，不看未来星历 |
| `reactive` | **反应式迁移（进影就搬）**：贪心分配；算力星一旦进入地影，立即将请求迁移到当前最快的亮星 |
| `ephemeris` | **星历分配（不迁移）**：利用已知星历预测整段解码与回传耗时来放置请求，运行中不迁移 |
| `contact_aware` | **接触感知迁移**：星历分配；按"迁移者提前量 + 源星变快 − 目标星变慢"的节点对净收益决定是否迁移，每颗目标星每周期只接一个（`externality: false` 为只算自己的 v1） |

实验结果与结论见 [`docs/notes/cosim_contact_trace.md`](docs/notes/cosim_contact_trace.md)，引擎设计见 [`docs/adr/0005-cosim-contact-trace.md`](docs/adr/0005-cosim-contact-trace.md)。

## 配置文件命名（`base` / `exp` / `scene`）

| 前缀 | 目录 | 作用 |
|------|------|------|
| `base_*` | `input/scenarios/` | **基础环境模板**（拓扑、GPU、温控、网络硬件参数）；不能直接作为命令行入口运行 |
| `exp_*` | `input/scenarios/` + `configs/` 软链 | **可运行实验**（定义请求流、对比策略、随机种子）；命令行直接跑 `configs/exp_*.yaml` |
| `scene_*` | `input/scenes/` | **场景背景说明**（解释实验对应的物理背景与简化假设）；仅供对照阅读，不能直接运行 |

文件名（去掉 `.yaml`）与文件内的顶层 `name:` 保持一致。`exp_*` 中的 `base:` 引用基础环境模板，`metadata.scene:` 引用对应的场景背景说明。

## 核心结论（Claims）

实验围绕两个核心结论展开（只统计限功切换之后每个生成 token 的耗时 TPOT 是否达标）：

- **结论 C1（重调度收益）**：卫星发生限功切换（亮星变暗、算力降频）后，**首次分配**把请求死锁在变慢的卫星上，吞吐大跌；**无消耗迁移**能把请求及时挪到算力充足的卫星，显著提升达标吞吐。
- **结论 C2（带宽争用代价）**：一旦迁移需要走星间链路传输状态（**占带宽迁移**），多条迁移流与背景流争抢有限带宽，**随机迁移**会因为传输变慢和超时中断，把免费迁移赚来的性能收益吐回去。

## 主实验结果：四星环拓扑（`exp_ring_s1_cost_claims_v3`，多种子均值）

以 **首次分配（`place_once`）** 为基线；括号内为相对基线的变化比例；**粗体**为该列最优。

| 调度策略 | 达标长请求数 (`good_long`) ↑ | 有效吞吐 (`goodput` tok/s) ↑ | 迁移流量 (`link_GB`) ↓ | 迁移超时中断 (`abort`) ↓ |
|---|---|---|---|---|
| 首次分配 (`place_once`) | 12.8 | 242.8 | **0** | **0** |
| **无消耗迁移 (`free`)** | **19.2 (+50%)** | **356.6 (+46.9%)** | **0** | **0** |
| 随机迁移 (`random`) | 15.0 (+17.2%) | 282.0 (+16.1%) | 11.0 | 2.6 |
| **自研迁移 (`ours`, CCE-Net)** | **19.2 (+50%)** | **356.6 (+46.9%)** | 26.1 | **0** |

- **结论 C1**：无消耗迁移相比首次分配显著更好（有效吞吐 **+46.9%**）。
- **结论 C2**：随机迁移因为占带宽和抢链路，有效吞吐从无消耗迁移的 356.6 掉回 282.0（回落约 **−20.9%**），平均每次实验发生 2.6 次迁移超时中断。
- **自研迁移 (`ours`)**：基于计算-通信净转化值打分网络（`CCE-Net`），将算力加速收益与路径拥塞代价压缩为单一标量分值 `v`，主动避开被背景流占满的边并通过空闲邻边接力卸载，实现 **0 超时中断**，有效吞吐追平无消耗迁移上界（**+46.9%**）。

## 双轨四星拓扑结果（`exp_orbit_planes_s1_cost_claims_v1`）

使用 `topology: orbit_planes`（2 颗遥感星 + 2 颗算力星组成跨轨网格），随机种子 1–5 均值；基线同样为 **首次分配（`place_once`）**。

| 调度策略 | 达标长请求数 (`good_long`) ↑ | 有效吞吐 (`goodput` tok/s) ↑ | 迁移流量 (`link_GB`) ↓ | 迁移超时中断 (`abort`) ↓ |
|---|---|---|---|---|
| 首次分配 (`place_once`) | 12.8 | 242.8 | **0** | **0** |
| **无消耗迁移 (`free`)** | **19.2 (+50%)** | **356.6 (+46.9%)** | **0** | **0** |
| 随机迁移 (`random`) | 16.4 (+28.1%) | 306.8 (+26.4%) | 13.9 | **0** |
| 随机迁移·无背景流 (`random_nobg`) | 16.4 (+28.1%) | 306.8 (+26.4%) | 14.5 | **0** |
| **自研迁移 (`ours`, CCE-Net)** | **19.2 (+50%)** | **356.6 (+46.9%)** | 20.8 | **0** |

- 在跨轨网格拓扑上，结论 C1 的提升幅度与环拓扑一致；随机迁移相比无消耗迁移回落约 **−14.0%**；自研迁移（`ours`）通过计算-通信转化打分选路，同样追平无消耗上界（**+46.9%**）。
- 运行命令：`.venv/bin/satsim configs/exp_orbit_planes_s1_cost_claims_v1.yaml`；设计背景见 `docs/adr/0003-orbit-planes-topology-snapshot.md`。

## 动态温控模式试跑（`exp_ring_s1_dynamic_thermal_v1`）

开启 `thermal.mode=dynamic`（卫星随负载发热自动降频），复用主实验负载。**目前单靠发热降频还跑不出结论 C1 的明显差距**：种子 1–3 下，达标长请求数分别为首次分配 12.0、无消耗迁移 8.67（−28%）、随机迁移 3.0（−75%）、自研迁移 10.67（−11%，0 超时中断）；这是因为四颗星在高负载下几乎同时过热降频，场上没有“凉快且算力满血”的目标星可供迁入。详细原因与改进方向见 `docs/notes/dynamic_thermal_cost_claims.md`。

## 跨轨瓶颈链路（E★）拥塞实验（`exp_orbit_planes_estar_contention_v1`）

场景设置：限功切换后，卫星 0 和 1 变暗降频，卫星 2 和 3 保持满血；占带宽迁移策略若将大批请求集中迁往卫星 2，所有迁移流都会挤在同一条跨轨瓶颈链路 E★=`(0,2)` 上。将链路带宽收紧到 `link_gbps=0.5`、超时设为 30s、关闭背景流，且每个请求最多尝试迁移一次。

| 调度策略 | 达标长请求数 (`good_long`) ↑ | 有效吞吐 (`goodput` tok/s) ↑ | 迁移流量 (`link_GB`) ↓ | 迁移超时中断 (`abort`) ↓ | 瓶颈边峰值并发流 ↓ |
|---|---|---|---|---|---|
| 首次分配 (`place_once`) | 6.4 | 129.1 | **0.0** | **0.0** | **0.0** |
| **无消耗迁移 (`free`)** | **19.2 (+200%)** | **356.6 (+176.3%)** | **0.0** | **0.0** | **0.0** |
| 占带宽迁移·瓶颈汇聚 (`random`) | 6.4 (+0%) | 129.1 (+0%) | 2.8 | 12.8 | 12.8 |
| 自研迁移 (`ours`, CCE-Net) | 13.6 (+112.5%) | 257.1 (+99.2%) | 19.0 | **0.0** | 1.0 |

- 无消耗迁移依然收益巨大；盲目占带宽迁移在瓶颈链路 E★ 上扎堆抢带宽（峰值 12.8 条流），导致 **所有迁移全部超时中断（成功迁出数为 0）**，性能跌回与首次分配完全相同（相比无消耗迁移损失 **−66.7%** 的达标长请求）。
- **自研迁移 (`ours`)** 将 E★ 上的并发流严格控制在 **1.0 条**，同时分流到平行跨轨边 `(1,3)`，实现 **0 超时中断**，在物理带宽仅 0.5 Gbps 的极端受限下将达标长请求提升 **+112.5%**、有效吞吐提升 **+99.2%**。
- 运行命令：`.venv/bin/python scripts/exp_orbit_estar_contention.py`（配置文件 `configs/exp_orbit_planes_estar_contention_v1.yaml`；场景说明见 `input/scenes/scene_orbit_planes_estar_contention_v1.yaml`）。

## 吞吐指标口径

总统计时长 **仿真总时长 (`horizon`)** = 请求到达期 `arrival_window_s` + 尾流排空期 `drain_s`。请求有完成时间戳 `t_done` 即视为已完成。

- **请求吞吐率 (`request_throughput_rps`)** = 已完成请求数 / `horizon`（单位：req/s）
- **总 Token 吞吐率 (`total_token_throughput_tok_s`)** = 已完成请求的（输入 prompt + 输出 output）token 总数 / `horizon`（单位：tok/s）
- **达标有效吞吐 (`goodput_tok_s`)** = 限功切换之后、生成速度（TPOT）满足延迟要求的输出 token 总数 / `arrival_window_s`（结论 C1/C2 主表使用的核心指标）

## 文档导航

1. 术语对照表：`GLOSSARY.md`
2. 为什么定这三档调度策略：`docs/adr/0002-three-tier-schedulers.md`
3. 静态轨道拓扑说明：`docs/adr/0003-orbit-planes-topology-snapshot.md`
4. 策略与温控插件规格：`docs/SPEC_PLUGIN_FRAMEWORK.md`
5. 可运行实验入口说明：`configs/README.md`
6. 自研迁移的并行思路看板：`docs/ideas/README.md`

代码目录：输入与场景生成在 `input/`（命令行入口在 `configs/exp_*.yaml`）；仿真引擎与调度策略在 `src/satsim/method/`；指标汇总与结论检验在 `src/satsim/output/`。
