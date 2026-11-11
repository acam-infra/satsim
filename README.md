# satsim

天基低轨算力星座的在线推理与跨星有状态迁移仿真器。仓库采用**“机理验证层（SimPy 静态拓扑）+ 真实星座层（OrbitSim 轨迹协同仿真）”**双层架构，主线代码与论文实验已统一收拢在 `main` 分支。

---

## 一、双层仿真架构总览

| 层次 | 命令行入口 | 代码目录 | 物理拓扑与温控 | 核心解决的问题与自研算法 |
|---|---|---|---|---|
| **L1. 机理验证层**<br>（SimPy 离散事件仿真） | `satsim` | `src/satsim/method/` | 四星环网 (`ring`) / 双轨四星网格 (`orbit_planes`)；支持定时限功 (`explicit`) 与一阶 RC 动态温控 (`dynamic`) | 干净剥离并验证**结论 C1（重调度收益）**与**结论 C2（带宽争用代价）**；实现 **CCE-Net (`rebalance_ours`)** 算网交换打分选路 |
| **L2. 真实星座层**<br>（OrbitSim 轨迹联合仿真） | `satsim-cosim` | `src/satsim/cosim/` | 读取真实星间接触迹线、时变 ISL 带宽与日照/本影相位演化（`input/traces/*.json`） | 解决局部星座过热/进影时的单跳拥塞踩踏与多模态原图回访开销；实现 **TORCH (`torch_lp`)** 无旋钮多跳热-轨流规划与数据锚点牵引 |

---

## 二、快速运行指南

### 1. 环境安装与测试
```bash
uv sync
uv run pytest -q
```

### 2. L1 机理验证层实验（SimPy）
```bash
.venv/bin/satsim configs/exp_ring_s1_cost_claims_v3.yaml          # 主实验：四星环拓扑三档 + CCE-Net 对比
.venv/bin/satsim configs/exp_orbit_planes_s1_cost_claims_v1.yaml  # 双轨四星静态跨轨拓扑完整对比
.venv/bin/python scripts/exp_orbit_estar_contention.py            # 跨轨瓶颈边（E★）多流拥塞实验
.venv/bin/satsim configs/exp_ring_s1_dynamic_thermal_v1.yaml      # 动态温控（发热降频）对照实验
.venv/bin/satsim configs/exp_poisson_smoke_v1.yaml                # 泊松到达冒烟测试
.venv/bin/satsim configs/exp_gamma_smoke_v1.yaml                  # Gamma 突发到达冒烟测试（cv 控制突发度）
.venv/bin/satsim configs/exp_sharegpt_s1_smoke_v1.yaml            # ShareGPT 真实请求长度冒烟测试
```

### 3. L2 真实星座协同仿真与多轨调度（OrbitSim CoSim + TORCH）

支持读取 orbitsim 导出的星间接触与地影轨迹（`input/traces/*.json`），或通过 `satsim.cosim.synth.synth_multi_sso()` **在内存即时合成多轨道面异构 SSO 星座**（如 `LTAN 10:30/13:30` 成像伴随影轨 + `LTAN 06:00` 晨昏全亮轨），将调度解耦为正交的 **任务分配器（`Placer`）$\times$ 运行中迁移器（`Migrator`）**：

```bash
# 1. 前置问题确认（Motivation）：多轨道面异构 SSO 下传统分配算法 (nearest/rr/random) 的计算负载错配诊断
uv run python scripts/exp_motivation_multi_sso.py --seeds 1,2,3 --rate 10 --sweep-rates 6,8,10,12

# 2. 单场景协同仿真与 3D 轨迹回放导出
uv run satsim-cosim configs/cosim_coop_w53_v1.yaml --viz ../orbitsim/viz/public/playback
uv run satsim-cosim configs/cosim_typhoon_agentic_v1.yaml --viz ../orbitsim/viz/public/playback

# 3. TORCH 论文核心实验套件（突发压力测试 + 三大场景 + Agentic 数据锚点分层）
uv run python scripts/cosim_burst_stress.py --policies none,deadline_rc,torch_notether,torch_lp --seeds 15
uv run python scripts/cosim_agentic_tether.py --out-dir runs/torch_paper_eval --seeds 15
uv run python scripts/verify_torch_milp.py                        # 内置求解器与 HiGHS 精确 MILP 对拍验证
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

---

## 三、L1 机理验证层结果（结论 C1 / C2 与 CCE-Net）

机理实验围绕两个核心结论展开（统计限功切换之后每个生成 token 的耗时 TPOT 是否达标）：
- **结论 C1（重调度收益）**：卫星发生限功切换（亮星变暗、算力降频）后，**首次分配 (`place_once`)** 把请求死锁在变慢的卫星上，吞吐大跌；**无消耗迁移 (`rebalance_free`)** 能把请求及时挪到算力充足的卫星，显著提升达标吞吐。
- **结论 C2（带宽争用代价）**：一旦迁移需要走星间链路传输状态（**占带宽迁移**），多条迁移流与背景流争抢有限带宽，**随机迁移 (`rebalance_random`)** 会因为传输变慢和超时中断，把免费迁移赚来的性能收益吐回去。

### 3.1 主实验结果：四星环拓扑（`exp_ring_s1_cost_claims_v3`，多种子均值）

以 **首次分配（`place_once`）** 为基线；括号内为相对基线的变化比例；**粗体**为该列最优。

| 策略 | 总体达标吞吐 (req/s) | 限功期达标吞吐 (req/s) | SLO 违约率 | 迁移总次数 | 迁移中断次数 | 迁移传输均时 (s) |
|---|---|---|---|---|---|---|
| `place_once`（基线） | 0.088 | 0.041 | 38.2% | 0 | 0 | - |
| `rebalance_free`（上限） | **0.126 (+43.2%)** | **0.095 (+131.7%)** | **11.4%** | 38.6 | 0 | 0 |
| `rebalance_random`（对比） | 0.094 (+6.8%) | 0.052 (+26.8%) | 33.1% | 29.4 | 4.8 | 18.3 |
| `rebalance_ours`（CCE-Net 自研） | 0.121 (+37.5%) | 0.089 (+117.1%) | 14.8% | 34.2 | 0.4 | 6.2 |

> 完整机理实验数据复现与多拓扑扩展详见各实验 yaml 与 `docs/notes/`。
