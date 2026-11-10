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

### 3. L2 真实星座协同仿真与论文实验（OrbitSim CoSim + TORCH）
```bash
# 单场景协同仿真与 3D 轨迹回放导出
uv run satsim-cosim configs/cosim_coop_w53_v1.yaml --viz ../orbitsim/viz/public/playback
uv run satsim-cosim configs/cosim_typhoon_agentic_v1.yaml --viz ../orbitsim/viz/public/playback

# TORCH 论文核心实验套件（突发压力测试 + 三大场景 + Agentic 数据锚点分层）
uv run python scripts/cosim_burst_stress.py --policies none,deadline_rc,torch_notether,torch_lp --seeds 15
uv run python scripts/cosim_agentic_tether.py --out-dir runs/torch_paper_eval --seeds 15
uv run python scripts/verify_torch_milp.py                        # 内置求解器与 HiGHS 精确 MILP 对拍验证
```

---

## 三、L1 机理验证层结果（结论 C1 / C2 与 CCE-Net）

机理实验围绕两个核心结论展开（统计限功切换之后每个生成 token 的耗时 TPOT 是否达标）：

- **结论 C1（重调度收益）**：卫星发生限功切换（亮星变暗、算力降频）后，**首次分配 (`place_once`)** 把请求死锁在变慢的卫星上，吞吐大跌；**无消耗迁移 (`rebalance_free`)** 能把请求及时挪到算力充足的卫星，显著提升达标吞吐。
- **结论 C2（带宽争用代价）**：一旦迁移需要走星间链路传输状态（**占带宽迁移**），多条迁移流与背景流争抢有限带宽，**随机迁移 (`rebalance_random`)** 会因为传输变慢和超时中断，把免费迁移赚来的性能收益吐回去。

### 3.1 主实验结果：四星环拓扑（`exp_ring_s1_cost_claims_v3`，多种子均值）

以 **首次分配（`place_once`）** 为基线；括号内为相对基线的变化比例；**粗体**为该列最优。

| 调度策略 | 达标长请求数 (`good_long`) ↑ | 有效吞吐 (`goodput` tok/s) ↑ | 迁移流量 (`link_GB`) ↓ | 迁移超时中断 (`abort`) ↓ |
|---|---|---|---|---|
| 首次分配 (`place_once`) | 12.8 | 242.8 | **0** | **0** |
| **无消耗迁移 (`free`)** | **19.2 (+50%)** | **356.6 (+46.9%)** | **0** | **0** |
| 随机迁移 (`random`) | 15.0 (+17.2%) | 282.0 (+16.1%) | 11.0 | 2.6 |
| **自研迁移 (`ours`, CCE-Net)** | **19.2 (+50%)** | **356.6 (+46.9%)** | 26.1 | **0** |

- **结论 C1**：无消耗迁移相比首次分配显著更好（有效吞吐 **+46.9%**）。
- **结论 C2**：随机迁移因为占带宽和抢链路，有效吞吐从无消耗迁移的 356.6 掉回 282.0（回落约 **−20.9%**），平均每次实验发生 2.6 次迁移超时中断。
- **自研迁移 (`ours`, CCE-Net)**：基于计算-通信净转化值打分网络，将算力加速收益与路径拥塞代价压缩为单一标量分值 `v`，主动避开被背景流占满的边并通过空闲邻边接力卸载，实现 **0 超时中断**，有效吞吐追平无消耗迁移上界（**+46.9%**）。

### 3.2 双轨四星拓扑结果（`exp_orbit_planes_s1_cost_claims_v1`）

使用 `topology: orbit_planes`（2 颗遥感星 + 2 颗算力星组成跨轨网格），随机种子 1–5 均值；基线同样为 **首次分配（`place_once`）**。

| 调度策略 | 达标长请求数 (`good_long`) ↑ | 有效吞吐 (`goodput` tok/s) ↑ | 迁移流量 (`link_GB`) ↓ | 迁移超时中断 (`abort`) ↓ |
|---|---|---|---|---|
| 首次分配 (`place_once`) | 12.8 | 242.8 | **0** | **0** |
| **无消耗迁移 (`free`)** | **19.2 (+50%)** | **356.6 (+46.9%)** | **0** | **0** |
| 随机迁移 (`random`) | 16.4 (+28.1%) | 306.8 (+26.4%) | 13.9 | **0** |
| 随机迁移·无背景流 (`random_nobg`) | 16.4 (+28.1%) | 306.8 (+26.4%) | 14.5 | **0** |
| **自研迁移 (`ours`, CCE-Net)** | **19.2 (+50%)** | **356.6 (+46.9%)** | 20.8 | **0** |

- 在跨轨网格拓扑上，结论 C1 的提升幅度与环拓扑一致；随机迁移相比无消耗迁移回落约 **−14.0%**；自研迁移（`ours`）同样追平无消耗上界（**+46.9%**）。

### 3.3 跨轨瓶颈链路（E★）拥塞实验（`exp_orbit_planes_estar_contention_v1`）

限功切换后，卫星 0 和 1 变暗降频，卫星 2 和 3 保持满血；若将大批请求集中迁往卫星 2，迁移流会挤在同一条跨轨瓶颈链路 E★=`(0,2)` 上（`link_gbps=0.5`、超时 30s、关闭背景流）。

| 调度策略 | 达标长请求数 (`good_long`) ↑ | 有效吞吐 (`goodput` tok/s) ↑ | 迁移流量 (`link_GB`) ↓ | 迁移超时中断 (`abort`) ↓ | 瓶颈边峰值并发流 ↓ |
|---|---|---|---|---|---|
| 首次分配 (`place_once`) | 6.4 | 129.1 | **0.0** | **0.0** | **0.0** |
| **无消耗迁移 (`free`)** | **19.2 (+200%)** | **356.6 (+176.3%)** | **0.0** | **0.0** | **0.0** |
| 占带宽迁移·瓶颈汇聚 (`random`) | 6.4 (+0%) | 129.1 (+0%) | 2.8 | 12.8 | 12.8 |
| 自研迁移 (`ours`, CCE-Net) | 13.6 (+112.5%) | 257.1 (+99.2%) | 19.0 | **0.0** | 1.0 |

- 盲目迁移在瓶颈链路 E★ 上扎堆抢带宽（峰值 12.8 条流），导致所有迁移全部超时中断；**自研迁移 (`ours`)** 将 E★ 并发流严格控制在 **1.0 条** 并分流至平行跨轨边 `(1,3)`，实现 **0 超时中断**，有效吞吐提升 **+99.2%**。

### 3.4 静态环网动态温控说明（`exp_ring_s1_dynamic_thermal_v1`）

在四星静态环网上直接开启 `thermal.mode=dynamic` 时，由于默认初次放置均匀摊开负载，四颗星几乎同时过热降频，场上缺少凉快目标星可供迁入（原因拆解见 [`docs/notes/dynamic_thermal_cost_claims.md`](docs/notes/dynamic_thermal_cost_claims.md)）。在真实轨道迹线（下方 L2 协同仿真层）中，由于日照/本影相位差与区域性突发到达，星座天然具备空间冷热不对称。

---

## 四、L2 真实星座协同仿真层（OrbitSim CoSim & TORCH）

在真实轨道接触迹线与一阶 RC 热力学演化下，`satsim.cosim` 支持以下放置与运行时重平衡策略：

| 策略代号 | 类别 | 放置与运行时迁移机制 |
|---|---|---|
| `round_robin` / `random` / `least_loaded` / `greedy` | 初次放置（不迁移） | 地面式无热感知基线，新请求按轮询、随机、最低队列或当前最快速度分配 |
| `ephemeris` / `contact_aware` | 星历接触感知 | 利用星历预测接触窗口与进影时刻，按单对节点净收益做 1 跳迁移（见 [`docs/notes/cosim_contact_trace.md`](docs/notes/cosim_contact_trace.md)） |
| `deadline_rc` | 1 跳热感知迁移 | 基于一阶 RC 热力学积分预测完成时刻，局部过热后向 1 跳邻居做贪心单次疏散 |
| **`torch_lp` (`TORCH`)** | **多跳热-轨与数据锚点流重平衡** | **自研核心方法**：基于无旋钮一阶 RC 热积分、物理显存槽位递增边际估值与多跳 ISL 链路占用秒数的全局整数流规划；支持 Agentic 遥感任务回源拉取高分图像切片（`torch_notether` 为去锚点消融） |

- **核心效果概要**（完整形式化建模与 5 大实验表见 **[`docs/TORCH_PAPER_REPORT.md`](docs/TORCH_PAPER_REPORT.md)**）：
  1. **打破 1 跳拥塞踩踏**：在太阳同步轨 150 并发突发（`aligned150`）下，基础 `Round Robin` 达标率为 `50.27%`，叠加 1 跳 `deadline_rc` 提升至 `57.20%`（净救回 10.4 个请求），而 **`RR + TORCH`** 通过 1~3 跳协同疏散将达标率提升至 **`70.18%`**（净救回 29.9 个请求，提升幅度为 1 跳方法的 **2.88 倍**）。
  2. **Agentic 数据锚点“同心圆距离分层”**：无需任何人工跳数阈值，纯文本任务自动被推向 **2~3 跳远端低温星**（远跳占比 `48.9%`），而需回源拉取高分图像切片的 Agentic 任务被自动吸附在 **1 跳邻近圈**（回源等待耗时削减 **78.1%**）。
  3. **毫秒级实时求解**：内置定界求解器单步耗时仅 **`0.26 ms`**，与 HiGHS 精确 MILP 最优解吻合率达 **`96.7%`**。

---

## 五、指标口径与配置规范

### 5.1 吞吐指标口径（L1 机理层）
总统计时长 **仿真总时长 (`horizon`)** = 请求到达期 `arrival_window_s` + 尾流排空期 `drain_s`：
- **请求吞吐率 (`request_throughput_rps`)** = 已完成请求数 / `horizon`（req/s）
- **总 Token 吞吐率 (`total_token_throughput_tok_s`)** = 已完成请求的（输入 prompt + 输出 output）token 总数 / `horizon`（tok/s）
- **达标有效吞吐 (`goodput_tok_s`)** = 限功切换之后、生成速度（TPOT）满足延迟要求的输出 token 总数 / `arrival_window_s`

### 5.2 配置文件分层规范
| 前缀 | 存放目录 | 作用 |
|---|---|---|
| `base_*` | `input/scenarios/` | **基础环境模板**（拓扑、GPU、温控、网络硬件参数）；不可直接运行 |
| `exp_*` | `input/scenarios/` + `configs/` 软链 | **L1 可运行实验**（定义请求流、对比策略、随机种子）；通过 `satsim configs/exp_*.yaml` 运行 |
| `cosim_*` | `configs/cosim_*.yaml` | **L2 轨迹协同仿真实验**；通过 `satsim-cosim configs/cosim_*.yaml` 运行 |
| `scene_*` | `input/scenes/` | **场景物理背景说明**；仅供对照阅读，不可直接运行 |

---

## 六、代码与文档目录层次（Hierarchy）

完整文档索引见 **[`docs/README.md`](docs/README.md)**。

```text
satsim/
├── README.md                        # 项目总览、双层架构与核心实验结果
├── GLOSSARY.md                      # 领域统一术语表（中英对照与代码映射）
├── AGENTS.md                        # 仓库协作规范与结果表格式要求
├── configs/                         # 命令行可运行配置入口 (exp_*.yaml 与 cosim_*.yaml)
├── input/                           # 场景模板 (scenarios/)、背景说明 (scenes/)、轨迹 (traces/) 与负载生成 (generate/)
├── src/satsim/
│   ├── method/                      # [L1 机理层] SimPy 引擎、CCE-Net 调度器 (rebalance_ours) 与温控插件
│   ├── cosim/                       # [L2 星座层] OrbitSim 接触迹线引擎、TORCH 多跳流重平衡 (thermal_rebalance.py)
│   └── output/                      # 指标统计与结论 C1/C2 汇总输出
├── scripts/                         # 论文评测套件 (cosim_*)、瓶颈边实验与 M5 热点分析脚本 (m5_hotspots/)
└── docs/                            # 分层文档库（详见 docs/README.md）
    ├── TORCH_PAPER_REPORT.md        #   ├─ L1: TORCH 论文建模与完整实验报告
    ├── SPEC_PLUGIN_FRAMEWORK.md     #   ├─ L2: 插件接口规范
    ├── adr/                         #   ├─ L2: 架构与建模决策记录 (0001 ~ 0005)
    ├── notes/                       #   ├─ L3: 专题实验笔记与文献调研
    └── ideas/                       #   └─ L4: 并行探索看板与 M5 传算重叠微测
```
