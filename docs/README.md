# 文档总览（Documentation Hierarchy）

本目录按**从论文结论 → 架构决策 → 专题笔记 → 并行探索**四个层次组织所有技术与科研文档。

```text
docs/
├── README.md                        # 本文件：文档层次总导航
├── TORCH_PAPER_REPORT.md            # [L1 论文主报告] TORCH 多跳热-轨与数据锚点流重平衡建模与评测
├── torch_paper_eval/                # [L1 论文评测数据] 15 seeds 多场景汇总数据 (summary.json / runs.csv)
├── SPEC_PLUGIN_FRAMEWORK.md         # [L2 框架规格] Scheduler / LinkCost / ThermalModel 插件接口规范
├── adr/                             # [L2 架构决策] 关键建模取舍记录 (0001 ~ 0005)
├── notes/                           # [L3 专题笔记] 阶段性实验复盘、文献调研与现象分析
├── ideas/                           # [L4 并行探索] 候选思路看板与独立消融微测 (S* / B* / M*)
└── agents/                          # [维护规范] 领域术语表与 ADR 维护指南
```

---

## L1. 论文核心报告与评测数据

| 文档 / 目录 | 对应引擎 | 核心内容 |
|---|---|---|
| [`TORCH_PAPER_REPORT.md`](TORCH_PAPER_REPORT.md) | `satsim-cosim` | **TORCH 完整论文形式化建模与五大实验报告**：一阶 RC 热力学积分、物理 KV 槽位边际估值、多跳链路占用秒数 MILP 规划、Agentic 数据锚点同心圆分层、无旋钮参数敏感度与 HiGHS 求解器实时性对拍 |
| [`torch_paper_eval/`](torch_paper_eval/) | `satsim-cosim` | `TORCH` 论文评测原始汇总结果（`runs.csv` 与 `summary.json`） |
| [`../README.md`](../README.md) | `satsim` & `satsim-cosim` | **仓库总入口与结论 C1 / C2 主实验表**（四星环网、双轨四星、跨轨瓶颈边 E★ 拥塞对比与 `CCE-Net` 结果） |

---

## L2. 架构与建模决策记录（ADR & Specs）

记录仿真器演进过程中“为什么这样建模”的固定决议：

| 编号 / 文件 | 主题 | 一句话结论 |
|---|---|---|
| [`SPEC_PLUGIN_FRAMEWORK.md`](SPEC_PLUGIN_FRAMEWORK.md) | 插件框架规格 | 定义 `Scheduler`、`LinkCost`、`ThermalModel` 的统一协议与注册入口 |
| [`adr/0001-closed-scenario-and-two-claims.md`](adr/0001-closed-scenario-and-two-claims.md) | 封闭场景与双结论 | 锁定结论 **C1（重调度收益）** 与 **C2（带宽争用代价）** 的最小可证场景 |
| [`adr/0002-three-tier-schedulers.md`](adr/0002-three-tier-schedulers.md) | 三档调度策略 | 确立 `place_once`（首次分配）、`rebalance_free`（无消耗迁移）、`rebalance_random` / `rebalance_ours`（占带宽迁移）三档对照体系 |
| [`adr/0003-orbit-planes-topology-snapshot.md`](adr/0003-orbit-planes-topology-snapshot.md) | 静态跨轨拓扑快照 | 引入 `orbit_planes`（遥感星 + 算力星跨轨网格）以暴露跨轨瓶颈边争用 |
| [`adr/0004-pluggable-thermal-model.md`](adr/0004-pluggable-thermal-model.md) | 可插拔温控模型 | 支持 `explicit`（定时限功切换）与 `dynamic`（一阶 RC 负载自发热降频）双模式 |
| [`adr/0005-cosim-contact-trace.md`](adr/0005-cosim-contact-trace.md) | OrbitSim 轨迹协同仿真 | 引入真实星间接触与本影迹线驱动的 `satsim.cosim` 仿真层 |

---

## L3. 专题实验笔记与调研（Notes）

针对特定物理现象、阶段性复盘或相关工作的深入记录：

- [`notes/research_approach.md`](notes/research_approach.md)：整体研究故事线与推进路线图。
- [`notes/cosim_contact_trace.md`](notes/cosim_contact_trace.md)：OrbitSim 接触迹线驱动下的早期四档策略（`greedy` / `reactive` / `ephemeris` / `contact_aware`）实验笔记与 3D 回放指南。
- [`notes/dynamic_thermal_cost_claims.md`](notes/dynamic_thermal_cost_claims.md)：四星静态环网开启纯动态自发热后结论 C1 消失的原因拆解（同步升温导致无凉快目标星）。*(注：后续针对该问题的 `pack + hetero` 专题探索归档于独立分支 `exp/dynamic-batch-hetero-v1`)*。
- [`notes/related_work_migration_traffic.md`](notes/related_work_migration_traffic.md)：有状态推理迁移与流量调度的相关工作调研。
- [`notes/multi_flow_throughput_traditional.md`](notes/multi_flow_throughput_traditional.md)：传统网络多流规划与系统吞吐方向调研笔记。

---

## L4. 并行思路探索看板（Ideas）

用于在不动主表的前提下独立探索新场景（`S*`）、新基线（`B*`）与新机制（`M*`）：

- **总看板与隔离规则**：[`ideas/README.md`](ideas/README.md)
- **已完成并合入 `main` 的探索**：
  - [`ideas/M5_traffic_compute_overlap.md`](ideas/M5_traffic_compute_overlap.md)：轨道入网突发与**热点同时性（Hotspot Concurrency）**刻画，以及单瓶颈边下**截止期排序 × 传算重叠**的 2×2 消融微测（配套脚本：`scripts/m5_hotspots/`）。
- **其他候选草案**：`S1`（地影波）、`S2`（温控空间不对称）、`B1`（贪心不看链路准入）、`M1`（路径重叠选星）、`M2`（错峰预迁移）、`M3`（传 KV vs 重算）、`M4`（截止感知准入）。
