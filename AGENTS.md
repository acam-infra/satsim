# satsim

## 目录结构与双层架构（Layout）

- `input/`：实验场景配置（`scenarios/`）、物理背景说明（`scenes/`）、星历接触迹线（`traces/`）与请求流生成器（`generate/`）。
- `src/satsim/method/`：**[L1 机理验证层]** SimPy 离散事件仿真引擎、全局调度器（`Scheduler`）、链路传输开销模型（`link_cost`）与可插拔温控（`thermal`）。
  - 论文三档调度策略：`place_once`（首次分配）｜`rebalance_free`（无消耗迁移）｜`rebalance_random`（随机迁移）/ `rebalance_ours`（自研迁移 **CCE-Net**，已实现）。
  - 温控与降频模式：`thermal.mode` 支持 `explicit`（定时限功）与 `dynamic`（一阶 RC 动态温控，见 ADR 0004）。
- `src/satsim/cosim/`：**[L2 真实星座层]** OrbitSim 接触迹线联合仿真引擎、**TORCH (`TorchRebalance` / `torch_lp`)** 无旋钮多跳热-轨流重平衡与 Agentic 数据锚点牵引（`thermal_rebalance.py`）。
- `src/satsim/output/`：结论 C1 / C2 指标统计与汇总输出。
- 文档层次总导航：`docs/README.md`
  - L1 论文核心报告：`docs/TORCH_PAPER_REPORT.md`（CoSim + TORCH）与根目录 `README.md`（SimPy C1/C2 主表）。
  - L2 插件与架构决策：`docs/SPEC_PLUGIN_FRAMEWORK.md` 与 `docs/adr/`（0001 ~ 0005）。
  - L3 专题实验笔记：`docs/notes/`。
  - L4 并行探索看板：`docs/ideas/README.md`（含已合入的 `M5` 轨道突发热点刻画与传算重叠微测）。

配置文件命名规范（详见 `configs/README.md`）：
- `base_*`：基础环境模板（定义硬件、拓扑、温控参数，不可直接运行）
- `exp_*`：L1 SimPy 可运行实验（在 `configs/exp_*.yaml` 下有同名软链入口）
- `cosim_*`：L2 OrbitSim 轨迹协同仿真实验（位于 `configs/cosim_*.yaml`）
- `scene_*`：场景背景说明（位于 `input/scenes/`，解释物理背景，不可直接运行）
- `base_*` / `exp_*` / `scene_*` 文件名（去后缀）必须与 YAML 顶层 `name:` 字段完全一致。

## 术语与架构决策（Language）

- 领域术语表：`GLOSSARY.md`
- 架构与建模决策记录：`docs/adr/`
- 如何维护术语与 ADR：`docs/agents/domain.md`

## Agent skills

本仓库内置 `.agents/skills/`（来自 `mattpocock/skills` 的项目级工程与效率技能）。各技能读取以下仓库配置契约：

### Issue tracker

任务与分阶段并发子票统一在 Multica 平台（`ccds-lab`）追踪，不在 GitHub Issues 双开。详见 `docs/agents/issue-tracker.md`。

### Triage labels

五类标准分流角色与 Multica 状态/标签的映射约定。详见 `docs/agents/triage-labels.md`。

### Domain docs

单上下文（Single-context）领域文档布局，以根目录 `GLOSSARY.md` 与 `docs/adr/` 为准。详见 `docs/agents/domain.md`。

## Lab

Research go / no-go → `api.gjam.tech`, workspace `ccds-lab`, entry `LAB-1466`.

## 结果表规范（必须遵守）

写 `README.md`、实验小结、结论对比表时统一遵循：

- **基线**：统一以 `place_once`（首次分配）为 100% 参照基线，不要拿 `free` 当基线。
- **百分比**：写在数值旁边，如 `19.2 (+50%)`、`264.2 (+8.8%)`，表示相对首次分配的增减；基线行自身不写百分比。
- **粗体**：标出该列最优值（通常是无消耗迁移 `free`）。
- **列**：保持精简，表头标明优劣方向 `↑` 或 `↓`。
- **中文标签**：策略名称优先用直白中文——`首次分配` / `无消耗迁移` / `随机迁移` / `自研迁移`，代码 id 可括注在后（如 `首次分配 (place_once)`）。
- **读法**：先看相对首次分配谁有显著提升（结论 C1）；再看占带宽迁移相比无消耗迁移掉回去多少（结论 C2，在表下用一句话说明即可，不必单开一列）。
