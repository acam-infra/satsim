# satsim

## 目录结构（Layout）

- `input/`：实验场景配置（`scenarios/`）、物理背景说明（`scenes/`）与请求流生成器（`generate/`）。
- `src/satsim/method/`：SimPy 仿真引擎、全局调度器（`Scheduler`）与链路传输开销模型（`link_cost`）。
- `src/satsim/output/`：结论 C1 / C2 指标统计与汇总输出。
- 论文三档调度策略：`place_once`（首次分配）｜`rebalance_free`（无消耗迁移）｜`rebalance_random`（随机迁移）/ `rebalance_ours`（自研迁移，TODO）。
- 插件与接口规格：`docs/SPEC_PLUGIN_FRAMEWORK.md`。
- 温控与降频模式：`thermal.mode` 支持 `explicit`（定时限功）与 `dynamic`（动态温控，见 ADR 0004）。
- 实验结果主表：`README.md`。
- 自研迁移的并行思路：`docs/ideas/`，每条思路一个文件、一个分支、一个调度器文件，隔离规则见 `docs/ideas/README.md`。

配置文件命名规范：
- `base_*`：基础环境模板（定义硬件、拓扑、温控参数，不可直接运行）
- `exp_*`：可运行实验（在 `configs/exp_*.yaml` 下有同名软链入口）
- `scene_*`：场景背景说明（位于 `input/scenes/`，解释物理背景，不可直接运行）
- 文件名（去后缀）必须与 YAML 顶层 `name:` 字段完全一致。

## 术语与架构决策（Language）

- 领域术语表：`GLOSSARY.md`
- 架构与建模决策记录：`docs/adr/`
- 如何维护术语与 ADR：`docs/agents/domain.md`

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
