# satsim

## Layout

`input/` scenarios + offline generate. `src/satsim/method/` engine + Scheduler + link_cost. `src/satsim/output/` C1/C2. Paper tiers: place_once | rebalance_free | rebalance_random / rebalance_ours(TODO). Spec: `docs/SPEC_PLUGIN_FRAMEWORK.md`. Results table: `README.md`.

Naming: `base_*` world templates, `exp_*` runnable experiments (`configs/exp_*.yaml` symlinks), `scene_*` narrative mirrors under `input/scenes/` (not runnable). Filename stem == YAML `name:`.

## Language

Claims, knobs, migration: `GLOSSARY.md`. Scenario / loop / scoring changes: `docs/adr/`. How to consume them: `docs/agents/domain.md`.

## Lab

Research go / no-go → `api.gjam.tech`, workspace `ccds-lab`, entry `LAB-1466`.

## 结果表（必须）

写 README / 实验小结 / claim 表时统一：

- **基线**：`place_once`（首次分配），不是 free。
- **百分比**：写在数值旁边，如 `19.2 (+50%)`、`264.2 (+8.8%)`；相对基线；基线行不写 %。
- **粗体**：该列最优（通常 free）。
- **列**：精简；需要时表头带 ↑↓。
- **中文标签**：臂名优先用首次分配 / 无消耗 / 有消耗·随机（亮星）等白话；代码 id 可括注。
- **读法**：先看相对首次谁大好；有代价臂再和 free 比是否掉回去（可在表下用一句说明，不必另开 vs-free 列）。

