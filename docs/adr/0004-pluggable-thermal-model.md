# 热路径收成可插拔 ThermalModel

`World.f_of` 原先内联读 `thermal.explicit_events` 再走 DVFS。为把 lab 里的自动温度曲线/动态节流接进来又不改 decode 与调度臂，把「节点此时钟倍率」收成 `ThermalModel` 协议：`clock_factor(node, t)` + 控制周期 `advance`。工厂键 `thermal.mode`：`explicit`（缺省，行为与旧路径一致）| `dynamic`（每节点一阶 RC + 滞回两档节流）。decode_proc 不出现热分支；新后端只加 policies/thermal 与 schema 准入。
