# 自热（dynamic）能否造出 C1/C2 数据？

日期：2026-11-03（Asia/Shanghai）。tip 含 `ThermalModel`（`01193fa` 起）。  
实验：`exp_ring_s1_dynamic_thermal_v1`；对照显式基线 `exp_ring_s1_cost_claims_v3`。

## 结论（诚实）

**目前不能造出我们需要的 C1 向数据。** 自热 alone 下：

- **C1**：`rebalance_free` **没有**相对 `place_once` 拉开 good_long / goodput_slo；本参数下甚至 **更差**（约 −28% good_long）。
- **原始吞吐** req/s、tok/s **全臂持平**（请求都在 drain 内完成）。
- **C2 向**：`rebalance_random` 相对 free 的 goodput **会掉**（链路 + abort），但 free 本身不是「好上界」，不能当作论文 C2 叙事的干净对照。
- 热诊断：四节点几乎都进节流（peak T≈86.7°C，累计节流时间 ∼1.5e3 s），**没有长期凉快的落脚点**。

根因（与旋钮探针一致）：

1. `DynamicThermal` 功率对负载 **二值**（`load>0 → P_busy`），忙着的星同等升温；s1 并发下很快 **全员节流**。
2. 显式 C1 靠的是 **换位后的空间不对称**（一星 f≈0.35、邻星仍 1.0）；自热造不出「暗星旁边还有亮星」的稳定格局。
3. `nmig==0` 一次性迁移：free 常在节流前因负载差就迁走，或把热 **摊到更多节点**，SLO 不升反降。

## 改了什么、为什么

| 项 | 相对 `base_ring_s1` / LAB 默认 | 动机 |
|---|---|---|
| `thermal.mode` | `dynamic` | 接 ADR 0004 自热路径 |
| `tau_s=25` | 示意短时常数 | 使忙节点在 horizon 内过阈 |
| `R_th=0.45` | 偏高 | Teq_busy≈155°C，确定能节流 |
| `T0_C=50` | 略暖启动 | 缩短首次触阈时间 |
| `T_throttle/release` | 84 / 78 | 沿用 LAB-1178/1214 叙事 |
| `f_throttle=0.40` | 低于 LAB 0.71 | 更接近显式 `P_cap=80→f≈0.35`，放大节流幅度 |
| `P_busy=300` | =P_max | 忙时满功率发热 |
| 工作负载 | 对齐 s1 v3（88 req、同 pad/t_swap 锚点） | 与显式表同口径；`explicit_events` 仍写入但 dynamic **不读** |

另加诊断字段（dynamic 行）：`thermal_peak_T_C_max` / `thermal_throttle_time_s_sum` / `thermal_nodes_ever_throttled`。

## 表：显式基线（`exp_ring_s1_cost_claims_v3`，seeds 1–5）

基线 **place_once**；括号相对基线；**粗体**列最优。

| 臂 | good_long ↑ | goodput_slo ↑ | req/s | tok/s | link_GB ↓ | abort ↓ |
|---|---|---|---|---|---|---|
| place_once | 12.8 | 242.8 | 0.0419 | 242.4 | 0 | 0 |
| **free** | **19.2 (+50%)** | **356.6 (+46.9%)** | 0.0419 | 242.4 | **0** | **0** |
| random | 15.0 (+17%) | 282.0 (+16%) | 0.0419 | 242.4 | 11.0 | 2.6 |

- C1：free ≫ 首次。C2：random 相对 free goodput 约 −21%。

## 表：自热（`exp_ring_s1_dynamic_thermal_v1`，seeds 1–3）

| 臂 | good_long ↑ | goodput_slo ↑ | req/s | tok/s | link_GB ↓ | abort ↓ | peakT °C | thr_s Σ | n_thr |
|---|---|---|---|---|---|---|---|---|---|
| place_once | 12.0 | 228.4 | 0.0419 | 254.9 | 0 | 0 | 86.7 | 1599 | 4.0 |
| free | 8.67 (−28%) | 169.1 (−26%) | 0.0419 | 254.9 | 0 | 0 | 86.7 | 1583 | 4.0 |
| random | 3.0 (−75%) | 68.4 (−70%) | 0.0419 | 254.9 | 37.0 | 4.7 | 86.7 | 1533 | 4.0 |
| random_nobg | 7.0 (−42%) | 139.5 (−39%) | 0.0419 | 254.9 | 34.2 | 0 | 86.7 | 1593 | 4.0 |

读法：相对首次，free **未**变好；raw tok/s 平坦；random 更差主要来自链路争用/abort，**不是**「先有免费再分配红利再被代价吃掉」的 C2 教科书形态。

## 探针（未入库配置）摘要

在同引擎上试过：抬/降 `R_th`·`T_throttle`·`tau`·`f_throttle`、减 `target_n`、8 节点、**负载比例功率** monkeypatch、仅 hot→cool 迁移、允许多次 `nmig`。  
常见结局：要么 **无人节流 → 三臂打平且 SLO 饱和**，要么 **多星同节流 → free ≲ 首次**。未稳定出现 free ≫ place_once。

## 下一步建议旋钮（优先序）

1. **空间不对称热源**（食/日照或子集星显式帽 + dynamic）：恢复「暗星旁有亮星」，这才是现有 C1 机制。  
2. **功率随负载连续化**（改 `_power`，非二值）+ 更大节点数/更稀并发，使 idle/轻载星能长期低于阈值。  
3. **调度**：节流触发后再迁、或放宽 `nmig`，避免节流前烧光迁移次数（单独不够，见探针）。  
4. 不要指望只拧 `tau/R_th/T_throttle` 就能在当前二值自热 + 四节点满载上复现 README 主表。

跑：

```bash
.venv/bin/satsim configs/exp_ring_s1_dynamic_thermal_v1.yaml
.venv/bin/satsim configs/exp_ring_s1_cost_claims_v3.yaml
```
