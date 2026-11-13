# 离散卫星功率配置与线性 RC 热量约束（`discrete_tiers` + 离线时空 Oracle / 在线多跳排空贪心）

## 背景与问题

在 `satsim.cosim` 既有动态热模型（`LAB-1779`）中：
1. `CoWorld.device_power(batch, f)` 采用空载 `P_idle = 60 W` 加随 `batch` 连续变化的指数饱和公式 `1 - (1 - floor) * exp(-batch / scale)`；
2. `DynamicThermal` 在 `84°C / 76°C` 之间做非线性滞回降频跳变（`f_throttle = 0.55`）。

这导致请求完成时间对节点驻留集合呈非可加非线性耦合，标准混合整数线性规划（`scipy.optimize.milp` / HiGHS）无法内嵌热量约束，只能依赖外层贪心逐个重跑分段 RC 积分器。

## 决策（含 LAB-1881 独立审计修正）

采用 **Expand-Contract（扩展不破坏旧行为）** 原则，在保留默认连续模式 `diffs=0` 零回归的前提下，引入可选的 **`power.mode: discrete_tiers`（离散功率模式）** 与双算法体系（**离线控制断面 `discrete_time_ilp (120s)` + 在线全周期 `discrete_rh_greedy`**）：

1. **离散功率档位、空星安全频率解耦与对称公平规则（`satsim.cosim.power_tiers`）**：
   - 每颗算力星配置离散档位集合 $k \in \{0, 1, \dots, K\}$：
     - $k = 0$（零请求空闲档）：$P_0 = 0\text{ W}$，$B^{\max}_0 = 0$，$f_0 = 0.0$；当驻留活跃请求数 $n_j(t) = 0$ 时强制落入该档，热物理积分功率严格归零，加速整批排空后的 RC 散热。
     - $k \in \{1, \dots, K\}$（工作档）：绑定三元组 $(P_k, B^{\max}_k, f_k)$（及可选的 `prefill_clock_f` 表达 `LAB-1849` Prefill/Decode 功率不对称性）。
   - **空星 `0W` 热功耗与算步评估频率 `clock_f > 0` 严格解耦（防除零不变量）**：
     - 放置与重平衡策略对空候选星调用 `CoWorld.iter_of(j, extra)`，以及请求刚 `admit` 进入空星但尚未到下一个 `_thermal_tick` 时，均通过 `effective_clock_f(spec, batch, ...)` 按 `max(batch, 1)` 取工作档 `clock_f > 0`，并在 `admit / remove / mig_done` 事件点即时同步节点档位，杜绝 `timing.iter_time` 除零崩溃。
   - **对称基线公平性（`prefer_high_clock = True` 默认物理调速器）**：
     - 在日照且满足一阶 RC 温限裕度的算力星上，默认选用当前合法最高算速档（避免把 `batch = 1..4` 轻载基线人为锁死在慢速档造成 `35%` 不公平降速惩罚）；一旦进入日影（$P_k > P_{\text{dark}}$）或逼近硬温限 $T_{\max}$ 自动回退低功档，同时允许规划策略通过 `set_node_tier(j, k)` 主动提前降档预冷。

2. **一阶 RC 线性热量递推与显式旁路旧滞回**：
   - 在 `discrete_tiers` 模式下，**显式旁路 `DynamicThermal._apply_hysteresis()` 的 `84°C/76°C` 滞回降频跳变**，仅保留一阶 RC 温度积分与峰值统计，防止双套节流叠加污染。
   - 在前瞻热窗口 $\Delta_H$（`thermal.thermal_horizon_s`，默认 `30.0 s`）上，令 $\alpha = e^{-\Delta_H / \tau_{\text{rc}}}$，结温演化关于档位二元变量 $z_{j,k}(t) \in \{0, 1\}$ 完全线性：
     $$T_j(t + \Delta_H) = \alpha T_j(t) + (1 - \alpha)\left(T_{\text{amb}} + R_{\text{th}} \sum_{k=0}^{K} P_k z_{j,k}(t)\right) \le T_{\max} + s_j$$

3. **离线控制断面 Oracle（`discrete_time_ilp`, `120s`）与在线闭环策略（`discrete_rh_greedy`）边界分离**：
   - **离线控制断面 Oracle（`discrete_time_ilp`，`time_limit_s = 120.0`）**：用于真实/校准星座热点断面的离线规模扫描与同目标对拍（同轴对比 `No-Plan Stay Baseline`、`1-Hop Eclipse Greedy`、`Multi-Hop 0W-Drain RH-Greedy` 与 `120s Time-ILP` 的求解耗时与目标函数值），**不挂入 `3600s × 5-seed` 在线每 10s 重平衡循环**（避免单组实验退化至数十小时）。
   - **在线结构感知滚动热贪心（`discrete_rh_greedy`）**：结合（a）**目标星存量请求连带拖慢外部性惩罚**、（b）**源星整批排空至 `0W` 的降温红利**、（c）**受控 2~3 跳 $K$-最短路中继绕开连片热区**、（d）**数据锚点 `tether` 拉取跳数惩罚**，毫秒级出解并接入 `3600s × 5-seed` 全仿真闭环，与仓内基线 `place_once`、`cool_aware`、`deadline_rc` 正面对比。

4. **观测指标与算力落点**：
   - 非 `discrete_tiers` 模式下保持既有 25 项汇总指标逐字 `diffs=0`；
   - `discrete_tiers` 模式下追加 `zero_power_frac`、`power_tier_fracs`、`energy_kj_total`、`tier_unsat_s`、`peak_temp_C` 与 `min_thermal_margin_C`；
   - 实验并发跑数统一落点于 **`cnb-compute / cnb-compute2`**（各 16C/32G 独占容器），不挤占 `jm` 在线 GPU 推理宿主。
