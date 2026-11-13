# 离散卫星功率配置与线性 RC 热量约束（`discrete_tiers` + 联合 MILP）

## 背景与问题

在 `satsim.cosim` 既有动态热模型（`LAB-1779`）中：
1. `CoWorld.device_power(batch, f)` 采用空载 `P_idle = 60 W` 加随 `batch` 连续变化的指数饱和公式 `1 - (1 - floor) * exp(-batch / scale)`；
2. `DynamicThermal` 在 `84°C / 76°C` 之间做非线性滞回降频跳变（`f_throttle = 0.55`）。

这导致请求完成时间对节点驻留集合呈非可加非线性耦合，标准混合整数线性规划（`scipy.optimize.milp` / HiGHS）无法内嵌热量约束，只能依赖外层贪心逐个重跑分段 RC 积分器。

## 决策

采用 **Expand-Contract（扩展不破坏旧行为）** 原则，在保留默认连续模式 `diffs=0` 零回归的前提下，引入可选的 **`power.mode: discrete_tiers`（离散功率模式）** 与 **`discrete_milp` 联合调度策略**：

1. **离散功率档位与零请求 0W（`satsim.cosim.power_tiers`）**：
   - 每颗算力星配置离散档位集合 $k \in \{0, 1, \dots, K\}$：
     - $k = 0$（零请求空闲档）：$P_0 = 0\text{ W}$，$B^{\max}_0 = 0$，$f_0 = 0.0$；当驻留活跃请求数 $n_j(t) = 0$ 时强制落入该档，功率严格归零，加速整批排空后的 RC 散热。
     - $k \in \{1, \dots, K\}$（工作档）：绑定固定三元组 $(P_k, B^{\max}_k, f_k)$。基准实验默认采用两档工作态（共三态）：低功档 `(150 W, max_batch=4, clock_f=0.65)`、高功档 `(300 W, max_batch=12, clock_f=1.0)`。
   - 日影段（`is_eclipse(j, t) == True`）自动屏蔽 $P_k > P_{\text{dark}}$ 的高功耗档（若配置了满足 $\le P_{\text{dark}}$ 的日影低功档则允许使用，否则日影下不接纳超包络工作档）。

2. **一阶 RC 线性热量递推与硬温限约束**：
   - 在前瞻热窗口 $\Delta_H$（`thermal.thermal_horizon_s`，默认 `30.0 s`）上，令 $\alpha = e^{-\Delta_H / \tau_{\text{rc}}}$，结温演化关于档位选择二元变量 $z_{j,k}(t) \in \{0, 1\}$ 完全线性：
     $$T_j(t + \Delta_H) = \alpha T_j(t) + (1 - \alpha)\left(T_{\text{amb}} + R_{\text{th}} \sum_{k=0}^{K} P_k z_{j,k}(t)\right) \le T_{\max} + s_j$$
   - 其中 $T_{\max}$（`thermal.T_max_C`，默认 `84.0 °C`）为硬温限上限，$s_j \ge 0$ 为带大惩罚系数 $M_{\text{pen}}$ 的软松弛变量，仅在极端全网过载无可行解时兜底防止求解器报错崩溃。

3. **时空联合多时隙 ILP（`120s` 上限）vs. 专用滚动热感知贪心双算法设计（`satsim.cosim.discrete_milp`）**：
   - **完整多时隙时空规划（`discrete_time_ilp`，`time_limit_s = 120.0`）**：在多时隙窗口 $\tau \in \{0, \dots, H-1\}$ 上联合决策请求驻留 $y_{r,j,\tau} \in \{0,1\}$、跨星迁移 $m_{r,e,\tau} \in \{0,1\}$ 与功率档位 $z_{j,k,\tau} \in \{0,1\}$，纳入跨时间流守恒、KV 容量、共享边预算、`(D1)-(D3)` 与多步线性 RC 递推 $T_j(\tau+1) \le T_{\max} + s_{j,\tau}$。变量规模按 $O(|\mathcal{R}|(|\mathcal{V}_c| + |\mathcal{E}|)H)$ 膨胀：小规模下由 HiGHS 给出精确最优上界，中等/真实星座规模下即使给足 **`120 s`** 求解上限也会因分支定界树爆炸触发超时或只能拿到劣质可行解。
   - **结构感知滚动热贪心（`discrete_rh_greedy`）**：针对离散功率阶跃结构定制三阶段多项式时间调度——（a）**源星 `0W` 排空联盟优先**：优先疏散日影临近/高温星并奖励能使源星降档（尤其整批排空至 `0W` 档以获得最大 RC 降温率）的迁移集合；（b）**目的星多步 RC 轨迹前瞻**：用 `rollout_rc_temperatures` 在 $O(H)$ 内校验未来 $H$ 步光照切换与温限裕度；（c）**边际效用/通信代价增量贪心**：在毫秒级耗时内稳定产出零违约高质量解，在真实规模 `120s` 预算对比中同时取得更短求解时间与更优最终优化目标（Goodput / 总效用）。

4. **观测指标对齐（`satsim.cosim.record`）**：
   - 保持既有 25 项汇总指标在非 `discrete_tiers` 模式下逐字不变；
   - 在 `discrete_tiers` 模式下输出 `zero_power_frac`（0W 空闲占比）、`power_tier_fracs`（各档位时间占比）、`energy_kj_total`（总能耗 kJ）、`peak_temp_C` 与 `min_thermal_margin_C`。
