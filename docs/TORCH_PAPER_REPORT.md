# TORCH: Thermo-Orbital Spatio-Temporal & Data-Anchor Tethered Multi-Hop Flow Rebalancing for Spaceborne Agentic Inference

> **Repository Summary & Paper Formulation Document**
> - **Core Algorithm Implementation**: [`src/satsim/cosim/thermal_rebalance.py`](../src/satsim/cosim/thermal_rebalance.py) (`TorchRebalance` / `torch_lp`)
> - **World & Agentic Tether Co-Simulation**: [`src/satsim/cosim/world.py`](../src/satsim/cosim/world.py) (`Req`, `World.migrate(r, dst, path)`, raw-image tether pull)
> - **Policy Registry**: [`src/satsim/cosim/policies.py`](../src/satsim/cosim/policies.py)
> - **Evaluation Scripts**:
>   - Standard Burst Stress: [`scripts/cosim_burst_stress.py`](../scripts/cosim_burst_stress.py)
>   - Agentic Tether & Multi-Scenario Suite: [`scripts/cosim_agentic_tether.py`](../scripts/cosim_agentic_tether.py)
> - **Unit Tests**: [`tests/test_cosim.py`](../tests/test_cosim.py) (`uv run pytest -q`)

---

## 1. 核心故事线：两阶段解耦与全网算力统筹 (Two-Stage Architecture)

在天基遥感星座中，遥感成像卫星（地面站过顶区域）在短时间内集中产生大量多模态遥感分析请求。我们将全网推理服务解耦为两个阶段：

1. **阶段一：常规初始放置 (`Round Robin`) + 长时推理暴露局部物理失衡**
   - 请求到达时，仅通过最基础的无感知 **Round Robin (RR)** 均匀分发给可见邻近算力卫星。
   - 随着推理推进（长序列 Decode + 多轮工具调用），由于各卫星所处光照相（日照区环境热辐射加剧升温 vs 本影区电池放电限功 $0.355\times$）、初始温度、以及并发 Batch 的差异，部分卫星迅速撞上 **热节流墙 ($T_u \ge 75^\circ\text{C}$，算力骤降至 $0.55\times / 0.25\times$)** 或 **本影能量墙**。
2. **阶段二：1-to-Many 多跳最小费用流全局重平衡 (`TORCH`)**
   - 当局部算力簇集体过热或限功时，传统的“1-to-1 最近邻单次迁移”（如 `deadline_rc`）会发生**拥塞踩踏**，且无法跨越已饱和的一跳邻居去利用全网远端处于低温日照黄金窗口的空闲算力卫星。
   - **TORCH** 在每个重平衡周期内求解一个**基于纯物理量纲（秒）的槽位边际分配与多跳链路流规划问题**，将局部堆积的滞留任务疏散到全网最优的 1 跳、2 跳、3 跳算力节点。

---

## 2. 最小形式化数学建模 (Knob-Free First-Principles Formulation)

模型中**没有任何人为硬编码的“安全并发上限 ($b_{\text{crit}}$)”、硬编码跳数截断或人为拍脑袋权重**。所有迁移与跳数分布均由一阶 RC 热力学、物理显存槽位上限及多跳 ISL 链路物理带宽自然涌现。

### 2.1 状态空间与纯物理参数

在重平衡时刻 $t$，星座拓扑为有向图 $\mathcal{G}(t) = (\mathcal{V}, \mathcal{E}(t))$，每条星间链路 $e \in \mathcal{E}(t)$ 具有物理可用速率 $B_e(t)$（$\text{Bytes/s}$）。

| 符号 | 物理定义与量纲 | 物理计算公式 |
| :--- | :--- | :--- |
| $T_u(t), \phi_u(t)$ | 卫星 $u$ 当前结温（$^\circ\text{C}$）与日照因子 $\phi_u \in [0, 1]$ | 一阶 RC 热阻网络实时状态：$C_{\text{th}} \frac{dT_u}{dt} = P_{\text{compute}}(b_u, T_u, \phi_u) - \frac{T_u - T_{\text{amb}}(\phi_u)}{R_{\text{th}}}$ |
| $\mu(b, T, \phi)$ | 并发度 $b$、温度 $T$、光照 $\phi$ 下的单步 Decode 吞吐率（$\text{tokens/s}$） | $\mu(b, T, \phi) = \mu_0(b) \cdot \min\big(\alpha_{\text{therm}}(T),\, \alpha_{\text{pow}}(\phi)\big)$ |
| $B^{\max}_u$ | 卫星 $u$ 的**物理显存（KV Cache）硬槽位上限** | 由星载 HBM/DRAM 容量严格决定（无人工 $b_{\text{crit}}$ 截断），剩余槽位 $K_v = B^{\max}_v - b_v$ |
| $L_r, D_r$ | 请求 $r$ 剩余生成长度（$\text{tokens}$）与截止期（$\text{s}$） | 任务固有属性；在任意给定节点与并发队列下，由 RC 数值积分器精确推演完成时刻 $\hat{t}_{\text{fin}}(r)$ |
| $a_r, M_r, S_{\text{tile}}$ | 原始遥感图像锚点卫星 $a_r$、剩余回访次数 $M_r$、单次切片字节 $S_{\text{tile}}$ | Agentic 遥感任务（如 `zoom_in_crop`）每次工具调用需从锚点星 $a_r$ 拉取原始高分图像切片 |

---

### 2.2 槽位递增边际估值 (Slot-Indexed Marginal Valuation)

为什么不能把目标星的“安全并发数”写死为硬约束？因为：
1. **热容时间裕度**：一颗刚出阴影区的低温卫星（$38^\circ\text{C}$）即使承接较大并发，其热容 $C_{\text{th}}$ 也足以支撑上百秒全速推理才触及 $75^\circ\text{C}$；
2. **相对速率增益**：即便目标星轻度热节流（$0.55\times$），对于困在本影区（$0.355\times$）或重度热崩溃（$0.25\times$）源节点上的任务而言，迁过去依然是正收益。

因此，目标卫星 $v$ 仅受**物理显存空余槽位** $K_v = B^{\max}_v - b_v$ 约束。当同一周期内有第 $k \in \{1, \dots, K_v\}$ 个任务迁入节点 $v$ 时（记此前已分配给 $v$ 的前 $k-1$ 个迁入任务集合为 $\mathcal{A}_v^{(k-1)}$），其完成时刻直接将节点 $v$ 的仿真队列更新为 $\mathcal{Q}_v \cup \mathcal{A}_v^{(k-1)} \cup \{r\}$ 并代入 **一阶 RC 热-轨积分器** `est_finish_s(w, r, v, hypothetical_nodes)`:

$$
\tau_{\text{mig}}(r, p, k) = t + \underbrace{\Delta t_{\text{KV}}(r, p)}_{\text{KV 多跳传输耗时 (s)}} + \underbrace{\Delta t_{\text{tether}}(r, v)}_{\text{数据锚点回访耗时 (s)}} + \underbrace{\Delta t_{\text{RC}}\big(r \,\big|\, \mathcal{Q}_v \cup \mathcal{A}_v^{(k-1)} \cup \{r\}, T_v(t), \phi_v(t)\big)}_{\text{包含 } k \text{ 个迁入任务真实升温耦合的 RC 积分推理耗时 (s)}}
$$

其中：
- **KV 多跳传输耗时**：$\Delta t_{\text{KV}}(r, p) = \sum_{e \in p} \frac{S_{\text{KV}}(r)}{B_e(t)}$
- **Agentic 数据锚点回访耗时**：$\Delta t_{\text{tether}}(r, v) = \sum_{e \in p(a_r \to v)} \frac{M_r \cdot S_{\text{tile}}}{B_e(t)}$

对应地，迁入节点 $v$ 第 $k$ 个槽位的**纯物理时间节省（秒）**为：

$$
\Delta t_{\text{saved}}(r, p, k) = \tau_{\text{stay}}(r, u) - \tau_{\text{mig}}(r, p, k)
$$

当任务满足截止期 $\tau_{\text{mig}}(r, p, k) \le D_r$ 时，优先赋予截止期挽救等效时间价值 $V_{\text{SLO}} \cdot \mathbb{I}[\tau_{\text{stay}} > D_r \ge \tau_{\text{mig}}]$：

$$
G(r, p, k) = V_{\text{SLO}} \cdot \mathbb{I}\big[\tau_{\text{stay}}(r, u) > D_r \ge \tau_{\text{mig}}(r, p, k)\big] + \Delta t_{\text{saved}}(r, p, k)
$$

> **核心物理性质**：如果向同一颗卫星 $v$ 塞入第 2 或第 3 个任务会导致 $v$ 加速升温进入 $75^\circ\text{C}/85^\circ\text{C}$ 热节流，或者导致 Batch 排队变慢，RC 积分器算出的 $\Delta t_{\text{RC}}$ 会自动变大，从而使第 $k$ 个槽位的边际收益 $G(r, p, k) \le 0$ 并被求解器自然拒绝——**无需任何人工设定的“安全并发数”**。

---

### 2.3 链路占用外部性与统一目标函数

多跳路径 $p$ 上的迁移会占用星间链路带宽。每条边 $e \in p$ 被请求 $r$ 占用的**链路秒数（Link-Occupation Seconds）**为：

$$
\tau_{\text{link}}(r, e) = \frac{S_{\text{KV}}(r)}{B_e(t)} \quad (\text{单位：s})
$$

定义二元决策变量 $x_{r,p,k} \in \{0, 1\}$ 表示是否将任务 $r$ 沿路径 $p$ 迁入终点 $v_p$ 的第 $k$ 个空闲显存槽位。全网协同规划的**最小形式化整数规划 (MILP)** 为：

$$
\max_{x \in \{0,1\}} \quad \sum_{r \in \mathcal{R}_{\text{mig}}} \sum_{p \in \mathcal{P}_r} \sum_{k=1}^{K_{v_p}} \Big( G(r, p, k) - \sum_{e \in p} \tau_{\text{link}}(r, e) \Big) \, x_{r,p,k}
$$

**服从三组纯物理约束（无人工旋钮）**：
1. **单任务路由唯一性**：$\sum_{p \in \mathcal{P}_r} \sum_{k=1}^{K_{v_p}} x_{r,p,k} \le 1, \quad \forall r \in \mathcal{R}_{\text{mig}}$
2. **源节点迁移速率上限**：$\sum_{r \in \mathcal{R}_u} \sum_{p \in \mathcal{P}_r} \sum_{k=1}^{K_{v_p}} x_{r,p,k} \le C^{\text{out}}_u, \quad \forall u \in \mathcal{V}$
3. **目标节点物理显存槽位唯一性**：$\sum_{r} \sum_{p: v_p = v} x_{r,p,k} \le 1, \quad \forall v \in \mathcal{V}, \, \forall k \in \{1, \dots, K_v\}$
4. **星间链路并发流物理上限**：$\sum_{r} \sum_{p \ni e} \sum_{k} x_{r,p,k} \le \Gamma_e, \quad \forall e \in \mathcal{E}(t)$

---

## 3. 实验设计与结论总表 (Experimental Verification Summary)

### 3.1 实验一：以 `Round Robin` 为基准的两阶段收益对比（60 Runs，15 Seeds/Policy）

验证核心故事线：**初始仅用最普通的 `Round Robin (RR)` 分发，对比“不迁移”、“1-to-1 单次最近邻迁移 (`deadline_rc`)”、“TORCH 单次多跳迁移”与“TORCH 周期多跳迁移”**：

| 对比组 (`aligned150`, 15 Seeds) | 机制特征 | 达标率 (SLO Hit Rate) | 达标请求数 (/150) | 相比 `RR` 净挽救请求数 | 相比 `RR + 1-to-1` 提升倍数 | P90 延迟 (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. `RR Only` (无迁移基线)** | 仅初始 Round Robin 均匀分发 | `50.27% ± 0.90%` | 75.4 | 基准 (0) | — | 520.9s |
| **2. `RR + 1-to-1 Once` (`deadline_rc`)** | 局部过热后仅向 1 跳邻居做贪心单次迁移 | `57.20% ± 2.97%` | 85.8 | **+10.4 个** (`+6.93%`) | $1.00\times$ (基准) | 485.3s |
| **3. `RR + TORCH Once` (本文单次)** | 局部过热后用 1-to-Many 多跳流规划做**单次**疏散 | `68.53% ± 3.13%` | 102.8 | **+27.4 个** (`+18.27%`) | **$2.64\times$** | 405.8s |
| **4. `RR + TORCH Full` (本文完整)** | 1-to-Many 多跳流规划 + 随轨道热态演化动态重平衡 | **`70.18% ± 2.41%`** | **105.3** | **+29.9 个** (`+19.91%`) | **$2.88\times$** | **395.9s** |

---

### 3.2 实验二：三大标准突发场景性能总表（105 Runs，7 策略 $\times$ 3 场景）

| 星座与负载场景 | `No Rebalance` | `Greedy Thermal` | `Deadline-RC` (1-Hop SOTA) | `TORCH-NoTether` (Ablation) | **`TORCH` (Full Ours)** | 相比最优基线净提升 | 迁移跳数分布 (`1H / 2H / 3H`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`aligned150`** (太阳同步轨 150 并发突发) | `42.2% ± 1.0%` | `59.4% ± 6.2%` | `68.2% ± 1.7%` | `72.4% ± 1.6%` | **`76.0% ± 2.2%`** | **`+7.82%`** (相对降违约 **24.6%**) | `53.2% / 36.7% / 10.1%` |
| **`regional120`** (区域热点 120 并发突发) | `47.5% ± 0.9%` | `66.1% ± 6.4%` | `76.4% ± 1.6%` | `80.4% ± 1.9%` | **`82.5% ± 2.2%`** | **`+6.11%`** (相对降违约 **25.9%**) | `63.6% / 29.8% / 6.6%` |
| **`aligned60`** (中等突发 60 并发) | `62.2% ± 2.1%` | `72.7% ± 7.5%` | `82.6% ± 3.3%` | `83.0% ± 2.1%` | **`85.4% ± 2.5%`** | **`+2.89%`** (相对降违约 **16.6%**) | `70.1% / 23.8% / 6.1%` |

---

### 3.3 实验三：Agentic 数据锚点“同心圆距离分层”验证

在 `aligned150` 下，对比无锚点意识的 `TORCH-NoTether` 与具备锚点拉取感知的 `TORCH`：
- **Text-Only 推理任务**（$M_r = 0$，无原始图像回访）：被大胆推送至 **2~3 跳远端低温日照星**（远跳占比 **48.9%**，平均跳数 **1.61 跳**）；
- **Agentic 遥感复判任务**（$M_r \ge 2$，需回源拉取高分切片）：被精准吸附在 **1 跳邻近轨道圈**（远跳占比压低至 **22.6%**，平均跳数 **1.25 跳**），使每任务回源拉取等待时间从 `10.02s` 降至 **`2.19s`（削减 78.1%）**，从而在相同算力星座上多救回 **3.60%** 的全局 SLO。

---

### 3.4 实验四：无旋钮物理参数连续敏感度验证 (Knob-Free Physical Emergence)

在**不修改算法任何一行代码或参数**的前提下，扫描外部物理环境参数，验证跳数分布与任务分层的自然涌现：

1. **原始遥感切片体积 $S_{\text{tile}}$ 扫描**（$0 \to 500\text{ MB}$）：
   - 当 $S_{\text{tile}} = 0\text{ MB}$ 时，Agentic 任务与 Text 任务无物理差异，两者平均迁移距离均为 **1.54 跳**（远跳占比 ~44%）；
   - 随着 $S_{\text{tile}}$ 增大到 $150\text{ MB} \to 300\text{ MB} \to 500\text{ MB}$，Agentic 任务远跳占比平滑降至 **`25.0%` $\to$ `11.1%` $\to$ `0.0%`**（自动收敛在 1 跳以内），而同星座中的 Text 任务依然保持 **44%~52%** 的 2~3 跳远端迁移。
2. **卫星辐射散热热阻 $R_{\text{th}}$ 扫描**（$0.22 \to 0.36\text{ }^\circ\text{C/W}$）：
   - 散热越好（$R_{\text{th}} = 0.22$），1 跳邻居热容量充足，仅需 `5.8` 次迁移即可达到 `87.3%` SLO；
   - 散热越差（$R_{\text{th}} = 0.36$），局部星座迅速热饱和，算法自动增加迁移频次至 `15.5` 次并向 2~3 跳远端扩散，相比 `deadline_rc` 的净优势扩大至 **`+8.6%`**。

---

### 3.5 实验五：精确 MILP (`scipy.optimize.milp` HiGHS) 与内置求解器实时性验证

在 `aligned150` 全部动态重平衡决策断面上的逐帧对拍结果：
- **精确 MILP (HiGHS)** 平均求解耗时仅 **`1.71 ms`**（P99 `5.65 ms`）；
- **内置零依赖定界求解器 (`_solve_assignment`)** 平均耗时仅 **`0.26 ms`**（最大 `1.07 ms`），与 HiGHS 精确整数规划最优解吻合率达 **`96.7%`**，完全满足星载计算机毫秒级实时调度要求。

---

## 4. 复现命令速查 (Quick Reproduction Commands)

```bash
# 1. 运行全部单元测试
uv run --frozen pytest -q

# 2. 运行标准突发压力测试 (对比 torch_lp 与 deadline_rc 等基线)
uv run --frozen python scripts/cosim_burst_stress.py \
  --policies none,deadline_rc,torch_notether,torch_lp \
  --seeds 15

# 3. 运行完整论文实验套件 (三大场景 + Agentic 同心圆距离分层 + 窄带 ISL 压力测试)
uv run --frozen python scripts/cosim_agentic_tether.py \
  --out-dir runs/torch_paper_eval \
  --seeds 15
```
