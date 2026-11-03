# 多流规划与系统吞吐：传统方向调研简报

> 面向 satsim（LEO 算力星座 + LLM KV 迁移争用）的对照笔记。问题表述尽量一般化；星座相关文献单独标薄。  
> 整理日期：2026-11-03（Asia/Shanghai）

---

## 1. 问题再表述（多流规划 ↔ 系统吞吐）

传统文献里，“多流同时共享瓶颈链路”几乎从不被当成“只求链路利用率最大”。核心张力是：

1. **单流指标好 ≠ 应用/作业指标好**  
   公平分带宽、把每条流 FCT 压短，不一定让 *一组相关流*（shuffle / coflow / 一次迁移任务）更快结束；作业往往卡在 **最慢那条流**。

2. **规划改变争用图**  
   路由、放置、迁移目的地一变，哪些流叠在同一条链路上就变；吞吐（goodput / 完成率 / 截止达成率）是 **争用结构 + 分配策略** 的联合结果，不是固定拓扑上的静态带宽表。

3. **决策变量常见三层**  
   - *路径/路由*：走哪条、是否分流（MCF / TE / ECMP）  
   - *速率/份额*：每流或每组分多少（max-min / PF / WFQ / NUM）  
   - *调度序/抢占*：谁先传完（SJF / EDF / coflow SRTF 启发式）

对本仓直觉的对应：多个 KV 迁移 / 背景流共享 ISL（流体带宽 + 优先级）→ 放置与迁移改路径重叠 ρ → 影响请求完成、超时 abort、系统有效吞吐。传统方向回答的是：**先建模争用如何吃带宽，再用公平或完成时间目标显式优化分配/调度**，而不是默认“带宽够用”。

一句话答「传统怎么解多流对吞吐的影响」：

> **把争用写成容量约束下的速率分配（或组完成时间调度），用公平准则或 makespan/CCT/截止目标决定份额与顺序；更高层再用 TE/MCF 选路降低重叠；应用层用 transfer/coflow/任务感知把“流集合”当成调度单元。吞吐影响用 CCT、FCT、截止满足率、作业完成时间量化，而不是只报链路利用率。**

---

## 2. 传统路线图谱

| 路线 | 问题框定 | 决策变量 | 典型目标 | 求解风格 | 如何量化“争用→吞吐” |
|------|----------|----------|----------|----------|---------------------|
| A. 带宽公平 / NUM | 多流共享链路容量 | 速率 \(x_r\) | max-min / \(\sum w\log x\) / 效用和 | 集中水填；分布式 AIMD/影子价格 | 均衡速率 → 完成时间 \(\approx\) size\(/x\)；总吞吐 vs 公平折中 |
| B. 数据中心流调度 | 短流/截止流抢带宽 | 优先级、暂停、是否接纳新流 | 最小 FCT；最大 deadline hit | 分布（PDQ）或近似理想优先级（pFabric） | FCT 分布、截止达成率；明确反对“盲目公平” |
| C. Transfer / Coflow | 作业卡在一组流的最慢者 | 组间调度序 + 组内带宽 | 最小 CCT / 截止 coflow | 中心启发式（Varys）；非全知（Aalo）；任务 FIFO（Baraat） | CCT、作业 JCT、截止 coflow 比例 |
| D. TE / MCF | 多商品需求压在拓扑上 | 路径与分流比 | 最小拥塞 / 最大可承载需求 | 离线 LP；半/全 oblivious；MPLS LSP | 最大链路利用率、可路由需求矩阵 |
| E. 作业–网络共调度 | 计算阶段依赖网络阶段 | 传输开始时刻、并行度、带宽份额 | 迁移/shuffle 完成时间；作业 makespan | 中心调度器 + 预测模型 | 迁移时长预测、进度、端到端作业时间 |
| F. LEO / ISL TE | 时变拓扑上的多流 | 路由、ISL 匹配、速率 | 负载均衡、吞吐 | SDN 控制器 / 对偶分解等 | 文献有但偏路由与容量；**与“可选迁移 + 算力日照”耦合仍薄** |

---

## 3. 代表工作 + 做法要点

### 3.1 带宽分配与拥塞（公平如何塑造吞吐）

**Bertsekas & Gallager,《Data Networks》（教材经典）— Max-Min Fairness**  
问题：在链路容量约束下为会话分配速率。决策：速率向量。目标：词典序最大化最小速率（水填：均匀抬高直至链路饱和，钉死瓶颈上的流，对其余继续）。风格：集中式构造；分布式可用迭代近似。量化：瓶颈公平份额直接决定每流完成时间；总吞吐往往低于“饿死长路径、喂饱短路径”的策略——**公平与聚合吞吐的经典折中**。

**Kelly,《Charging and rate control for elastic traffic》, ETT 1997；Kelly / Maulloo / Tan, JORS 1998 — Proportional Fairness & NUM**  
问题：弹性流如何共享网络。决策：用户愿付 \(w_r\) 与网络分配 \(x_r\)。目标：\(\max \sum_r w_r \log x_r\)（比例公平；max-min 是极限特例）。风格：**网络效用最大化（NUM）** 分解为用户问题 + 网络问题；原/对偶动态（AIMD 类或影子价格）分布式收敛，可推广到选路。量化：均衡速率由路径影子价格和决定；吞吐是效用最优下的副产品，不是单独 \(\max \sum x\)。

**Massoulié & Roberts,《Bandwidth sharing: objectives and algorithms》, ToN 2002**  
把 max-min、比例公平、以及“最小化潜在时延”（\(\sum 1/x\)）放在同一框架比较，并讨论窗口/调度如何逼近这些目标。要点：**目标函数一换，争用下的完成时间排序就换**——选公平准则本身就是在选吞吐形态。

**Demers / Keshav / Shenker,《Analysis and Simulation of a Fair Queueing Algorithm》, SIGCOMM 1989（WFQ 谱系）**  
问题：包级如何近似处理器共享、隔离恶意流。决策：按 finish-time 的加权公平排队。目标：保护与加权份额。风格：交换机/路由器本地。量化：权重 → 长期带宽份额 → 与流体模型对接；是后来“加权/优先级流体带宽”仿真的操作基础。

### 3.2 数据中心：单流完成时间与截止

**Hong / Caesar / Godfrey,《Finishing Flows Quickly with Preemptive Scheduling》(PDQ), SIGCOMM 2012**  
问题：公平共享对 FCT / deadline 次优。决策：分布地逼近 EDF（有截止）与 SJF（无截止），可暂停低优先流。目标：更多流赶上截止、更小平均 FCT。风格：交换机协作、分布式；有多路径扩展。量化：相对 TCP/RCP/D3 的 FCT 与截止达标；**直接论证“多流争用时公平 ≠ 系统有效吞吐”**。

**Alizadeh et al.,《pFabric: Minimal Near-Optimal Datacenter Transport》, SIGCOMM 2013**  
问题：理想优先级调度下的近优 FCT。决策：主机打剩余大小/截止优先级，交换机小缓冲 + 优先级调度/丢弃。风格：干净理论下限对照，部署成本高。量化：短流 FCT（含高分位）逼近理想；说明 **调度序对争用下完成时间的一阶影响**，速率控制可退居二线。

**Popa et al.,《FairCloud: Sharing the Network in Cloud Computing》, SIGCOMM 2012**  
问题：多租户云网络在支付比例、最小带宽保障、网络比例性之间不可兼得。决策：PS-L / PS-P / PS-N 等共享策略。目标：在 tradeoff 面上选点。量化：小作业 shuffle 时间等应用级指标；**带宽分配策略会透过争用结构打到作业吞吐**。

### 3.3 Transfer / Coflow：组完成时间（最接近“多迁移争用”）

**Chowdhury et al.,《Managing Data Transfers in Computer Clusters with Orchestra》, SIGCOMM 2011**  
问题：MapReduce 等作业卡在 **transfer**（阶段间全体流），而非单流。决策：Inter-Transfer Controller 做 FIFO/优先级/加权；Transfer Controller 优化 shuffle（WSS）与 broadcast（Cornet）。目标：缩短 transfer / 高优先级 transfer。风格：应用层中心协调。量化：broadcast 完成时间、高优 transfer 加速；点明 **flow-level 管理对作业吞吐不够**。

**Chowdhury / Zhong / Stoica,《Efficient Coflow Scheduling with Varys》, SIGCOMM 2014**  
问题：inter-coflow 调度。决策：将问题连到 *concurrent open shop with coupled resources*；启发式最小化 CCT 或服务截止。目标：CCT / 更多 coflow 赶上 deadline；保持高利用率、防饿死。风格：**中心、需先验 coflow 信息（clairvoyant）**。量化：相对 per-flow fair / prioritization 的 CCT 与作业时间；EC2 + Facebook trace。

**Chowdhury / Stoica,《Efficient Coflow Scheduling Without Prior Knowledge》(Aalo), SIGCOMM 2015**  
问题：无大小/端点先验 + 集群动态（pipeline、失败、多波）。决策：D-CLAS——按已发送字节分粗优先级队列，队列间优先、队列内 FIFO。风格：非全知、仍需松散协调。量化：CCT 接近有先验方案，动态下更稳。

**Dogar et al.,《Decentralized Task-Aware Scheduling for Data Center Networks》(Baraat), SIGCOMM 2014**  
问题：任务（多流）感知但要去中心。决策：FIFO-LM——默认按任务到达序，重任务动态增加复用以减轻 HOL。风格：交换机侧 task-id / 优先级类，少显式协调。量化：相对 pFabric 等的尾部完成时间。与 coflow 中心方案形成对照：**完成时间目标可走分布式近似**。

### 3.4 流量工程与多商品流

**多商品流（MCF）LP（网络优化标准模型）**  
问题：给定需求矩阵，为每个 OD 选路/分流，满足链路容量。决策：路径流变量。目标：常见为最小化最大链路利用率（拥塞）、或可行性。风格：离线/周期重优化；输出可下发为 MPLS LSP 或权重。量化：**可承载的需求规模、拥塞比**——规划层直接决定多流是否“一开始就挤在一起”。

**Awduche et al., RFC 2702（1999）— Requirements for Traffic Engineering Over MPLS**  
工程侧把 TE 需求写成显式 LSP、流量干线、约束路由、并行干线负载分担等；MCF/启发式的结果常通过 MPLS TE 落地。与 ECMP 对照：ECMP 局部哈希均衡，**不做面向全局目标的规划**。

**Applegate & Cohen 等；Räcke 谱系 — Oblivious Routing**  
问题：需求未知或剧变时仍控制最坏拥塞。决策：对每个 OD 预先固定（可分流）路由模板。目标：最小化相对最优 MCF 的最坏拥塞比。风格：离线多项式可解 LP / 理论 polylog 保证。量化：鲁棒拥塞；适合“不能每时刻全局重算 TE”的场景——与星座上 **少数候选、难做全网实时 TE** 的约束同构。

### 3.5 作业 / 迁移与网络共调度

**Kherbache 等（BtrPlace / mVM；如 TCC 2017《Scheduling Live Migration of Virtual Machines》一线）**  
问题：多台 VM 热迁移共享网络时，顺序与并行度决定总迁移时间与能耗。决策：开始时刻、并行集合、带宽；用脏页率与拓扑估计迁移时长。目标：迁移日程完成时间、约束（同步、截止、功耗）。风格：中心调度 + 预测。量化：时长预测误差、日程 makespan。**“迁移是可选且有开销的操作”在 VM 文献里出现，但通常迁移是必须执行的重配置，而非与“留在原节点继续算”的在线权衡。**

**Llumnix（Sun et al., OSDI 2024）— LLM 服务内 KV 热迁移**  
问题：推理实例间动态重调度 in-flight 请求。决策：迁谁、迁去哪；利用 KV append-only 做并行拷贝以降停机。目标：负载均衡与尾延迟。风格：集群内调度。量化：迁移停机、实例水位。对照：通常假设 **DC 内高速网络、迁移收益明确**，很少把多迁移路径争用写成一等公民的流体瓶颈模型，也无日照/功率耦合。

### 3.6 LEO / ISL 多流（薄，单列）

可见工作多集中在：**时变拓扑路由、SDN-LEO TE、激光 ISL 匹配 + 流路由联合优化**（例如 TNSM 2022 一线的 Software-Defined LEO TE；近年 arXiv/期刊上的 ISL matching + Lagrangian 对偶等）。它们证明星座侧也在做多流容量与路由，但 **与“算力随光照变化 + 可选 KV 迁移 + 请求超时 abort”的闭环仍少**——借用时宜当 TE/容量层，不宜直接当本问题的现成解。

---

## 4. 常用目标函数与度量

| 层级 | 目标 / 度量 | 含义 |
|------|-------------|------|
| 链路/网络 | 利用率、最大拥塞比、可路由需求 | 规划层是否把流拆开 |
| 速率公平 | max-min 份额、\(\sum \log x\)、加权份额 | 争用下稳态带宽 |
| 单流 | FCT、deadline miss 率 | 公平往往伤短流/截止流 |
| 组/作业 | CCT、transfer time、JCT | **最慢流**决定；与 KV 迁移“整次搬完才生效”同构 |
| 系统有效 | goodput、按时完成数、超时 abort | 与 satsim 的完成/abort 对齐 |
| 公平–效率 | Jain 指数 vs \(\sum x\) 或 vs 作业完成 | 显式报告折中 |

建模争用时的常用技巧（可对照本仓 ρ / path-overlap）：

- **流级流体**：链路 \(\sum_r x_r \le C_e\)，完成时间 \(T_r = s_r / x_r\)（或时变积分）。  
- **组完成**：CCT \(= \max_{f \in \text{coflow}} T_f\) 或 bottleneck 估计。  
- **优先级流体**：高优先级吃满，残余给低优先级（与 satsim 背景流/迁移优先级同类）。  
- **重叠矩阵**：谁与谁共享边 → 等价于 coflow 文献里的耦合资源 / 本仓 path-overlap。

---

## 5. 与本仓设定的差距 / 可借用点

| 本仓特点 | 传统文献覆盖 | 可借用 | 缺口 |
|----------|--------------|--------|------|
| (a) 功率/日照 → 算力速率时变，且与网络同时变 | 几乎只做网络或只做计算 | NUM/TE 的“时变容量”视角；共调度里的预测 | **算力钟频与链路争用联合目标**少见 |
| (b) 迁移是可选且占带宽的动作（可留在原节点） | 迁移日程多假定“必须迁” | 把迁移时延当开销写入效用；Llumnix 式收益估计 | **在线决策：迁 vs 留** 与多流争用耦合弱 |
| (c) 超时 abort | PDQ/Varys 的 deadline 线 | EDF、截止 coflow admission、丢弃注定错过的流 | abort 后对 KV/队列状态与吞吐定义需自洽 |
| (d) 少数局部候选，非全局实时 TE | Oblivious / 分布式 PDQ·Baraat；局部启发式 | **不要上全网 MCF 每个决策**；用 overlap 分数 + 局部 argmin | 星座ISL时变 + 少候选的专门算法仍空 |
| 流体带宽 + 优先级 | WFQ / 优先级调度 / 残余带宽 | 可直接作仿真带宽层 | 与放置分数（iter_time）要同一量纲闭环 |
| 指标：goodput、完成、abort | CCT、JCT、deadline hit | 汇报时对齐“组完成/截止”而非只报链路占用 | 需定义迁移进行中算不算有效服务 |

**对研究问题的收敛建议**：传统答案不是“再做一个公平带宽”，而是——

1. **把一次迁移（或一次请求相关的迁移集合）当成 coflow/transfer 单元**，目标用 CCT 或“超时前完成概率”，而不是单流 FCT。  
2. **放置/选目的地 = 改多商品路径重叠**（小规模可用 overlap 启发式，对照 MCF/oblivious 的思想，不必每步解 LP）。  
3. **带宽层用显式准则**（优先级残余 / 加权公平 / 截止优先）并报告对完成率·abort 的影响，避免隐式“均分”。  
4. **额外状态机**（日照算力、stay vs migrate、abort）是相对 Varys/Kelly/Llumnix 的真正缺口，适合作为贡献切口。

---

## 6. 建议阅读短清单（8–12）

1. **Kelly, Maulloo, Tan.** *Rate control for communication networks: shadow prices, proportional fairness and stability.* JORS, 1998. — NUM/比例公平与分布式收敛的根。  
2. **Kelly.** *Charging and rate control for elastic traffic.* ETT, 1997. — 愿付与比例公平分解；max-min 极限。  
3. **Massoulié & Roberts.** *Bandwidth sharing: objectives and algorithms.* IEEE/ACM ToN, 2002. — 多种公平目标如何改变完成时间形态。  
4. **Chowdhury, Zaharia, Ma, Jordan, Stoica.** *Managing Data Transfers in Computer Clusters with Orchestra.* SIGCOMM, 2011. — transfer 级调度；作业吞吐依赖整组流。  
5. **Chowdhury, Zhong, Stoica.** *Efficient Coflow Scheduling with Varys.* SIGCOMM, 2014. — inter-coflow、CCT/截止、耦合资源调度。  
6. **Chowdhury & Stoica.** *Efficient Coflow Scheduling Without Prior Knowledge (Aalo).* SIGCOMM, 2015. — 非全知、在线、防饿死。  
7. **Hong, Caesar, Godfrey.** *Finishing Flows Quickly with Preemptive Scheduling (PDQ).* SIGCOMM, 2012. — 争用下公平为何伤吞吐/截止。  
8. **Alizadeh et al.** *pFabric: Minimal Near-Optimal Datacenter Transport.* SIGCOMM, 2013. — 优先级调度与 FCT 下限直觉。  
9. **Popa et al.** *FairCloud: Sharing the Network in Cloud Computing.* SIGCOMM, 2012. — 多租户带宽政策 → 作业级后果。  
10. **Dogar, Karagiannis, Ballani, Rowstron.** *Decentralized Task-Aware Scheduling for Data Center Networks (Baraat).* SIGCOMM, 2014. — 任务感知的分布式近似。  
11. **Applegate & Cohen.** *Making Routing Robust to Changing Traffic Demands: Algorithms and Evaluation.* (oblivious routing LP 一线；可与 Räcke 理论对照). — 需求不确定时的规划，对照“非实时全网 TE”。  
12. **Kherbache et al.** *Scheduling Live Migration of Virtual Machines*（BtrPlace/mVM 一线，如 IEEE TCC 2017）. — 多迁移共享网络的日程与时长预测；可选读 **Llumnix (OSDI 2024)** 作 LLM KV 迁移对照。

补充（教材/标准，按需）：Bertsekas & Gallager *Data Networks*（max-min 水填）；RFC 2702（MPLS TE 需求）；Demers/Keshav/Shenker SIGCOMM’89（公平排队）。

---

## 7. 收束

传统方向对「多流规划如何影响系统吞吐」的解法是分层的：**TE/MCF 决定和谁争，公平/NUM/优先级决定争的时候分多少，coflow/任务调度决定谁先结束**；应用吞吐用 **组完成时间与截止** 衡量。对本仓，最可迁移的是 coflow/transfer 目标与 overlap 感知选路，加上截止/abort；真正要新做的是 **日照时变算力 × 可选迁移 × 局部候选** 的闭环，而不是再发明一种链路公平。
