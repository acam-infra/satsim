# 轨面舱段

这个上下文只给卫星上的推理换位用：换位之后驻留要不要改地方，以及改地方时星间链路付多少。

## Language

**轨面舱段**:
同一轨道面里的一小段计算节点，用来讨论换位后的驻留。
_Avoid_: 全星座, Walker

**功率帽换位**:
某一时刻算力上限从一台节点挪到另一台。
_Avoid_: 热轨道, 食季

**首次分配**:
请求到来时放一次。功率帽换位之后不再动。这一次解决不了换位。
_Avoid_: 静态放置, 免费改放置

**运行时重分配**:
功率帽换位之后，把请求再放到仍有功率的节点。
_Avoid_: 免费迁移, 随意迁移, 计费迁移

**链路代价**:
多条流抢同一条有带宽上限的链路时，运行时重分配分到的速率不够、搬不完。
_Avoid_: byte·hop, 搬走的字节数

**已准入旋钮**:
场景声明里允许改的具名参数。
_Avoid_: 扩展, 插件

**场景镜像**:
用已经生效的旋钮讲的轨道叙事，不另给一套物理。
_Avoid_: 第二套仿真

**合成过境**:
压进一段墙钟的到达簇，用来代替真实观测轨迹。
_Avoid_: STAC 复现, GPU 实测

**全局调度器 (scheduler)**:
一个策略同时覆盖初次分配 `place` 与再分配 `maybe_rebalance`。YAML 写 `scheduler` + `link_cost`（+ `hysteresis` / `knobs`）。
_Avoid_: 并列 placement/rebalance 两套钩子, route/migrate/cfg

**link_cost: off|full|net**:
迁移是否上链路及如何计代价。与 scheduler 有准入搭配（如 place_once 只配 off）。
_Avoid_: 把 off 写成布尔（YAML 须加引号）

**线下到达生成**:
`trace.kind: synthetic_pass` 或 `poisson`；仿真开始前生成完整请求序列再回放。
_Avoid_: 在引擎里边跑边采样当默认

**Request**:
中立请求类型（`satsim.domain.Request`）；`pin`/`pout` 为兼容别名。

