# 0005 协同仿真: orbitsim 接触轨迹驱动的时变拓扑 (`satsim.cosim`)

ADR 0003 的 `orbit_planes` 只在启动时算一次拓扑, 也只用一次定时限功切换当扰动。要回答
"遥感星 → 算力星的接触窗口、地影对在轨推理调度到底有多大影响", 需要拓扑和功率跟着轨道变。

决定:

- 几何由 orbitsim 提供, 支持两种输入路径:
  1. **静态轨迹文件**: `orbitsim.contact_trace` 采样为事件区间 (`orbitsim.contact_trace/v1` JSON), 存放在 `input/traces/` 以保证跨仓库复现与前端回放对齐;
  2. **内存即时合成 (`satsim.cosim.synth`)**: 科研实验直接调用 `synth_multi_sso()` 参数化合成多轨道面异构 SSO 星座 (如 LTAN 10:30/13:30 影轨 + LTAN 06:00 晨昏亮轨) 并返回 `Trace` 对象, 无需先导 JSON。
- 新引擎放在独立子包 `satsim.cosim`, 不动 `satsim.method` 的封闭场景与 golden 测试。复用 `timing.iter_time` / `clock_factor` / `kv_bytes_per_tok`, 网络仍是流体 max-min 公平模型, 增加: 按链路类型的容量; 流经过的链路断开时该流失败 (不重路由)。
- 请求来自遥感星 (RS), 结果必须回到发起的 RS 才算交付。RS 只是端点, 不做中继。地影内算力星功率上限 `P_dark` (默认 80 W → 时钟 0.35×)。
- **正交策略解耦 (`Policy = Placer × Migrator`)**:
  - **任务分配器 (`Placer`)**: `nearest` (就近最短跳数卸载) / `rr` (轮询分配) / `random` (随机分配) / `least_loaded` (JSQ 最短队列, 不看明暗) / `greedy` (看当前队列与当前明暗) / `ephemeris` (看星历预测的完整交付时间);
  - **运行中迁移器 (`Migrator`)**: `none` (不迁移) / `reactive` (进影才搬) / `contact_aware` (按节点对净收益与目标限流迁移)。
- **计算负载诊断 (`satsim.cosim.probe.LoadProbe`)**: 采样分轨道面分流比例、直投暗星比例、暗/亮星实时队列积压比 (`q_dark_over_lit`)、Jain 均衡度与解码耗时膨胀比。
- 回放: `satsim-cosim --viz <dir>` 写出嵌入 orbitsim 场景的回放 JSON, orbitsim `viz/play.html` 用同一闭式解画星, 网络事件与轨道共用时钟。

刻意不做: 包级路由与链路切换时的重捕获时间; prefill 算力; 地面站; 热模型与电池 (地影 = 固定限功); 背景流 (上传和 KV 迁移就是全部流量)。
