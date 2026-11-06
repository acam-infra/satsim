# 0005 协同仿真: orbitsim 接触轨迹驱动的时变拓扑 (`satsim.cosim`)

ADR 0003 的 `orbit_planes` 只在启动时算一次拓扑, 也只用一次定时限功切换当扰动。要回答
"遥感星 → 算力星的接触窗口、地影对在轨推理调度到底有多大影响", 需要拓扑和功率跟着轨道变。

决定:

- 几何留在 orbitsim: `orbitsim.contact_trace` 把闭式传播采样成事件区间 (每条候选链路的
  `[建立, 断开)`, 每颗星的地影 `[进, 出)`), 导出为 `orbitsim.contact_trace/v1` JSON。satsim 不含任何
  轨道计算, 只读轨迹; 轨迹文件复制在 `input/traces/` 以保证可复现。
- 新引擎放在独立子包 `satsim.cosim`, 不动 `satsim.method` 的封闭场景与 golden 测试。复用
  `timing.iter_time` / `clock_factor` / `kv_bytes_per_tok`, 网络仍是流体 max-min 公平模型, 增加:
  按链路类型的容量; 流经过的链路断开时该流失败 (不重路由)。
- 请求来自遥感星 (RS), 结果必须回到发起的 RS 才算交付。RS 只是端点, 不做中继。
  地影内算力星功率上限 `P_dark` (默认 80 W → 时钟 0.35×)。
- 策略分四档: greedy (不看轨道) / reactive (进影才搬) / ephemeris (用星历放置, 不迁移) /
  contact_aware (星历放置 + 按预测交付时间迁移)。轨道是确定的, 策略读轨迹做预测等于
  持有星历, 不算作弊; ephemeris 专门用来剥离"放置就能拿到的收益"。
- 回放: `satsim-cosim --viz <dir>` 写出嵌入 orbitsim 场景的回放 JSON, orbitsim `viz/play.html`
  用同一闭式解画星, 网络事件与轨道共用时钟。

刻意不做: 包级路由与链路切换时的重捕获时间; prefill 算力; 地面站; 热模型与电池 (地影 = 固定限功);
背景流 (上传和 KV 迁移就是全部流量)。
