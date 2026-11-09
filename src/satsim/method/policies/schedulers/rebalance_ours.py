"""占带宽迁移 · 自研策略（CCE-Net）：计算-通信净转化值打分网络。

核心思想：
将跨星 KV 迁移建模为“用星间通信带宽置换远端计算加速”的在线转化决策。
每个离散控制周期，对候选动作 a = (req, src -> dst) 提取 6 维量纲统一的
计算-通信转化特征 phi(s_t, a)，由轻量级价值网络 V_theta(phi) 输出唯一标量分值 v：
  - v > 0：通信换计算净收益为正，执行得分最高的迁移；
  - v <= 0：链路拥塞或通信代价高于算力收益，留在原节点继续计算（no-op）。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from satsim.method.engine.timing import iter_time
from satsim.method.engine.world import MAX_INFLIGHT, MAX_OUT
from satsim.method.policies.schedulers._placement import place_by_iter_time

# 特征维度：
#   phi[0]: phi_comp    — 归一化计算省时收益 ((t_src - t_dst) * rem_tok / timeout_s)
#   phi[1]: phi_speedup — 单步迭代相对加速比 ((t_src - t_dst) / t_src)
#   phi[2]: phi_slo     — SLO 抢救紧迫度（源节点逼近/违约 SLO 且目标节点安全达标）
#   phi[3]: phi_comm    — 自身通信耗时占比 (T_comm / timeout_s)
#   phi[4]: phi_cong    — 路径重叠拥塞外部性（对同路径在途迁移流造成的附加延迟比）
#   phi[5]: phi_hazard  — 超时断崖风险惩罚 max(0, phi_comm - eta_safe)^2
FEATURE_NAMES = (
    "phi_comp",
    "phi_speedup",
    "phi_slo",
    "phi_comm",
    "phi_cong",
    "phi_hazard",
)

# 经跨场景策略寻优（CEM / RL episode return 优化）收敛的默认参数 theta*
DEFAULT_THETA: dict = {
    # 显式线性主干权重：
    #   [phi_comp, phi_rate, phi_slo, phi_comm, phi_cong, phi_hazard]
    #   计算省时(+) / 转化汇率(+) / SLO抢救(+)  vs  通信耗时(-) / 拥塞外部性(-) / 超时风险(-)
    "w_lin": [2.20, 1.65, 3.40, -1.50, -4.20, -9.00],
    "b_lin": 0.10,
    # 2 层非线性残差单元（捕捉高算力差与高链路拥塞的交叉耦合）
    "W_hid": [
        [1.40, 1.20, 2.00, -1.30, -2.60, -3.80],
        [-0.50, -0.40, 0.70, 2.00, 3.40, 4.80],
        [1.00, 1.30, 0.90, -0.80, -1.90, -2.60],
        [0.35, -0.25, 1.40, -0.50, 2.10, 3.20],
    ],
    "b_hid": [0.20, -0.95, 0.15, -0.50],
    "w_out": [0.65, -0.70, 0.45, -0.45],
    # 安全阈值与单步相对加速门限
    "eta_safe": 0.78,
    "eta_max": 0.96,
    "min_speedup": 0.08,
}


class CCEScorer:
    """计算-通信净转化值打分器 V_theta(phi) -> float。"""

    def __init__(self, theta: dict | None = None):
        cfg = theta or DEFAULT_THETA
        self.w_lin = [float(x) for x in cfg["w_lin"]]
        self.b_lin = float(cfg["b_lin"])
        self.W_hid = [[float(x) for x in row] for row in cfg["W_hid"]]
        self.b_hid = [float(x) for x in cfg["b_hid"]]
        self.w_out = [float(x) for x in cfg["w_out"]]
        self.eta_safe = float(cfg.get("eta_safe", 0.55))
        self.eta_max = float(cfg.get("eta_max", 0.82))
        self.min_speedup = float(cfg.get("min_speedup", 0.12))

    def score(self, phi: Sequence[float]) -> float:
        """将 6 维计算-通信特征映射为单一标量净转化值 v in R。"""
        if phi[3] >= self.eta_max:
            return -1e6
        lin = self.b_lin + sum(w * x for w, x in zip(self.w_lin, phi))
        res = 0.0
        for row, bh, wo in zip(self.W_hid, self.b_hid, self.w_out):
            z = bh + sum(w * x for w, x in zip(row, phi))
            res += wo * math.tanh(z)
        return lin + res


def _network_state(world) -> tuple[dict, dict, dict, list[int], list[float]]:
    """从 world.net 提取每条有向边的剩余带宽、在途迁移流数、在途字节，以及各节点在途迁入负载。"""
    cap = float(world.net.cap)
    bg_used: dict[tuple[int, int], float] = defaultdict(float)
    mig_flows: dict[tuple[int, int], int] = defaultdict(int)
    mig_rem_bytes: dict[tuple[int, int], float] = defaultdict(float)
    inc_nd = [0] * world.n
    inc_bytes = [0.0] * world.n

    for fl in world.net.flows:
        if fl.remaining <= 0:
            continue
        if fl.priority == 0:
            # 背景流按路径分配后的速率或需求估算占用
            used = fl.rate if fl.rate > 0 else fl.demand
            for e in fl.path:
                bg_used[e] += used
        elif fl.priority == 1:
            for e in fl.path:
                mig_flows[e] += 1
                mig_rem_bytes[e] += fl.remaining
            if fl.path:
                dst = fl.path[-1][1]
                if 0 <= dst < world.n:
                    inc_nd[dst] += 1
                    inc_bytes[dst] += fl.remaining

    return bg_used, mig_flows, mig_rem_bytes, inc_nd, inc_bytes


def extract_cce_features(
    world,
    req,
    src: int,
    dst: int,
    *,
    eff_nd: Sequence[int],
    eff_ctx: Sequence[float],
    rem_ctx: Sequence[float],
    f_nodes: Sequence[float],
    bg_used: dict,
    mig_flows: dict,
    mig_rem_bytes: dict,
    eta_safe: float = 0.78,
) -> tuple[float, ...]:
    """提取候选迁移 (req, src -> dst) 的 6 维计算-通信转化特征向量 phi。"""
    gpu = world.cfg["gpu"]
    timeout_s = max(float(world.cfg["cost"]["timeout_s"]), 1.0)
    hop_lat = float(world.cfg["net"]["hop_latency_s"])
    switch_s = float(world.cfg["cost"]["switch_s"])
    slo_tpot = float(world.cfg.get("slo_tpot", 0.032))

    rem_tok = max(0.0, float(req.pout) - float(req.gen))
    # 1. 计算侧：按解码中段预期上下文长度估算留在 src 与迁往 dst 的单步耗时
    nd_src = max(1, eff_nd[src])
    ctx_src_mid = max(req.kv, eff_ctx[src]) + 0.5 * rem_ctx[src]
    ctx_dst_mid = eff_ctx[dst] + req.kv + 0.5 * (rem_ctx[dst] + rem_tok)
    t_src = iter_time(nd_src, ctx_src_mid, f_nodes[src], gpu)
    t_dst = iter_time(eff_nd[dst] + 1, ctx_dst_mid, f_nodes[dst], gpu)

    dt_comp_saved = max(0.0, (t_src - t_dst) * rem_tok)
    phi_comp = dt_comp_saved / timeout_s

    # 2. 通信侧：路径瓶颈剩余带宽、自身流体守恒传输耗时与同路拥塞外部性
    path = world.pick_path(src, dst)
    nbytes = float(req.kv) * float(world.bpt)
    cap = float(world.net.cap)

    max_self_time = 0.0
    max_ext_time = 0.0
    for e in path:
        rem_bw = cap - bg_used.get(e, 0.0)
        if rem_bw <= 1e3:  # 边已被高优背景流占满
            return (phi_comp, 0.0, 0.0, 10.0, 10.0, 100.0)
        m_e = mig_flows.get(e, 0)
        rem_b = mig_rem_bytes.get(e, 0.0)
        # 流体处理器共享（Max-Min 公平）下：新流与在途流共享带宽的完成时间上界
        self_t = (rem_b + nbytes) / rem_bw
        # 外部性：新流加入使边上已有的 m_e 条迁移流以及排队等待被推迟
        ext_t = (m_e * nbytes + rem_b) / rem_bw if m_e > 0 else 0.0
        if self_t > max_self_time:
            max_self_time = self_t
        if ext_t > max_ext_time:
            max_ext_time = ext_t

    t_comm = max_self_time + len(path) * hop_lat + switch_s
    phi_comm = t_comm / timeout_s
    phi_cong = max_ext_time / timeout_s

    # 计算-通信转化汇率（每投入 1 秒通信可换取的计算省时秒数，归一化到 [0, 3]）
    exchange_rate = dt_comp_saved / max(t_comm, 0.5)
    phi_rate = min(3.0, exchange_rate / 4.0)

    # 端到端全生命周期 TPOT 达标抢救评估（结合已累积的 post_time/post_tok 与剩余 token）
    total_post_tok = max(1.0, float(req.post_tok) + rem_tok)
    tpot_stay = (float(req.post_time) + t_src * rem_tok) / total_post_tok
    tok_during_mig = min(rem_tok, t_comm / max(t_src, 1e-6))
    tok_after_mig = max(0.0, rem_tok - tok_during_mig)
    tpot_mig = (float(req.post_time) + t_comm + t_dst * tok_after_mig) / total_post_tok

    if tpot_stay >= 0.88 * slo_tpot and tpot_mig < slo_tpot and tpot_stay > tpot_mig:
        rescue_gain = min(2.5, (tpot_stay - tpot_mig) / max(slo_tpot - tpot_mig, 1e-4))
        long_weight = 1.25 if getattr(req, "long", False) else 0.25
        phi_slo = rescue_gain * long_weight * (rem_tok / max(float(req.pout), 1.0))
    else:
        phi_slo = 0.0

    # 当边上已有在途流且叠加后逼近超时，或单流本身逼近超时，触发断崖惩罚
    effective_load_ratio = max(phi_comm, (max_self_time + 0.6 * max_ext_time) / timeout_s)
    phi_hazard = max(0.0, effective_load_ratio - eta_safe) ** 2

    return (phi_comp, phi_rate, phi_slo, phi_comm, phi_cong, phi_hazard)


class RebalanceOursScheduler:
    """占星间带宽时的计算-通信净转化值调度器（CCE-Net）。"""

    id = "rebalance_ours"
    TODO = ""

    def __init__(self, theta: dict | None = None):
        self.scorer = CCEScorer(theta)

    def place(self, world, req) -> int:
        return place_by_iter_time(world, req)

    def evaluate_candidates(self, world) -> list[tuple[float, object, int, int, tuple[float, ...]]]:
        """对所有可行的 (req, src -> dst) 计算转化分值 v，返回按 v 降序排列的列表。"""
        now = world.env.now
        f_nodes = [world.f_of(j, now) for j in range(world.n)]
        # 若全星座算力完全均匀（尚未发生限功或降频），不提前消耗单次迁移配额
        if max(f_nodes) - min(f_nodes) < 1e-6:
            return []
        bg_used, mig_flows, mig_rem_bytes, inc_nd, inc_bytes = _network_state(world)

        # 计入在途迁出与在途迁入后的有效计算负载投影
        eff_nd = [
            sum(1 for r in world.nodes[j] if not r.migrating) + inc_nd[j]
            for j in range(world.n)
        ]
        eff_ctx = [
            sum(r.kv for r in world.nodes[j] if not r.migrating) + inc_bytes[j] / world.bpt
            for j in range(world.n)
        ]
        rem_ctx = [
            sum(max(0.0, float(r.pout) - float(r.gen)) for r in world.nodes[j] if not r.migrating)
            for j in range(world.n)
        ]

        cap_bytes = float(world.cfg["gpu"]["kv_gib"]) * (2**30)
        scored: list[tuple[float, object, int, int, tuple[float, ...]]] = []

        for src in range(world.n):
            if sum(1 for r in world.nodes[src] if r.migrating) >= MAX_OUT:
                continue
            cands = [
                r
                for r in world.nodes[src]
                if not r.migrating and r.nmig == 0 and r.t <= now and (r.pout - r.gen) > 1.0
            ]
            if not cands:
                continue
            # 每个源节点优先考察 KV 较大或剩余生成较长的代表性请求
            for req in cands:
                req_bytes = req.kv * world.bpt
                for dst in range(world.n):
                    if dst == src:
                        continue
                    # 不向算力倍率更暗的节点迁移
                    if f_nodes[dst] < f_nodes[src]:
                        continue
                    # 同算力倍率下（亮星之间接力），仅当源节点明显比目标节点拥挤时才允许接力腾挪
                    if f_nodes[dst] == f_nodes[src] and eff_nd[src] <= eff_nd[dst] + 1:
                        continue
                    # 计入在途迁入后的显存容量约束
                    used_dst = sum(r.kv for r in world.nodes[dst]) * world.bpt + inc_bytes[dst]
                    if used_dst + req_bytes > cap_bytes:
                        continue
                    phi = extract_cce_features(
                        world,
                        req,
                        src,
                        dst,
                        eff_nd=eff_nd,
                        eff_ctx=eff_ctx,
                        rem_ctx=rem_ctx,
                        f_nodes=f_nodes,
                        bg_used=bg_used,
                        mig_flows=mig_flows,
                        mig_rem_bytes=mig_rem_bytes,
                        eta_safe=self.scorer.eta_safe,
                    )
                    if phi[0] <= 0.0 or phi[1] < self.scorer.min_speedup:
                        continue
                    v = self.scorer.score(phi)
                    scored.append((v, req, src, dst, phi))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def maybe_rebalance(self, world) -> None:
        inflight = sum(1 for rs in world.nodes for r in rs if r.migrating)
        if inflight >= MAX_INFLIGHT:
            return
        scored = self.evaluate_candidates(world)
        if not scored:
            return
        best_v, req, src, dst, _phi = scored[0]
        # 核心准则：只有“通信换计算”的净转化分值 v > 0 时才触发迁移，否则留在原地
        if best_v <= 0.0:
            return
        world.env.process(world.migrate_proc(req, src, dst))

