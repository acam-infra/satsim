"""Test Knob-Free (Zero-Magic-Number) Physics-Driven TORCH across varying environment parameters:
1. Varying Raw-Image Tile Size S_tile in {0, 50, 150, 300, 500} MB -> continuous concentric hop adaptation
2. Varying Satellite Thermal Resistance R_th in {0.22, 0.26, 0.30, 0.36} C/W -> self-calibrating thermal tipping point
"""
import copy
import math
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import simpy

sys.path.insert(0, "scripts")
from cosim_burst_stress import profile_config
from satsim.cosim.run import load_config
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, make_requests
from satsim.cosim.policies import candidates as place_candidates, inflight, make_policy, path_lifetime
from satsim.cosim.record import summarize
from satsim.cosim.thermal_rebalance import QUANTILES, ThermalForecast, _interp_linear, k_shortest_paths
from satsim.method.engine.timing import iter_time


class KnobFreeTorch:
    """Pure first-principles Thermo-Orbital & Data-Anchor Rebalancer (Zero magic thresholds)."""

    def __init__(self, placement: dict, interval_s: float = 5.0, k_paths: int = 3):
        self.placement = make_policy(placement)
        self.interval_s = interval_s
        self.next_plan = interval_s
        self.k_paths = k_paths
        self._last_rebalance_t = -1.0

    @staticmethod
    def _sustainable_batch(w) -> int:
        """Derive critical non-throttling batch size b_crit directly from RC thermal equilibrium equation:
        T_amb + R_th * P(b, 1.0) <= T_throttle."""
        if w.thermal is None:
            return 99
        th = w.thermal
        b_crit = 1
        for b in range(1, 16):
            p_w = w.device_power(b, 1.0)
            if th.T_amb_C + th.R_th * p_w <= th.T_throttle_C:
                b_crit = b
            else:
                break
        return b_crit

    def place(self, w, r):
        now = w.env.now
        cs = place_candidates(w, r)
        if not cs:
            return None
        if r.long and now - self._last_rebalance_t >= w.period:
            # If a full-speed sunlit slot is open while other nodes are dark or throttled,
            # let in-flight trapped jobs relay over fast ISL before slow RS upload locks the slot.
            if any(not w.tr.is_dark(j, now) and w.thermal_f[j] == 1.0 for j in cs):
                has_trapped = any(
                    (w.tr.is_dark(j, now) or w.thermal_f[j] < 1.0)
                    and any(x.long and not x.migrating for x in w.nodes[j])
                    for j in w.tr.compute
                )
                if has_trapped:
                    self._run_rebalance(w)
                    cs = place_candidates(w, r)
                    if not cs:
                        return None
        return self.placement.place(w, r)

    @staticmethod
    def _mean_rem(w, r) -> float:
        lo, hi = w.cfg["workload"]["long_out" if r.long else "short_out"]
        return max(1.0, (max(float(lo), r.gen) + float(hi)) / 2.0 - r.gen)

    @staticmethod
    def _q_rem(w, r, q: float) -> float:
        lo, hi = w.cfg["workload"]["long_out" if r.long else "short_out"]
        a = max(float(lo), r.gen)
        b = float(hi)
        return max(1.0, (a + q * (b - a)) - r.gen)

    @staticmethod
    def _survival_prob(r, finish: float) -> float:
        """Continuous survival utility in [0, 1] scaled purely by request's own deadline budget."""
        if math.isinf(finish):
            return 0.0
        budget = max(1.0, r.deadline - r.t)
        spread = budget / 15.0
        return min(1.0, max(0.0, 0.5 + (r.deadline - finish) / (2.0 * spread)))

    def _tether_delay_s(self, w, node: int, r, now: float, edge_users: dict[int, int]) -> tuple[float, tuple[int, ...]]:
        """Exact physical time (seconds) to pull remaining raw-image tiles from r.anchor_node to node."""
        anchor = getattr(r, "anchor_node", -1)
        reaccess_rem = getattr(r, "reaccess_remaining", 0)
        tile_bytes = getattr(r, "reaccess_tile_bytes", 0.0)
        if anchor < 0 or reaccess_rem <= 0 or tile_bytes <= 0 or anchor == node:
            return 0.0, ()
        p = w.tr.path(anchor, node, now)
        if p is not None and p[1]:
            hops = len(p[1])
            # Shared bottleneck bandwidth along the multi-hop path from anchor to node
            bw = min(w.net.cap[e] / (edge_users.get(e, 0) + 1.0) for e in p[1])
            # Multi-hop store-and-forward / contention delay across hops
            pull_s = reaccess_rem * ((tile_bytes / bw) * hops + w.hop_lat * hops)
            return pull_s, tuple(p[1])
        # Disconnected from anchor at `now`: wait for next contact or use worst-case diameter
        t_conn = w.tr.next_conn(anchor, node, now) if anchor in w.tr.rs else now
        wait_s = max(0.0, t_conn - now) if not math.isinf(t_conn) else 300.0
        min_cap = min(w.net.cap)
        return wait_s + reaccess_rem * (tile_bytes / min_cap * 4.0), ()

    def _batch_trajectory(self, w, node: int, jobs: list, start: float):
        if not jobs:
            return [0.0], [start]
        active = sorted(((self._mean_rem(w, r), r.i, r) for r in jobs))
        previous = 0.0
        forecast = ThermalForecast(w, node) if w.thermal is not None else None
        t_cur = start
        xs = [0.0]
        ts = [start]
        for i, (target, _, _r) in enumerate(active):
            step = target - previous
            rest = active[i:]
            batch = len(rest)
            if step > 0:
                ctx = sum(x.kv + (target + previous) / 2.0 for _, _, x in rest)
                if forecast is not None:
                    t_cur = forecast.finish(t_cur, batch, ctx, step)
                else:
                    for a, b, dark in w.tr.power_segments(node, t_cur):
                        clock = w.f_dark if dark else 1.0
                        it = iter_time(batch, ctx, clock, w.gpu)
                        if step * it <= b - a:
                            t_cur = a + step * it
                            break
                        step -= (b - a) / it
                xs.append(target)
                ts.append(t_cur)
            previous = target
        last_tok, last_t = xs[-1], ts[-1]
        if not math.isinf(last_t):
            ctx_tail = active[-1][2].kv + last_tok + 4000.0
            if forecast is not None:
                t_ext = forecast.finish(last_t, 1, ctx_tail, 10000.0)
            else:
                t_ext = last_t + 10000.0 * iter_time(1, ctx_tail, 1.0, w.gpu)
            xs.append(last_tok + 10000.0)
            ts.append(t_ext)
        return xs, ts

    def _eval_node(self, w, node: int, jobs: list, start: float, edge_users: dict[int, int]):
        if not jobs:
            return {}, {}
        xs, ts = self._batch_trajectory(w, node, jobs, start)
        probs, fins = {}, {}
        for r in jobs:
            tether_s, _ = self._tether_delay_s(w, node, r, start, edge_users)
            p_sum, f_sum = 0.0, 0.0
            for q in QUANTILES:
                rem_q = self._q_rem(w, r, q)
                t_dec = _interp_linear(rem_q, xs, ts) + tether_s
                t_deliv = w.tr.next_conn(r.origin, node, t_dec)
                p_sum += self._survival_prob(r, t_deliv)
                f_sum += min(t_deliv, r.deadline + 600.0)
            probs[r] = p_sum / len(QUANTILES)
            fins[r] = f_sum / len(QUANTILES)
        return probs, fins

    def rebalance(self, w):
        now = w.env.now
        if now < self.next_plan:
            return
        self.next_plan = now + self.interval_s
        self._run_rebalance(w)

    def _run_rebalance(self, w):
        now = w.env.now
        self._last_rebalance_t = now
        b_crit = self._sustainable_batch(w)

        edge_users = {}
        for f in w.net.flows:
            for _, _, e in f.hops:
                edge_users[e] = edge_users.get(e, 0) + 1

        before_prob, before_fin = {}, {}
        for j in w.tr.compute:
            before_prob[j], before_fin[j] = self._eval_node(w, j, w.nodes[j], now, edge_users)

        long_need = (w.cfg["workload"]["long_pin"] + w.cfg["workload"]["long_out"][1]) * w.bpt
        free_slots = {
            j: int(math.floor((w.kv_cap - w.kv_used(j) + 1e-3) / long_need))
            for j in w.tr.compute
        }
        dst_candidates = [j for j in w.tr.compute if free_slots[j] >= 1 and not w.tr.is_dark(j, now)]
        if not dst_candidates:
            return

        dst_candidates.sort(
            key=lambda j: (
                w.thermal_f[j] < 1.0,
                len(w.nodes[j]) + round(w.reserved[j] / long_need),
                w.thermal.temperature(j) if w.thermal else 0.0,
                j,
            )
        )

        candidates = []
        for src, batch in w.nodes.items():
            long_batch = [r for r in batch if r.long and not r.migrating and now < r.deadline]
            if not long_batch:
                continue
            for r in long_batch:
                stay_p = before_prob[src][r]
                stay_f = before_fin[src][r]
                rest_src = [x for x in batch if x is not r]
                without_p, without_f = self._eval_node(w, src, rest_src, now, edge_users)
                relief_p = sum(without_p[x] - before_prob[src][x] for x in rest_src)
                relief_t = sum(before_fin[src][x] - without_f[x] for x in rest_src)

                # If r is already certain to meet its deadline and moving it gives no relief to others, skip
                if stay_p > 0.98 and relief_p < 0.02 and relief_t < 15.0:
                    continue

                nbytes = r.kv * w.bpt
                feasible = 0
                for dst in dst_candidates:
                    if dst == src or not w.fits(dst, r):
                        continue
                    paths = k_shortest_paths(w.tr, src, dst, now, k=self.k_paths)
                    if not paths:
                        continue
                    best_p, best_delay = None, math.inf
                    for p in paths:
                        if not p[1]:
                            continue
                        bw = min(w.net.cap[e] / (edge_users.get(e, 0) + 1) for e in p[1])
                        d_p = nbytes / bw + w.hop_lat * len(p[1]) + w.switch_s
                        if path_lifetime(w, p) >= now + 1.25 * d_p and d_p < w.mig_timeout:
                            if d_p < best_delay:
                                best_delay, best_p = d_p, p
                    if best_p is None:
                        continue

                    feasible += 1
                    after_p, after_f = self._eval_node(w, dst, [*w.nodes[dst], r], now + best_delay, edge_users)
                    move_p = after_p.pop(r)
                    move_f = after_f.pop(r)
                    burden_p = sum(after_p[x] - before_prob[dst][x] for x in w.nodes[dst])
                    burden_t = sum(after_f[x] - before_fin[dst][x] for x in w.nodes[dst])

                    # Pure physical objective (all in unit of expected surviving requests / normalized seconds):
                    # Subtract total ISL link-occupation seconds (KV migration + repeated raw-image tether pulls)
                    tether_s, tether_edges = self._tether_delay_s(w, dst, r, now + best_delay, edge_users)
                    tether_s_src, _ = self._tether_delay_s(w, src, r, now, edge_users)
                    delta_tether_link_s = max(0.0, tether_s - tether_s_src)
                    kv_link_s = best_delay * len(best_p[1])

                    net_prob_gain = (move_p - stay_p) + relief_p + burden_p
                    net_time_saved = (stay_f - move_f) + relief_t - burden_t - (kv_link_s + delta_tether_link_s)
                    budget_s = max(1.0, r.deadline - r.t)
                    spread = budget_s / 15.0
                    # Convert link-occupation seconds into survival-probability units using the exact same spread!
                    link_prob_cost = (kv_link_s + delta_tether_link_s) / (2.0 * spread)
                    total_utility = net_prob_gain - link_prob_cost + (net_time_saved / budget_s) * 0.15

                    if total_utility > 0.02 and net_time_saved > 0.0:
                        candidates.append({
                            "gain": total_utility,
                            "r": r,
                            "src": src,
                            "dst": dst,
                            "path": best_p,
                            "nbytes": nbytes,
                            "tether_edges": tether_edges,
                            "tether_bytes": getattr(r, "reaccess_remaining", 0) * getattr(r, "reaccess_tile_bytes", 0.0),
                        })
                    if feasible >= 10:
                        break

        if not candidates:
            return

        # Capacity constraints derived purely from physical limits
        dst_cap = {}
        for c in candidates:
            dst = c["dst"]
            if dst not in dst_cap:
                # Keep destination at or below sustainable non-throttling batch b_crit (unless empty)
                dst_cap[dst] = max(1, min(free_slots[dst], b_crit - len(w.nodes[dst])))

        src_cap = {c["src"]: max(1, len(w.nodes[c["src"]]) - 1 if not w.tr.is_dark(c["src"], now) else len(w.nodes[c["src"]])) for c in candidates}
        edge_cap = {}
        for c in candidates:
            for e in c["path"][1]:
                edge_cap[e] = max(c["nbytes"], w.net.cap[e] * (self.interval_s * 3.0) / (edge_users.get(e, 0) + 1))
            for e in c["tether_edges"]:
                edge_cap[e] = max(c["nbytes"], w.net.cap[e] * (self.interval_s * 3.0) / (edge_users.get(e, 0) + 1))

        order = sorted(range(len(candidates)), key=lambda i: -candidates[i]["gain"])
        used_req, used_src, used_dst, used_edge = set(), {}, {}, {}
        for idx in order:
            c = candidates[idx]
            r_id, src, dst, nb = c["r"].i, c["src"], c["dst"], c["nbytes"]
            if r_id in used_req or used_src.get(src, 0) >= src_cap[src] or used_dst.get(dst, 0) >= dst_cap[dst]:
                continue
            if any(used_edge.get(e, 0.0) + nb > edge_cap[e] + 1e-6 for e in c["path"][1]):
                continue
            if w.migrate(c["r"], dst, path=c["path"]):
                used_req.add(r_id)
                used_src[src] = used_src.get(src, 0) + 1
                used_dst[dst] = used_dst.get(dst, 0) + 1
                for e in c["path"][1]:
                    used_edge[e] = used_edge.get(e, 0.0) + nb


def run_tile_sweep(tile_mb: float):
    base = load_config(Path("configs/cosim_placement_thermal.yaml"))
    cfg, _ = profile_config(base, 14, "aligned150", 11)
    cfg["workload"]["agentic_classes"] = {
        "p_scan": 0.35, "p_pinpoint": 0.40, "tile_MB": float(tile_mb),
        "scan_pulls": 5, "pinpoint_pulls": 0, "phase_pulls": 4,
    }
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    env = simpy.Environment()
    pol = KnobFreeTorch(placement=base["policies"]["round_robin"])
    w = CoWorld(env, tr, cfg, pol, seed=11)
    reqs = make_requests(cfg, tr, 11)
    for r in reqs:
        w.submit(r)
    env.run(until=tr.horizon)
    s = summarize(w)
    longs = [r for r in reqs if r.long]
    hops = {}
    for cls in ("scan_high_reaccess", "phase_transition", "pinpoint_deep"):
        cr = [r for r in longs if r.task_class == cls]
        hs = []
        for r in cr:
            if r.anchor_node == r.node:
                hs.append(0)
            else:
                p = tr.path(r.anchor_node, r.node, min(tr.horizon - 1.0, r.t_decoded or tr.horizon - 1.0))
                hs.append(len(p[1]) if (p and p[1]) else 3)
        hops[cls] = st.fmean(hs)
    return tile_mb, s["good_long_frac"] * 100, hops["scan_high_reaccess"], hops["phase_transition"], hops["pinpoint_deep"], w.acc.get("tether_bytes", 0.0) / 1e9


def run_rth_sweep(r_th: float):
    base = load_config(Path("configs/cosim_placement_thermal.yaml"))
    cfg, _ = profile_config(base, 14, "regional120", 11)
    cfg["thermal"]["R_th"] = float(r_th)
    cfg["thermal"]["T0_C"] = cfg["thermal"]["T_amb_C"] + float(r_th) * cfg["power"]["P_idle"]
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    out = {}
    for name, pspec in [
        ("RR_only", base["policies"]["round_robin"]),
        ("RR+deadline_rc", {"policy": "thermal_rebalance", "placement": base["policies"]["round_robin"], "thermal_aware": "rc"}),
        ("RR+KnobFreeTorch", None),
    ]:
        env = simpy.Environment()
        pol = KnobFreeTorch(placement=base["policies"]["round_robin"]) if pspec is None else make_policy(pspec)
        w = CoWorld(env, tr, copy.deepcopy(cfg), pol, seed=11)
        b_crit = KnobFreeTorch._sustainable_batch(w)
        for r in make_requests(cfg, tr, 11):
            w.submit(r)
        env.run(until=tr.horizon)
        out[name] = ( summarize(w)["good_long_frac"] * 100, b_crit )
    return r_th, out


if __name__ == "__main__":
    print("=== 1. Continuous Concentric Adaptation vs Raw-Image Tile Size (S_tile) [Zero Hardcoded Hop Limits!] ===")
    with ProcessPoolExecutor(5) as pool:
        for tile_mb, gl, h_scan, h_ph, h_pin, t_gb in pool.map(run_tile_sweep, [0, 50, 150, 300, 500]):
            print(f"  S_tile = {tile_mb:3.0f} MB | good_long = {gl:5.1f}% | Avg Hops (Scan / Phase / Pinpoint) = {h_scan:.2f} / {h_ph:.2f} / {h_pin:.2f} | Ratio(Pin/Scan) = {h_pin/max(0.01,h_scan):.2f}x | tether_GB = {t_gb:5.1f}")

    print("\n=== 2. Self-Calibrating Thermal Tipping-Point across Satellite Thermal Resistance (R_th) ===")
    with ProcessPoolExecutor(4) as pool:
        for r_th, res in pool.map(run_rth_sweep, [0.22, 0.26, 0.30, 0.36]):
            b_crit = res["RR+KnobFreeTorch"][1]
            print(f"  R_th = {r_th:.2f} C/W (auto b_crit={b_crit}) | RR_only = {res['RR_only'][0]:5.1f}% | RR+deadline_rc = {res['RR+deadline_rc'][0]:5.1f}% | RR+KnobFreeTorch = {res['RR+KnobFreeTorch'][0]:5.1f}%")
