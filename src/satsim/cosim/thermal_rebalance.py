"""Deadline-aware migration for long requests on thermally constrained nodes."""
from __future__ import annotations

import math

from satsim.cosim.policies import (
    MAX_INFLIGHT, inflight, make_policy, path_lifetime, thermal_service_clock, transfer_estimate,
)
from satsim.method.engine.timing import iter_time


class ThermalRebalance:
    def __init__(self, placement: dict, externality: bool = True, interval_s: float = 30.0,
                 deadline_aware: bool = True, thermal_aware: bool | str = True,
                 max_inflight: int | None = None, max_mig_per_req: int | None = None):
        self.placement = make_policy(placement)
        self.externality = externality
        self.deadline_aware = deadline_aware
        self.thermal_aware = thermal_aware
        self.rc_forecast = thermal_aware == "rc"
        self.interval_s = interval_s
        self.next_plan = interval_s
        self.max_inflight = MAX_INFLIGHT if max_inflight is None else int(max_inflight)
        self.max_mig_per_req = 1 if max_mig_per_req is None else int(max_mig_per_req)

    def place(self, w, r):
        return self.placement.place(w, r)

    @staticmethod
    def _remaining(w, r):
        lo, hi = w.cfg["workload"]["long_out" if r.long else "short_out"]
        # Condition the workload prior on this request still being alive at r.gen.
        return max(1.0, (max(lo, r.gen) + hi) / 2 - r.gen)

    def _batch_eta(self, w, node, jobs, start):
        active = sorted(((self._remaining(w, r), r.i, r) for r in jobs))
        previous = 0.0
        result = {}
        forecast = ThermalForecast(w, node) if self.rc_forecast and w.thermal is not None else None
        for i, (target, _, r) in enumerate(active):
            step = target - previous
            rest = active[i:]
            batch = len(rest)
            if step > 0:
                ctx = sum(x.kv + (target + previous) / 2 for _, _, x in rest)
                if forecast is not None:
                    start = forecast.finish(start, batch, ctx, step)
                else:
                    lit_clock = thermal_service_clock(w, node, batch, step, ctx) if self.thermal_aware else 1.0
                    for a, b, dark in w.tr.power_segments(node, start):
                        clock = min(lit_clock, w.f_dark) if dark else lit_clock
                        it = iter_time(batch, ctx, clock, w.gpu)
                        if step * it <= b - a:
                            start = a + step * it
                            break
                        step -= (b - a) / it
            result[r] = w.tr.next_conn(r.origin, node, start)
            previous = target
        return result

    @staticmethod
    def _value(r, finish):
        # A soft deadline avoids a discontinuous score under length forecast error.
        spread = min(60.0, (r.deadline - r.t) / 6)
        return min(1.0, max(0.0, 0.5 + (r.deadline - finish) / (2 * spread)))

    def rebalance(self, w):
        now = w.env.now
        if now < self.next_plan:
            return
        self.next_plan = now + self.interval_s
        budget = self.max_inflight - inflight(w)
        if budget <= 0:
            return
        ready = [j for j in w.tr.compute if w.reserved[j] == 0]
        ready.sort(key=lambda j: (w.iter_of(j), w.kv_used(j), j))
        before = {j: self._batch_eta(w, j, w.nodes[j], now) for j in w.tr.compute}
        moves = []
        for src, batch in w.nodes.items():
            for r in batch:
                if not r.long or r.migrating or r.n_mig >= self.max_mig_per_req or now >= r.deadline:
                    continue
                stay = before[src][r]
                if self.deadline_aware and stay < r.deadline - 60:
                    continue
                relief = 0.0
                if self.externality:
                    without = self._batch_eta(w, src, [x for x in batch if x is not r], now)
                    relief = sum(self._value(x, t) - self._value(x, before[src][x]) for x, t in without.items())
                feasible = 0
                for dst in ready:
                    if src == dst or not w.fits(dst, r):
                        continue
                    path = w.tr.path(src, dst, now)
                    if path is None:
                        continue
                    delay = transfer_estimate(w, path, r.kv * w.bpt)
                    if path_lifetime(w, path) < now + 1.25 * delay or delay >= w.mig_timeout:
                        continue
                    feasible += 1
                    after = self._batch_eta(w, dst, [*w.nodes[dst], r], now + delay)
                    mover = after.pop(r)
                    gain = (self._value(r, mover) - self._value(r, stay) if self.deadline_aware
                            else (stay - mover) / 120.0)
                    if self.externality:
                        gain += relief + sum(self._value(x, t) - self._value(x, before[dst][x]) for x, t in after.items())
                    if gain > 0.10:
                        moves.append((gain, -delay, r.i, src, dst, r))
                    if feasible == 8:
                        break
        used = set()
        for _, _, _, src, dst, r in sorted(moves, key=lambda x: x[:-1], reverse=True):
            if src in used or dst in used:
                continue
            if w.migrate(r, dst):
                used.update((src, dst))
                budget -= 1
                if budget == 0:
                    break


class ThermalForecast:
    """Project the existing RC model until each predicted batch departure or thermal transition."""

    def __init__(self, world, node):
        self.w = world
        self.node = node
        self.model = world.thermal
        self.temp = self.model.temperature(node)
        self.throttled = world.thermal_f[node] < 1

    def advance(self, start, dt, batch):
        """Integrate the node's temperature for ``dt`` seconds with ``batch`` residents decoding.

        Token accounting is skipped on purpose: this is the "what happens while the request is
        still uploading / queued" half of a projection, where the caller only needs the temperature
        at the far end of the interval and whether the throttle state flipped.
        """
        w, model = self.w, self.model
        if dt <= 0:
            return start
        target = start + dt
        for now, end, dark in w.tr.power_segments(self.node, start):
            orbit_clock = w.f_dark if dark else 1.0
            while now < min(end, target):
                clock = min(orbit_clock, max(model.f_min, model.f_throttle)) if self.throttled else orbit_clock
                power = w.device_power(batch, clock)
                equilibrium = model.T_amb_C + model.R_th * power
                boundary = model.T_release_C if self.throttled else model.T_throttle_C
                can_cross = equilibrium < boundary if self.throttled else equilibrium > boundary
                until_cross = math.inf
                if can_cross:
                    ratio = (self.temp - equilibrium) / (boundary - equilibrium)
                    until_cross = model.tau_s * math.log(max(1.0, ratio))
                step = min(min(end, target) - now, until_cross)
                if step == until_cross:
                    self.throttled = not self.throttled
                    self.temp = boundary + 1e-3 if self.throttled else boundary - 1e-3
                else:
                    self.temp = equilibrium + (self.temp - equilibrium) * math.exp(-step / model.tau_s)
                now += step
            if now >= target:
                break
        return target

    def finish(self, start, batch, context, tokens):
        w, model = self.w, self.model
        for now, end, dark in w.tr.power_segments(self.node, start):
            orbit_clock = w.f_dark if dark else 1.0
            while now < end:
                clock = min(orbit_clock, max(model.f_min, model.f_throttle)) if self.throttled else orbit_clock
                power = w.device_power(batch, clock)
                equilibrium = model.T_amb_C + model.R_th * power
                boundary = model.T_release_C if self.throttled else model.T_throttle_C
                can_cross = equilibrium < boundary if self.throttled else equilibrium > boundary
                until_cross = math.inf
                if can_cross:
                    ratio = (self.temp - equilibrium) / (boundary - equilibrium)
                    until_cross = model.tau_s * math.log(max(1.0, ratio))
                iteration = iter_time(batch, context, clock, w.gpu)
                service = tokens * iteration
                dt = min(service, end - now, until_cross)
                if dt == until_cross:
                    self.temp = boundary
                    self.throttled = not self.throttled
                else:
                    self.temp = equilibrium + (self.temp - equilibrium) * math.exp(-dt / model.tau_s)
                now += dt
                if dt == service:
                    return now
                tokens -= dt / iteration
        return math.inf


QUANTILES = (0.14, 0.35, 0.56, 0.78)


def _interp_linear(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    import bisect

    k = bisect.bisect_right(xs, x) - 1
    dx = xs[k + 1] - xs[k]
    if dx <= 0:
        return ys[k]
    return ys[k] + (x - xs[k]) / dx * (ys[k + 1] - ys[k])


def k_shortest_paths(
    trace, s: int, d: int, t: float, k: int = 3, max_hops: int = 6
) -> list[tuple[list[int], list[int]]]:
    """Up to k simple paths at time t avoiding intermediate RS nodes (cached per epoch)."""
    cache = getattr(trace, "_k_path_cache", None)
    if cache is None:
        cache = {}
        trace._k_path_cache = cache
    key = (trace.epoch_index(t), s, d, k, max_hops)
    cached = cache.get(key)
    if cached is not None:
        return cached
    p0 = trace.path(s, d, t)
    if p0 is None or not p0[1]:
        paths = [] if p0 is None else [p0]
        cache[key] = paths
        return paths
    paths = [p0]
    if k <= 1:
        return paths
    adj = trace.adjacency(t)
    for ban_e in p0[1]:
        if len(paths) >= k:
            break
        prev: dict[int, tuple[int, int] | None] = {s: None}
        q = [s]
        idx = 0
        while idx < len(q) and d not in prev:
            u = q[idx]
            idx += 1
            if u != s and trace.role[u] == "rs":
                continue
            for v, e in adj[u]:
                if e == ban_e or v in prev:
                    continue
                prev[v] = (u, e)
                q.append(v)
        if d in prev:
            nodes, eids = [d], []
            x = d
            while prev[x] is not None:
                u, e = prev[x]
                nodes.append(u)
                eids.append(e)
                x = u
            cand = (nodes[::-1], eids[::-1])
            if len(cand[1]) <= max_hops and cand not in paths:
                paths.append(cand)
    cache[key] = paths
    return paths


class TorchRebalance:
    """TORCH: Thermo-Orbital Spatio-Temporal Multi-Hop Rebalancer.

    Combines five mechanisms for spaceborne agentic LLM inference:
    1. Pre-Admission Relay Rebalancing: when a sunlit slot opens under burst overload, trapped
       dark/throttled in-flight requests relay across fast 10 Gbps ISLs (~1s) while decoding
       at source, vacating the dark/cooling slot for the slow 500 MB RS upload.
    2. Multi-Quantile RC Thermal-Orbital Trajectory Evaluation: evaluates completion probability
       across output-length quantiles on a single-pass piecewise batch trajectory.
    3. Thermal Hysteresis Tipping-Point Control: respects the batch=2 sustainable equilibrium
       (< 84 C) vs batch >= 3 throttling cliff, and supports cooling destination lookahead.
    4. 1-to-Many Multi-Path Constrained Assignment (Branch-and-Bound over k-shortest paths):
       jointly enforces per-source fan-out, per-destination safe thermal slots, and shared ISL
       multi-hop link capacity budgets.
    5. Agentic Data-Anchor Tether Awareness (optional): penalizes or restricts migrating
       high-reaccess tasks (Type-1 wide-area scan) far from their anchor satellite u_0 while
       routing zero-reaccess deep-reasoning tasks (Type-2/Type-3 phase-2) to distant cold nodes.

    ``viable_slack_s`` and ``pre_admission_relay`` are the **admission gate**, not migration
    quota. They keep their existing parameter names and can be ablated independently of
    ``rebalance_invariants``; the latter only controls interval / inflight / per-request moves.
    """

    id = "torch"

    def __init__(
        self,
        placement: dict,
        interval_s: float = 5.0,
        max_inflight: int = 24,
        max_mig_per_req: int = 4,
        max_src_out: int = 3,
        k_paths: int = 3,
        viable_slack_s: float = 485.0,
        tether_aware: bool = True,
        pre_admission_relay: bool = True,
    ):
        from satsim.cosim.policies import candidates as place_candidates

        self._place_candidates = place_candidates
        self.placement = make_policy(placement)
        self.interval_s = interval_s
        self.next_plan = interval_s
        self.max_inflight = max_inflight
        self.max_mig_per_req = max_mig_per_req
        self.max_src_out = max_src_out
        self.k_paths = k_paths
        self.viable_slack_s = viable_slack_s
        self.tether_aware = tether_aware
        self.pre_admission_relay = pre_admission_relay
        self._last_rebalance_t = -1.0

    def place(self, w, r):
        now = w.env.now
        cs = self._place_candidates(w, r)
        if not cs:
            return None

        if r.long:
            if self.pre_admission_relay and now > 90.0 and now - self._last_rebalance_t >= 1.5:
                if any(not w.tr.is_dark(j, now) and (w.thermal_f[j] == 1.0 or len(w.nodes[j]) <= 1) for j in cs):
                    has_trapped = any(
                        (w.tr.is_dark(j, now) or w.thermal_f[j] < 1.0)
                        and any(
                            x.long and not x.migrating and x.n_mig < self.max_mig_per_req
                            for x in w.nodes[j]
                        )
                        for j in w.tr.compute
                    )
                    if has_trapped:
                        self._run_rebalance(w, urgent_relay=True)
                        cs = self._place_candidates(w, r)
                        if not cs:
                            return None

            if self.viable_slack_s > 0:
                slack_r = r.deadline - now
                if slack_r < self.viable_slack_s:
                    waiting_long = [
                        x for x in w.reqs
                        if x is not r and x.long and x.state == "wait" and x.t <= now
                    ]
                    if len(waiting_long) >= 12:
                        long_need = r.reserved_tokens * w.bpt
                        avail_slots = sum(
                            int(math.floor((w.kv_cap - w.kv_used(j) + 1e-3) / long_need)) for j in cs
                        )
                        viable_waiting = sum(
                            1 for x in waiting_long
                            if self.viable_slack_s <= (x.deadline - now) <= 720.0
                        )
                        if viable_waiting >= max(4, avail_slots):
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
    def _soft_val(r, finish: float) -> float:
        if math.isinf(finish):
            return 0.0
        spread = min(55.0, (r.deadline - r.t) / 6.0)
        base = min(1.0, max(0.0, 0.5 + (r.deadline - finish) / (2.0 * spread)))
        if base == 0.0 and finish < r.deadline + 360.0:
            base = 0.08 * max(0.0, 1.0 - (finish - r.deadline) / 360.0)
        return base

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

    def _tether_extra_s(
        self, w, node: int, r, now: float, edge_users: dict[int, int] | None = None
    ) -> tuple[float, tuple[int, ...]]:
        """Physical time (seconds) and ISL edges to pull remaining raw-image tiles from anchor_node."""
        if not self.tether_aware:
            return 0.0, ()
        anchor = getattr(r, "anchor_node", -1)
        reaccess_rem = getattr(r, "reaccess_remaining", 0)
        tile_bytes = getattr(r, "reaccess_tile_bytes", 0.0)
        if anchor < 0 or reaccess_rem <= 0 or tile_bytes <= 0 or anchor == node:
            return 0.0, ()
        p = w.tr.path(anchor, node, now)
        if p is not None and p[1]:
            hops = len(p[1])
            if edge_users is not None:
                bw = min(w.net.cap[e] / (edge_users.get(e, 0) + 1.0) for e in p[1])
            else:
                bw = min(w.net.cap[e] for e in p[1])
            pull_s = reaccess_rem * (tile_bytes / bw + w.hop_lat * hops)
            return pull_s, tuple(p[1])
        min_cap = min(w.net.cap)
        # 无路径时按 4 跳估计
        return reaccess_rem * (tile_bytes / min_cap + w.hop_lat * 4), ()

    def _eval_job_on_traj(
        self, w, node: int, r, xs: list[float], ts: list[float], start: float,
        edge_users: dict[int, int] | None = None,
    ):
        val_sum = 0.0
        fin_sum = 0.0
        tether_s, _ = self._tether_extra_s(w, node, r, start, edge_users)
        for q in QUANTILES:
            rem_q = self._q_rem(w, r, q)
            t_dec = _interp_linear(rem_q, xs, ts) + tether_s
            t_deliv = w.tr.next_conn(r.origin, node, t_dec)
            val_sum += self._soft_val(r, t_deliv)
            fin_sum += min(t_deliv, r.deadline + 500.0)
        n_q = len(QUANTILES)
        return val_sum / n_q, fin_sum / n_q

    def _eval_node(
        self, w, node: int, jobs: list, start: float, edge_users: dict[int, int] | None = None
    ):
        if not jobs:
            return {}, {}
        xs, ts = self._batch_trajectory(w, node, jobs, start)
        scores, fins = {}, {}
        for r in jobs:
            scores[r], fins[r] = self._eval_job_on_traj(w, node, r, xs, ts, start, edge_users)
        return scores, fins

    def rebalance(self, w):
        now = w.env.now
        if now < self.next_plan:
            return
        self.next_plan = now + self.interval_s
        self._run_rebalance(w, urgent_relay=False)

    def _run_rebalance(self, w, urgent_relay: bool = False):
        now = w.env.now
        self._last_rebalance_t = now
        cur_inflight = inflight(w)
        budget = self.max_inflight - cur_inflight
        if budget <= 0:
            return

        edge_users = {}
        for f in w.net.flows:
            for _, _, e in f.hops:
                edge_users[e] = edge_users.get(e, 0) + 1

        before_score, before_fin = {}, {}
        for j in w.tr.compute:
            before_score[j], before_fin[j] = self._eval_node(w, j, w.nodes[j], now, edge_users)

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
            long_batch = [r for r in batch if r.long]
            if not long_batch:
                continue
            src_dark = w.tr.is_dark(src, now)
            src_thr = w.thermal_f[src] < 1.0
            if urgent_relay and not (src_dark or src_thr):
                continue

            for r in long_batch:
                if r.migrating or r.n_mig >= self.max_mig_per_req or now >= r.deadline:
                    continue
                stay_s = before_score[src][r]
                stay_f = before_fin[src][r]
                if urgent_relay and stay_s > 0.78:
                    continue

                rest_src = [x for x in batch if x is not r]
                without_score, without_fin = self._eval_node(w, src, rest_src, now, edge_users)
                relief_s = sum(without_score[x] - before_score[src][x] for x in rest_src)
                relief_t = sum(before_fin[src][x] - without_fin[x] for x in rest_src)

                if stay_s > 0.96 and relief_s < 0.03 and relief_t < 15.0:
                    continue

                nbytes = r.kv * w.bpt
                feasible = 0
                for dst in dst_candidates:
                    if dst == src or not w.fits(dst, r):
                        continue
                    paths = k_shortest_paths(w.tr, src, dst, now, k=self.k_paths)
                    if not paths:
                        continue
                    best_p = None
                    best_delay = math.inf
                    for p in paths:
                        if not p[1]:
                            continue
                        bw = min(w.net.cap[e] / (edge_users.get(e, 0) + 1) for e in p[1])
                        d_p = nbytes / bw + w.hop_lat * len(p[1]) + w.switch_s
                        if path_lifetime(w, p) >= now + 1.25 * d_p and d_p < w.mig_timeout:
                            if d_p < best_delay:
                                best_delay = d_p
                                best_p = p
                    if best_p is None:
                        continue

                    feasible += 1
                    delay = best_delay
                    p = best_p
                    after_score, after_fin = self._eval_node(w, dst, [*w.nodes[dst], r], now + delay, edge_users)
                    move_s = after_score.pop(r)
                    move_f = after_fin.pop(r)

                    burden_s = sum(after_score[x] - before_score[dst][x] for x in w.nodes[dst])
                    burden_t = sum(after_fin[x] - before_fin[dst][x] for x in w.nodes[dst])

                    # Pure link-occupation seconds (KV transfer + raw-image tether pulls across hops)
                    tether_s_dst, tether_edges = self._tether_extra_s(w, dst, r, now + delay, edge_users)
                    tether_s_src, _ = self._tether_extra_s(w, src, r, now, edge_users)
                    delta_tether_link_s = max(0.0, tether_s_dst - tether_s_src)
                    kv_link_s = delay * len(p[1])

                    net_slo_gain = (move_s - stay_s) + relief_s + burden_s
                    net_time_saved = (stay_f - move_f) + relief_t - burden_t - (kv_link_s + delta_tether_link_s)
                    spread = min(55.0, (r.deadline - r.t) / 6.0)
                    link_prob_cost = (kv_link_s + delta_tether_link_s) / (4.0 * spread)
                    gain = net_slo_gain - link_prob_cost + (net_time_saved / max(1.0, r.deadline - r.t)) * 0.15

                    min_gain = 0.10 if urgent_relay else 0.04
                    if gain > min_gain and net_time_saved > 0.0:
                        candidates.append({
                            "gain": gain,
                            "r": r,
                            "src": src,
                            "dst": dst,
                            "path": p,
                            "nbytes": nbytes,
                            "tether_edges": tether_edges,
                            "delay": delay,
                            "relief_s": relief_s,
                            "relief_t": relief_t,
                            "stay_s": stay_s,
                            "stay_f": stay_f,
                            "link_s": kv_link_s + delta_tether_link_s,
                        })
                    if feasible >= 10:
                        break

        if not candidates:
            return

        selected = self._solve_assignment(w, candidates, free_slots, edge_users, budget, now, urgent_relay)
        for i in selected:
            c = candidates[i]
            w.migrate(c["r"], c["dst"], path=c["path"])

    def _solve_assignment(self, w, candidates, free_slots, edge_users, budget, now, urgent_relay):
        # Pure physical capacities:
        # 1. Source fan-out bounded only by number of resident requests
        src_cap = {
            c["src"]: len(w.nodes[c["src"]]) for c in candidates
        }
        # 2. Destination bounded ONLY by physical free KV memory slots (NO artificial safe-batch cap!)
        dst_cap = {
            c["dst"]: free_slots[c["dst"]] for c in candidates
        }
        # 3. Shared ISL link capacity over planning window
        edge_cap = {}
        for c in candidates:
            for e in c["path"][1]:
                if e not in edge_cap:
                    edge_cap[e] = max(c["nbytes"], w.net.cap[e] * 15.0 / (edge_users.get(e, 0) + 1))

        # Incremental Slot-Indexed Marginal Evaluation: when multiple tasks target the same dst,
        # re-evaluate the 2nd/3rd inbound task against the updated destination batch [*w.nodes[dst], r_1, ..., r_k]
        # so the RC thermal integrator naturally rejects moves that cause thermal throttling!
        order = sorted(range(len(candidates)), key=lambda i: -candidates[i]["gain"])
        selected: list[int] = []
        used_req: set[int] = set()
        used_src: dict[int, int] = {}
        dst_added: dict[int, list] = {}
        used_edge: dict[int, float] = {}

        for idx in order:
            if len(selected) >= budget:
                break
            c = candidates[idx]
            r = c["r"]
            src = c["src"]
            dst = c["dst"]
            nb = c["nbytes"]
            edges = c["path"][1]
            if r.i in used_req or used_src.get(src, 0) >= src_cap[src]:
                continue
            added = dst_added.get(dst, [])
            if len(added) >= dst_cap[dst]:
                continue
            if any(used_edge.get(e, 0.0) + nb > edge_cap[e] + 1e-6 for e in edges):
                continue

            # If dst already has inbound migrations selected in this tick, evaluate true marginal gain
            # under the combined batch [*w.nodes[dst], *added, r] via the RC thermal trajectory
            if added:
                prev_jobs = [*w.nodes[dst], *added]
                prev_score, prev_fin = self._eval_node(w, dst, prev_jobs, now + c["delay"], edge_users)
                new_jobs = [*prev_jobs, r]
                after_score, after_fin = self._eval_node(w, dst, new_jobs, now + c["delay"], edge_users)
                move_s = after_score.pop(r)
                move_f = after_fin.pop(r)
                burden_s = sum(after_score[x] - prev_score[x] for x in prev_jobs)
                burden_t = sum(after_fin[x] - prev_fin[x] for x in prev_jobs)
                net_slo_gain = (move_s - c["stay_s"]) + c["relief_s"] + burden_s
                net_time_saved = (c["stay_f"] - move_f) + c["relief_t"] - burden_t - c["link_s"]
                spread = min(55.0, (r.deadline - r.t) / 6.0)
                marginal_gain = (
                    net_slo_gain
                    - c["link_s"] / (4.0 * spread)
                    + (net_time_saved / max(1.0, r.deadline - r.t)) * 0.15
                )
                if marginal_gain <= (0.10 if urgent_relay else 0.04) or net_time_saved <= 0.0:
                    continue

            selected.append(idx)
            used_req.add(r.i)
            used_src[src] = used_src.get(src, 0) + 1
            dst_added.setdefault(dst, []).append(r)
            for e in edges:
                used_edge[e] = used_edge.get(e, 0.0) + nb

        return selected
