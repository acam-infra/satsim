"""Top-conference-inspired strong baselines for discrete power-thermal ablation (LAB-1916 / LAB-1919).

1. ``LlumnixStyleMigOnly`` (inspired by Llumnix, USENIX OSDI 2024):
   Priority-aware placement & live KV-cache migration isolating tight-slack urgent requests from
   loose-slack interference, WITHOUT proactive ``explicit_tier_idx`` low-tier locking.
2. ``TapasReactiveThermal`` (inspired by TAPAS, ACM ASPLOS 2025 & Hot AI in Cold Space, 2026):
   Current-temperature admission routing + reactive thermal guardband tier step-down near ``T_max_C``
   + Thermal-Load Balancing (TLB) migration, strictly blind to Mixed-Criticality Slack.
3. ``McsSlackDvfsNoMig`` (inspired by Hosseinimotlagh et al., IEEE RTAS 2020 MCS+DVFS):
   Local resident queue tightest-slack ``explicit_tier_idx`` mapping (``P_low`` / ``P_high``),
   with strictly zero cross-satellite ISL migrations.
"""
from __future__ import annotations

import math

from satsim.cosim.policies import (
    OrbitLoad,
    candidates as place_candidates,
    inflight,
    make_policy,
    path_lifetime,
    upload_options,
)
from satsim.cosim.power_tiers import parse_discrete_power_spec
from satsim.cosim.thermal_rebalance import k_shortest_paths
from satsim.cosim.trace import INF
from satsim.method.engine.timing import iter_time


def _est_remaining_tokens(w, r, quantile: float = 0.5) -> float:
    """Conditioned non-clairvoyant remaining token estimate (falls back to r.remaining in toy tests)."""
    wl = (getattr(w, "cfg", None) or {}).get("workload") or {}
    key = "long_out" if r.long else "short_out"
    if key in wl:
        lo, hi = wl[key]
        a = max(float(lo), float(r.gen))
        b = max(a, float(hi))
        return max(1.0, (a + quantile * (b - a)) - float(r.gen))
    return max(1.0, float(r.remaining))


def _is_urgent(w, r, now: float, urgent_slack_ratio: float) -> bool:
    """Classify request criticality from explicit tag or remaining slack-to-ideal-decode ratio."""
    explicit_urg = getattr(r, "urgent", None)
    if explicit_urg is not None:
        return bool(explicit_urg)
    crit = getattr(r, "criticality", None)
    if crit is not None:
        return str(crit).lower() in ("hi", "high", "urgent", "critical")
    rem_tok = _est_remaining_tokens(w, r, quantile=0.5)
    ideal_it = iter_time(1, r.pin + r.gen + rem_tok / 2.0, 1.0, w.gpu)
    ideal_s = max(1e-6, rem_tok * ideal_it)
    return (r.deadline - now) / ideal_s <= urgent_slack_ratio


def _transfer_time_s(w, path, nbytes: float, edge_users: dict[int, int]) -> float:
    """Estimate KV migration transfer duration over shared ISL links (including optional delta-KV)."""
    if not path[1]:
        return 0.0
    bw = min(w.net.cap[e] / (edge_users.get(e, 0) + 1.0) for e in path[1])
    if bw <= 0.0:
        return math.inf
    t_net = nbytes / bw
    if getattr(w, "delta_kv", False) and path[0]:
        it_src = w.iter_of(path[0][0])
        ratio = (w.bpt / it_src) / bw if it_src > 0.0 else 0.0
        if ratio >= 0.9:
            return math.inf
        t_net *= 1.0 + ratio + ratio * ratio
    return t_net + w.hop_lat * len(path[1]) + w.switch_s


def _est_delivery_time(w, node: int, r, start_t: float, rem_tok: float, *, joining: bool) -> float:
    """Estimate delivery completion timestamp on ``node`` starting at ``start_t``."""
    if rem_tok <= 1e-9:
        return w.tr.next_conn(r.origin, node, start_t)
    extra = r if joining else None
    f_lit = w.f_of(node, start_t) if not w.is_eclipse(node, start_t) else 1.0
    it_lit = w.iter_of(node, extra, f=f_lit)
    it_dark = w.iter_of(node, extra, f=min(w.f_dark, f_lit))
    left = rem_tok
    decode_end = INF
    for a, b, dark in w.tr.power_segments(node, start_t):
        it = it_dark if dark else it_lit
        span = b - a
        if left * it <= span:
            decode_end = a + left * it
            break
        left -= span / it
    if decode_end >= INF:
        return INF
    return w.tr.next_conn(r.origin, node, decode_end)


class LlumnixStyleMigOnly:
    """OSDI'24 Llumnix-inspired stateful migration baseline without explicit low-tier locking."""

    id = "llumnix_style_mig_only"

    def __init__(
        self,
        placement: dict | None = None,
        interval_s: float = 5.0,
        max_inflight: int = 24,
        max_mig_per_req: int = 4,
        urgent_slack_ratio: float = 2.2,
        k_paths: int = 3,
        min_gain_s: float = 8.0,
    ):
        self.placement = make_policy(placement) if placement is not None else None
        self.interval_s = float(interval_s) if interval_s is not None else None
        self.next_plan = float(interval_s) if interval_s is not None else 0.0
        self.max_inflight = int(max_inflight)
        self.max_mig_per_req = int(max_mig_per_req)
        self.urgent_slack_ratio = float(urgent_slack_ratio)
        self.k_paths = int(k_paths)
        self.min_gain_s = float(min_gain_s)

    def place(self, w, r) -> int | None:
        if self.placement is not None:
            return self.placement.place(w, r)
        now = float(w.env.now)
        opts = list(upload_options(w, r))
        if not opts:
            opts = [(j, 0.0) for j in place_candidates(w, r)]
        if not opts:
            return None
        urgent = _is_urgent(w, r, now, self.urgent_slack_ratio)
        rem_tok = _est_remaining_tokens(w, r, quantile=0.5)

        def _score(item: tuple[int, float]):
            j, upload_s = item
            t_arr = now + upload_s
            dark = 1 if w.is_eclipse(j, t_arr) else 0
            hot = 1 if w.is_hot(j) else 0
            urg_cnt = sum(1 for x in w.nodes[j] if _is_urgent(w, x, now, self.urgent_slack_ratio))
            est_s = upload_s + rem_tok * w.iter_of(j, r)
            if urgent:
                return (dark, hot, urg_cnt, est_s, len(w.nodes[j]), j)
            return (1 if urg_cnt > 0 else 0, dark, hot, len(w.nodes[j]), est_s, j)

        return min(opts, key=_score)[0]

    def rebalance(self, w) -> None:
        # Intentionally never sets explicit_tier_idx: active sunlit cool nodes sprint at default 300W tier.
        now = float(w.env.now)
        if self.interval_s is not None:
            if now < self.next_plan:
                return None
            self.next_plan = now + self.interval_s

        budget = self.max_inflight - inflight(w)
        if budget <= 0:
            return None

        edge_users: dict[int, int] = {}
        claimed_dst: set[int] = {j for j in w.tr.compute if w.reserved[j] > 0}
        for f in w.net.flows:
            for _, _, e in f.hops:
                edge_users[e] = edge_users.get(e, 0) + 1
            if f.tag in ("mig", "mig_delta") and f.hops:
                claimed_dst.add(f.hops[-1][1])

        urgent_map: dict[object, bool] = {}
        urg_count: dict[int, int] = {}
        for j in w.tr.compute:
            cnt = 0
            for x in w.nodes[j]:
                is_u = _is_urgent(w, x, now, self.urgent_slack_ratio)
                urgent_map[x] = is_u
                if is_u:
                    cnt += 1
            urg_count[j] = cnt

        proposals = []
        for src in w.tr.compute:
            residents = w.nodes[src]
            if not residents:
                continue
            src_dark = w.is_eclipse(src, now)
            src_hot = w.is_hot(src)
            has_urgent_on_src = urg_count[src] > 0

            for r in residents:
                if r.state not in ("decode", "run") or r.migrating or r.n_mig >= self.max_mig_per_req:
                    continue
                if r.remaining <= 0 or now >= r.deadline:
                    continue

                r_urg = urgent_map[r]
                isolate_lo = (not r_urg) and has_urgent_on_src and len(residents) >= 2
                rem_r = _est_remaining_tokens(w, r, quantile=0.5)
                stay_deliv = _est_delivery_time(w, src, r, now, rem_r, joining=False)
                at_risk = stay_deliv > r.deadline - 20.0

                if not (isolate_lo or src_dark or src_hot or (r_urg and (at_risk or len(residents) >= 2))):
                    continue

                it_src = w.iter_of(src)
                hi_relief_s = 0.0
                if isolate_lo and len(residents) >= 2:
                    rest = [x for x in residents if x is not r]
                    it_without = iter_time(
                        max(1, len(rest)), sum(x.kv for x in rest), w.f_of(src, now), w.gpu
                    )
                    d_it = max(0.0, it_src - it_without)
                    hi_relief_s = sum(
                        _est_remaining_tokens(w, x, quantile=0.5) * d_it for x in rest if urgent_map[x]
                    )

                nbytes = r.kv * w.bpt
                best_dst = None
                best_path = None
                best_score = -math.inf

                for dst in w.tr.compute:
                    if dst == src or dst in claimed_dst or not w.fits(dst, r):
                        continue
                    if w.is_eclipse(dst, now) or w.is_hot(dst):
                        continue
                    if not r_urg and urg_count[dst] > 0:
                        continue

                    paths = k_shortest_paths(w.tr, src, dst, now, k=self.k_paths)
                    for p in paths:
                        if not p[1]:
                            continue
                        delay = _transfer_time_s(w, p, nbytes, edge_users)
                        if delay >= 0.85 * w.mig_timeout or path_lifetime(w, p) < now + 1.15 * delay:
                            continue
                        rem_after = max(0.0, rem_r - (delay / it_src if it_src > 0.0 else 0.0))
                        move_deliv = _est_delivery_time(w, dst, r, now + delay, rem_after, joining=True)
                        if move_deliv >= INF:
                            continue

                        if isolate_lo:
                            if move_deliv > r.deadline:
                                continue
                            gain = hi_relief_s + max(0.0, stay_deliv - move_deliv) - 0.1 * delay
                            min_req = 0.05
                        else:
                            gain = stay_deliv - move_deliv
                            min_req = self.min_gain_s

                        if gain >= min_req and gain > best_score:
                            best_score = gain
                            best_dst = dst
                            best_path = p

                if best_dst is not None and best_path is not None:
                    prio = 0 if r_urg else (1 if isolate_lo else 2)
                    proposals.append(((prio, -round(best_score, 6), r.deadline, r.i), r, best_dst, best_path))

        proposals.sort(key=lambda item: item[0])
        for _, r, dst, p in proposals:
            if budget <= 0:
                break
            if dst in claimed_dst:
                continue
            if w.migrate(r, dst, path=p):
                claimed_dst.add(dst)
                budget -= 1
        return None


class TapasReactiveThermal:
    """ASPLOS'25 TAPAS & HotCarbon'26 TLB reactive thermal baseline (slack-blind)."""

    id = "tapas_reactive_thermal"

    def __init__(
        self,
        placement: dict | None = None,
        interval_s: float = 5.0,
        max_inflight: int = 24,
        max_mig_per_req: int = 4,
        guardband_c: float = 6.0,
        hysteresis_c: float = 2.0,
        cool_delta_c: float = 8.0,
        temp_weight_s_per_c: float = 2.5,
        k_paths: int = 3,
        min_gain_s: float = 8.0,
    ):
        self.placement = make_policy(placement) if placement is not None else None
        self.interval_s = float(interval_s) if interval_s is not None else None
        self.next_plan = float(interval_s) if interval_s is not None else 0.0
        self.max_inflight = int(max_inflight)
        self.max_mig_per_req = int(max_mig_per_req)
        self.guardband_c = float(guardband_c)
        self.hysteresis_c = float(hysteresis_c)
        self.cool_delta_c = float(cool_delta_c)
        self.temp_weight_s_per_c = float(temp_weight_s_per_c)
        self.k_paths = int(k_paths)
        self.min_gain_s = float(min_gain_s)

    def _t_max_and_amb(self, w) -> tuple[float, float]:
        spec = parse_discrete_power_spec(getattr(w, "cfg", {}))
        th = (getattr(w, "cfg", None) or {}).get("thermal") or {}
        t_amb = float(th.get("T_amb_C", 20.0))
        if spec is not None:
            return spec.t_max_c, t_amb
        if getattr(w, "thermal", None) is not None:
            return float(getattr(w.thermal, "T_max_C", getattr(w.thermal, "T_throttle_C", 84.0))), t_amb
        return math.inf, t_amb

    def _reactive_tier_for_node(self, w, node: int) -> int | None:
        pm = getattr(w, "power_model", None)
        if pm is None or not hasattr(pm, "set_explicit_tier"):
            return None
        spec = getattr(pm, "spec", None) or parse_discrete_power_spec(getattr(w, "cfg", {}))
        if spec is None:
            return None
        batch = len(w.nodes[node])
        if batch <= 0:
            return None
        t_cur = float(pm.temperature(node))
        t_max = spec.t_max_c
        cur_exp = pm._explicit_tier_idx[node]
        low_fitting = [tk for tk in sorted(spec.work_tiers, key=lambda t: (t.power_w, t.index)) if tk.max_batch >= batch]
        low_idx = low_fitting[0].index if low_fitting else spec.work_tiers[-1].index

        # Slack-blind reactive guardband: step down near T_max_C, release after hysteresis cooling.
        if t_cur >= t_max - self.guardband_c:
            return low_idx
        if cur_exp is not None and t_cur > t_max - (self.guardband_c + self.hysteresis_c):
            return low_idx
        return None

    def on_arrival(self, w, node: int, r) -> None:
        pm = getattr(w, "power_model", None)
        if pm is not None and hasattr(pm, "set_explicit_tier"):
            pm.set_explicit_tier(node, self._reactive_tier_for_node(w, node))

    def on_departure(self, w, node: int, r) -> None:
        pm = getattr(w, "power_model", None)
        if pm is not None and hasattr(pm, "set_explicit_tier"):
            pm.set_explicit_tier(node, self._reactive_tier_for_node(w, node))

    def place(self, w, r) -> int | None:
        if self.placement is not None:
            return self.placement.place(w, r)
        # Strictly slack-blind admission routing: never reads r.deadline.
        now = float(w.env.now)
        opts = list(upload_options(w, r))
        if not opts:
            opts = [(j, 0.0) for j in place_candidates(w, r)]
        if not opts:
            return None
        t_max, t_amb = self._t_max_and_amb(w)
        est_tok = _est_remaining_tokens(w, r, quantile=0.5)

        def _score(item: tuple[int, float]):
            j, upload_s = item
            t_arr = now + upload_s
            t_j = float(w.temperature(j))
            dark = w.is_eclipse(j, t_arr)
            in_guardband = (t_j >= t_max - self.guardband_c) or w.is_hot(j)
            batch = len(w.nodes[j]) + 1 + int(w.reserved[j] > 0)
            ctx = (
                sum(x.kv for x in w.nodes[j])
                + r.pin
                + est_tok / 2.0
                + w.reserved[j] / w.bpt / 2.0
            )
            f_eff = w.f_of(j, t_arr)
            est_decode_s = est_tok * iter_time(batch, ctx, f_eff, w.gpu)
            thermal_cost_s = max(0.0, t_j - t_amb) * self.temp_weight_s_per_c
            return (
                1 if (dark or in_guardband) else 0,
                upload_s + est_decode_s + thermal_cost_s,
                t_j,
                batch,
                j,
            )

        return min(opts, key=_score)[0]

    def rebalance(self, w) -> None:
        # 1. Local reactive temperature guardband DVFS update (slack-blind)
        pm = getattr(w, "power_model", None)
        if pm is not None and hasattr(pm, "set_explicit_tier"):
            for j in w.tr.compute:
                desired = self._reactive_tier_for_node(w, j)
                if pm._explicit_tier_idx[j] != desired:
                    pm.set_explicit_tier(j, desired)
                    if hasattr(w, "_sync_discrete_node"):
                        w._sync_discrete_node(j, len(w.nodes[j]))

        # 2. Cross-satellite Thermal-Load Balancing (TLB) on rebalance_invariants cadence (slack-blind)
        now = float(w.env.now)
        if self.interval_s is not None:
            if now < self.next_plan:
                return None
            self.next_plan = now + self.interval_s

        budget = self.max_inflight - inflight(w)
        if budget <= 0:
            return None

        t_max, _ = self._t_max_and_amb(w)
        cool_ceil_c = t_max - (self.guardband_c + self.hysteresis_c)

        edge_users: dict[int, int] = {}
        claimed_dst: set[int] = {j for j in w.tr.compute if w.reserved[j] > 0}
        for f in w.net.flows:
            for _, _, e in f.hops:
                edge_users[e] = edge_users.get(e, 0) + 1
            if f.tag in ("mig", "mig_delta") and f.hops:
                claimed_dst.add(f.hops[-1][1])

        proposals = []
        for src in w.tr.compute:
            residents = w.nodes[src]
            if not residents:
                continue
            t_src = float(w.temperature(src))
            src_dark = w.is_eclipse(src, now)
            src_hot = w.is_hot(src) or (t_src >= t_max - self.guardband_c)
            if not (src_dark or src_hot):
                continue

            it_src = w.iter_of(src)
            for r in residents:
                if r.state not in ("decode", "run") or r.migrating or r.n_mig >= self.max_mig_per_req:
                    continue
                if r.remaining <= 0:
                    continue

                rem_r = _est_remaining_tokens(w, r, quantile=0.5)
                stay_deliv = _est_delivery_time(w, src, r, now, rem_r, joining=False)
                nbytes = r.kv * w.bpt

                best_dst = None
                best_path = None
                best_gain = self.min_gain_s - 1e-9

                for dst in w.tr.compute:
                    if dst == src or dst in claimed_dst or not w.fits(dst, r):
                        continue
                    if w.is_eclipse(dst, now) or w.is_hot(dst):
                        continue
                    t_dst = float(w.temperature(dst))
                    if t_dst > cool_ceil_c:
                        continue
                    if not src_dark and (t_src - t_dst) < self.cool_delta_c:
                        continue

                    paths = k_shortest_paths(w.tr, src, dst, now, k=self.k_paths)
                    for p in paths:
                        if not p[1]:
                            continue
                        delay = _transfer_time_s(w, p, nbytes, edge_users)
                        if delay >= 0.85 * w.mig_timeout or path_lifetime(w, p) < now + 1.15 * delay:
                            continue
                        rem_after = max(0.0, rem_r - (delay / it_src if it_src > 0.0 else 0.0))
                        move_deliv = _est_delivery_time(w, dst, r, now + delay, rem_after, joining=True)
                        if move_deliv >= INF:
                            continue
                        gain = stay_deliv - move_deliv
                        if gain > best_gain:
                            best_gain = gain
                            best_dst = dst
                            best_path = p

                if best_dst is not None and best_path is not None:
                    # Slack-blind ordering: hottest/most-delayed move first, never sorted by r.deadline
                    proposals.append(((-round(t_src, 3), -round(best_gain, 6), r.i), r, best_dst, best_path))

        proposals.sort(key=lambda item: item[0])
        for _, r, dst, p in proposals:
            if budget <= 0:
                break
            if dst in claimed_dst:
                continue
            if w.migrate(r, dst, path=p):
                claimed_dst.add(dst)
                budget -= 1
        return None


class McsSlackDvfsNoMig:
    """RTAS'20 MCS+DVFS local slack-driven tier governor without cross-satellite migration."""

    id = "mcs_slack_dvfs_no_mig"

    def __init__(
        self,
        placement: dict | None = None,
        interval_s: float = 5.0,
        max_inflight: int = 24,
        max_mig_per_req: int = 4,
        slack_budget_ratio: float = 0.80,
        rem_quantile: float = 0.75,
    ):
        self.placement = make_policy(placement) if placement is not None else OrbitLoad(thermal_mode="dvfs")
        self.interval_s = float(interval_s) if interval_s is not None else None
        self.next_plan = float(interval_s) if interval_s is not None else 0.0
        self.max_inflight = int(max_inflight)
        self.max_mig_per_req = int(max_mig_per_req)
        self.slack_budget_ratio = float(slack_budget_ratio)
        self.rem_quantile = float(rem_quantile)

    def place(self, w, r) -> int | None:
        return self.placement.place(w, r)

    def _select_tier_for_node(self, w, node: int) -> int | None:
        pm = getattr(w, "power_model", None)
        if pm is None or not hasattr(pm, "set_explicit_tier"):
            return None
        spec = getattr(pm, "spec", None) or parse_discrete_power_spec(getattr(w, "cfg", {}))
        if spec is None:
            return None
        jobs = w.nodes[node]
        if not jobs:
            return None

        now = float(w.env.now)
        batch = len(jobs)
        is_dark = bool(w.tr.is_dark(node, now))
        cur_t = float(pm.temperature(node))
        alpha, offset, gain = pm._rc_probe_coeffs
        rems = {x: _est_remaining_tokens(w, x, quantile=self.rem_quantile) for x in jobs}
        ctx = sum(x.kv + rems[x] / 2.0 for x in jobs)
        live = [x for x in jobs if x.deadline > now]

        feasible_work = []
        for tk in sorted(spec.work_tiers, key=lambda t: (t.power_w, t.index)):
            if is_dark and tk.power_w > spec.p_dark_w + 1e-9:
                continue
            if alpha * cur_t + offset + gain * tk.power_w > spec.t_max_c + 1e-9:
                continue
            feasible_work.append(tk)
            if tk.max_batch < batch:
                continue
            if not live:
                # All resident jobs already missed deadline: stay on lowest-power fitting tier
                return tk.index
            it_tk = iter_time(batch, ctx, tk.clock_f, w.gpu)
            all_meet = True
            for x in live:
                avail_s = x.deadline - now
                dec_end = now + rems[x] * it_tk
                deliver_t = w.tr.next_conn(x.origin, node, dec_end)
                est_total_s = (deliver_t - now) if deliver_t < INF else (rems[x] * it_tk)
                if est_total_s > self.slack_budget_ratio * avail_s:
                    all_meet = False
                    break
            if all_meet:
                return tk.index

        fitting = [tk for tk in feasible_work if tk.max_batch >= batch]
        if fitting:
            return max(fitting, key=lambda t: (t.clock_f, -t.power_w)).index
        return None

    def on_arrival(self, w, node: int, r) -> None:
        pm = getattr(w, "power_model", None)
        if pm is not None and hasattr(pm, "set_explicit_tier"):
            pm.set_explicit_tier(node, self._select_tier_for_node(w, node))

    def on_departure(self, w, node: int, r) -> None:
        pm = getattr(w, "power_model", None)
        if pm is not None and hasattr(pm, "set_explicit_tier"):
            pm.set_explicit_tier(node, self._select_tier_for_node(w, node))

    def rebalance(self, w) -> None:
        now = float(w.env.now)
        if self.interval_s is not None and now >= self.next_plan:
            self.next_plan = now + self.interval_s
        pm = getattr(w, "power_model", None)
        if pm is None or not hasattr(pm, "set_explicit_tier"):
            return None
        for j in w.tr.compute:
            desired = self._select_tier_for_node(w, j)
            if pm._explicit_tier_idx[j] != desired:
                pm.set_explicit_tier(j, desired)
                if hasattr(w, "_sync_discrete_node"):
                    w._sync_discrete_node(j, len(w.nodes[j]))
        # Strictly zero cross-satellite migrations (RTAS'20 local MCS+DVFS baseline).
        return None
