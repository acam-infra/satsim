"""Discrete power-tier assignment solvers, online multi-hop 0W-drain migrator, and offline Time-ILP."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from satsim.cosim.interfaces import AssignmentSolver, CandidateMove
from satsim.cosim.policies import inflight, path_lifetime
from satsim.cosim.power_tiers import (
    DiscretePowerSpec,
    parse_discrete_power_spec,
    rc_linear_coeffs,
    rollout_rc_temperatures,
    select_tier_for_batch,
)
from satsim.cosim.thermal_rebalance import ThermalForecast, TorchRebalance, k_shortest_paths
from satsim.method.engine.timing import iter_time


def _infer_capacities(w, candidates: list[CandidateMove]) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
    """Infer source fan-out, destination KV slot, and edge bandwidth capacities from ``w`` and ``candidates``."""
    src_cap: dict[int, int] = {}
    dst_cap: dict[int, int] = {}
    edge_cap: dict[int, float] = {}

    nodes = getattr(w, "nodes", {})
    for c in candidates:
        if c.src not in src_cap:
            if "src_cap" in c.meta:
                src_cap[c.src] = int(c.meta["src_cap"])
            else:
                src_cap[c.src] = max(1, len(nodes.get(c.src, ())))
        if c.dst not in dst_cap:
            if "dst_cap" in c.meta:
                dst_cap[c.dst] = int(c.meta["dst_cap"])
            else:
                dst_cap[c.dst] = max(1, len(candidates))
        for e in c.path[1]:
            if e not in edge_cap:
                if "edge_cap" in c.meta and e in c.meta["edge_cap"]:
                    edge_cap[e] = float(c.meta["edge_cap"][e])
                else:
                    net = getattr(w, "net", None)
                    cap_e = float(net.cap[e]) if net is not None else 1e9
                    edge_cap[e] = max(c.nbytes, cap_e * 15.0)
    return src_cap, dst_cap, edge_cap


class IncrementalGreedySolver:
    """Incremental slot-indexed greedy solver with inbound marginal re-evaluation."""

    id = "incremental_greedy"

    def solve(
        self,
        w: object,
        candidates: list[CandidateMove],
        *,
        budget: int,
        now: float,
    ) -> list[CandidateMove]:
        if budget <= 0 or not candidates:
            return []

        src_cap, dst_cap, edge_cap = _infer_capacities(w, candidates)
        order = sorted(range(len(candidates)), key=lambda i: (-candidates[i].gain, i))

        selected: list[CandidateMove] = []
        used_req: set[int] = set()
        used_src: dict[int, int] = {}
        dst_added: dict[int, list] = {}
        used_edge: dict[int, float] = {}

        for idx in order:
            if len(selected) >= budget:
                break
            c = candidates[idx]
            if c.req_id in used_req or used_src.get(c.src, 0) >= src_cap[c.src]:
                continue
            added = dst_added.get(c.dst, [])
            if len(added) >= dst_cap[c.dst]:
                continue
            edges = c.path[1]
            if any(used_edge.get(e, 0.0) + c.nbytes > edge_cap[e] + 1e-6 for e in edges):
                continue

            if added:
                marginal_eval = c.meta.get("marginal_eval")
                if callable(marginal_eval):
                    ok, _marginal_gain = marginal_eval(added)
                    if not ok:
                        continue

            selected.append(c)
            used_req.add(c.req_id)
            used_src[c.src] = used_src.get(c.src, 0) + 1
            dst_added.setdefault(c.dst, []).append(c.req)
            for e in edges:
                used_edge[e] = used_edge.get(e, 0.0) + c.nbytes

        return selected


class StepMilpSolver:
    """Single-step HiGHS MILP assignment solver over scored ``CandidateMove`` items."""

    id = "step_milp"

    def __init__(self, time_limit_s: float = 1.0):
        self.time_limit_s = float(time_limit_s)

    def solve(
        self,
        w: object,
        candidates: list[CandidateMove],
        *,
        budget: int,
        now: float,
    ) -> list[CandidateMove]:
        m = len(candidates)
        if budget <= 0 or m == 0:
            return []

        src_cap, dst_cap, edge_cap = _infer_capacities(w, candidates)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        ub: list[float] = []
        row_idx = 0

        # (C1) Unique move per request
        by_req: dict[int, list[int]] = {}
        for i, c in enumerate(candidates):
            by_req.setdefault(c.req_id, []).append(i)
        for idxs in by_req.values():
            if len(idxs) > 1:
                for i in idxs:
                    rows.append(row_idx); cols.append(i); vals.append(1.0)
                ub.append(1.0); row_idx += 1

        # (C2) Source fan-out capacity
        by_src: dict[int, list[int]] = {}
        for i, c in enumerate(candidates):
            by_src.setdefault(c.src, []).append(i)
        for src, idxs in by_src.items():
            cap_s = src_cap[src]
            if len(idxs) > cap_s:
                for i in idxs:
                    rows.append(row_idx); cols.append(i); vals.append(1.0)
                ub.append(float(cap_s)); row_idx += 1

        # (C3) Destination KV capacity
        by_dst: dict[int, list[int]] = {}
        for i, c in enumerate(candidates):
            by_dst.setdefault(c.dst, []).append(i)
        for dst, idxs in by_dst.items():
            cap_d = dst_cap[dst]
            if len(idxs) > cap_d:
                for i in idxs:
                    rows.append(row_idx); cols.append(i); vals.append(1.0)
                ub.append(float(cap_d)); row_idx += 1

        # (C4) Shared ISL edge bandwidth capacity
        by_edge: dict[int, list[tuple[int, float]]] = {}
        for i, c in enumerate(candidates):
            for e in c.path[1]:
                by_edge.setdefault(e, []).append((i, c.nbytes))
        for e, items in by_edge.items():
            if sum(nb for _, nb in items) > edge_cap[e] + 1e-9:
                for i, nb in items:
                    if nb > 0.0:
                        rows.append(row_idx); cols.append(i); vals.append(nb)
                ub.append(edge_cap[e]); row_idx += 1

        # (C5) Global inflight budget
        if m > budget:
            for i in range(m):
                rows.append(row_idx); cols.append(i); vals.append(1.0)
            ub.append(float(budget)); row_idx += 1

        obj = np.array([-c.gain for c in candidates], dtype=float)
        if row_idx > 0:
            mat = coo_matrix((vals, (rows, cols)), shape=(row_idx, m)).tocsc()
            res = milp(
                c=obj,
                integrality=np.ones(m),
                bounds=Bounds(0, 1),
                constraints=LinearConstraint(mat, -np.inf, np.array(ub, dtype=float)),
                options={"time_limit": self.time_limit_s},
            )
            if res.x is None:
                return IncrementalGreedySolver().solve(w, candidates, budget=budget, now=now)
            return [candidates[i] for i in range(m) if res.x[i] > 0.5]

        selected = [candidates[i] for i in range(m) if candidates[i].gain > 0.0]
        return selected[:budget]


class DiscreteThermalMigrator(TorchRebalance):
    """Discrete power-tier aware multi-hop migrator with true 0W drain, link gate, and hard thermal cap."""

    def __init__(
        self,
        placement: dict | None = None,
        solver: AssignmentSolver | None = None,
        interval_s: float = 5.0,
        max_inflight: int = 24,
        max_mig_per_req: int = 4,
        max_src_out: int = 3,
        k_paths: int = 3,
        viable_slack_s: float = 485.0,
        tether_aware: bool = True,
        pre_admission_relay: bool = True,
        energy_weight: float = 0.08,
        empty_bonus: float = 0.12,
        wake_penalty: float = 0.14,
        relay_min_free_ratio: float = 0.40,
        explicit_tier: bool = False,
        on_arrival_tier: bool = False,
        low_tier_pack: bool = False,
        tier_slack_margin_s: float = 10.0,
        tier_slack_ratio: float = 0.0,
        suppress_low_tier_self_mig: bool = False,
        urgent_tier_guard: bool = False,
        eclipse_lookahead_s: float = 0.0,
        thermal_precool_horizon_s: float = 0.0,
        urgent_isolation: bool = False,
        id_override: str | None = None,
    ):
        if placement is None and low_tier_pack and not explicit_tier:
            placement = {"policy": "orbit_load", "thermal_mode": "rc_staged"}
        elif placement is None and explicit_tier:
            placement = {"policy": "orbit_load"}
        super().__init__(
            placement=placement,
            interval_s=interval_s,
            max_inflight=max_inflight,
            max_mig_per_req=max_mig_per_req,
            max_src_out=max_src_out,
            k_paths=k_paths,
            viable_slack_s=viable_slack_s,
            tether_aware=tether_aware,
            pre_admission_relay=pre_admission_relay,
        )
        self.solver: AssignmentSolver = solver if solver is not None else IncrementalGreedySolver()
        self.explicit_tier = bool(explicit_tier)
        self.on_arrival_tier = bool(on_arrival_tier)
        self.low_tier_pack = bool(low_tier_pack)
        self.tier_slack_margin_s = float(tier_slack_margin_s)
        self.tier_slack_ratio = float(tier_slack_ratio)
        self.suppress_low_tier_self_mig = bool(suppress_low_tier_self_mig)
        self.urgent_tier_guard = bool(urgent_tier_guard)
        self.eclipse_lookahead_s = float(eclipse_lookahead_s)
        self.thermal_precool_horizon_s = float(thermal_precool_horizon_s)
        self.urgent_isolation = bool(urgent_isolation)
        if id_override is not None:
            self.id = str(id_override)
        elif self.explicit_tier:
            if self.thermal_precool_horizon_s > 0.0 or self.urgent_isolation:
                self.id = "discrete_rh_tier_v5"
            elif self.eclipse_lookahead_s > 0.0:
                self.id = "discrete_rh_tier_v4"
            elif self.suppress_low_tier_self_mig or self.tier_slack_ratio > 0.0 or self.urgent_tier_guard:
                self.id = "discrete_rh_tier_v3"
            else:
                self.id = "discrete_rh_tier_v2" if (self.on_arrival_tier or self.low_tier_pack) else "discrete_rh_tier"
        else:
            self.id = "discrete_rh_greedy" if self.solver.id == "incremental_greedy" else f"discrete_{self.solver.id}"
        self.energy_weight = float(energy_weight)
        self.empty_bonus = float(empty_bonus)
        self.wake_penalty = float(wake_penalty)
        self.relay_min_free_ratio = float(relay_min_free_ratio)

    @staticmethod
    def _spec(w) -> DiscretePowerSpec | None:
        return parse_discrete_power_spec(getattr(w, "cfg", {}))

    @classmethod
    def _node_temp_c(cls, w, node: int) -> float:
        if hasattr(w, "temperature"):
            return float(w.temperature(node))
        pm = getattr(w, "power_model", None)
        if pm is not None:
            return float(pm.temperature(node))
        thermal = getattr(w, "thermal", None)
        return float(thermal.temperature(node)) if thermal is not None else 0.0

    @classmethod
    def _node_power_w(cls, w, node: int, batch: int, t: float) -> float:
        pm = getattr(w, "power_model", None)
        is_dark = w.tr.is_dark(node, t)
        if pm is not None:
            return float(pm.operating_point(node, batch, is_dark=is_dark, t=t).power_w)
        spec = cls._spec(w)
        if spec is not None:
            cur_t = cls._node_temp_c(w, node)
            return float(select_tier_for_batch(spec, batch, is_eclipse=is_dark, current_temp_c=cur_t).power_w)
        return float(w.device_power(batch, w.f_of(node, t)))

    @classmethod
    def _t_max_c(cls, w) -> float | None:
        spec = cls._spec(w)
        if spec is not None:
            return spec.t_max_c
        thermal = getattr(w, "thermal", None)
        if thermal is not None:
            return float(getattr(thermal, "T_max_C", getattr(thermal, "T_throttle_C", 84.0)))
        return None

    @classmethod
    def _check_thermal_safe(cls, w, dst: int, jobs: list, start_t: float) -> bool:
        t_max = cls._t_max_c(w)
        if t_max is None or not jobs:
            return True

        spec = cls._spec(w)
        horizon_s = spec.thermal_horizon_s if spec is not None else 30.0
        cur_temp = cls._node_temp_c(w, dst)
        if cur_temp >= t_max - 1e-6:
            return False

        pm = getattr(w, "power_model", None)
        if pm is not None:
            now = float(getattr(getattr(w, "env", None), "now", start_t))
            if hasattr(pm, "advance_temp"):
                t_arr = pm.advance_temp(dst, now, max(0.0, start_t - now), len(w.nodes[dst]), trace=w.tr)
                if t_arr >= t_max - 1e-6:
                    return False
                end_temp = pm.advance_temp(
                    dst, start_t, horizon_s, len(jobs), trace=w.tr, initial_temp_c=t_arr
                )
                return end_temp <= t_max + 1e-6
            ctx = sum(x.kv for x in jobs)
            _, end_temp = pm.project_finish(dst, start_t, len(jobs), ctx, 200.0, trace=w.tr, gpu=w.gpu)
            return end_temp <= t_max + 1e-6

        thermal = getattr(w, "thermal", None)
        if thermal is None:
            return True
        if spec is not None:
            p_w = cls._node_power_w(w, dst, len(jobs), start_t)
            traj = rollout_rc_temperatures(
                cur_temp,
                [p_w, p_w],
                tau_s=thermal.tau_s,
                r_th=thermal.R_th,
                t_amb_c=thermal.T_amb_C,
                dt_s=max(1.0, horizon_s / 2.0),
            )
            return max(traj) <= t_max + 1e-6

        fc = ThermalForecast(w, dst)
        fc.advance(start_t, horizon_s, len(jobs))
        return fc.temp <= t_max + 1e-6

    def _is_urgent(self, r, w=None, now: float | None = None) -> bool:
        explicit_urg = getattr(r, "urgent", None)
        if explicit_urg is not None:
            return bool(explicit_urg)
        crit = getattr(r, "criticality", None)
        if crit is not None:
            return str(crit).lower() in ("hi", "high", "urgent", "critical")
        if w is not None and now is not None:
            rem_tok = self._est_rem(w, r)
            ideal_it = iter_time(1, r.pin + r.gen + rem_tok / 2.0, 1.0, w.gpu)
            ideal_s = max(1e-6, rem_tok * ideal_it)
            return (r.deadline - now) / ideal_s <= 2.2
        return False

    def _tier_margin(self, deadline: float, now: float, ratio_override: float | None = None) -> float:
        rem_slack = max(0.0, deadline - now)
        ratio = self.tier_slack_ratio if ratio_override is None else ratio_override
        if ratio > 0.0:
            return max(self.tier_slack_margin_s, ratio * rem_slack)
        return min(self.tier_slack_margin_s, 0.1 * rem_slack)

    def _eval_node_for_mig(self, w, node: int, jobs: list, start: float, edge_users: dict[int, int]):
        if not self.suppress_low_tier_self_mig:
            return self._eval_node(w, node, jobs, start, edge_users)
        pm = getattr(w, "power_model", None)
        if (
            pm is None
            or not hasattr(pm, "_explicit_tier_idx")
            or pm._explicit_tier_idx[node] is None
            or w.tr.is_dark(node, start)
            or w.thermal_f[node] < 1.0
        ):
            return self._eval_node(w, node, jobs, start, edge_users)
        saved = pm._explicit_tier_idx[node]
        pm.set_explicit_tier(node, None)
        try:
            return self._eval_node(w, node, jobs, start, edge_users)
        finally:
            pm.set_explicit_tier(node, saved)

    def _collect_candidates(self, w, urgent_relay: bool = False) -> list[CandidateMove]:
        now = w.env.now
        edge_users: dict[int, int] = {}
        for f in w.net.flows:
            for _, _, e in f.hops:
                edge_users[e] = edge_users.get(e, 0) + 1

        before_score, before_fin = {}, {}
        for j in w.tr.compute:
            before_score[j], before_fin[j] = self._eval_node_for_mig(w, j, w.nodes[j], now, edge_users)

        long_need = (w.cfg["workload"]["long_pin"] + w.cfg["workload"]["long_out"][1]) * w.bpt
        free_slots = {
            j: int(math.floor((w.kv_cap - w.kv_used(j) + 1e-3) / long_need))
            for j in w.tr.compute
        }
        dst_candidates = [j for j in w.tr.compute if free_slots[j] >= 1 and not w.tr.is_dark(j, now)]
        if not dst_candidates:
            return []

        dst_candidates.sort(
            key=lambda j: (
                w.thermal_f[j] < 1.0,
                len(w.nodes[j]) == 0,  # Prefer already-active cool nodes over waking 0W idle nodes
                len(w.nodes[j]) + round(w.reserved[j] / long_need),
                self._node_temp_c(w, j),
                j,
            )
        )

        spec = self._spec(w)
        ref_power_w = max((t.power_w for t in spec.tiers), default=300.0) if spec is not None else float(
            w.cfg["power"].get("P_max", 300.0)
        )
        edge_cap_map = {
            e: max(1e-6, w.net.cap[e] * 15.0 / (edge_users.get(e, 0) + 1))
            for e in range(len(w.net.cap))
        }
        min_gain = 0.10 if urgent_relay else 0.04

        candidates: list[CandidateMove] = []
        for src, batch in w.nodes.items():
            long_batch = [r for r in batch if r.long]
            if not long_batch:
                continue
            src_dark = w.tr.is_dark(src, now)
            src_thr = w.thermal_f[src] < 1.0
            if urgent_relay and not (src_dark or src_thr):
                continue

            n_s = len(batch)
            p_src_before = self._node_power_w(w, src, n_s, now)
            p_src_after = self._node_power_w(w, src, max(0, n_s - 1), now)
            empties_src_to_0w = n_s == 1

            for r in long_batch:
                if r.migrating or r.n_mig >= self.max_mig_per_req or now >= r.deadline:
                    continue
                stay_s = before_score[src][r]
                stay_f = before_fin[src][r]
                if urgent_relay and stay_s > 0.78:
                    continue

                rest_src = [x for x in batch if x is not r]
                without_score, without_fin = self._eval_node_for_mig(w, src, rest_src, now, edge_users)
                relief_s = sum(without_score[x] - before_score[src][x] for x in rest_src)
                relief_t = sum(before_fin[src][x] - without_fin[x] for x in rest_src)

                # Do not skip single-tail requests when 0W drain consolidation is possible
                if not empties_src_to_0w and stay_s > 0.96 and relief_s < 0.03 and relief_t < 15.0:
                    continue

                nbytes = r.kv * w.bpt
                dst_paths: list[tuple[int, tuple[list[int], list[int]], float]] = []
                for dst in dst_candidates:
                    if dst == src or not w.fits(dst, r):
                        continue
                    paths = k_shortest_paths(w.tr, src, dst, now, k=self.k_paths)
                    for p in paths:
                        if not p[1]:
                            continue
                        bw = min(w.net.cap[e] / (edge_users.get(e, 0) + 1) for e in p[1])
                        d_p = nbytes / bw + w.hop_lat * len(p[1]) + w.switch_s
                        if path_lifetime(w, p) >= now + 1.25 * d_p and d_p < w.mig_timeout:
                            dst_paths.append((dst, p, d_p))
                            break
                    if len(dst_paths) >= 14:
                        break

                # Evaluate 1-hop candidates before >=2-hop relays so only genuinely thermally-safe
                # positive-gain 1-hop options suppress multi-hop escape out of a regional hotspot.
                dst_paths.sort(key=lambda item: len(item[1][1]) > 1)
                has_good_1hop = False
                feasible = 0
                for dst, p, delay in dst_paths:
                    hops = len(p[1])
                    # 2~3 hop congestion gate: only use multi-hop when 1-hop has no good cool option
                    # AND relay edges have enough uncongested headroom.
                    if hops >= 2:
                        if has_good_1hop:
                            continue
                        if any(1.0 / (edge_users.get(e, 0) + 1) < self.relay_min_free_ratio for e in p[1]):
                            continue

                    if not self._check_thermal_safe(w, dst, [*w.nodes[dst], r], now + delay):
                        continue

                    feasible += 1
                    n_d = len(w.nodes[dst])
                    wakes_dst_from_0w = n_d == 0
                    p_dst_before = self._node_power_w(w, dst, n_d, now + delay)
                    p_dst_after = self._node_power_w(w, dst, n_d + 1, now + delay)
                    delta_power_w = (p_src_after - p_src_before) + (p_dst_after - p_dst_before)

                    after_score, after_fin = self._eval_node_for_mig(w, dst, [*w.nodes[dst], r], now + delay, edge_users)
                    move_s = after_score.pop(r)
                    move_f = after_fin.pop(r)
                    burden_s = sum(after_score[x] - before_score[dst][x] for x in w.nodes[dst])
                    burden_t = sum(after_fin[x] - before_fin[dst][x] for x in w.nodes[dst])

                    tether_s_dst, _ = self._tether_extra_s(w, dst, r, now + delay, edge_users)
                    tether_s_src, _ = self._tether_extra_s(w, src, r, now, edge_users)
                    delta_tether_link_s = max(0.0, tether_s_dst - tether_s_src)
                    kv_link_s = delay * hops

                    net_slo_gain = (move_s - stay_s) + relief_s + burden_s
                    net_time_saved = (stay_f - move_f) + relief_t - burden_t - (kv_link_s + delta_tether_link_s)
                    spread = min(55.0, (r.deadline - r.t) / 6.0)
                    link_prob_cost = (kv_link_s + delta_tether_link_s) / (4.0 * spread)

                    power_bonus = (-delta_power_w / max(1.0, ref_power_w)) * self.energy_weight
                    if empties_src_to_0w and not wakes_dst_from_0w:
                        power_bonus += self.empty_bonus
                    elif wakes_dst_from_0w and not empties_src_to_0w:
                        power_bonus -= self.wake_penalty
                        if spec is not None and spec.wakeup_energy_j > 0.0:
                            power_bonus -= (spec.wakeup_energy_j / max(1.0, ref_power_w * 10.0)) * self.energy_weight

                    gain = net_slo_gain - link_prob_cost + (net_time_saved / max(1.0, r.deadline - r.t)) * 0.15 + power_bonus

                    if gain > min_gain and (net_time_saved > 0.0 or (empties_src_to_0w and delta_power_w < 0.0)):
                        if hops == 1:
                            has_good_1hop = True
                        def _make_marginal(
                            _dst=dst, _r=r, _delay=delay, _stay_s=stay_s, _stay_f=stay_f,
                            _relief_s=relief_s, _relief_t=relief_t, _link_s=kv_link_s + delta_tether_link_s,
                            _p_src_delta=p_src_after - p_src_before, _empties=empties_src_to_0w,
                        ):
                            def _eval_marginal(added_reqs: list) -> tuple[bool, float]:
                                prev_jobs = [*w.nodes[_dst], *added_reqs]
                                new_jobs = [*prev_jobs, _r]
                                if not self._check_thermal_safe(w, _dst, new_jobs, now + _delay):
                                    return False, -1.0
                                prev_sc, prev_fn = self._eval_node_for_mig(w, _dst, prev_jobs, now + _delay, edge_users)
                                aft_sc, aft_fn = self._eval_node_for_mig(w, _dst, new_jobs, now + _delay, edge_users)
                                m_s = aft_sc.pop(_r)
                                m_f = aft_fn.pop(_r)
                                b_s = sum(aft_sc[x] - prev_sc[x] for x in prev_jobs)
                                b_t = sum(aft_fn[x] - prev_fn[x] for x in prev_jobs)
                                n_slo = (m_s - _stay_s) + _relief_s + b_s
                                n_t = (_stay_f - m_f) + _relief_t - b_t - _link_s
                                p_d_prev = self._node_power_w(w, _dst, len(prev_jobs), now + _delay)
                                p_d_new = self._node_power_w(w, _dst, len(new_jobs), now + _delay)
                                d_pw = _p_src_delta + (p_d_new - p_d_prev)
                                p_bon = (-d_pw / max(1.0, ref_power_w)) * self.energy_weight + (self.empty_bonus if _empties else 0.0)
                                sp = min(55.0, (_r.deadline - _r.t) / 6.0)
                                mg = n_slo - _link_s / (4.0 * sp) + (n_t / max(1.0, _r.deadline - _r.t)) * 0.15 + p_bon
                                ok = mg > min_gain and (n_t > 0.0 or (_empties and d_pw < 0.0))
                                return ok, mg
                            return _eval_marginal

                        candidates.append(
                            CandidateMove(
                                req_id=r.i,
                                req=r,
                                src=src,
                                dst=dst,
                                path=p,
                                nbytes=nbytes,
                                delay_s=delay,
                                gain=gain,
                                delta_power_w=delta_power_w,
                                empties_src_to_0w=empties_src_to_0w,
                                wakes_dst_from_0w=wakes_dst_from_0w,
                                meta={
                                    "src_cap": len(batch),
                                    "dst_cap": free_slots[dst],
                                    "edge_cap": edge_cap_map,
                                    "marginal_eval": _make_marginal(),
                                },
                            )
                        )
                    if feasible >= 10:
                        break

        return candidates

    def _est_rem(self, w, r) -> float:
        wl = getattr(w, "cfg", {}).get("workload") or {}
        key = "long_out" if getattr(r, "long", False) else "short_out"
        if key in wl:
            return self._mean_rem(w, r)
        return max(1.0, float(getattr(r, "remaining", 1.0)))

    def _select_explicit_tier_for_node(self, w, node: int) -> int | None:
        pm = getattr(w, "power_model", None)
        spec = self._spec(w)
        if pm is None or spec is None or not hasattr(pm, "set_explicit_tier"):
            return None
        jobs = w.nodes[node]
        if not jobs:
            return None
        now = float(w.env.now)
        if self.urgent_tier_guard and any(self._is_urgent(x, w, now) for x in jobs):
            return None
        batch = len(jobs)
        is_dark = bool(w.tr.is_dark(node, now))
        approaching_dark = (
            self.eclipse_lookahead_s > 0.0
            and not is_dark
            and bool(w.tr.is_dark(node, now + self.eclipse_lookahead_s))
        )
        cur_t = float(pm.temperature(node))
        # Pre-eclipse sprint: when about to enter an eclipse with P_dark < P_high and thermal
        # headroom is safe, keep high tier to drain tokens before the dark slowdown.
        if (
            approaching_dark
            and spec.p_dark_w < max(t.power_w for t in spec.work_tiers) - 1e-9
            and cur_t < spec.t_max_c - 6.0
        ):
            return None
        alpha, offset, gain = pm._rc_probe_coeffs
        if self.thermal_precool_horizon_s > 0.0:
            # Gate proactive low-tier pre-cooling: only step down when approaching thermal cap.
            thermal_gate = cur_t >= spec.t_max_c - 4.0
            if not thermal_gate and hasattr(pm, "advance_temp"):
                saved = pm._explicit_tier_idx[node]
                pm.set_explicit_tier(node, None)
                try:
                    t_proj = pm.advance_temp(node, now, self.thermal_precool_horizon_s, batch, trace=w.tr)
                finally:
                    pm.set_explicit_tier(node, saved)
                thermal_gate = t_proj >= spec.t_max_c - 2.0
            if not thermal_gate:
                return None
        ctx = sum(x.kv + self._est_rem(w, x) / 2.0 for x in jobs)
        eff_ratio = self.tier_slack_ratio
        if self.eclipse_lookahead_s > 0.0 and cur_t >= spec.t_max_c - 8.0:
            eff_ratio = min(eff_ratio, 0.20) if eff_ratio > 0.0 else 0.20
        for tk in sorted(spec.work_tiers, key=lambda t: (t.power_w, t.index)):
            if tk.max_batch < batch:
                continue
            if is_dark and tk.power_w > spec.p_dark_w + 1e-9:
                continue
            if alpha * cur_t + offset + gain * tk.power_w > spec.t_max_c + 1e-9:
                continue
            it = iter_time(batch, ctx, tk.clock_f, w.gpu)
            all_safe = True
            for x in jobs:
                rem = self._est_rem(w, x)
                margin = self._tier_margin(x.deadline, now, ratio_override=eff_ratio)
                if now + rem * it > x.deadline - margin:
                    all_safe = False
                    break
            if all_safe:
                return tk.index
        return None

    def _update_explicit_tiers(self, w, nodes: list[int] | None = None) -> None:
        if not self.explicit_tier:
            return
        pm = getattr(w, "power_model", None)
        if pm is None or not hasattr(pm, "set_explicit_tier"):
            return
        for j in (w.tr.compute if nodes is None else nodes):
            pm.set_explicit_tier(j, self._select_explicit_tier_for_node(w, j))

    def on_arrival(self, w, node: int, r) -> None:
        if self.explicit_tier and self.on_arrival_tier:
            self._update_explicit_tiers(w, [node])

    def _low_tier_pack_place(self, w, r) -> int | None:
        from satsim.cosim.policies import upload_options

        pm = getattr(w, "power_model", None)
        spec = self._spec(w)
        if pm is None or spec is None or len(spec.work_tiers) < 2 or not hasattr(pm, "_tier_idx"):
            return None
        now = float(w.env.now)
        if self.urgent_tier_guard and self._is_urgent(r, w, now):
            return None
        low_tier = min(spec.work_tiers, key=lambda t: (t.power_w, t.index))
        if low_tier.max_batch < 2:
            return None
        est_rem_r = self._est_rem(w, r)
        best_score = None
        best_node: int | None = None
        alpha, offset, gain = pm._rc_probe_coeffs
        for j, upload_s in upload_options(w, r):
            if not w.nodes[j]:
                continue
            if self.urgent_tier_guard and any(self._is_urgent(x, w, now) for x in w.nodes[j]):
                continue
            if pm._tier_idx[j] != low_tier.index and pm._explicit_tier_idx[j] != low_tier.index:
                continue
            new_batch = len(w.nodes[j]) + 1 + int(w.reserved[j] > 0)
            if new_batch > low_tier.max_batch:
                continue
            start = now + upload_s
            if w.tr.is_dark(j, start) and low_tier.power_w > spec.p_dark_w + 1e-9:
                continue
            cur_t = float(pm.temperature(j))
            if alpha * cur_t + offset + gain * low_tier.power_w > spec.t_max_c + 1e-9:
                continue
            ctx = (
                sum(x.kv + self._est_rem(w, x) / 2.0 for x in w.nodes[j])
                + r.pin
                + est_rem_r / 2.0
                + w.reserved[j] / w.bpt / 2.0
            )
            it = iter_time(new_batch, ctx, low_tier.clock_f, w.gpu)
            margin_r = self._tier_margin(r.deadline, start)
            est_fin_r = start + est_rem_r * it
            if est_fin_r > r.deadline - margin_r:
                continue
            it_cur = w.iter_of(j)
            residents_ok = True
            for x in w.nodes[j]:
                rem_x = max(0.0, self._est_rem(w, x) - (upload_s / it_cur if it_cur > 0 else 0.0))
                margin_x = self._tier_margin(x.deadline, start)
                if start + rem_x * it > x.deadline - margin_x:
                    residents_ok = False
                    break
            if not residents_ok:
                continue
            score = (est_fin_r, new_batch, upload_s, j)
            if best_score is None or score < best_score:
                best_score = score
                best_node = j
        return best_node

    def _urgent_isolated_place(self, w, r) -> int | None:
        from satsim.cosim.policies import upload_options

        base_dst = super().place(w, r)
        if base_dst is None:
            return None
        now = float(w.env.now)
        if self._is_urgent(r, w, now):
            return base_dst
        if not any(self._is_urgent(x, w, now) for x in w.nodes[base_dst]):
            return base_dst
        if not hasattr(self.placement, "_delivery"):
            return base_dst
        tokens = w.expected_tokens(r)
        best_score = None
        best_node: int | None = None
        for j, upload_s in upload_options(w, r):
            if any(self._is_urgent(x, w, now) for x in w.nodes[j]):
                continue
            fin = self.placement._delivery(w, r, tokens, j, upload_s)
            if fin > r.deadline:
                continue
            score = (fin, len(w.nodes[j]), j)
            if best_score is None or score < best_score:
                best_score = score
                best_node = j
        return best_node if best_node is not None else base_dst

    def place(self, w, r):
        if self.low_tier_pack:
            packed = self._low_tier_pack_place(w, r)
            if packed is not None:
                return packed
        if self.urgent_isolation:
            isolated = self._urgent_isolated_place(w, r)
            if isolated is not None:
                return isolated
        return super().place(w, r)

    def _run_rebalance(self, w, urgent_relay: bool = False):
        now = w.env.now
        self._last_rebalance_t = now
        self._update_explicit_tiers(w)
        budget = self.max_inflight - inflight(w)
        if budget <= 0:
            return
        candidates = self._collect_candidates(w, urgent_relay=urgent_relay)
        if not candidates:
            return
        selected = self.solver.solve(w, candidates, budget=budget, now=now)
        for c in selected:
            w.migrate(c.req, c.dst, path=c.path)


@dataclass(frozen=True)
class SnapshotProblem:
    """Multi-slot discrete power-thermal snapshot for 4-way oracle vs online comparison."""

    n_nodes: int
    n_slots: int
    dt_s: float
    tiers: tuple[tuple[float, int, float], ...]  # (power_w, max_batch, clock_f)
    t0_c: list[float]
    is_dark: list[list[bool]]  # [node][slot]
    p_dark_w: float
    tau_s: float
    r_th: float
    t_amb_c: float
    t_max_c: float
    adj: dict[int, list[tuple[int, int]]]  # node -> [(nbr, edge_id)]
    edge_cap_gbps: list[float]
    req_init_node: list[int]
    req_value: list[float]
    req_kv_gb: list[float]
    energy_lambda: float = 0.002


@dataclass(frozen=True)
class SnapshotResult:
    """Evaluated outcome for one snapshot solver."""

    name: str
    objective: float
    solve_ms: float
    thermal_violations: int
    energy_kj: float
    zero_w_frac: float
    timed_out: bool = False


def _evaluate_node_slots(
    prob: SnapshotProblem,
    assign: list[list[int]],  # assign[r][t] -> node
    tier_assign: list[list[int]] | None = None,
) -> tuple[float, int, float, float]:
    """Compute (objective, thermal_violations, energy_kj, zero_w_frac) for a node assignment."""
    n_req = len(prob.req_init_node)
    alpha, offset, gain = rc_linear_coeffs(prob.tau_s, prob.r_th, prob.t_amb_c, prob.dt_s)
    total_energy_j = 0.0
    zero_slots = 0
    thermal_violations = 0
    service_score = 0.0

    # Per-node slot occupancy and RC rollout
    for j in range(prob.n_nodes):
        cur_t = prob.t0_c[j]
        for t in range(prob.n_slots):
            batch = sum(1 for r in range(n_req) if assign[r][t] == j)
            if batch == 0:
                p_w = 0.0
                zero_slots += 1
            else:
                dark = prob.is_dark[j][t]
                if tier_assign is not None and 1 <= tier_assign[j][t] < len(prob.tiers):
                    tk = prob.tiers[tier_assign[j][t]]
                    if tk[1] >= batch and (not dark or tk[0] <= prob.p_dark_w + 1e-9) and (
                        alpha * cur_t + offset + gain * tk[0] <= prob.t_max_c + 1e-6
                    ):
                        chosen = tk
                    else:
                        chosen = None
                else:
                    chosen = None
                if chosen is None:
                    feasible = [
                        (pw, mb, cf)
                        for pw, mb, cf in prob.tiers[1:]
                        if (not dark or pw <= prob.p_dark_w + 1e-9)
                        and alpha * cur_t + offset + gain * pw <= prob.t_max_c + 1e-9
                    ]
                    if not feasible:
                        feasible = [min(prob.tiers[1:], key=lambda x: x[0])]
                    fitting = [x for x in feasible if x[1] >= batch]
                    chosen = max(fitting, key=lambda x: (x[2], -x[0])) if fitting else max(feasible, key=lambda x: x[1])
                p_w = chosen[0]
                cf = chosen[2]
                # Reward requests served at node j in slot t scaled by clock_f and slot weight
                for r in range(n_req):
                    if assign[r][t] == j:
                        service_score += (prob.req_value[r] / prob.n_slots) * cf
            cur_t = alpha * cur_t + offset + gain * p_w
            if cur_t > prob.t_max_c + 1e-6:
                thermal_violations += 1
            total_energy_j += p_w * prob.dt_s

    energy_kj = total_energy_j / 1000.0
    zero_w_frac = zero_slots / max(1, prob.n_nodes * prob.n_slots)
    obj = service_score - prob.energy_lambda * energy_kj
    return obj, thermal_violations, energy_kj, zero_w_frac


def _shortest_path_edges(
    prob: SnapshotProblem, src: int, dst: int, max_hops: int
) -> list[int] | None:
    if src == dst:
        return []
    prev: dict[int, tuple[int, int] | None] = {src: None}
    q = [src]
    idx = 0
    while idx < len(q) and dst not in prev:
        u = q[idx]
        idx += 1
        for v, e in prob.adj.get(u, ()):
            if v not in prev:
                prev[v] = (u, e)
                q.append(v)
    if dst not in prev:
        return None
    edges: list[int] = []
    cur = dst
    while prev[cur] is not None:
        u, e = prev[cur]  # type: ignore[misc]
        edges.append(e)
        cur = u
    edges.reverse()
    return edges if len(edges) <= max_hops else None


def solve_snapshot_no_plan(prob: SnapshotProblem) -> SnapshotResult:
    t0 = time.perf_counter()
    assign = [[prob.req_init_node[r]] * prob.n_slots for r in range(len(prob.req_init_node))]
    obj, viol, ekj, zw = _evaluate_node_slots(prob, assign)
    return SnapshotResult("No-Plan", obj, (time.perf_counter() - t0) * 1000.0, viol, ekj, zw)


def solve_snapshot_greedy(prob: SnapshotProblem, *, max_hops: int, name: str) -> SnapshotResult:
    """Fast rolling-horizon greedy snapshot solver (1-Hop or 3-Hop 0W-drain)."""
    t0 = time.perf_counter()
    n_req = len(prob.req_init_node)
    alpha, offset, gain = rc_linear_coeffs(prob.tau_s, prob.r_th, prob.t_amb_c, prob.dt_s)
    assign = [[prob.req_init_node[r]] * prob.n_slots for r in range(n_req)]
    cur_t = list(prob.t0_c)
    max_cap = max(mb for _, mb, _ in prob.tiers)

    for t in range(prob.n_slots):
        cur_node = [prob.req_init_node[r] if t == 0 else assign[r][t - 1] for r in range(n_req)]
        batch = [0] * prob.n_nodes
        for r in range(n_req):
            batch[cur_node[r]] += 1
        edge_used_gb = [0.0] * len(prob.edge_cap_gbps)

        # Order requests: dark/hot or single-tail sources first
        req_order = sorted(
            range(n_req),
            key=lambda r: (
                not prob.is_dark[cur_node[r]][t],
                batch[cur_node[r]] != 1,
                -prob.req_value[r],
            ),
        )
        for r in req_order:
            src = cur_node[r]
            best_dst = src
            best_delta = 0.0
            best_edges: list[int] = []

            # Helper: instantaneous node value & power at batch b
            def _node_tier(j: int, b: int) -> tuple[float, float, bool]:
                if b <= 0:
                    return 0.0, 0.0, True
                dark = prob.is_dark[j][t]
                feas = [
                    (pw, mb, cf)
                    for pw, mb, cf in prob.tiers[1:]
                    if (not dark or pw <= prob.p_dark_w + 1e-9)
                    and alpha * cur_t[j] + offset + gain * pw <= prob.t_max_c + 1e-9
                ]
                if not feas:
                    return 0.0, 0.0, False
                fit = [x for x in feas if x[1] >= b]
                ch = max(fit, key=lambda x: (x[2], -x[0])) if fit else max(feas, key=lambda x: x[1])
                return ch[0], ch[2], True

            p_s0, cf_s0, ok_s0 = _node_tier(src, batch[src])
            p_s1, cf_s1, _ = _node_tier(src, batch[src] - 1)
            src_score_before = batch[src] * cf_s0 if ok_s0 else 0.0
            src_score_after = (batch[src] - 1) * cf_s1

            for dst in range(prob.n_nodes):
                if dst == src or batch[dst] >= max_cap:
                    continue
                if prob.is_dark[dst][t] and not prob.is_dark[src][t]:
                    continue
                edges = _shortest_path_edges(prob, src, dst, max_hops)
                if edges is None:
                    continue
                if any(
                    edge_used_gb[e] + prob.req_kv_gb[r] > prob.edge_cap_gbps[e] * prob.dt_s / 8.0 + 1e-9
                    for e in edges
                ):
                    continue
                # Multi-hop congestion gate
                if len(edges) >= 2 and any(edge_used_gb[e] > 0.4 * prob.edge_cap_gbps[e] * prob.dt_s / 8.0 for e in edges):
                    continue

                p_d0, cf_d0, ok_d0 = _node_tier(dst, batch[dst])
                p_d1, cf_d1, ok_d1 = _node_tier(dst, batch[dst] + 1)
                if not ok_d1:
                    continue
                dst_score_before = batch[dst] * cf_d0 if ok_d0 else 0.0
                dst_score_after = (batch[dst] + 1) * cf_d1

                delta_svc = ((src_score_after - src_score_before) + (dst_score_after - dst_score_before)) * (
                    prob.req_value[r] / prob.n_slots
                )
                delta_ekj = ((p_s1 - p_s0) + (p_d1 - p_d0)) * prob.dt_s / 1000.0
                delta_obj = delta_svc - prob.energy_lambda * delta_ekj
                if delta_obj > best_delta + 1e-9:
                    best_delta = delta_obj
                    best_dst = dst
                    best_edges = edges

            if best_dst != src:
                batch[src] -= 1
                batch[best_dst] += 1
                cur_node[r] = best_dst
                for e in best_edges:
                    edge_used_gb[e] += prob.req_kv_gb[r]
            assign[r][t] = cur_node[r]

        for j in range(prob.n_nodes):
            if batch[j] == 0:
                p_w = 0.0
            else:
                dark = prob.is_dark[j][t]
                feas = [
                    (pw, mb, cf)
                    for pw, mb, cf in prob.tiers[1:]
                    if (not dark or pw <= prob.p_dark_w + 1e-9)
                    and alpha * cur_t[j] + offset + gain * pw <= prob.t_max_c + 1e-9
                ]
                if not feas:
                    feas = [min(prob.tiers[1:], key=lambda x: x[0])]
                fit = [x for x in feas if x[1] >= batch[j]]
                p_w = (max(fit, key=lambda x: (x[2], -x[0])) if fit else max(feas, key=lambda x: x[1]))[0]
            cur_t[j] = alpha * cur_t[j] + offset + gain * p_w

    solve_ms = (time.perf_counter() - t0) * 1000.0
    obj, viol, ekj, zw = _evaluate_node_slots(prob, assign)
    return SnapshotResult(name, obj, solve_ms, viol, ekj, zw)


class SnapshotTimeIlpSolver:
    """Offline multi-slot time-indexed MILP oracle (``120s Time-ILP``) using HiGHS."""

    def __init__(self, time_limit_s: float = 120.0, max_hops: int = 3):
        self.time_limit_s = float(time_limit_s)
        self.max_hops = int(max_hops)

    def solve(self, prob: SnapshotProblem) -> SnapshotResult:
        t0 = time.perf_counter()
        n_req = len(prob.req_init_node)
        n_nodes = prob.n_nodes
        n_slots = prob.n_slots
        n_tiers = len(prob.tiers)
        alpha, offset, gain = rc_linear_coeffs(prob.tau_s, prob.r_th, prob.t_amb_c, prob.dt_s)

        # Variable indexing:
        # y[r, j, t] in {0, 1}: request r at node j in slot t
        # z[j, t, k] in {0, 1}: node j in slot t uses tier k (k=0 is 0W idle)
        # T[j, t] continuous: temperature at end of slot t
        # w[r, j, t] continuous: exact McCormick envelope of y[r,j,t] * sum_k cf_k * z[j,t,k] * (val_r/T)
        # m[r, u, v, t] in {0, 1}: migration of request r from u to v at start of slot t
        n_y = n_req * n_nodes * n_slots
        n_z = n_nodes * n_slots * n_tiers
        n_T = n_nodes * n_slots
        n_w = n_req * n_nodes * n_slots

        # Precompute pairwise shortest-path edges once per (u, v) instead of per (t, r, u, v)
        pair_edges: dict[tuple[int, int], list[int]] = {}
        for u in range(n_nodes):
            for v in range(n_nodes):
                if u == v:
                    continue
                edges = _shortest_path_edges(prob, u, v, self.max_hops)
                if edges is not None:
                    pair_edges[(u, v)] = edges

        # Precompute feasible migrations (r, u, v, t, edges); at t=0 only u == req_init_node[r] is valid
        mig_tuples: list[tuple[int, int, int, int, list[int]]] = []
        for t in range(n_slots):
            for r in range(n_req):
                u_candidates = (prob.req_init_node[r],) if t == 0 else range(n_nodes)
                for u in u_candidates:
                    for v in range(n_nodes):
                        edges = pair_edges.get((u, v))
                        if edges is not None:
                            mig_tuples.append((r, u, v, t, edges))
        n_m = len(mig_tuples)
        total_vars = n_y + n_z + n_T + n_w + n_m

        def y_idx(r: int, j: int, t: int) -> int:
            return (r * n_nodes + j) * n_slots + t

        def z_idx(j: int, t: int, k: int) -> int:
            return n_y + (j * n_slots + t) * n_tiers + k

        def t_idx(j: int, t: int) -> int:
            return n_y + n_z + j * n_slots + t

        def w_idx(r: int, j: int, t: int) -> int:
            return n_y + n_z + n_T + (r * n_nodes + j) * n_slots + t

        def m_idx(idx: int) -> int:
            return n_y + n_z + n_T + n_w + idx

        integrality = np.ones(total_vars, dtype=int)
        integrality[n_y + n_z : n_y + n_z + n_T + n_w] = 0  # T and w are continuous

        lb = np.zeros(total_vars, dtype=float)
        ub = np.ones(total_vars, dtype=float)
        lb[n_y + n_z : n_y + n_z + n_T] = -50.0
        ub[n_y + n_z : n_y + n_z + n_T] = prob.t_max_c
        for r in range(n_req):
            rv = prob.req_value[r] / max(1, n_slots)
            for j in range(n_nodes):
                for t in range(n_slots):
                    ub[w_idx(r, j, t)] = rv

        # Eclipse tier forbiddance
        for j in range(n_nodes):
            for t in range(n_slots):
                if prob.is_dark[j][t]:
                    for k in range(1, n_tiers):
                        if prob.tiers[k][0] > prob.p_dark_w + 1e-9:
                            ub[z_idx(j, t, k)] = 0.0

        obj = np.zeros(total_vars, dtype=float)
        for r in range(n_req):
            for j in range(n_nodes):
                for t in range(n_slots):
                    obj[w_idx(r, j, t)] = -1.0
        for j in range(n_nodes):
            for t in range(n_slots):
                for k in range(n_tiers):
                    pw = prob.tiers[k][0]
                    obj[z_idx(j, t, k)] = prob.energy_lambda * (pw * prob.dt_s / 1000.0)
        for idx in range(n_m):
            obj[m_idx(idx)] = 1e-6  # Tiny symmetry-breaking tie-breaker against zero-sum cycles

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        c_lb: list[float] = []
        c_ub: list[float] = []
        row = 0

        def add_eq(items: list[tuple[int, float]], rhs: float) -> None:
            nonlocal row
            for col, v in items:
                rows.append(row); cols.append(col); vals.append(v)
            c_lb.append(rhs); c_ub.append(rhs); row += 1

        def add_le(items: list[tuple[int, float]], rhs: float) -> None:
            nonlocal row
            for col, v in items:
                rows.append(row); cols.append(col); vals.append(v)
            c_lb.append(-np.inf); c_ub.append(rhs); row += 1

        # 1. Each request resides on exactly one node per slot
        for r in range(n_req):
            for t in range(n_slots):
                add_eq([(y_idx(r, j, t), 1.0) for j in range(n_nodes)], 1.0)

        # 2. Tier mutual exclusion: sum_k z[j,t,k] == 1
        for j in range(n_nodes):
            for t in range(n_slots):
                add_eq([(z_idx(j, t, k), 1.0) for k in range(n_tiers)], 1.0)

        # 3. 0W sleep switch & tier batch capacity: sum_r y[r,j,t] - sum_k B_k * z[j,t,k] <= 0
        for j in range(n_nodes):
            for t in range(n_slots):
                items = [(y_idx(r, j, t), 1.0) for r in range(n_req)]
                items.extend((z_idx(j, t, k), -float(prob.tiers[k][1])) for k in range(n_tiers))
                add_le(items, 0.0)

        # 3b. Big-M-free McCormick envelope for w[r,j,t] = (val_r/T) * y[r,j,t] * (sum_k cf_k * z[j,t,k])
        for r in range(n_req):
            rv = prob.req_value[r] / max(1, n_slots)
            for j in range(n_nodes):
                for t in range(n_slots):
                    add_le([(w_idx(r, j, t), 1.0), (y_idx(r, j, t), -rv)], 0.0)
                    tier_items = [(w_idx(r, j, t), 1.0)]
                    tier_items.extend(
                        (z_idx(j, t, k), -rv * prob.tiers[k][2]) for k in range(n_tiers) if prob.tiers[k][2] > 0.0
                    )
                    add_le(tier_items, 0.0)

        # 4. 1st-order RC temperature dynamics: T[j,t] - alpha*T[j,t-1] - gain*sum_k P_k*z[j,t,k] == offset
        for j in range(n_nodes):
            for t in range(n_slots):
                items = [(t_idx(j, t), 1.0)]
                items.extend((z_idx(j, t, k), -gain * prob.tiers[k][0]) for k in range(n_tiers))
                if t == 0:
                    add_eq(items, offset + alpha * prob.t0_c[j])
                else:
                    items.append((t_idx(j, t - 1), -alpha))
                    add_eq(items, offset)

        # 5. Flow conservation + departure/arrival tightening (eliminates phantom LP circulations)
        in_by_rjt: dict[tuple[int, int, int], list[int]] = {}
        out_by_rjt: dict[tuple[int, int, int], list[int]] = {}
        edge_items: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for idx, (r, u, v, t, edges) in enumerate(mig_tuples):
            out_by_rjt.setdefault((r, u, t), []).append(idx)
            in_by_rjt.setdefault((r, v, t), []).append(idx)
            for e in edges:
                edge_items.setdefault((e, t), []).append((m_idx(idx), prob.req_kv_gb[r]))

        for r in range(n_req):
            for j in range(n_nodes):
                for t in range(n_slots):
                    in_list = in_by_rjt.get((r, j, t), ())
                    out_list = out_by_rjt.get((r, j, t), ())
                    items = [(y_idx(r, j, t), 1.0)]
                    for idx in in_list:
                        items.append((m_idx(idx), -1.0))
                    for idx in out_list:
                        items.append((m_idx(idx), 1.0))
                    if t == 0:
                        rhs = 1.0 if prob.req_init_node[r] == j else 0.0
                        add_eq(items, rhs)
                        if out_list:
                            add_le([(m_idx(idx), 1.0) for idx in out_list], rhs)
                    else:
                        items.append((y_idx(r, j, t - 1), -1.0))
                        add_eq(items, 0.0)
                        if out_list:
                            out_le = [(m_idx(idx), 1.0) for idx in out_list]
                            out_le.append((y_idx(r, j, t - 1), -1.0))
                            add_le(out_le, 0.0)
                    if in_list:
                        in_le = [(m_idx(idx), 1.0) for idx in in_list]
                        in_le.append((y_idx(r, j, t), -1.0))
                        add_le(in_le, 0.0)

        # 6. Multi-hop edge bandwidth capacity per slot
        for (e, _t), items in edge_items.items():
            add_le(items, prob.edge_cap_gbps[e] * prob.dt_s / 8.0)

        mat = coo_matrix((vals, (rows, cols)), shape=(row, total_vars)).tocsc()
        res = milp(
            c=obj,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=LinearConstraint(mat, np.array(c_lb, dtype=float), np.array(c_ub, dtype=float)),
            options={"time_limit": self.time_limit_s},
        )
        solve_ms = (time.perf_counter() - t0) * 1000.0
        timed_out = res.status == 1  # HiGHS status 1 = time/iteration limit reached

        if res.x is None:
            return SnapshotResult("120s Time-ILP", -math.inf, solve_ms, 0, 0.0, 0.0, timed_out=True)

        assign = [[prob.req_init_node[r]] * n_slots for r in range(n_req)]
        for r in range(n_req):
            for t in range(n_slots):
                best_j = max(range(n_nodes), key=lambda j: res.x[y_idx(r, j, t)])
                assign[r][t] = best_j

        tier_assign = [
            [max(range(n_tiers), key=lambda k: res.x[z_idx(j, t, k)]) for t in range(n_slots)]
            for j in range(n_nodes)
        ]

        eval_obj, viol, ekj, zw = _evaluate_node_slots(prob, assign, tier_assign=tier_assign)
        return SnapshotResult("120s Time-ILP", eval_obj, solve_ms, viol, ekj, zw, timed_out=timed_out)

