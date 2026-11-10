import time
import statistics as st
import sys
from pathlib import Path
import numpy as np
import simpy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

sys.path.insert(0, "scripts")
from cosim_burst_stress import profile_config
from satsim.cosim.run import load_config
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, make_requests
from satsim.cosim.record import summarize
from satsim.cosim.thermal_rebalance import TorchRebalance


class MilpVerifiedTorch(TorchRebalance):
    def __init__(self, *args, use_milp=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_milp = use_milp
        self.solve_times_ms = []
        self.var_counts = []
        self.exact_match = 0
        self.total_solves = 0

    def _solve_assignment(self, w, candidates, free_slots, edge_users, budget, now, urgent_relay):
        t0 = time.perf_counter()
        bnb_sel = super()._solve_assignment(w, candidates, free_slots, edge_users, budget, now, urgent_relay)
        t_bnb = (time.perf_counter() - t0) * 1000.0

        if not self.use_milp:
            self.solve_times_ms.append(t_bnb)
            self.var_counts.append(len(candidates))
            self.total_solves += 1
            self.exact_match += 1
            return bnb_sel

        t1 = time.perf_counter()
        m = len(candidates)
        rows, cols, vals, ub = [], [], [], []
        row_idx = 0

        # (C1) Unique destination/path per request
        by_req = {}
        for i, c in enumerate(candidates):
            by_req.setdefault(c["r"].i, []).append(i)
        for idxs in by_req.values():
            if len(idxs) > 1:
                for i in idxs:
                    rows.append(row_idx); cols.append(i); vals.append(1.0)
                ub.append(1.0); row_idx += 1

        # (C2) Source 1-to-Many fan-out capacity
        by_src = {}
        for i, c in enumerate(candidates):
            by_src.setdefault(c["src"], []).append(i)
        for src, idxs in by_src.items():
            is_urgent = w.tr.is_dark(src, now) or w.thermal_f[src] < 1.0
            cap_s = min(self.max_src_out if is_urgent else 2, len(w.nodes[src]))
            if len(idxs) > cap_s:
                for i in idxs:
                    rows.append(row_idx); cols.append(i); vals.append(1.0)
                ub.append(float(cap_s)); row_idx += 1

        # (C3) Destination thermal tipping-point & KV slot capacity
        by_dst = {}
        for i, c in enumerate(candidates):
            by_dst.setdefault(c["dst"], []).append(i)
        for dst, idxs in by_dst.items():
            dst_temp = w.thermal.temperature(dst) if w.thermal else 0.0
            if urgent_relay:
                safe_s = free_slots[dst]
            else:
                max_b = 2 if dst_temp >= 68.0 else 3
                safe_s = max(1, min(free_slots[dst], max_b - len(w.nodes[dst])))
            if len(idxs) > safe_s:
                for i in idxs:
                    rows.append(row_idx); cols.append(i); vals.append(1.0)
                ub.append(float(safe_s)); row_idx += 1

        # (C4) Shared multi-hop ISL edge bandwidth capacity (KV flow + raw-image tether flow)
        by_edge = {}
        for i, c in enumerate(candidates):
            for e in c["path"][1]:
                by_edge.setdefault(e, []).append((i, c["nbytes"]))
            for e in c.get("tether_edges", ()):
                by_edge.setdefault(e, []).append((i, c.get("tether_edge_bytes", 0.0)))
        for e, items in by_edge.items():
            if len(items) > 1:
                max_b = max(max(nb for _, nb in items), w.net.cap[e] * 15.0 / (edge_users.get(e, 0) + 1))
                for i, nb in items:
                    if nb > 0:
                        rows.append(row_idx); cols.append(i); vals.append(nb)
                ub.append(max_b); row_idx += 1

        if m > budget:
            for i in range(m):
                rows.append(row_idx); cols.append(i); vals.append(1.0)
            ub.append(float(budget)); row_idx += 1

        obj = np.array([-c["gain"] for c in candidates], dtype=float)
        if row_idx > 0:
            mat = coo_matrix((vals, (rows, cols)), shape=(row_idx, m)).tocsc()
            res = milp(
                c=obj,
                integrality=np.ones(m),
                bounds=Bounds(0, 1),
                constraints=LinearConstraint(mat, -np.inf, np.array(ub, dtype=float)),
            )
            milp_sel = [i for i in range(m) if res.x is not None and res.x[i] > 0.5]
        else:
            milp_sel = list(range(min(m, budget)))
        t_milp = (time.perf_counter() - t1) * 1000.0
        self.solve_times_ms.append(t_milp)
        self.var_counts.append(m)
        self.total_solves += 1
        g_bnb = sum(candidates[i]["gain"] for i in bnb_sel)
        g_milp = sum(candidates[i]["gain"] for i in milp_sel)
        if abs(g_bnb - g_milp) / max(1e-6, abs(g_milp)) < 0.02:
            self.exact_match += 1
        return milp_sel


base = load_config(Path("configs/cosim_placement_thermal.yaml"))
rr = base["policies"]["round_robin"]
for use_milp, tag in [(True, "HiGHS_Exact_MILP"), (False, "Fast_BnB_Solver")]:
    cfg, _ = profile_config(base, 14, "regional120", 11)
    cfg["workload"]["agentic_classes"] = {
        "p_scan": 0.35, "p_pinpoint": 0.40, "tile_MB": 250.0,
        "scan_pulls": 5, "pinpoint_pulls": 0, "phase_pulls": 4,
    }
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    env = simpy.Environment()
    pol = MilpVerifiedTorch(placement=rr, use_milp=use_milp)
    w = CoWorld(env, tr, cfg, pol, seed=11)
    for r in make_requests(cfg, tr, 11):
        w.submit(r)
    env.run(until=tr.horizon)
    s = summarize(w)
    gl = s["good_long_frac"] * 100
    migs = s["migrations"]
    mean_v = st.fmean(pol.var_counts)
    max_v = max(pol.var_counts)
    mean_ms = st.fmean(pol.solve_times_ms)
    p99_ms = sorted(pol.solve_times_ms)[int(0.99 * len(pol.solve_times_ms))]
    print(
        f"{tag:18s} | good_long={gl:.2f}% | migs={migs} | "
        f"vars: mean={mean_v:.1f}, max={max_v} | "
        f"solve_time: mean={mean_ms:.3f}ms, p99={p99_ms:.3f}ms | "
        f"opt_match={pol.exact_match}/{pol.total_solves} ({pol.exact_match/pol.total_solves*100:.1f}%)"
    )
