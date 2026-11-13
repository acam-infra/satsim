"""4-way snapshot benchmark: No-Plan vs 1-Hop vs discrete_rh_greedy vs 120s Time-ILP."""
from __future__ import annotations

import argparse
import random

from satsim.cosim.discrete_milp import (
    SnapshotProblem,
    SnapshotResult,
    SnapshotTimeIlpSolver,
    solve_snapshot_greedy,
    solve_snapshot_no_plan,
)


def make_snapshot(n_nodes: int, n_slots: int, n_reqs: int, seed: int = 42) -> SnapshotProblem:
    rng = random.Random(seed)
    tiers = (
        (0.0, 0, 0.0),
        (150.0, 3, 0.70),
        (300.0, 8, 1.00),
    )
    # Ring + chord topology
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_nodes)}
    edge_cap: list[float] = []

    def add_edge(u: int, v: int, cap_gbps: float) -> None:
        eid = len(edge_cap)
        edge_cap.append(cap_gbps)
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    for i in range(n_nodes):
        add_edge(i, (i + 1) % n_nodes, 10.0)
        if n_nodes >= 6 and i % 3 == 0:
            add_edge(i, (i + 3) % n_nodes, 10.0)

    is_dark = [[(j + t) % 4 == 0 for t in range(n_slots)] for j in range(n_nodes)]
    t0_c = [38.0 + (22.0 if j % 5 == 0 else rng.uniform(0.0, 10.0)) for j in range(n_nodes)]
    req_init = [rng.randrange(n_nodes) for _ in range(n_reqs)]
    req_val = [rng.uniform(0.85, 1.15) for _ in range(n_reqs)]
    req_kv = [2.0 for _ in range(n_reqs)]

    return SnapshotProblem(
        n_nodes=n_nodes,
        n_slots=n_slots,
        dt_s=15.0,
        tiers=tiers,
        t0_c=t0_c,
        is_dark=is_dark,
        p_dark_w=150.0,
        tau_s=240.0,
        r_th=0.30,
        t_amb_c=20.0,
        t_max_c=84.0,
        adj=adj,
        edge_cap_gbps=edge_cap,
        req_init_node=req_init,
        req_value=req_val,
        req_kv_gb=req_kv,
        energy_lambda=0.005,
    )


def run_benchmark(label: str, prob: SnapshotProblem, ilp_time_limit_s: float) -> list[SnapshotResult]:
    no_plan = solve_snapshot_no_plan(prob)
    one_hop = solve_snapshot_greedy(prob, max_hops=1, name="1-Hop")
    rh_greedy = solve_snapshot_greedy(prob, max_hops=3, name="discrete_rh_greedy")
    time_ilp = SnapshotTimeIlpSolver(time_limit_s=ilp_time_limit_s, max_hops=3).solve(prob)
    results = [no_plan, one_hop, rh_greedy, time_ilp]

    ilp_obj = time_ilp.objective if time_ilp.objective > 0 else 1.0
    print(f"\n=== {label} (N={prob.n_nodes}, T={prob.n_slots}, R={len(prob.req_init_node)}) ===")
    print(f"{'Method':20s} | {'Obj':>8s} | {'%ILP':>7s} | {'Time(ms)':>10s} | {'ThermViol':>9s} | {'Energy(kJ)':>10s} | {'0W%':>7s}")
    print("-" * 86)
    for r in results:
        pct = (r.objective / ilp_obj * 100.0) if ilp_obj > 0 and r.objective > -1e9 else 0.0
        to_tag = "*" if r.timed_out else " "
        print(
            f"{r.name:20s} | {r.objective:8.3f} | {pct:6.1f}% | {r.solve_ms:9.2f}{to_tag} | "
            f"{r.thermal_violations:9d} | {r.energy_kj:10.2f} | {r.zero_w_frac*100:6.1f}%"
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="4-way discrete MILP snapshot benchmark")
    ap.add_argument("--small-time-limit", type=float, default=120.0)
    ap.add_argument("--large-time-limit", type=float, default=120.0)
    args = ap.parse_args()

    small_prob = make_snapshot(n_nodes=10, n_slots=4, n_reqs=12, seed=42)
    run_benchmark("Small-Scale Snapshot", small_prob, args.small_time_limit)

    large_prob = make_snapshot(n_nodes=60, n_slots=4, n_reqs=72, seed=7)
    run_benchmark("60-Sat Large-Scale Snapshot", large_prob, args.large_time_limit)


if __name__ == "__main__":
    main()
