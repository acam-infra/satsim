"""Tests for discrete power-thermal migrators, AssignmentSolvers, and protocol-driven invariants."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from satsim.cosim import policies
from satsim.cosim.interfaces import AssignmentSolver, CandidateMove, Migrator, OperatingPoint
from satsim.cosim.run import apply_rebalance_invariants
from satsim.cosim.thermal_rebalance import ThermalForecast


def test_custom_migrator_receives_rebalance_invariants_without_whitelist(monkeypatch) -> None:
    class CustomMigrator:
        id = "custom_proto_mig"

        def __init__(self, interval_s: float = 10.0, max_inflight: int = 8, max_mig_per_req: int = 2):
            self.interval_s = interval_s
            self.max_inflight = max_inflight
            self.max_mig_per_req = max_mig_per_req

        def rebalance(self, w) -> None:
            return None

    assert isinstance(CustomMigrator(), Migrator)
    monkeypatch.setitem(policies.MIGRATORS, "custom_proto_mig", CustomMigrator)
    monkeypatch.setitem(policies.POLICIES, "custom_proto_mig", CustomMigrator)

    inv = {"interval_s": 5.0, "max_inflight": 24, "max_mig_per_req": 4}
    out_comp = apply_rebalance_invariants(
        {"rebalance_invariants": inv},
        {"place": "nearest", "migrate": "custom_proto_mig"},
    )
    assert (out_comp["interval_s"], out_comp["max_inflight"], out_comp["max_mig_per_req"]) == (5.0, 24, 4)

    out_direct = apply_rebalance_invariants(
        {"rebalance_invariants": inv},
        {"policy": "custom_proto_mig"},
    )
    assert (out_direct["interval_s"], out_direct["max_inflight"], out_direct["max_mig_per_req"]) == (5.0, 24, 4)

    with pytest.raises(ValueError, match="conflicts"):
        apply_rebalance_invariants(
            {"rebalance_invariants": inv},
            {"place": "nearest", "migrate": "custom_proto_mig", "max_inflight": 12},
        )


def test_thermal_forecast_delegates_to_power_model() -> None:
    calls = []

    class StubPowerModel:
        mode = "discrete_tiers"

        def temperature(self, node: int) -> float:
            return 42.5 + node

        def project_finish(self, node, start, batch, context, tokens, *, trace, gpu):
            calls.append((node, start, batch, context, tokens, trace, gpu))
            return (start + 17.5, 63.2)

    w = SimpleNamespace(
        power_model=StubPowerModel(),
        thermal=None,
        thermal_f=[1.0, 1.0],
        tr="stub_trace",
        gpu={"kv_gib": 24.0},
    )
    fc = ThermalForecast(w, 1)
    assert fc.temp == pytest.approx(43.5)
    fin = fc.finish(10.0, 3, 12000.0, 500.0)
    assert fin == pytest.approx(27.5)
    assert fc.temp == pytest.approx(63.2)
    assert calls == [(1, 10.0, 3, 12000.0, 500.0, "stub_trace", {"kv_gib": 24.0})]


def test_solvers_satisfy_assignment_solver_protocol_and_constraints() -> None:
    from satsim.cosim.discrete_milp import IncrementalGreedySolver, StepMilpSolver

    greedy = IncrementalGreedySolver()
    milp_solver = StepMilpSolver()
    assert isinstance(greedy, AssignmentSolver)
    assert isinstance(milp_solver, AssignmentSolver)

    r1 = SimpleNamespace(i=1)
    r2 = SimpleNamespace(i=2)
    w = SimpleNamespace(
        nodes={1: [r1, r2], 2: [], 3: []},
        kv_cap=24.0,
        kv_used=lambda j: 0.0,
        net=SimpleNamespace(cap={0: 10.0, 1: 10.0}),
    )
    cands = [
        CandidateMove(req_id=1, req=r1, src=1, dst=2, path=([1, 2], [0]), nbytes=1.0, delay_s=0.1, gain=0.8),
        CandidateMove(req_id=1, req=r1, src=1, dst=3, path=([1, 3], [1]), nbytes=1.0, delay_s=0.1, gain=0.4),
        CandidateMove(req_id=2, req=r2, src=1, dst=3, path=([1, 3], [1]), nbytes=1.0, delay_s=0.1, gain=0.6),
    ]
    for solver in (greedy, milp_solver):
        sel = solver.solve(w, cands, budget=2, now=0.0)
        assert [c.req_id for c in sel] == [1, 2]
        assert {(c.req_id, c.dst) for c in sel} == {(1, 2), (2, 3)}

        sel_b1 = solver.solve(w, cands, budget=1, now=0.0)
        assert len(sel_b1) == 1
        assert sel_b1[0].dst == 2


def test_discrete_migrator_prefers_0w_drain_and_avoids_waking_idle() -> None:
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator
    from satsim.cosim.trace import Trace
    from satsim.cosim.world import CRequest, CoWorld
    import simpy

    tr = Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "drain-0w",
        "horizon_s": 1500.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0, 1500]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 1500]]},
            {"u": 1, "v": 3, "kind": "intra", "up": [[0, 1500]]},
        ],
        "dark": [[], [], [], []],
    })
    cfg = {
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7,
                "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
        "power": {
            "mode": "discrete_tiers",
            "P_idle": 0,
            "P_max": 300,
            "gamma": 2.4,
            "f_min": 0.35,
            "P_dark": 150,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
                {"name": "low", "power_w": 150.0, "max_batch": 4, "clock_f": 0.85},
                {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0},
            ],
        },
        "thermal": {
            "tau_s": 240.0, "R_th": 0.30, "T_amb_C": 20.0, "T0_C": 38.0, "T_max_C": 84.0,
            "T_throttle_C": 84.0, "T_release_C": 76.0, "f_throttle": 0.55,
            "P_busy": 300.0, "busy_util_floor": 0.40, "batch_scale": 4.0,
        },
        "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
        "migration": {"timeout_s": 120.0, "switch_s": 0.02},
        "workload": {"arrival_window_s": 100.0, "rate_per_rs_per_min": 0, "long_out": [18000, 24000],
                     "short_out": [200, 800], "long_pin": 4000, "short_pin": 500},
        "slo": {"long_deadline_s": 900.0, "short_deadline_s": 90.0, "max_stall_frac": 0.1},
        "delivery": "final",
        "control_period_s": 2.0,
    }
    env = simpy.Environment()
    pol = policies.make_policy({"place": "nearest", "migrate": "discrete_rh_greedy", "interval_s": 0.0})
    w = CoWorld(env, tr, cfg, pol, seed=1)

    # Node 1 has 1 single-tail long request; Node 2 is active cool (1 request, low tier max_batch=4);
    # Node 3 is completely idle (0W).
    r_src = CRequest(i=1, origin=0, t=0.0, pin=4000, pout=20000, long=True, deadline=900.0, upload_bytes=1e6)
    r_src.state = "run"
    r_src.node = 1
    r_dst = CRequest(i=2, origin=0, t=0.0, pin=4000, pout=20000, long=True, deadline=900.0, upload_bytes=1e6)
    r_dst.state = "run"
    r_dst.node = 2
    w.nodes[1].append(r_src)
    w.nodes[2].append(r_dst)

    assert isinstance(pol.migrator, DiscreteThermalMigrator)
    cands = pol.migrator._collect_candidates(w, urgent_relay=False)
    by_dst = {c.dst: c for c in cands if c.req_id == r_src.i}

    # Consolidating into active cool node 2 empties node 1 to 0W and saves 150W;
    # waking idle node 3 gives no power saving and is penalized.
    assert 2 in by_dst
    assert by_dst[2].empties_src_to_0w is True
    assert by_dst[2].wakes_dst_from_0w is False
    assert by_dst[2].delta_power_w < 0.0
    if 3 in by_dst:
        assert by_dst[2].gain > by_dst[3].gain


def test_discrete_migrator_blocks_3hop_from_starving_1hop_under_congestion() -> None:
    from satsim.cosim.trace import Trace
    from satsim.cosim.world import CRequest, CoWorld
    import simpy

    # Linear chain 1 - 2 - 3 - 4, where edge (1,2) already has an active flow (congested).
    # Node 1 has a request that could go 1->2->3->4 (3-hop) ONLY if relay edges were uncongested,
    # whereas 1-hop node 2 is available or 3-hop must be blocked when relay edge (1,2) is busy.
    tr = Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "hop-congestion",
        "horizon_s": 300.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
            {"id": 4, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0, 300]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 300]]},
            {"u": 2, "v": 3, "kind": "intra", "up": [[0, 300]]},
            {"u": 3, "v": 4, "kind": "intra", "up": [[0, 300]]},
        ],
        "dark": [[], [[0, 300]], [[0, 300]], [[0, 300]], []],  # 1,2,3 dark; 4 sunlit (3 hops away from 1)
    })
    cfg = {
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7,
                "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
        "power": {
            "mode": "discrete_tiers",
            "P_idle": 0,
            "P_max": 300,
            "gamma": 2.4,
            "f_min": 0.35,
            "P_dark": 150,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
                {"name": "low", "power_w": 150.0, "max_batch": 4, "clock_f": 0.65},
                {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0},
            ],
        },
        "thermal": {
            "tau_s": 240.0, "R_th": 0.30, "T_amb_C": 20.0, "T0_C": 38.0, "T_max_C": 84.0,
            "T_throttle_C": 84.0, "T_release_C": 76.0, "f_throttle": 0.55,
            "P_busy": 300.0, "busy_util_floor": 0.40, "batch_scale": 4.0,
        },
        "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
        "migration": {"timeout_s": 120.0, "switch_s": 0.02},
        "workload": {"arrival_window_s": 100.0, "rate_per_rs_per_min": 0, "long_out": [18000, 24000],
                     "short_out": [200, 800], "long_pin": 4000, "short_pin": 500},
        "slo": {"long_deadline_s": 900.0, "short_deadline_s": 90.0, "max_stall_frac": 0.1},
        "delivery": "final",
        "control_period_s": 2.0,
    }
    env = simpy.Environment()
    pol = policies.make_policy({"place": "nearest", "migrate": "discrete_rh_greedy", "interval_s": 0.0})
    w = CoWorld(env, tr, cfg, pol, seed=1)

    r1 = CRequest(i=1, origin=0, t=0.0, pin=4000, pout=20000, long=True, deadline=900.0, upload_bytes=1e6)
    r1.state = "run"
    r1.node = 1
    w.nodes[1].append(r1)

    # Simulate congested relay edges (1,2) and (2,3) by injecting active flows on edge 1 and 2
    w.net.flows.append(SimpleNamespace(hops=[(1, 2, 1), (2, 3, 2)]))
    w.net.flows.append(SimpleNamespace(hops=[(1, 2, 1), (2, 3, 2)]))

    cands = pol.migrator._collect_candidates(w, urgent_relay=False)
    # 3-hop path 1->2->3->4 must be blocked because relay edges are congested
    assert all(len(c.path[1]) < 3 for c in cands)


def test_discrete_migrator_hard_rejects_over_tmax_and_registers_step_milp() -> None:
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator
    from satsim.cosim.trace import Trace
    from satsim.cosim.world import CRequest, CoWorld
    import simpy

    pol_milp = policies.make_policy({"place": "greedy", "migrate": "discrete_step_milp"})
    assert isinstance(pol_milp.migrator, DiscreteThermalMigrator)
    assert pol_milp.migrator.id == "discrete_step_milp"

    pol_standalone = policies.make_policy({"policy": "discrete_rh_greedy", "placement": {"policy": "greedy"}})
    assert isinstance(pol_standalone, DiscreteThermalMigrator)
    assert pol_standalone.id == "discrete_rh_greedy"

    tr = Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "tmax-cap",
        "horizon_s": 1500.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0, 1500]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 1500]]},
        ],
        "dark": [[], [[0, 1500]], []],
    })
    cfg = {
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7,
                "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
        "power": {
            "mode": "discrete_tiers",
            "P_idle": 0,
            "P_max": 300,
            "gamma": 2.4,
            "f_min": 0.35,
            "P_dark": 150,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
                {"name": "low", "power_w": 150.0, "max_batch": 4, "clock_f": 0.65},
                {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0},
            ],
        },
        "thermal": {
            "tau_s": 240.0, "R_th": 0.30, "T_amb_C": 20.0, "T0_C": 38.0, "T_max_C": 84.0,
            "T_throttle_C": 84.0, "T_release_C": 76.0, "f_throttle": 0.55,
            "P_busy": 300.0, "busy_util_floor": 0.40, "batch_scale": 4.0,
        },
        "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
        "migration": {"timeout_s": 120.0, "switch_s": 0.02},
        "workload": {"arrival_window_s": 100.0, "rate_per_rs_per_min": 0, "long_out": [18000, 24000],
                     "short_out": [200, 800], "long_pin": 4000, "short_pin": 500},
        "slo": {"long_deadline_s": 900.0, "short_deadline_s": 90.0, "max_stall_frac": 0.1},
        "delivery": "final",
        "control_period_s": 2.0,
    }
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, pol_standalone, seed=1)
    # Set dst node 2 right above T_max_C=84.0 via PowerThermalModel seam so it is hard-rejected
    w.power_model.set_temperature(2, 84.5)
    r1 = CRequest(i=1, origin=0, t=0.0, pin=4000, pout=20000, long=True, deadline=900.0, upload_bytes=1e6)
    r1.state = "run"
    r1.node = 1
    w.nodes[1].append(r1)
    cands = pol_standalone._collect_candidates(w, urgent_relay=False)
    assert all(c.dst != 2 for c in cands)


def test_overheated_1hop_does_not_block_uncongested_2hop_escape_and_run_aggregates_dicts() -> None:
    from satsim.cosim.run import run as cosim_run
    from satsim.cosim.trace import Trace
    from satsim.cosim.world import CRequest, CoWorld
    from pathlib import Path
    import json
    import simpy
    import tempfile

    # Chain 1 - 2 - 3: node 1 dark, 1-hop node 2 sunlit but overheated (84.5C > T_max=84C),
    # 2-hop node 3 sunlit and cool (38C). Overheated 1-hop must NOT block 2-hop escape to node 3.
    tr_dict = {
        "schema": "orbitsim.contact_trace/v1",
        "name": "escape-2hop",
        "horizon_s": 1500.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0, 1500]]},
            {"u": 0, "v": 3, "kind": "rs", "up": [[0, 1500]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 1500]]},
            {"u": 2, "v": 3, "kind": "intra", "up": [[0, 1500]]},
        ],
        "dark": [[], [[0, 1500]], [], []],
    }
    tr = Trace(tr_dict)
    cfg = {
        "name": "mini_discrete_run",
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7,
                "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
        "power": {
            "mode": "discrete_tiers",
            "P_dark": 150,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
                {"name": "low", "power_w": 150.0, "max_batch": 4, "clock_f": 0.65},
                {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0},
            ],
        },
        "thermal": {
            "tau_s": 240.0, "R_th": 0.30, "T_amb_C": 20.0, "T0_C": 38.0, "T_max_C": 84.0,
            "thermal_horizon_s": 30.0,
        },
        "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
        "migration": {"timeout_s": 120.0, "switch_s": 0.02},
        "workload": {
            "arrival_window_s": 60.0, "rate_per_rs_per_min": 2.0, "p_long": 0.5,
            "long_out": [18000, 24000], "short_out": [200, 400],
            "long_pin": 4000, "short_pin": 300, "long_upload_MB": 20, "short_upload_MB": 5,
        },
        "slo": {"deadline_factor_long": 2.0, "deadline_factor_short": 4.0, "min_deadline_s": 60.0, "max_stall_frac": 0.1},
        "policy": {"stall_weight": 1.0},
        "delivery": "final",
        "control_period_s": 2.0,
        "policies": {"discrete_rh_greedy": {"place": "nearest", "migrate": "discrete_rh_greedy", "interval_s": 5.0}},
        "seeds": [1, 2],
    }
    env = simpy.Environment()
    pol = policies.make_policy({"place": "nearest", "migrate": "discrete_rh_greedy", "interval_s": 0.0})
    w = CoWorld(env, tr, cfg, pol, seed=1)
    w.power_model.set_temperature(2, 84.5)
    r1 = CRequest(i=1, origin=0, t=0.0, long=True, pin=4000, pout=20000, upload_bytes=1e6, deadline=380.0)
    r1.state = "run"
    r1.node = 1
    w.nodes[1].append(r1)
    cands = pol.migrator._collect_candidates(w, urgent_relay=False)
    assert [c.dst for c in cands] == [3]

    # Also verify satsim.cosim.run.run() cleanly aggregates dict-valued power_tier_fracs across seeds
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td) / "t.json"
        tp.write_text(json.dumps(tr_dict), encoding="utf-8")
        cfg["_dir"] = td
        cfg["trace"] = "t.json"
        payload = cosim_run(cfg, quiet=True)
        mean_fracs = payload["mean"]["discrete_rh_greedy"]["power_tier_fracs"]
        assert isinstance(mean_fracs, dict)
        assert abs(sum(mean_fracs.values()) - 1.0) < 1e-6


def test_snapshot_time_ilp_and_rh_greedy_on_small_snapshot() -> None:
    from satsim.cosim.discrete_milp import (
        SnapshotProblem,
        SnapshotTimeIlpSolver,
        solve_snapshot_greedy,
        solve_snapshot_no_plan,
    )

    prob = SnapshotProblem(
        n_nodes=4,
        n_slots=3,
        dt_s=15.0,
        tiers=((0.0, 0, 0.0), (150.0, 2, 0.70), (300.0, 6, 1.00)),
        t0_c=[40.0, 40.0, 40.0, 40.0],
        is_dark=[[True, True, True], [False, False, False], [False, False, False], [False, False, False]],
        p_dark_w=150.0,
        tau_s=240.0,
        r_th=0.30,
        t_amb_c=20.0,
        t_max_c=84.0,
        adj={0: [(1, 0)], 1: [(0, 0), (2, 1)], 2: [(1, 1), (3, 2)], 3: [(2, 2)]},
        edge_cap_gbps=[10.0, 10.0, 10.0],
        req_init_node=[0, 2, 3],
        req_value=[1.0, 1.0, 1.0],
        req_kv_gb=[1.0, 1.0, 1.0],
        energy_lambda=0.005,
    )
    no_plan = solve_snapshot_no_plan(prob)
    one_hop = solve_snapshot_greedy(prob, max_hops=1, name="1-Hop")
    rh = solve_snapshot_greedy(prob, max_hops=3, name="discrete_rh_greedy")
    ilp = SnapshotTimeIlpSolver(time_limit_s=10.0, max_hops=3).solve(prob)

    assert rh.thermal_violations == 0
    assert ilp.thermal_violations == 0
    assert rh.objective >= one_hop.objective - 1e-6
    assert rh.objective >= no_plan.objective
    assert rh.zero_w_frac >= no_plan.zero_w_frac


def test_discrete_rh_tier_v4_eclipse_lookahead():
    """discrete_rh_tier_v4 decouples low_tier_pack, uses orbit_load dvfs placement, and pre-sprints before eclipse."""
    from satsim.cosim.policies import make_policy

    pol = make_policy({"policy": "discrete_rh_tier_v4"})
    assert pol.id == "discrete_rh_tier_v4"
    assert pol.explicit_tier is True
    assert pol.low_tier_pack is False
    assert pol.eclipse_lookahead_s == 30.0
    assert pol.tier_slack_ratio == 0.45
    assert pol.placement.id == "orbit_thermal"


def test_discrete_rh_tier_v4_pre_eclipse_sprint_and_dynamic_urgent():
    """approaching_dark with thermal headroom keeps high tier (None); dynamic slack classifies urgent reqs."""
    from types import SimpleNamespace
    from satsim.cosim.policies import make_policy

    pol = make_policy({"policy": "discrete_rh_tier_v4"})
    spec = SimpleNamespace(
        p_dark_w=80.0,
        t_max_c=80.0,
        work_tiers=[
            SimpleNamespace(index=1, power_w=75.0, max_batch=4, clock_f=0.50),
            SimpleNamespace(index=2, power_w=300.0, max_batch=12, clock_f=1.00),
        ],
    )
    pol._spec = lambda _w: spec  # type: ignore[method-assign]
    pm = SimpleNamespace(
        set_explicit_tier=lambda *_a, **_kw: None,
        temperature=lambda _n: 50.0,
        _rc_probe_coeffs=(0.98, 0.4, 0.005),
    )
    req_loose = SimpleNamespace(pin=500, gen=0, remaining=200, deadline=500.0)
    w = SimpleNamespace(
        env=SimpleNamespace(now=100.0),
        power_model=pm,
        nodes={1: [req_loose]},
        tr=SimpleNamespace(is_dark=lambda _n, t: t >= 120.0),
        gpu={"t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7, "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
        cfg={},
    )
    assert pol._select_explicit_tier_for_node(w, 1) is None
    req_urg = SimpleNamespace(pin=4000, gen=0, remaining=20000, deadline=110.0)
    assert pol._is_urgent(req_urg, w, 100.0) is True
    assert pol._is_urgent(req_loose, w, 100.0) is False
