"""Co-simulation: trace lookups, fluid network over dying links, and policies on a toy orbit."""
import copy

import pytest
import simpy

from satsim.cosim.net import FluidNet
from satsim.cosim.policies import make_policy
from satsim.cosim.trace import INF, Trace
from satsim.cosim.world import CoWorld, CRequest, deadline_for


def toy_trace() -> Trace:
    """rs0 — a(1) — b(2): the rs link is up [0, 300) and [600, 1000); a—b always up.
    a is in the umbra [100, 700), b never is."""
    return Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "toy",
        "horizon_s": 1000.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0, 300], [600, 1000]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 1000]]},
        ],
        "dark": [[], [[100, 700]], []],
    })


CFG = {
    "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
    "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7,
            "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
    "power": {"P_idle": 60, "P_max": 300, "gamma": 2.4, "f_min": 0.35, "P_dark": 80},
    "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
    "migration": {"timeout_s": 120.0, "switch_s": 0.02},
    "workload": {"arrival_window_s": 1000},
    "slo": {"deadline_factor_long": 3.0, "deadline_factor_short": 4.0, "min_deadline_s": 60,
            "max_stall_frac": 0.1},
    "policy": {"stall_weight": 1.0},
    "delivery": "final",
    "control_period_s": 2.0,
}


def test_trace_reachability_and_power():
    tr = toy_trace()
    assert tr.connected(0, 2, 50) and not tr.connected(0, 2, 400)
    assert tr.next_conn(0, 2, 400) == 600
    assert tr.conn_until(0, 1, 50) == 300
    assert tr.conn_until(0, 1, 700) == INF  # runs to the horizon
    assert tr.disconnected_time(0, 1, 200, 700) == pytest.approx(300)
    assert tr.is_dark(1, 100) and not tr.is_dark(1, 700)
    assert list(tr.power_segments(1, 50))[:2] == [(50, 100, False), (100, 700, True)]
    assert tr.path(0, 2, 50) == ([0, 1, 2], [0, 1])
    assert tr.path(0, 2, 400) is None
    assert tr.change_times() == [100.0, 300.0, 600.0, 700.0]


def test_rs_is_never_a_relay():
    tr = Trace({
        "schema": "orbitsim.contact_trace/v1", "name": "relay", "horizon_s": 10.0,
        "nodes": [{"id": 0, "role": "compute"}, {"id": 1, "role": "rs"}, {"id": 2, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0, 10]]}, {"u": 1, "v": 2, "kind": "rs", "up": [[0, 10]]}],
        "dark": [[], [], []],
    })
    assert tr.path(0, 2, 1) is None
    assert tr.path(0, 1, 1) is not None


def test_fluid_net_shares_a_link_and_aborts_on_cut():
    env = simpy.Environment()
    net = FluidNet(env, [8e9 / 8, 8e9 / 8])  # 8 Gbps -> 1 GB/s each
    f1 = net.add("a", [0, 1], [0], 1e9)
    f2 = net.add("b", [0, 1, 2], [0, 1], 1e9)
    env.run(until=1.0)
    assert f1.rate == pytest.approx(0.5e9) and f2.rate == pytest.approx(0.5e9)
    env.run(until=2.0 + 1e-6)
    assert f1.done.triggered and f2.done.triggered and not f1.failed
    f3 = net.add("c", [1, 2], [1], 1e9)
    env.run(until=2.5)
    dead = net.abort_dead(lambda e: e != 1)
    assert dead == [f3] and f3.failed and f3.done.triggered


def _one_long_request(policy: str, *, t: float = 30.0, pout: int = 8000, factor: float = 3.0):
    tr = toy_trace()
    env = simpy.Environment()
    cfg = copy.deepcopy(CFG)
    cfg["slo"]["deadline_factor_long"] = factor
    w = CoWorld(env, tr, cfg, make_policy({"policy": policy}))
    r = CRequest(0, 0, t, True, 2000, pout, 50e6, deadline_for(cfg, t, 2000, pout, True))
    w.submit(r)
    env.run(until=tr.horizon)
    return w, r


def test_greedy_decodes_in_the_dark_and_misses_delivery():
    # a (node 1) is the only node one hop away at t=30 and as fast as b; greedy ties to the lower id.
    # 16k tokens: ~70 s in sunlight, the rest at the umbra clock -> decode ends inside the rs gap
    w, r = _one_long_request("greedy", pout=16000, factor=1.5)
    assert r.trail == [1]
    assert r.dark_tok > 0
    assert 300 < r.t_decoded < 600
    assert r.t_delivered is not None and r.t_delivered >= 600  # waited for the rs link to return
    assert not w.good(r)


def test_contact_aware_avoids_the_umbra():
    w, r = _one_long_request("contact_aware", pout=16000, factor=1.5)
    # placed on b (never dark) or moved there before the umbra; either way little dark decoding
    assert r.trail[-1] == 2
    assert r.dark_tok < 100
    g, rg = _one_long_request("greedy", pout=16000, factor=1.5)
    assert r.t_decoded < rg.t_decoded
    assert w.good(r) and not g.good(rg)


def test_reactive_migrates_when_its_node_enters_the_umbra():
    w, r = _one_long_request("reactive")
    assert r.trail == [1, 2]
    assert w.acc["mig"] == 1 and w.acc["mig_bytes"] > 0


def test_migration_cut_by_link_drop_is_counted():
    tr = Trace({
        "schema": "orbitsim.contact_trace/v1", "name": "cut", "horizon_s": 500.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [{"u": 0, "v": 1, "kind": "rs", "up": [[0, 500]]},
                  {"u": 1, "v": 2, "kind": "intra", "up": [[0, 100.2]]}],
        "dark": [[], [[100, 500]], []],
    })
    cfg = copy.deepcopy(CFG)
    cfg["net"]["link_gbps"]["intra"] = 0.1  # 1.3 GB of KV needs ~100 s: the link dies first
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, make_policy({"policy": "reactive"}))
    r = CRequest(0, 0, 1.0, True, 8000, 16000, 1e6, 1e9)
    w.submit(r)
    env.run(until=tr.horizon)
    assert w.acc["mig"] == 0
    assert r.mig_abort >= 1
    assert r.trail == [1]


def test_runs_are_deterministic(tmp_path):
    a, ra = _one_long_request("contact_aware")
    b, rb = _one_long_request("contact_aware")
    assert (ra.t_delivered, ra.trail) == (rb.t_delivered, rb.trail)


def test_placement_baseline_variants_keep_their_distinct_behavior():
    env = simpy.Environment(initial_time=120)
    world = CoWorld(env, toy_trace(), CFG, make_policy({"policy": "greedy"}))
    request = CRequest(0, 0, 120, False, 500, 1000, 1e6, 210)
    assert make_policy({"policy": "round_robin"}).place(world, request) == 2
    assert make_policy({"policy": "round_robin_all"}).place(world, request) == 1

    world.nodes[1].append(CRequest(1, 0, 0, True, 4000, 20000, 1e6, 900))
    world.nodes[2].extend(CRequest(i, 0, 0, False, 500, 1000, 1e6, 90) for i in (2, 3))
    assert make_policy({"policy": "least_loaded"}).place(world, request) == 1
    assert make_policy({"policy": "least_reserved_kv"}).place(world, request) == 2


def test_buffered_release_preserves_each_job_and_deadline_budget():
    from satsim.cosim.world import make_requests

    cfg = copy.deepcopy(CFG)
    cfg["workload"].update(rate_per_rs_per_min=8, p_long=0.5, long_out=[2000, 4000], short_out=[100, 200],
                           long_pin=400, short_pin=50, long_upload_MB=100, short_upload_MB=10,
                           reserve_output_budget=True)
    tr = toy_trace()
    smooth = make_requests(cfg, tr, 5)
    cfg["workload"]["release_burst"] = {"period_s": 200, "duration_s": 40, "source_phase_s": 0}
    buffered = make_requests(cfg, tr, 5)
    assert smooth and len(smooth) == len(buffered)
    for a, b in zip(smooth, buffered):
        assert (a.origin, a.pin, a.pout, a.upload_bytes) == (b.origin, b.pin, b.pout, b.upload_bytes)
        assert b.t % 200 < 40
        assert b.deadline - b.t == pytest.approx(a.deadline - a.t)
        assert b.reserved_tokens == b.pin + (4000 if b.long else 200)


def test_sampled_thermal_feedback_slows_decode_without_umbra_power_cap():
    cfg = copy.deepcopy(CFG)
    cfg["power"]["P_dark"] = cfg["power"]["P_max"]
    thermal = dict(tau_s=8.0, R_th=0.50, T_amb_C=20.0, T0_C=50.0, T_throttle_C=84.0,
                   T_release_C=76.0, f_throttle=0.55, P_busy=300.0, busy_util_floor=0.40, batch_scale=4.0)
    completion = []
    for use_heat in (False, True):
        if use_heat:
            cfg["thermal"] = thermal
        env = simpy.Environment()
        w = CoWorld(env, toy_trace(), cfg, make_policy({"policy": "greedy"}))
        r = CRequest(0, 0, 0, True, 1000, 12000, 0, 300)
        w.submit(r)
        env.run(until=300)
        completion.append(r.t_delivered)
        assert r.trail == [1]
        if use_heat:
            assert w.thermal.diagnostics()["throttle_time_s"][1] > 0
    assert completion[1] > completion[0]


def test_torch_rebalance_migrates_out_of_umbra_and_supports_agentic_tether():
    cfg = copy.deepcopy(CFG)
    cfg["workload"].update(
        long_out=[8000, 16000],
        short_out=[200, 1000],
        long_pin=2000,
        short_pin=500,
    )
    cfg["thermal"] = dict(
        tau_s=60.0, R_th=0.30, T_amb_C=20.0, T0_C=38.0, T_throttle_C=84.0,
        T_release_C=76.0, f_throttle=0.55, P_busy=300.0, busy_util_floor=0.40, batch_scale=4.0,
    )
    tr = toy_trace()
    env = simpy.Environment()
    pol = make_policy({"policy": "torch_rebalance", "placement": {"policy": "greedy"}})
    w = CoWorld(env, tr, cfg, pol)
    r = CRequest(
        0, 0, 30.0, True, 2000, 12000, 20e6, 550.0,
        output_budget=16000, task_class="pinpoint_deep",
    )
    w.submit(r)
    env.run(until=tr.horizon)
    assert r.trail == [1, 2]
    assert w.good(r)

    # Verify agentic raw-image tether pull accounting when migrated away from anchor_node
    env2 = simpy.Environment()
    pol2 = make_policy({"policy": "reactive"})
    w2 = CoWorld(env2, tr, cfg, pol2)
    r2 = CRequest(
        0, 0, 30.0, True, 2000, 8000, 20e6, 900.0,
        output_budget=16000, task_class="scan_high_reaccess",
        reaccess_total=3, reaccess_tile_bytes=10e6,
    )
    w2.submit(r2)
    env2.run(until=tr.horizon)
    assert r2.anchor_node == 1
    assert r2.node == 2
    assert r2.reaccess_done == 3
    assert r2.tether_bytes > 0 and r2.tether_s > 0
