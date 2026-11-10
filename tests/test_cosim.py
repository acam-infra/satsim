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


def test_dispatch_baselines_rr_nearest_random():
    tr = toy_trace()
    cfg = copy.deepcopy(CFG)

    # Round-Robin alternates between reachable compute nodes 1 and 2
    env = simpy.Environment()
    w_rr = CoWorld(env, tr, cfg, make_policy({"place": "rr", "migrate": "none"}))
    r1 = CRequest(0, 0, 10.0, False, 500, 200, 1e6, 1e9)
    r2 = CRequest(1, 0, 11.0, False, 500, 200, 1e6, 1e9)
    r3 = CRequest(2, 0, 12.0, False, 500, 200, 1e6, 1e9)
    for r in (r1, r2, r3):
        w_rr.submit(r)
    env.run(until=tr.horizon)
    assert [r1.trail[0], r2.trail[0], r3.trail[0]] == [1, 2, 1]

    # Nearest always picks 1-hop node 1 over 2-hop node 2 when KV fits
    env2 = simpy.Environment()
    w_near = CoWorld(env2, tr, cfg, make_policy({"place": "nearest", "migrate": "none"}))
    n1 = CRequest(0, 0, 10.0, False, 500, 200, 1e6, 1e9)
    n2 = CRequest(1, 0, 11.0, False, 500, 200, 1e6, 1e9)
    w_near.submit(n1)
    w_near.submit(n2)
    env2.run(until=tr.horizon)
    assert [n1.trail[0], n2.trail[0]] == [1, 1]

