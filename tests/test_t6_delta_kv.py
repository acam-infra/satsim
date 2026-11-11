"""T6: tail KV produced while the main migration stream is in flight."""
from __future__ import annotations

import copy

import simpy

from satsim.cosim.policies import make_policy
from satsim.cosim.record import summarize
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, CRequest


def toy_trace() -> Trace:
    """rs0 - a(1) - b(2); a enters eclipse at 100 s, b is always sunlit."""
    return Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "t6-toy",
        "horizon_s": 500.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0, 500]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 500]]},
        ],
        "dark": [[], [[100, 500]], []],
    })


CFG = {
    "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
    "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7,
            "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
    "power": {"P_idle": 60, "P_max": 300, "gamma": 2.4, "f_min": 0.35, "P_dark": 80},
    "net": {"link_gbps": {"intra": 0.3, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
    "migration": {"timeout_s": 120.0, "switch_s": 0.02},
    "workload": {"arrival_window_s": 1000},
    "slo": {"deadline_factor_long": 3.0, "deadline_factor_short": 4.0, "min_deadline_s": 60,
            "max_stall_frac": 0.1},
    "policy": {"stall_weight": 1.0},
    "delivery": "final",
    "control_period_s": 2.0,
}


def _run(delta_kv: bool | None):
    cfg = copy.deepcopy(CFG)
    if delta_kv is not None:
        cfg["migration"]["delta_kv"] = delta_kv
    env = simpy.Environment()
    w = CoWorld(env, toy_trace(), cfg, make_policy({"policy": "reactive"}))
    r = CRequest(0, 0, 1.0, True, 8000, 16000, 1e6, 1e9)
    w.submit(r)
    env.run(until=w.tr.horizon)
    return w, r, summarize(w)


def test_default_is_off_and_identical_to_explicit_false():
    default, r0, s0 = _run(None)
    explicit, r1, s1 = _run(False)
    assert default.delta_kv is False and explicit.delta_kv is False
    assert s0["delta_kv_GB"] == 0.0 and s1["delta_kv_GB"] == 0.0
    assert s0["mig_GB"] == s1["mig_GB"]
    assert s0["migrations"] == 1
    assert r0.trail == r1.trail == [1, 2]


def test_low_bandwidth_delta_kv_adds_tail_bytes_and_increases_mig_GB():
    off, _, s_off = _run(False)
    on, _, s_on = _run(True)
    assert on.delta_kv is True
    assert s_off["migrations"] == s_on["migrations"] == 1
    assert s_on["delta_kv_GB"] > 0.0
    assert s_on["mig_GB"] > s_off["mig_GB"]
    assert s_on["mig_GB"] == on.acc["mig_bytes"] / 1e9
    assert on.acc["mig_bytes"] > off.acc["mig_bytes"]
    assert on.acc["delta_kv_bytes"] > 0.0
    assert s_on["mig_abort"] == 0


def test_summarize_reports_zero_without_the_feature():
    _, _, s = _run(False)
    assert s["delta_kv_GB"] == 0.0
    assert "delta_kv_GB" in s
