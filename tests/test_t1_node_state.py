"""T1: node-state helpers use the eclipse / hot / cool terms."""
import copy
import math

import pytest
import simpy

from satsim.cosim.policies import make_policy
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld


def toy_trace() -> Trace:
    return Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "t1-toy",
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

THERMAL = dict(
    tau_s=60.0, R_th=0.30, T_amb_C=20.0, T0_C=38.0, T_throttle_C=84.0,
    T_release_C=76.0, f_throttle=0.55, P_busy=300.0, busy_util_floor=0.40, batch_scale=4.0,
)


def _world(thermal: bool = True) -> CoWorld:
    cfg = copy.deepcopy(CFG)
    if thermal:
        cfg["thermal"] = THERMAL
    env = simpy.Environment(initial_time=150)
    return CoWorld(env, toy_trace(), cfg, make_policy({"policy": "greedy"}))


def test_eclipse_hot_and_cool_states():
    w = _world()
    # node 1 is in eclipse at t=150 and is made thermally throttled by the fixture.
    w.thermal_f[1] = 0.55
    w.thermal._T[1] = 90.0
    # node 2 is sunlit and has plenty of thermal headroom.
    w.thermal_f[2] = 1.0
    w.thermal._T[2] = 50.0

    assert w.is_eclipse(1, 150)
    assert not w.is_eclipse(2, 150)
    assert w.is_hot(1)
    assert not w.is_hot(2)
    assert w.thermal_headroom_C(1) == pytest.approx(-6.0)
    assert w.thermal_headroom_C(2) == pytest.approx(34.0)
    assert not w.is_cool(1, 10)
    assert w.is_cool(2, 10)


def test_cool_requires_both_not_hot_and_headroom():
    w = _world()
    w.thermal_f[1] = 0.55
    w.thermal._T[1] = 40.0
    assert w.is_hot(1)  # throttled even though cool in absolute temperature
    assert not w.is_cool(1, 10)

    w.thermal_f[1] = 1.0
    w.thermal._T[1] = 75.0
    assert w.is_hot(1) is False
    assert w.thermal_headroom_C(1) == pytest.approx(9.0)
    assert not w.is_cool(1, 10)
    assert w.is_cool(1, 9.0)


def test_no_thermal_model_has_infinite_headroom():
    w = _world(thermal=False)
    assert w.thermal_headroom_C(1) == math.inf
    assert not w.is_hot(1)
    assert w.is_cool(1, 10)
