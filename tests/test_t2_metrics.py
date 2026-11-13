"""T2: badput / waste / all-request P95 / throttle-time metrics."""
import copy

import pytest
import simpy

from satsim.cosim.policies import make_policy
from satsim.cosim.record import summarize
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, CRequest


def toy_trace() -> Trace:
    return Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "t2-toy",
        "horizon_s": 1000.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0, 1000]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 1000]]},
        ],
        "dark": [[], [], []],
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


def _world(thermal: bool = False) -> CoWorld:
    cfg = copy.deepcopy(CFG)
    if thermal:
        cfg["thermal"] = THERMAL
    env = simpy.Environment()
    return CoWorld(env, toy_trace(), cfg, make_policy({"policy": "greedy"}))


def test_badput_waste_all_p95_and_throttle_fields():
    w = _world()

    on_time = CRequest(0, 0, 10.0, True, 100, 100, 1e6, 200.0)
    on_time.gen = 100.0
    on_time.t_decoded = 90.0
    on_time.t_delivered = 100.0

    late = CRequest(1, 0, 20.0, True, 100, 100, 1e6, 100.0)
    late.gen = 80.0
    late.t_decoded = 130.0
    late.t_delivered = 150.0

    undelivered = CRequest(2, 0, 30.0, False, 100, 100, 1e6, 500.0)
    undelivered.gen = 40.0

    w.reqs = [on_time, late, undelivered]
    w.acc["throttle_s"] = 12.5
    s = summarize(w)

    assert s["badput_tok_s"] == pytest.approx((80.0 + 40.0) / 1000.0)
    assert s["waste_ratio"] == pytest.approx(120.0 / (100.0 + 120.0))
    assert s["undelivered_frac"] == pytest.approx(1.0 / 3.0)
    # all: 90, 130, horizon - arrival = 970 -> p95 = 886; long only: 90, 130 -> p95 = 128.
    assert s["e2e_p95_all_s"] == pytest.approx(886.0)
    assert s["long_p95_all_s"] == pytest.approx(128.0)
    assert s["throttle_s"] == pytest.approx(12.5)


def test_throttle_time_accumulates_per_control_tick():
    w = _world(thermal=True)
    w.thermal._throttled[1] = True
    w.thermal._f[1] = w.thermal.f_throttle
    w.thermal_f[1] = w.thermal.f_throttle
    w._thermal_tick()
    assert w.acc["throttle_s"] == pytest.approx(w.period)
    assert summarize(w)["throttle_s"] == pytest.approx(w.period)


def test_throttle_time_is_zero_without_thermal_model():
    w = _world(thermal=False)
    assert summarize(w)["throttle_s"] == 0.0


def test_throttle_time_accumulates_for_discrete_tier_model():
    w = _world(thermal=False)
    w.cfg["power"] = {
        "mode": "discrete_tiers",
        "P_dark": 80.0,
        "tiers": [
            {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
            {"name": "low", "power_w": 75.0, "max_batch": 4, "clock_f": 0.50},
            {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0},
        ],
    }
    from types import SimpleNamespace
    from satsim.cosim.power_model import DiscreteTierModel

    w.power_model = DiscreteTierModel(w.cfg, w.tr.n)
    w.power_model.thermal_trip_s = 7.5
    w.power_model.thermal_trip_count = 2
    w._thermal_tick()
    assert w.acc["throttle_s"] == pytest.approx(7.5)
    assert w.acc["thermal_trip_count"] == 2
    assert summarize(w)["throttle_s"] == pytest.approx(7.5)
