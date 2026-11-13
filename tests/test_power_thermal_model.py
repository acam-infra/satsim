"""Tests for PowerThermalModel adapters, CoWorld integration, and discrete metrics (LAB-1874)."""
from __future__ import annotations

import copy
import math

import pytest
import simpy

from satsim.cosim.interfaces import PowerThermalModel
from satsim.cosim.policies import make_policy
from satsim.cosim.power_model import (
    ContinuousDVFSModel,
    DiscreteTierModel,
    OrbitOnlyPowerModel,
    make_power_thermal_model,
)
from satsim.cosim.power_tiers import (
    ACC_ENERGY_J,
    ACC_TIER_TIME_S,
    ACC_TIER_UNSAT_S,
    ACC_ZERO_POWER_S,
)
from satsim.cosim.record import summarize
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, CRequest


def _toy_trace(*, dark_intervals: list[list[list[float]]] | None = None) -> Trace:
    return Trace(
        {
            "schema": "orbitsim.contact_trace/v1",
            "name": "dp1-toy",
            "horizon_s": 200.0,
            "nodes": [
                {"id": 0, "role": "rs"},
                {"id": 1, "role": "compute"},
                {"id": 2, "role": "compute"},
            ],
            "edges": [
                {"u": 0, "v": 1, "kind": "rs", "up": [[0, 200]]},
                {"u": 1, "v": 2, "kind": "intra", "up": [[0, 200]]},
            ],
            "dark": dark_intervals if dark_intervals is not None else [[], [], [[50.0, 150.0]]],
        }
    )


def _base_cfg() -> dict:
    return {
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": 24.0,
            "t_fixed_s": 0.003,
            "t_weights_s": 0.01,
            "t_kv_tok_s": 1e-7,
            "t_seq_s": 5e-5,
            "mem_freq_exp": 0.6,
        },
        "power": {"P_idle": 60, "P_max": 300, "gamma": 2.4, "f_min": 0.35, "P_dark": 80},
        "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
        "migration": {"timeout_s": 120.0, "switch_s": 0.02},
        "workload": {"arrival_window_s": 200},
        "slo": {
            "deadline_factor_long": 3.0,
            "deadline_factor_short": 4.0,
            "min_deadline_s": 60,
            "max_stall_frac": 0.1,
        },
        "policy": {"stall_weight": 1.0},
        "delivery": "final",
        "control_period_s": 2.0,
    }


def _discrete_cfg(*, t0_c: float = 40.0, t_max_c: float = 82.0) -> dict:
    cfg = _base_cfg()
    cfg["power"] = {
        "mode": "discrete_tiers",
        "P_dark": 160.0,
        "tiers": [
            {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
            {"name": "low", "power_w": 150.0, "max_batch": 4, "clock_f": 0.65, "prefill_clock_f": 0.35},
            {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
        ],
    }
    cfg["thermal"] = {
        "tau_s": 60.0,
        "R_th": 0.25,
        "T_amb_C": 20.0,
        "T0_C": t0_c,
        "T_max_C": t_max_c,
        "thermal_horizon_s": 20.0,
        # Include legacy hysteresis thresholds to verify DiscreteTierModel ignores them.
        "T_throttle_C": 70.0,
        "T_release_C": 60.0,
        "f_throttle": 0.55,
    }
    return cfg


def _continuous_cfg() -> dict:
    cfg = _base_cfg()
    cfg["thermal"] = {
        "tau_s": 60.0,
        "R_th": 0.30,
        "T_amb_C": 20.0,
        "T0_C": 38.0,
        "T_throttle_C": 84.0,
        "T_release_C": 76.0,
        "f_throttle": 0.55,
        "P_busy": 300.0,
        "busy_util_floor": 0.40,
        "batch_scale": 4.0,
    }
    return cfg


def test_make_power_thermal_model_selects_and_conforms_to_protocol() -> None:
    orbit_m = make_power_thermal_model(_base_cfg(), 3)
    cont_m = make_power_thermal_model(_continuous_cfg(), 3)
    disc_m = make_power_thermal_model(_discrete_cfg(), 3)

    assert isinstance(orbit_m, OrbitOnlyPowerModel)
    assert isinstance(cont_m, ContinuousDVFSModel)
    assert isinstance(disc_m, DiscreteTierModel)

    for m in (orbit_m, cont_m, disc_m):
        assert isinstance(m, PowerThermalModel)
        op_empty = m.operating_point(1, 0, is_dark=False, t=0.0)
        assert op_empty.clock_f > 0.0
        assert op_empty.power_w >= 0.0


def test_discrete_empty_node_zero_power_no_div_zero_and_no_hysteresis_crosstalk() -> None:
    cfg = _discrete_cfg(t0_c=72.0, t_max_c=82.0)
    tr = _toy_trace()
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, make_policy({"policy": "greedy"}))

    assert isinstance(w.power_model, DiscreteTierModel)
    # Empty node probing: iter_of and f_of must be strictly positive and never raise ZeroDivisionError
    assert w.f_of(1, 0.0) == pytest.approx(1.0)
    assert w.iter_of(1) > 0.0
    # Empty node physical power is strictly 0.0 W
    assert w.power_model.operating_point(1, 0, is_dark=False, t=0.0).power_w == 0.0

    # At T0=72C > legacy T_throttle_C=70C, DiscreteTierModel must NOT be throttled by legacy hysteresis;
    # sunlit light load (batch=1) uses high tier (clock_f=1.0) as long as T_next <= T_max_C=82C
    op_light = w.power_model.operating_point(1, 1, is_dark=False, t=0.0)
    assert op_light.clock_f == pytest.approx(1.0)
    assert op_light.power_w == pytest.approx(300.0)
    assert not op_light.throttled

    # Empty node cools at fastest RC slope (0 W -> toward T_amb_C=20C)
    w.power_model.step(10.0, {1: 0, 2: 0}, tr.is_dark)
    assert w.power_model.temperature(1) < 72.0
    expected_t10 = 20.0 + (72.0 - 20.0) * math.exp(-10.0 / 60.0)
    assert w.power_model.temperature(1) == pytest.approx(expected_t10, rel=1e-6)


def test_discrete_coworld_end_to_end_and_summarize_metrics() -> None:
    cfg = _discrete_cfg(t0_c=40.0, t_max_c=82.0)
    tr = _toy_trace()
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, make_policy({"policy": "greedy"}))

    r1 = CRequest(0, 0, 5.0, False, 200, 150, 1e6, 100.0)
    w.submit(r1)
    env.run(until=tr.horizon)

    assert r1.t_delivered is not None
    s = summarize(w)
    for key in (
        "zero_power_frac",
        "power_tier_fracs",
        "energy_kj_total",
        "tier_unsat_s",
        "peak_temp_C",
        "min_thermal_margin_C",
    ):
        assert key in s
    assert 0.0 < s["zero_power_frac"] <= 1.0
    assert math.isclose(sum(s["power_tier_fracs"].values()), 1.0, rel_tol=1e-6)
    assert s["energy_kj_total"] > 0.0
    assert s["peak_temp_C"] is not None
    assert s["min_thermal_margin_C"] == pytest.approx(82.0 - s["peak_temp_C"])


def test_continuous_mode_summarize_keys_unchanged() -> None:
    tr = _toy_trace()
    env_cont = simpy.Environment()
    w_cont = CoWorld(env_cont, tr, _continuous_cfg(), make_policy({"policy": "greedy"}))
    env_cont.run(until=50.0)
    s_cont = summarize(w_cont)
    assert "zero_power_frac" not in s_cont
    assert "energy_kj_total" not in s_cont
    assert len(s_cont) == 26


def test_project_finish_across_all_adapters() -> None:
    tr = _toy_trace(dark_intervals=[[], [[0.0, 200.0]], []])
    for cfg in (_base_cfg(), _continuous_cfg(), _discrete_cfg(t0_c=80.0, t_max_c=82.0)):
        m = make_power_thermal_model(cfg, tr.n)
        t_sun, temp_sun = m.project_finish(2, 0.0, 2, 1000.0, 500.0, trace=tr, gpu=cfg["gpu"])
        t_dark, _ = m.project_finish(1, 0.0, 2, 1000.0, 500.0, trace=tr, gpu=cfg["gpu"])
        assert 0.0 < t_sun < 200.0
        assert t_dark > t_sun
        if isinstance(m, DiscreteTierModel):
            assert temp_sun <= 82.0 + 1e-6


