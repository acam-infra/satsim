from __future__ import annotations

import math
import pytest

from satsim.cosim.power_tiers import (
    ACC_ENERGY_J,
    ACC_TIER_TIME_S,
    ACC_TIER_UNSAT_S,
    ACC_ZERO_POWER_S,
    POWER_MODE_CONTINUOUS,
    POWER_MODE_DISCRETE,
    allowed_tier_indices,
    effective_clock_f,
    is_discrete_power,
    parse_discrete_power_spec,
    rc_linear_coeffs,
    rollout_rc_temperatures,
    select_tier_for_batch,
)


def _sample_discrete_cfg(
    *,
    include_zero: bool = True,
    p_dark: float = 160.0,
    prefer_high_clock: bool | None = None,
) -> dict:
    tiers = [
        {"name": "low", "power_w": 150.0, "max_batch": 4, "clock_f": 0.65, "prefill_clock_f": 0.35},
        {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
    ]
    if include_zero:
        tiers.insert(0, {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0})
    power_cfg: dict = {
        "mode": POWER_MODE_DISCRETE,
        "P_dark": p_dark,
        "tiers": tiers,
    }
    if prefer_high_clock is not None:
        power_cfg["prefer_high_clock"] = prefer_high_clock
    return {
        "power": power_cfg,
        "thermal": {
            "T_max_C": 82.0,
            "thermal_horizon_s": 25.0,
        },
    }


def test_legacy_continuous_mode_returns_none():
    assert not is_discrete_power({})
    assert not is_discrete_power({"power": {"mode": POWER_MODE_CONTINUOUS}})
    assert parse_discrete_power_spec({"power": {"P_idle": 60, "P_max": 300}}) is None


def test_parse_discrete_power_spec_auto_inserts_zero_tier_and_sorts():
    cfg = _sample_discrete_cfg(include_zero=False)
    spec = parse_discrete_power_spec(cfg)
    assert spec is not None
    assert len(spec.tiers) == 3
    assert spec.prefer_high_clock is True
    assert spec.idle_tier.index == 0
    assert spec.idle_tier.power_w == 0.0
    assert spec.idle_tier.max_batch == 0
    assert spec.idle_tier.clock_f == 0.0
    assert [t.max_batch for t in spec.work_tiers] == [4, 12]
    assert spec.work_tiers[0].effective_prefill_f == 0.35
    assert spec.work_tiers[1].effective_prefill_f == 1.0
    assert spec.max_capacity == 12
    assert spec.t_max_c == 82.0
    assert spec.thermal_horizon_s == 25.0


def test_parse_discrete_power_spec_rejects_nonzero_idle_power():
    bad_cfg = {
        "power": {
            "mode": POWER_MODE_DISCRETE,
            "P_dark": 100.0,
            "tiers": [
                {"name": "bad_idle", "power_w": 15.0, "max_batch": 0, "clock_f": 0.0},
                {"name": "work", "power_w": 150.0, "max_batch": 4, "clock_f": 0.7},
            ],
        }
    }
    with pytest.raises(ValueError, match="zero-request tier"):
        parse_discrete_power_spec(bad_cfg)


def test_rc_linear_coeffs_exact_step():
    alpha, offset, gain = rc_linear_coeffs(tau_s=60.0, r_th=0.22, t_amb_c=20.0, dt_s=30.0)
    expected_alpha = math.exp(-0.5)
    assert math.isclose(alpha, expected_alpha)
    assert math.isclose(offset, (1.0 - expected_alpha) * 20.0)
    assert math.isclose(gain, (1.0 - expected_alpha) * 0.22)
    t0 = 50.0
    p_w = 300.0
    t1 = alpha * t0 + offset + gain * p_w
    t_inf = 20.0 + 0.22 * p_w
    assert math.isclose(t1, t_inf + (t0 - t_inf) * expected_alpha)

    traj = rollout_rc_temperatures(50.0, [300.0, 0.0], tau_s=60.0, r_th=0.22, t_amb_c=20.0, dt_s=30.0)
    assert len(traj) == 2
    assert math.isclose(traj[0], t1)
    assert traj[1] < traj[0]


def test_allowed_tiers_fair_sunlit_selection_and_safe_empty_clock():
    spec = parse_discrete_power_spec(_sample_discrete_cfg(p_dark=160.0))
    assert spec is not None
    assert allowed_tier_indices(spec, is_eclipse=False) == [0, 1, 2]
    assert allowed_tier_indices(spec, is_eclipse=True) == [0, 1]

    # Zero requests -> strict 0 W physical power tier, but strictly positive candidate clock_f!
    t_idle = select_tier_for_batch(spec, 0, is_eclipse=False)
    assert t_idle.index == 0 and t_idle.power_w == 0.0 and t_idle.clock_f == 0.0
    assert effective_clock_f(spec, 0, is_eclipse=False) == 1.0
    assert effective_clock_f(spec, 0, is_eclipse=True) == 0.65

    # Symmetric baseline fairness: sunlit cool node uses fastest safe tier even for light load (batch=3)
    assert select_tier_for_batch(spec, 3, is_eclipse=False).name == "high"
    # Optional min-power mode or explicit planner pre-cooling tier override selects low tier
    assert select_tier_for_batch(spec, 3, is_eclipse=False, prefer_high_clock=False).name == "low"
    assert select_tier_for_batch(spec, 3, is_eclipse=False, explicit_tier_idx=1).name == "low"
    # Explicit low tier is ignored if batch exceeds low tier's max_batch=4
    assert select_tier_for_batch(spec, 6, is_eclipse=False, explicit_tier_idx=1).name == "high"

    # In eclipse, high tier (300 W > 160 W) is pruned -> clamps to low tier
    assert select_tier_for_batch(spec, 3, is_eclipse=True).name == "low"
    assert select_tier_for_batch(spec, 8, is_eclipse=True).name == "low"

    # Thermal guardrail prunes high tier when T_next would exceed 82 C
    coeffs = rc_linear_coeffs(tau_s=60.0, r_th=0.22, t_amb_c=25.0, dt_s=30.0)
    # At T=80C, high tier (300W -> T_inf=91C) pushes T_next > 82C, while low tier (150W -> T_inf=58C) cools down
    assert select_tier_for_batch(spec, 3, is_eclipse=False, current_temp_c=80.0, rc_step_coeffs=coeffs).name == "low"
    assert select_tier_for_batch(spec, 8, is_eclipse=False, current_temp_c=80.0, rc_step_coeffs=coeffs).name == "low"
    assert effective_clock_f(spec, 0, is_eclipse=False, current_temp_c=80.0, rc_step_coeffs=coeffs) == 0.65

    # Canonical accumulator keys are stable strings
    assert {ACC_ZERO_POWER_S, ACC_TIER_TIME_S, ACC_ENERGY_J, ACC_TIER_UNSAT_S} == {
        "zero_power_s",
        "tier_time_s",
        "energy_j",
        "tier_unsat_s",
    }
