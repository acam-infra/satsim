"""Unit tests for discrete_rh_tier_v5 thermal/eclipse-gated pre-cooling and urgent isolation."""
from __future__ import annotations

from types import SimpleNamespace

from satsim.cosim.discrete_milp import DiscreteThermalMigrator
from satsim.cosim.policies import make_policy
from satsim.cosim.power_model import DiscreteTierModel


def _make_mock_world(cur_temp: float = 42.0, dark_after_s: float | None = None):
    cfg = {
        "gpu": {"t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7, "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
        "power": {
            "mode": "discrete_tiers",
            "P_dark": 80.0,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
                {"name": "low", "power_w": 75.0, "max_batch": 4, "clock_f": 0.50},
                {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0},
            ],
        },
        "thermal": {"tau_s": 60.0, "R_th": 0.25, "T_amb_C": 20.0, "T0_C": cur_temp, "T_max_C": 76.0},
        "workload": {"long_out": [18000, 36000], "short_out": [200, 1200]},
    }
    pm = DiscreteTierModel(cfg, 1)
    def _power_segments(j, t):
        if dark_after_s is None:
            return [(t, 1e9, False)]
        if t < dark_after_s:
            return [(t, dark_after_s, False), (dark_after_s, 1e9, True)]
        return [(t, 1e9, True)]

    tr = SimpleNamespace(
        compute=[0],
        is_dark=lambda j, t: (dark_after_s is not None and t >= dark_after_s),
        power_segments=_power_segments,
    )
    gpu = cfg["gpu"]
    req = SimpleNamespace(long=True, remaining=1000, gen=0, kv=4000, deadline=5000.0, urgent=False)
    w = SimpleNamespace(
        cfg=cfg,
        env=SimpleNamespace(now=0.0),
        nodes={0: [req]},
        power_model=pm,
        tr=tr,
        gpu=gpu,
    )
    return w


def test_v5_factory_and_thermal_gate():
    pol = make_policy({"policy": "discrete_rh_tier_v5"})
    assert isinstance(pol, DiscreteThermalMigrator)
    assert pol.id == "discrete_rh_tier_v5"

    # Cool sunlit node: gate stays closed -> None (stay on high tier)
    w_cool = _make_mock_world(cur_temp=40.0, dark_after_s=None)
    assert pol._select_explicit_tier_for_node(w_cool, 0) is None

    # Hot sunlit node near T_max_C (76.0): thermal gate opens -> low tier (index 1)
    w_hot = _make_mock_world(cur_temp=73.0, dark_after_s=None)
    assert pol._select_explicit_tier_for_node(w_hot, 0) == 1

    # Cool node approaching eclipse within 30s: pre-eclipse sprint stays on high tier (None)
    w_ecl = _make_mock_world(cur_temp=40.0, dark_after_s=20.0)
    assert pol._select_explicit_tier_for_node(w_ecl, 0) is None
