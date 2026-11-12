"""LAB-1842: OrbitLoad thermal-awareness ladder + per-plane-index dispatch shares."""
from __future__ import annotations

import copy
import simpy
import pytest

from satsim.cosim.policies import OrbitLoad, make_policy
from satsim.cosim.probe import LoadProbe, analyze_run
from satsim.cosim.run import load_config
from satsim.cosim.world import CoWorld, make_requests
from satsim.cosim.synth import synth_multi_sso


def test_orbit_load_mode_ids_and_rejects_unknown():
    assert OrbitLoad().id == "orbit_thermal"                       # default -> dvfs, zero regression
    assert OrbitLoad(thermal_mode="off").id == "orbit_load"
    assert OrbitLoad(thermal_mode="hot_only").id == "orbit_hot_only"
    assert OrbitLoad(thermal_mode="dvfs").id == "orbit_thermal"
    assert OrbitLoad(thermal_mode="rc_const").id == "orbit_thermal_rc_const"
    assert OrbitLoad(thermal_mode="rc_staged").id == "orbit_thermal_rc"
    # legacy spellings keep working
    assert OrbitLoad(thermal_aware=True).id == "orbit_thermal"
    assert OrbitLoad(thermal_aware=False).id == "orbit_load"
    assert OrbitLoad(thermal_aware="hot_only").id == "orbit_hot_only"
    assert OrbitLoad(thermal_aware="rc").id == "orbit_thermal_rc"
    assert OrbitLoad(thermal_mode="rc_staged", thermal_aware=False).id == "orbit_thermal_rc"
    with pytest.raises(ValueError):
        OrbitLoad(thermal_mode="warmish")


def test_orbit_load_modes_reach_policy_factory():
    for spec, expected in (({"thermal_mode": "off"}, "orbit_load"),
                           ({"thermal_mode": "hot_only"}, "orbit_hot_only"),
                           ({"thermal_aware": True}, "orbit_thermal"),
                           ({"thermal_mode": "rc_const"}, "orbit_thermal_rc_const"),
                           ({"thermal_mode": "rc_staged"}, "orbit_thermal_rc")):
        assert make_policy({"policy": "orbit_load", **spec}).id == expected
    composite = make_policy({"place": "orbit_load", "migrate": "none",
                             "place_kwargs": {"thermal_mode": "off"}})
    assert composite.placer.id == "orbit_load"


def test_plane_idx_shares_survive_ltan_collapse():
    """Two syntactic planes sharing one LTAN collapse in `plane_shares` but not in `plane_idx_shares`."""
    cfg = load_config(__import__("pathlib").Path(__file__).resolve().parents[1]
                      / "configs" / "cosim_hotspot_stress_v1.yaml")
    trace = synth_multi_sso(compute_ltans=(6.0, 6.0), sats_per_plane=5, horizon_s=600.0, dt_s=10.0)
    assert trace.raw_stats["kinds"]["inter"]["mean_up_edges"] > 0  # degenerate but non-empty
    cfg = copy.deepcopy(cfg)
    cfg["workload"]["arrival_window_s"] = 600
    env = simpy.Environment()
    probe = LoadProbe(sample_s=10.0)
    w = CoWorld(env, trace, cfg, make_policy({"policy": "nearest"}), seed=1)
    probe.attach(w)
    for r in make_requests(cfg, trace, 1):
        w.submit(r)
    env.run(until=trace.horizon)
    row = analyze_run(w, probe)
    assert row["plane_shares"].keys() == {"LTAN06.0h"}, "LTAN tag collapses as expected"
    assert set(row["plane_idx_shares"]) == {"p0", "p1"}
    assert abs(sum(row["plane_idx_shares"].values()) - 1.0) < 1e-9


def test_thermal_forecast_advance_idle_node_stays_cool():
    """An idle node advances at P_idle; a 4-deep batch crosses the 84 C wall in the same window."""
    from satsim.cosim.thermal_rebalance import ThermalForecast

    cfg = load_config(__import__("pathlib").Path(__file__).resolve().parents[1]
                      / "configs" / "cosim_hotspot_stress_v1.yaml")
    trace = synth_multi_sso(compute_ltans=(6.0,), sats_per_plane=4, horizon_s=1200.0, dt_s=10.0)
    cfg = copy.deepcopy(cfg)
    cfg["workload"]["arrival_window_s"] = 1200
    env = simpy.Environment()
    w = CoWorld(env, trace, cfg, make_policy({"policy": "nearest"}), seed=1)
    node = trace.compute[0]
    idle = ThermalForecast(w, node)
    idle.advance(0.0, 900.0, 0)
    busy = ThermalForecast(w, node)
    busy.advance(0.0, 900.0, 4)
    assert idle.temp < busy.temp
    assert idle.temp < cfg["thermal"]["T_throttle_C"]
