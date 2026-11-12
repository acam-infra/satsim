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
    assert OrbitLoad().id == "orbit_thermal"
    assert OrbitLoad(thermal_aware=True).id == "orbit_thermal"
    assert OrbitLoad(thermal_aware=False).id == "orbit_load"
    assert OrbitLoad(thermal_aware="hot_only").id == "orbit_hot_only"
    assert OrbitLoad(thermal_aware="rc").id == "orbit_thermal_rc"
    with pytest.raises(ValueError):
        OrbitLoad(thermal_aware="warmish")


def test_orbit_load_modes_reach_policy_factory():
    for thermal_aware, expected in ((False, "orbit_load"), ("hot_only", "orbit_hot_only"),
                                    (True, "orbit_thermal"), ("rc", "orbit_thermal_rc")):
        pol = make_policy({"policy": "orbit_load", "thermal_aware": thermal_aware})
        assert pol.id == expected


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
