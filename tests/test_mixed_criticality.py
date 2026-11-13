"""Tests for mixed-criticality request generation and summary metrics."""
from __future__ import annotations

import copy
from pathlib import Path

from satsim.cosim.record import summarize
from satsim.cosim.run import load_config, run_one
from satsim.cosim.trace import Trace
from satsim.cosim.world import make_requests


def test_mixed_criticality_preserves_arrivals_and_tightens_urgent_deadlines():
    cfg = load_config(Path("configs/cosim_discrete_power_w53_v1.yaml"))
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    base_reqs = make_requests(cfg, tr, seed=11)
    assert all(not r.urgent for r in base_reqs)

    mc_cfg = copy.deepcopy(cfg)
    mc_cfg["slo"]["urgent_frac"] = 0.25
    mc_cfg["slo"]["urgent_deadline_factor_long"] = 1.25
    mc_cfg["slo"]["deadline_factor_long"] = 2.60
    mc_reqs = make_requests(mc_cfg, tr, seed=11)

    assert len(mc_reqs) == len(base_reqs)
    urg_longs = [r for r in mc_reqs if r.urgent]
    norm_longs = [r for r in mc_reqs if r.long and not r.urgent]
    assert urg_longs and norm_longs
    for b, m in zip(base_reqs, mc_reqs):
        assert (b.t, b.origin, b.long, b.pin, b.pout) == (m.t, m.origin, m.long, m.pin, m.pout)
        if m.urgent:
            assert m.deadline < b.deadline
        elif m.long:
            assert m.deadline > b.deadline


def test_mixed_criticality_summary_metrics_smoke():
    cfg = load_config(Path("configs/cosim_discrete_power_w53_v1.yaml"))
    cfg["workload"]["arrival_window_s"] = 300
    cfg["slo"]["urgent_frac"] = 0.30
    cfg["slo"]["urgent_deadline_factor_long"] = 1.25
    cfg["slo"]["deadline_factor_long"] = 2.50
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    w, _ = run_one(cfg, tr, {"policy": "discrete_rh_tier_v4"}, seed=11, record=False)
    s = summarize(w)
    assert s["n_urgent"] > 0
    assert 0.0 <= s["good_urgent_frac"] <= 1.0
    assert 0.0 <= s["good_normal_long_frac"] <= 1.0
