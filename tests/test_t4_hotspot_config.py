"""T4: the stage-1 hotspot config changes only locality and baseline policies."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from satsim.cosim.run import load_config
from satsim.cosim.trace import Trace
from satsim.cosim.world import make_requests


ROOT = Path(__file__).resolve().parents[1]


def test_hotspot_config_matches_trace_and_base_total_rate():
    cfg = load_config(ROOT / "configs" / "cosim_hotspot_v1.yaml")
    base = load_config(ROOT / "configs" / "cosim_placement_thermal.yaml")
    trace = Trace.load((ROOT / "configs" / cfg["trace"]).resolve())

    rates = cfg["workload"]["source_rates_per_min"]
    assert len(rates) == len(trace.rs)
    assert rates[0] == 4 * rates[1] == 4 * rates[2]
    assert sum(rates) == len(trace.rs) * base["workload"]["rate_per_rs_per_min"]
    assert cfg["name"] == "cosim_hotspot_v1"
    assert list(cfg["policies"]) == ["round_robin", "nearest", "random"]
    assert cfg["policies"]["round_robin"] == {"policy": "round_robin_all"}
    assert cfg["seeds"] == [1, 2, 3, 4, 5]
    assert cfg["migration"] == base["migration"]
    assert cfg["workload"]["long_out"] == base["workload"]["long_out"]
    assert cfg["slo"] == base["slo"]


def test_hotspot_config_builds_requests_for_every_rs():
    cfg = load_config(ROOT / "configs" / "cosim_hotspot_v1.yaml")
    trace = Trace.load((ROOT / "configs" / cfg["trace"]).resolve())
    reqs = make_requests(cfg, trace, 1)
    origins = {r.origin: sum(1 for x in reqs if x.origin == r.origin) for r in reqs}
    assert set(origins) == set(trace.rs)
    # The hotspot RS has the highest arrival rate; Poisson samples keep the 4x ordering here.
    assert origins[trace.rs[0]] > 2 * max(origins[trace.rs[1]], origins[trace.rs[2]])
    assert len(reqs) > 0


def test_hotspot_config_has_no_banned_terms():
    text = (ROOT / "configs" / "cosim_hotspot_v1.yaml").read_text(encoding="utf-8").lower()
    assert "阴影" not in text and "shadow" not in text
    assert json.loads((ROOT / "input" / "traces" / "coop-w53.trace.json").read_text())["schema"]
