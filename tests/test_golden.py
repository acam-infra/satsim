"""Golden regression: fixed claim excerpt for the main synthetic_pass scenario."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from satsim.__main__ import run_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "input" / "scenarios" / "stac_s1_cost_claims_v3.yaml"
GOLDEN = ROOT / "tests" / "golden" / "stac_s1_cost_claims_v3.claims_excerpt.json"


def _excerpt(payload: dict) -> dict:
    c1, c2 = payload["claims"]["C1"], payload["claims"]["C2"]
    return {
        "name": payload["name"],
        "T_SWAP": payload["T_SWAP"],
        "C1": {
            "first_good_long": c1["first_good_long"],
            "rebalance_good_long": c1["rebalance_good_long"],
            "goodput_gain": round(c1["goodput_gain"], 10),
        },
        "C2": {
            "link_cap_gbps": c2["link_cap_gbps"],
            "goodput_drop": round(c2["goodput_drop"], 10),
            "nobg_goodput_drop": c2["nobg"]["goodput_drop"],
            "net_tracks_link": c2["net_tracks_link"],
            "rebalance_link_abort_mean": c2["rebalance_link_abort_mean"],
        },
    }


def test_main_claims_match_golden():
    want = json.loads(GOLDEN.read_text())
    got = _excerpt(run_spec(SPEC, outdir=None))
    assert got["name"] == want["name"]
    assert got["T_SWAP"] == pytest.approx(want["T_SWAP"])
    assert got["C1"]["first_good_long"] == pytest.approx(want["C1"]["first_good_long"])
    assert got["C1"]["rebalance_good_long"] == pytest.approx(want["C1"]["rebalance_good_long"])
    assert got["C1"]["goodput_gain"] == pytest.approx(want["C1"]["goodput_gain"], rel=0, abs=1e-9)
    assert got["C2"]["link_cap_gbps"] == pytest.approx(want["C2"]["link_cap_gbps"])
    assert got["C2"]["goodput_drop"] == pytest.approx(want["C2"]["goodput_drop"], rel=0, abs=1e-9)
    assert got["C2"]["nobg_goodput_drop"] == pytest.approx(want["C2"]["nobg_goodput_drop"], abs=1e-12)
    assert got["C2"]["net_tracks_link"] is want["C2"]["net_tracks_link"]
    assert got["C2"]["rebalance_link_abort_mean"] == pytest.approx(want["C2"]["rebalance_link_abort_mean"])
