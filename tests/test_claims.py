from pathlib import Path

import pytest

from satsim.__main__ import run_spec
from satsim.input.spec import SpecError, resolve_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "input" / "scenarios" / "stac_s1_cost_claims_v3.yaml"
SCENE = ROOT / "input" / "scenes" / "inferred_orbit_pod_v3.yaml"


def test_scene_is_not_a_run():
    with pytest.raises(SpecError, match="叙事"):
        resolve_spec(SCENE)


def _detached(text: str, dest: Path) -> Path:
    text = text.replace("base: construct_s1.yaml", f"base: {ROOT / 'input' / 'scenarios' / 'construct_s1.yaml'}")
    text = text.replace(
        "scene: ../scenes/inferred_orbit_pod_v3.yaml",
        f"scene: {SCENE}",
    )
    dest.write_text(text)
    return dest


def test_unknown_policy_is_refused(tmp_path: Path):
    text = SPEC.read_text().replace(
        "  first:\n",
        "  invented:\n    route: late\n  first:\n",
        1,
    )
    with pytest.raises(SpecError, match="未准入的策略"):
        resolve_spec(_detached(text, tmp_path / "bad.yaml"))


def test_unknown_override_is_refused(tmp_path: Path):
    text = SPEC.read_text().replace(
        "  gpu.kv_gib: 24.0\n",
        "  gpu.kv_gib: 24.0\n  thermal.T_max: 1\n",
        1,
    )
    with pytest.raises(SpecError, match="未准入的覆盖项"):
        resolve_spec(_detached(text, tmp_path / "bad.yaml"))


def test_benefit_and_link_cost():
    payload = run_spec(SPEC, outdir=None)
    c1 = payload["claims"]["C1"]
    c2 = payload["claims"]["C2"]
    assert c1["good_long_gain"] > 0
    assert c1["goodput_gain"] > 0
    assert c2["link_cap_gbps"] == pytest.approx(10.0)
    assert c2["max_flows_on_link"] > 1
    assert c2["goodput_drop"] > 0.05
    assert c2["rebalance_link_abort_mean"] > 0
    assert abs(c2["nobg"]["goodput_drop"]) < 1e-9
    assert c2["net_tracks_link"] is True
