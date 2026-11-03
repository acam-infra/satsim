from pathlib import Path

import pytest

from satsim.__main__ import run_spec
from satsim.input.load import resolve_spec
from satsim.input.schema import SpecError

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "input" / "scenarios" / "exp_ring_s1_cost_claims_v3.yaml"
SCENE = ROOT / "input" / "scenes" / "scene_inferred_orbit_pod_v3.yaml"


def test_scene_is_not_a_run():
    with pytest.raises(SpecError, match="场景背景说明"):
        resolve_spec(SCENE)


def _detached(text: str, dest: Path) -> Path:
    text = text.replace(
        "base: base_ring_s1.yaml",
        f"base: {ROOT / 'input' / 'scenarios' / 'base_ring_s1.yaml'}",
    )
    text = text.replace(
        "scene: ../scenes/scene_inferred_orbit_pod_v3.yaml",
        f"scene: {SCENE}",
    )
    dest.write_text(text)
    return dest


def test_policy_missing_scheduler_refused(tmp_path: Path):
    text = SPEC.read_text().replace(
        "  place_once:\n",
        "  invented:\n    hysteresis: 0.15\n  place_once:\n",
        1,
    )
    with pytest.raises(SpecError, match="scheduler|不允许的字段"):
        resolve_spec(_detached(text, tmp_path / "bad.yaml"))


def test_unknown_override_is_refused(tmp_path: Path):
    text = SPEC.read_text().replace(
        "  gpu.kv_gib: 24.0\n",
        "  gpu.kv_gib: 24.0\n  thermal.T_max: 1\n",
        1,
    )
    with pytest.raises(SpecError, match="不允许的覆盖项"):
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
    # 无背景时随机占带宽迁移仍可能慢于 free，但超时中断应减少
    assert c2["nobg"]["rebalance_link_abort_mean"] < c2["rebalance_link_abort_mean"]
    assert payload["claims"]["ours"]["status"] == "todo"
