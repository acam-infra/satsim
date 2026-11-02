"""全局 scheduler、泊松生成、configs 入口。"""
from __future__ import annotations

from pathlib import Path

import pytest

from satsim.__main__ import run_spec
from satsim.domain import Request
from satsim.input.generate.dispatch import generate_arrivals
from satsim.input.load import resolve_spec
from satsim.input.schema import SpecError
from satsim.method.registry import bundle_from_policy

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "input" / "scenarios" / "stac_s1_cost_claims_v3.yaml"
POISSON = ROOT / "input" / "scenarios" / "poisson_smoke_v1.yaml"
SCENE = ROOT / "input" / "scenes" / "inferred_orbit_pod_v3.yaml"
CONFIGS = ROOT / "configs" / "s1_cost_claims_v3.yaml"


def _detached(text: str, dest: Path) -> Path:
    text = text.replace(
        "base: construct_s1.yaml",
        f"base: {ROOT / 'input' / 'scenarios' / 'construct_s1.yaml'}",
    )
    text = text.replace(
        "scene: ../scenes/inferred_orbit_pod_v3.yaml",
        f"scene: {SCENE}",
    )
    text = text.replace(
        "scene: ../scenes/poisson_smoke_v1.yaml",
        f"scene: {ROOT / 'input' / 'scenes' / 'poisson_smoke_v1.yaml'}",
    )
    dest.write_text(text)
    return dest


def test_configs_symlink_runs():
    payload = run_spec(CONFIGS, outdir=None)
    assert payload["claims"]["C1"]["goodput_gain"] > 0


def test_bundle_from_scheduler_policy():
    _, _, jobs = resolve_spec(MAIN)
    name, pol, c2, seed = next(j for j in jobs if j[0] == "rebalance_over_link")
    bundle = bundle_from_policy(pol)
    assert pol["scheduler"] == "rebalance_over_link"
    assert pol["link_cost"] == "full"
    assert bundle.scheduler.id == "rebalance_over_link"
    assert bundle.link_cost.mode == "full"


def test_poisson_generate_and_run():
    spec, cfg, jobs = resolve_spec(POISSON)
    assert spec["trace"]["kind"] == "poisson"
    reqs, anchor, count = generate_arrivals(spec["trace"], float(cfg["workload"]["p_long"]), 1)
    assert isinstance(reqs[0], Request)
    assert len(reqs) >= 1
    payload = run_spec(POISSON, outdir=None)
    assert payload["T_SWAP"] == pytest.approx(40.0)
    assert "place_once" in payload["claims"]["policies"]
    assert "rebalance_local" in payload["claims"]["policies"]


def test_placement_rebalance_fields_refused(tmp_path: Path):
    text = MAIN.read_text().replace(
        "  place_once:\n    scheduler: place_once\n    link_cost: \"off\"\n",
        "  place_once:\n    placement: first\n    rebalance: none\n    link_cost: \"off\"\n",
        1,
    )
    with pytest.raises(SpecError, match="未准入字段|需要 scheduler"):
        resolve_spec(_detached(text, tmp_path / "legacy.yaml"))


def test_scheduler_link_cost_mismatch_refused(tmp_path: Path):
    text = MAIN.read_text().replace(
        "  place_once:\n    scheduler: place_once\n    link_cost: \"off\"\n",
        "  place_once:\n    scheduler: place_once\n    link_cost: \"full\"\n",
        1,
    )
    with pytest.raises(SpecError, match="link_cost 只准入"):
        resolve_spec(_detached(text, tmp_path / "bad_lc.yaml"))
