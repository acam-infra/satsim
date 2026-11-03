"""ShareGPT 到达：CSV 长度 + 合成到达。"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from satsim.__main__ import run_spec
from satsim.domain import Request
from satsim.input.generate.dispatch import generate_arrivals
from satsim.input.generate.sharegpt import ShareGPTArrival
from satsim.input.load import resolve_spec
from satsim.input.schema import SpecError, validate_trace

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "input" / "datasets" / "sharegpt_lengths_v1.csv"
SPEC = ROOT / "input" / "scenarios" / "sharegpt_s1_smoke_v1.yaml"


def _write_csv(path: Path, rows: list[tuple[int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_len", "output_len"])
        w.writerows(rows)


def test_sharegpt_fake_csv_generate(tmp_path: Path):
    csv_path = tmp_path / "lens.csv"
    _write_csv(csv_path, [(10, 20), (500, 100), (8, 8), (1000, 50)])
    trace = dict(
        kind="sharegpt",
        path=str(csv_path),
        window_s=10,
        rate_per_s=1.0,
        arrival="deterministic",
        num_prompts=5,
        long_threshold_tokens=200,
        max_seqlen=4096,
        jitter_s=0.0,
    )
    validate_trace(trace)
    reqs, anchor, count = ShareGPTArrival().generate(trace, p_long=0.2, seed=1)
    assert len(reqs) == 5
    assert isinstance(reqs[0], Request)
    assert reqs[0].t == pytest.approx(0.0)
    assert reqs[1].t == pytest.approx(1.0)
    assert any(r.long for r in reqs)
    assert count >= 1
    # dispatch
    reqs2, _, _ = generate_arrivals(trace, 0.2, 2)
    assert len(reqs2) == 5


def test_sharegpt_schema_requires_path():
    with pytest.raises(SpecError, match="path"):
        validate_trace({"kind": "sharegpt", "window_s": 10, "rate_per_s": 1.0})


def test_sharegpt_spec_resolves_and_runs():
    assert CSV.is_file(), "需要先跑 scripts/prep_sharegpt_lengths.py"
    spec, cfg, jobs = resolve_spec(SPEC)
    assert spec["trace"]["kind"] == "sharegpt"
    assert len(jobs) == 9  # 3 policies × 3 seeds
    reqs, _, _ = generate_arrivals(spec["trace"], float(cfg["workload"]["p_long"]), 1)
    assert len(reqs) == 120
    assert all(r.prompt_tokens >= 4 and r.output_tokens >= 4 for r in reqs)
    payload = run_spec(SPEC, outdir=None)
    assert payload["T_SWAP"] == pytest.approx(30.0)
    pols = payload["claims"]["policies"]
    assert "place_once" in pols and "rebalance_free" in pols and "rebalance_random" in pols
