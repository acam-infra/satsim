"""Unit tests for satsim minimal suite runner (T5)."""
from __future__ import annotations

import csv
from pathlib import Path
import re
import subprocess
import sys
import yaml

import pytest

from satsim.cosim.suite import (
    apply_dotted_set,
    apply_set_block,
    format_stat,
    main,
    run_suite,
)


def test_cli_help():
    """Verify `satsim-suite --help` returns code 0."""
    res = subprocess.run(
        [sys.executable, "-m", "satsim.cosim.suite", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "satsim co-simulation suite runner" in res.stdout
    assert "--out" in res.stdout
    assert "--jobs" in res.stdout
    assert "--set" in res.stdout


def test_dotted_and_set_block():
    """Verify dot-separated paths and set: block applications."""
    cfg = {
        "workload": {"arrival_window_s": 3600, "rate": 10},
        "net": {"link_gbps": {"inter": 10.0, "intra": 10.0}},
    }
    # Dotted set
    apply_dotted_set(cfg, "workload.arrival_window_s", 15.0)
    assert cfg["workload"]["arrival_window_s"] == 15.0
    assert cfg["workload"]["rate"] == 10  # sibling preserved

    apply_dotted_set(cfg, "net.link_gbps.inter", 20.0)
    assert cfg["net"]["link_gbps"]["inter"] == 20.0
    assert cfg["net"]["link_gbps"]["intra"] == 10.0

    # New nested key
    apply_dotted_set(cfg, "foo.bar.baz", 123)
    assert cfg["foo"]["bar"]["baz"] == 123

    # Set block dict with rebalance_invariants
    set_block = {
        "workload.arrival_window_s": 5.0,
        "rebalance_invariants": {
            "interval_s": 5.0,
            "max_inflight": 2,
            "max_mig_per_req": 1,
        },
    }
    apply_set_block(cfg, set_block)
    assert cfg["workload"]["arrival_window_s"] == 5.0
    assert cfg["rebalance_invariants"] == {
        "interval_s": 5.0,
        "max_inflight": 2,
        "max_mig_per_req": 1,
    }

    # Set block list with key=value strings
    apply_set_block(cfg, ["workload.arrival_window_s=25.0", "net.link_gbps.intra=5.0"])
    assert cfg["workload"]["arrival_window_s"] == 25.0
    assert cfg["net"]["link_gbps"]["intra"] == 5.0


def test_format_stat():
    """Verify paired diff formatting as 'mean ± sd (n)'."""
    assert format_stat(0.0, 0.0, 2) == "+0.0000 ± 0.0000 (2)"
    assert format_stat(120.5, 12.3, 5) == "+120.5 ± 12.3 (5)"
    assert format_stat(-0.00005, 0.00001, 3) == "-5.000e-05 ± 1.000e-05 (3)"
    assert format_stat(0.0, 0.0, 0) == "-"


def test_minimal_suite_run(tmp_path: Path):
    """Run a minimal suite and assert CSV and summary.md outputs."""
    base_cfg_path = Path(__file__).parent.parent / "configs" / "cosim_placement_thermal.yaml"
    assert base_cfg_path.exists(), f"Base config {base_cfg_path} must exist"

    suite_content = {
        "name": "test_t5_minimal",
        "base": str(base_cfg_path),
        "set": {
            "workload.arrival_window_s": 10.0,
            "rebalance_invariants": {
                "interval_s": 5.0,
                "max_inflight": 2,
            },
        },
        "seeds": [1, 2],
        "policies": {
            "round_robin": {"policy": "round_robin_all"},
            "nearest": {"policy": "nearest"},
        },
        "axes": {
            "net.link_gbps.inter": [5.0, 10.0],
        },
        "report": {
            "baseline": "round_robin",
            "metrics": ["good", "stall_s_mean", "migrations"],
        },
    }

    suite_yaml = tmp_path / "test_suite.yaml"
    suite_yaml.write_text(yaml.safe_dump(suite_content), encoding="utf-8")

    out_dir = tmp_path / "suite_out"
    res = run_suite(suite_yaml, out_dir=out_dir, jobs=2)

    csv_path = Path(res["runs_csv"])
    summary_path = Path(res["summary_md"])

    assert csv_path.exists()
    assert summary_path.exists()

    # 1. Assert CSV line count:
    # 2 axes values × 2 policies × 2 seeds = 8 rows (+ 1 header row = 9 lines)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 8, f"Expected 8 data rows, got {len(reader)}"

    # Check that axes values, policies, seeds are present
    policies_seen = {r["policy"] for r in reader}
    seeds_seen = {int(r["seed"]) for r in reader}
    inter_seen = {float(r["net.link_gbps.inter"]) for r in reader}

    assert policies_seen == {"round_robin", "nearest"}
    assert seeds_seen == {1, 2}
    assert inter_seen == {5.0, 10.0}

    # 2. Assert summary.md content and alignment
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "# Suite Summary: test_t5_minimal" in summary_text
    assert "**Baseline**: `round_robin`" in summary_text
    assert "2 axis combo(s) × 2 policy(ies) × 2 seed(s)" in summary_text
    assert "Axes: `net.link_gbps.inter = 5.0`" in summary_text
    assert "Axes: `net.link_gbps.inter = 10.0`" in summary_text

    # Verify table headers
    assert "| policy | good | stall_s_mean | migrations |" in summary_text

    # Verify paired differences pattern 'mean ± sd (n)'
    # Matches strings like "+0.0000 ± 0.0000 (2)" or "-0.1234 ± 0.0500 (2)"
    stat_pattern = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*±\s*\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*\(\d+\)")
    matches = stat_pattern.findall(summary_text)
    assert len(matches) > 0, f"Expected formatted stats in summary, text:\n{summary_text}"
    # Each row in each table should contain formatted stats
    # 2 axis combinations × 2 policies × 3 metrics = 12 stat cells
    assert len(matches) == 12


def test_suite_cli_main(tmp_path: Path):
    """Verify satsim-suite CLI main function execution with overrides."""
    base_cfg_path = Path(__file__).parent.parent / "configs" / "cosim_placement_thermal.yaml"
    suite_content = {
        "name": "test_cli_suite",
        "base": str(base_cfg_path),
        "set": {
            "workload.arrival_window_s": 5.0,
        },
        "seeds": [1],
        "policies": {
            "round_robin": {"policy": "round_robin_all"},
        },
    }
    suite_yaml = tmp_path / "cli_suite.yaml"
    suite_yaml.write_text(yaml.safe_dump(suite_content), encoding="utf-8")

    out_dir = tmp_path / "cli_out"
    main([
        str(suite_yaml),
        "--out", str(out_dir),
        "--jobs", "1",
        "--set", "workload.arrival_window_s=6.0",
    ])

    assert (out_dir / "runs.csv").exists()
    assert (out_dir / "summary.md").exists()


def test_coupled_axis_sets_both_paths():
    from satsim.cosim.suite import apply_dotted_set

    cfg = {"net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}}}
    apply_dotted_set(cfg, "net.link_gbps.intra,net.link_gbps.inter", 0.3)
    assert cfg["net"]["link_gbps"] == {"intra": 0.3, "inter": 0.3, "rs": 5.0}
