"""Tests for O(1) kv_used tracking, k_shortest_paths cache, sharded/resumable suite, and parallel cosim."""
from __future__ import annotations

from pathlib import Path

from satsim.cosim.run import load_config, run
from satsim.cosim.suite import run_suite
from satsim.cosim.thermal_rebalance import k_shortest_paths
from satsim.cosim.trace import Trace
from satsim.cosim.world import CRequest, _TrackedNodeList


def test_tracked_node_list_maintains_sum():
    lst = _TrackedNodeList()
    r1 = CRequest(0, 0, 0.0, True, 4000, 10000, 1e6, 900.0)
    r2 = CRequest(1, 0, 0.0, False, 500, 200, 1e5, 90.0)
    lst.append(r1)
    assert lst.reserved_tokens_sum == r1.reserved_tokens
    lst.extend([r2])
    assert lst.reserved_tokens_sum == r1.reserved_tokens + r2.reserved_tokens
    lst.remove(r1)
    assert lst.reserved_tokens_sum == r2.reserved_tokens
    popped = lst.pop()
    assert popped is r2
    assert lst.reserved_tokens_sum == 0
    lst.extend([r1, r2])
    lst.clear()
    assert lst.reserved_tokens_sum == 0


def test_k_shortest_paths_cached_matches_first_call():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "cosim_placement_thermal.yaml")
    tr = Trace.load((Path(cfg["_dir"]) / cfg["trace"]).resolve())
    p1 = k_shortest_paths(tr, tr.compute[0], tr.compute[5], 0.0, k=3)
    p2 = k_shortest_paths(tr, tr.compute[0], tr.compute[5], 0.0, k=3)
    assert p1 == p2
    assert hasattr(tr, "_k_path_cache")


def test_suite_shard_merge_and_resume(tmp_path):
    base_cfg = Path(__file__).resolve().parents[1] / "configs" / "cosim_placement_thermal.yaml"
    suite_yaml = tmp_path / "mini_suite.yaml"
    suite_yaml.write_text(
        f"""name: mini_test
base: {base_cfg}
seeds: [11, 12]
set:
  workload.arrival_window_s: 60
axes:
  workload.rate_per_rs_per_min: [10, 14]
policies:
  round_robin: {{policy: round_robin}}
  orbit_thermal: {{policy: orbit_load}}
report:
  baseline: round_robin
""",
        encoding="utf-8",
    )

    out_full = tmp_path / "full"
    res_full = run_suite(suite_yaml, out_dir=out_full, jobs=2)
    assert res_full["total_runs"] == 8

    out_sharded = tmp_path / "sharded"
    run_suite(suite_yaml, out_dir=out_sharded, jobs=1, shard="1/2")
    run_suite(suite_yaml, out_dir=out_sharded, jobs=1, shard="2/2")
    # Re-running with --resume should be a no-op and not duplicate rows
    res_resume = run_suite(suite_yaml, out_dir=out_sharded, jobs=1, shard="1/2", resume=True)
    assert res_resume["total_runs"] == 4
    res_merge = run_suite(suite_yaml, out_dir=out_sharded, merge=True)
    assert res_merge["total_runs"] == 8
    assert (out_sharded / "summary.md").is_file()


def test_cosim_parallel_matches_serial():
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "cosim_placement_thermal.yaml",
        ["workload.arrival_window_s=60"],
    )
    cfg["seeds"] = [11, 12]
    cfg["policies"] = {k: cfg["policies"][k] for k in ("round_robin", "orbit_thermal")}
    res1 = run(cfg, quiet=True, jobs=1)
    res2 = run(cfg, quiet=True, jobs=2)
    assert res1["mean"]["orbit_thermal"]["good_long"] == res2["mean"]["orbit_thermal"]["good_long"]
    assert res1["mean"]["round_robin"]["good_long"] == res2["mean"]["round_robin"]["good_long"]
