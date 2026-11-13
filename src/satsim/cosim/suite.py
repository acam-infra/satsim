"""Minimal suite runner for satsim co-simulations.

Executes a Cartesian product of axes × policy × seed in parallel, writing:
  - runs/<name>/runs.csv: raw results per (axes, policy, seed)
  - runs/<name>/summary.md: paired differences against baseline per axis group
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import itertools
import os
from pathlib import Path
import statistics as st
import sys
import yaml

from satsim.cosim.record import summarize
from satsim.cosim.run import _resolve, load_config, run_one
from satsim.cosim.trace import Trace

_TRACE_CACHE: dict[str, Trace] = {}


def apply_dotted_set(cfg: dict, path: str, val) -> None:
    """Set one or more nested dictionary values using dot-separated (optionally comma-coupled) paths."""
    if "," in path:
        for sub_path in path.split(","):
            sub = sub_path.strip()
            if sub:
                apply_dotted_set(cfg, sub, val)
        return
    node = cfg
    *head, last = path.split(".")
    for k in head:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[last] = copy.deepcopy(val)


def apply_set_block(cfg: dict, block: dict | list) -> None:
    """Apply the 'set:' configuration block onto cfg."""
    if isinstance(block, list):
        for item in block:
            key, _, raw = item.partition("=")
            apply_dotted_set(cfg, key, yaml.safe_load(raw))
    elif isinstance(block, dict):
        for k, v in block.items():
            if "." in k:
                apply_dotted_set(cfg, k, v)
            elif isinstance(v, dict) and k not in ("rebalance_invariants",):
                if k not in cfg or not isinstance(cfg[k], dict):
                    cfg[k] = {}
                apply_set_block(cfg[k], v)
            else:
                cfg[k] = copy.deepcopy(v)


def format_stat(mean: float, sd: float, n: int) -> str:
    """Format paired difference as 'mean ± sd (n)'."""
    if n == 0:
        return "-"
    if abs(mean) < 1e-4 and mean != 0.0:
        return f"{mean:+.3e} ± {sd:.3e} ({n})"
    elif abs(mean) >= 100 or abs(sd) >= 100:
        return f"{mean:+.1f} ± {sd:.1f} ({n})"
    else:
        return f"{mean:+.4f} ± {sd:.4f} ({n})"


def _worker_task(task_args):
    cell_cfg, trace_path, pol_name, pol_spec, seed, axis_comb = task_args
    if trace_path not in _TRACE_CACHE:
        _TRACE_CACHE[trace_path] = Trace.load(trace_path)
    trace = _TRACE_CACHE[trace_path]
    w, _ = run_one(cell_cfg, trace, pol_spec, seed, record=False)
    metrics = summarize(w)
    return axis_comb, pol_name, seed, metrics


def _cell_key(axis_names: list[str], axis_comb: dict, pol_name: str, seed: int | str) -> tuple:
    return (*(str(axis_comb.get(k, "")) for k in axis_names), str(pol_name), str(seed))


def _load_existing_rows(csv_path: Path, axis_names: list[str]) -> tuple[list[dict], set[tuple]]:
    if not csv_path.is_file():
        return [], set()
    rows: list[dict] = []
    done: set[tuple] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict = {}
            for k, v in raw.items():
                if k in axis_names or k == "policy":
                    row[k] = v
                elif k == "seed":
                    row[k] = int(v)
                elif v == "" or v is None:
                    row[k] = None
                else:
                    try:
                        row[k] = int(v) if v.isdigit() or (v.startswith("-") and v[1:].isdigit()) else float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
            done.add(_cell_key(axis_names, row, row["policy"], row["seed"]))
    return rows, done


def parse_shard(spec: str) -> tuple[int, int]:
    idx_s, sep, total_s = spec.partition("/")
    if not sep:
        raise ValueError(f"invalid --shard {spec!r}; expected I/N (1-indexed)")
    idx, total = int(idx_s), int(total_s)
    if total < 1 or not (1 <= idx <= total):
        raise ValueError(f"invalid --shard {spec!r}; expected 1 <= I <= N")
    return idx, total


def build_suite_plan(suite_path: Path, overrides: list[str] = ()) -> dict:
    suite_data = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    base_ref = suite_data.get("base")
    if base_ref:
        base_path = (suite_path.parent / base_ref).resolve()
        base_cfg = load_config(base_path)
    else:
        base_cfg = copy.deepcopy(suite_data)
        if "_dir" not in base_cfg:
            base_cfg["_dir"] = str(suite_path.parent.resolve())

    name = suite_data.get("name", base_cfg.get("name", "suite"))
    base_cfg["name"] = name

    set_block = suite_data.get("set", {})
    if set_block:
        apply_set_block(base_cfg, set_block)

    for item in overrides:
        key, _, raw = item.partition("=")
        apply_dotted_set(base_cfg, key, yaml.safe_load(raw))

    seeds = [int(s) for s in suite_data.get("seeds", base_cfg.get("seeds", [1]))]
    policies = suite_data.get("policies", base_cfg.get("policies", {}))
    axes_block = suite_data.get("axes", {}) or {}
    report_block = suite_data.get("report", {}) or {}

    if axes_block:
        axis_names = list(axes_block.keys())
        axis_values = [axes_block[k] if isinstance(axes_block[k], list) else [axes_block[k]]
                       for k in axis_names]
        combinations = [dict(zip(axis_names, prod)) for prod in itertools.product(*axis_values)]
    else:
        axis_names = []
        combinations = [{}]

    tasks = []
    for comb in combinations:
        cell_cfg = copy.deepcopy(base_cfg)
        for k, v in comb.items():
            apply_dotted_set(cell_cfg, k, v)
        trace_path = str(_resolve(cell_cfg, "trace"))
        for pol_name, pol_spec in policies.items():
            for seed in seeds:
                tasks.append((cell_cfg, trace_path, pol_name, pol_spec, seed, comb))

    return {
        "name": name,
        "suite_data": suite_data,
        "axis_names": axis_names,
        "combinations": combinations,
        "policies": policies,
        "seeds": seeds,
        "tasks": tasks,
        "report_block": report_block,
    }


def write_suite_summary(
    out_dir: Path,
    name: str,
    axis_names: list[str],
    combinations: list[dict],
    policies: dict,
    seeds: list[int],
    rows: list[dict],
    report_block: dict,
) -> Path:
    metric_keys = [k for k in (rows[0].keys() if rows else []) if k not in (*axis_names, "policy", "seed")]
    baseline_name = report_block.get("baseline")
    if not baseline_name and policies:
        baseline_name = next(iter(policies))

    report_metrics = report_block.get("metrics")
    if not report_metrics:
        report_metrics = metric_keys

    md_lines = [
        f"# Suite Summary: {name}",
        "",
        f"- **Baseline**: `{baseline_name}`",
        f"- **Configurations**: {len(combinations)} axis combo(s) × {len(policies)} policy(ies) × {len(seeds)} seed(s)",
        "",
    ]

    for comb in combinations:
        if comb:
            comb_str = ", ".join(f"`{k} = {v}`" for k, v in comb.items())
            md_lines.append(f"## Axes: {comb_str}")
        else:
            md_lines.append("## Results (All Cells)")
        md_lines.append("")

        cell_data: dict[str, dict[int, dict[str, float]]] = {p: {} for p in policies}
        for row in rows:
            if all(str(row.get(k, "")) == str(v) for k, v in comb.items()):
                cell_data[row["policy"]][int(row["seed"])] = row

        headers = ["policy"] + report_metrics
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for pol_name in policies:
            row_cells = [pol_name if pol_name != baseline_name else f"{pol_name} (baseline)"]
            for m in report_metrics:
                if baseline_name in cell_data and pol_name in cell_data:
                    common_seeds = sorted(set(cell_data[pol_name].keys()) & set(cell_data[baseline_name].keys()))
                    diffs = [
                        cell_data[pol_name][s][m] - cell_data[baseline_name][s][m]
                        for s in common_seeds
                        if cell_data[pol_name][s].get(m) is not None and cell_data[baseline_name][s].get(m) is not None
                    ]
                    n = len(diffs)
                    mean = st.fmean(diffs) if n > 0 else 0.0
                    sd = st.stdev(diffs) if n > 1 else 0.0
                    row_cells.append(format_stat(mean, sd, n))
                else:
                    row_cells.append("-")
            md_lines.append("| " + " | ".join(row_cells) + " |")
        md_lines.append("")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")
    return summary_path


def run_suite(
    suite_path: Path,
    out_dir: Path | None = None,
    jobs: int | None = None,
    overrides: list[str] = (),
    shard: str | None = None,
    resume: bool = False,
    merge: bool = False,
) -> dict:
    plan = build_suite_plan(suite_path, overrides=overrides)
    name = plan["name"]
    suite_data = plan["suite_data"]
    axis_names = plan["axis_names"]
    combinations = plan["combinations"]
    policies = plan["policies"]
    seeds = plan["seeds"]
    tasks = plan["tasks"]
    report_block = plan["report_block"]

    if out_dir is None:
        out_dir = Path("runs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if merge:
        shard_files = sorted(out_dir.glob("runs.shard-*.csv"))
        if not shard_files:
            raise FileNotFoundError(f"no runs.shard-*.csv files found in {out_dir}")
        merged_rows: list[dict] = []
        seen: set[tuple] = set()
        for sf in shard_files:
            part_rows, _ = _load_existing_rows(sf, axis_names)
            for r in part_rows:
                k = _cell_key(axis_names, r, r["policy"], r["seed"])
                if k not in seen:
                    seen.add(k)
                    merged_rows.append(r)
        csv_path = out_dir / "runs.csv"
        if merged_rows:
            fieldnames = list(merged_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(merged_rows)
        summary_path = write_suite_summary(
            out_dir, name, axis_names, combinations, policies, seeds, merged_rows, report_block
        )
        return {
            "name": name,
            "runs_csv": str(csv_path),
            "summary_md": str(summary_path),
            "total_runs": len(merged_rows),
        }

    shard_info = parse_shard(shard) if shard else None
    if shard_info is not None:
        s_idx, s_total = shard_info
        tasks = tasks[s_idx - 1::s_total]
        csv_path = out_dir / f"runs.shard-{s_idx}-of-{s_total}.csv"
    else:
        csv_path = out_dir / "runs.csv"

    rows, done_keys = _load_existing_rows(csv_path, axis_names) if resume else ([], set())
    pending = [
        t for t in tasks
        if _cell_key(axis_names, t[5], t[2], t[4]) not in done_keys
    ]

    raw_jobs = jobs if jobs is not None else int(suite_data.get("jobs", 1))
    max_workers = (os.cpu_count() or 1) if raw_jobs == 0 else max(1, raw_jobs)
    max_workers = min(max_workers, max(1, len(pending)))

    fieldnames: list[str] | None = list(rows[0].keys()) if rows else None
    write_header = not (resume and csv_path.is_file() and rows)
    mode = "a" if (resume and csv_path.is_file() and rows) else "w"

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer: csv.DictWriter | None = csv.DictWriter(f, fieldnames=fieldnames) if fieldnames else None
        if write_header and writer is not None:
            writer.writeheader()
            f.flush()

        def _record(res):
            nonlocal fieldnames, writer
            axis_comb, pol_name, seed, metrics = res
            row = {**axis_comb, "policy": pol_name, "seed": seed, **metrics}
            rows.append(row)
            if writer is None:
                fieldnames = [*axis_names, "policy", "seed", *metrics.keys()]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(row)
            f.flush()

        if max_workers <= 1 or len(pending) <= 1:
            for t in pending:
                _record(_worker_task(t))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_worker_task, t) for t in pending]
                for fut in concurrent.futures.as_completed(futures):
                    _record(fut.result())

    if shard_info is not None:
        return {
            "name": name,
            "runs_csv": str(csv_path),
            "summary_md": "",
            "total_runs": len(rows),
        }

    summary_path = write_suite_summary(
        out_dir, name, axis_names, combinations, policies, seeds, rows, report_block
    )
    return {
        "name": name,
        "runs_csv": str(csv_path),
        "summary_md": str(summary_path),
        "total_runs": len(rows),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="satsim co-simulation suite runner")
    ap.add_argument("suite", help="Path to suite YAML file")
    ap.add_argument("--out", default="", help="Output directory (default: runs/<name>)")
    ap.add_argument("--jobs", "-j", type=int, default=None,
                    help="Process pool concurrency (0 = auto cpu_count)")
    ap.add_argument("--shard", default="", help="Run 1-indexed shard I/N, e.g. 1/4")
    ap.add_argument("--resume", action="store_true", help="Skip cells already present in output CSV")
    ap.add_argument("--merge", action="store_true", help="Merge runs.shard-*.csv in --out and build summary.md")
    ap.add_argument("--set", action="append", default=[], help="CLI overrides (e.g. workload.arrival_window_s=10)")
    args = ap.parse_args(argv)

    suite_path = Path(args.suite)
    out_dir = Path(args.out) if args.out else None
    res = run_suite(
        suite_path,
        out_dir=out_dir,
        jobs=args.jobs,
        overrides=args.set,
        shard=args.shard or None,
        resume=args.resume,
        merge=args.merge,
    )
    print(f"Suite completed: {res['total_runs']} runs executed.")
    print(f"  CSV:     {res['runs_csv']}")
    if res["summary_md"]:
        print(f"  Summary: {res['summary_md']}")


if __name__ == "__main__":
    main()
