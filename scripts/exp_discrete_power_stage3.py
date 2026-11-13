"""Stage 3 evaluation: 5-seed main tables (coop-sso2 & coop-w53) + BW/Thermal frontier sweep."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import csv
import json
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from satsim.cosim.record import summarize
from satsim.cosim.run import load_config, run_one
from satsim.cosim.trace import Trace

METRICS = [
    "good_long_frac",
    "goodput_tok_s",
    "waste_ratio",
    "long_p95_all_s",
    "link_GB_hops",
    "tether_GB",
    "zero_power_frac",
    "energy_kj_total",
    "min_thermal_margin_C",
    "migrations",
]

_TRACE_CACHE: dict[str, Trace] = {}


def _get_trace(trace_path: str) -> Trace:
    if trace_path not in _TRACE_CACHE:
        _TRACE_CACHE[trace_path] = Trace.load(Path(trace_path))
    return _TRACE_CACHE[trace_path]


def _worker(task: dict) -> dict:
    cfg = copy.deepcopy(task["cfg"])
    trace = _get_trace(task["trace_path"])
    pol_spec = cfg["policies"][task["policy"]]
    t0 = time.perf_counter()
    w, _ = run_one(cfg, trace, pol_spec, task["seed"], record=False)
    s = summarize(w)
    row = {
        "scenario": task["scenario"],
        "sweep": task["sweep"],
        "param": task["param"],
        "policy": task["policy"],
        "seed": task["seed"],
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    for m in METRICS:
        row[m] = s.get(m)
    return row


def _build_tasks(repo_root: Path, sweep_seeds: list[int]) -> list[dict]:
    tasks: list[dict] = []
    # 1. 5-seed main tables on coop-sso2 and coop-w53
    for sc_name, cfg_rel in [
        ("coop_sso2", "configs/cosim_discrete_power_v1.yaml"),
        ("coop_w53", "configs/cosim_discrete_power_w53_v1.yaml"),
    ]:
        cfg = load_config(repo_root / cfg_rel)
        cfg["seeds"] = [11, 12, 13, 14, 15]
        trace_path = str((Path(cfg["_dir"]) / cfg["trace"]).resolve())
        for pol in cfg["policies"]:
            for seed in cfg["seeds"]:
                tasks.append({
                    "scenario": sc_name,
                    "sweep": "main",
                    "param": "default",
                    "policy": pol,
                    "seed": seed,
                    "cfg": cfg,
                    "trace_path": trace_path,
                })

    # 2. Frontier sweeps on coop-w53
    w53_cfg = load_config(repo_root / "configs/cosim_discrete_power_w53_v1.yaml")
    w53_trace = str((Path(w53_cfg["_dir"]) / w53_cfg["trace"]).resolve())
    sweep_pols = ["place_once", "deadline_rc", "discrete_rh_greedy"]

    for bw in [1.0, 2.5, 5.0, 10.0, 25.0]:
        c = copy.deepcopy(w53_cfg)
        c["net"]["link_gbps"] = {"intra": bw, "inter": bw, "rs": bw}
        for pol in sweep_pols:
            for seed in sweep_seeds:
                tasks.append({
                    "scenario": "coop_w53",
                    "sweep": "bw_gbps",
                    "param": str(bw),
                    "policy": pol,
                    "seed": seed,
                    "cfg": c,
                    "trace_path": w53_trace,
                })

    for tmax in [72.0, 76.0, 80.0, 84.0]:
        c = copy.deepcopy(w53_cfg)
        c["thermal"]["T_max_C"] = tmax
        for pol in sweep_pols:
            for seed in sweep_seeds:
                tasks.append({
                    "scenario": "coop_w53",
                    "sweep": "t_max_c",
                    "param": str(tmax),
                    "policy": pol,
                    "seed": seed,
                    "cfg": c,
                    "trace_path": w53_trace,
                })
    return tasks


def _aggregate(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["sweep"], r["param"], r["policy"]), []).append(r)

    summary: dict = {}
    for (sc, sw, param, pol), grp in sorted(groups.items()):
        key = f"{sc}|{sw}|{param}|{pol}"
        entry = {"scenario": sc, "sweep": sw, "param": param, "policy": pol, "n": len(grp)}
        for m in METRICS:
            vals = [g[m] for g in grp if g[m] is not None]
            entry[m] = round(st.fmean(vals), 4) if vals else None
        summary[key] = entry
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/stage3_discrete_power")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sweep-seeds", nargs="+", type=int, default=[11])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["scenario", "sweep", "param", "policy", "seed", "wall_s", *METRICS]
    csv_path = out_dir / "runs.csv"
    rows: list[dict] = []
    done_keys: set[tuple[str, str, str, str, int]] = set()
    if args.resume and csv_path.exists():
        for r in csv.DictReader(csv_path.open("r", encoding="utf-8")):
            k = (r["scenario"], r["sweep"], r["param"], r["policy"], int(r["seed"]))
            done_keys.add(k)
            parsed = dict(r)
            parsed["seed"] = int(r["seed"])
            parsed["wall_s"] = float(r["wall_s"])
            for m in METRICS:
                parsed[m] = float(r[m]) if r.get(m) not in (None, "", "None") else None
            rows.append(parsed)

    all_tasks = _build_tasks(repo_root, args.sweep_seeds)
    tasks = [
        t for t in all_tasks
        if (t["scenario"], t["sweep"], t["param"], t["policy"], t["seed"]) not in done_keys
    ]
    print(f"Running {len(tasks)} Stage 3 runs ({len(done_keys)} resumed) with {args.workers} workers...", flush=True)
    t0 = time.perf_counter()
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        csv_f.flush()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for idx, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                rows.append(r)
                writer.writerow(r)
                csv_f.flush()
                print(
                    f"  [{idx}/{len(tasks)}] {r['scenario']} {r['sweep']}={r['param']} "
                    f"{r['policy']} s={r['seed']} wall={r['wall_s']}s "
                    f"elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

    rows.sort(key=lambda r: (r["scenario"], r["sweep"], r["param"], r["policy"], r["seed"]))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = _aggregate(rows)
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {csv_path} and {json_path} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
