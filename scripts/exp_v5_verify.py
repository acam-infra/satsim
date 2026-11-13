"""Verify discrete_rh_tier_v5 on w53_nonheat_wakeup (seed=11, 5s/24/4)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from satsim.cosim.record import summarize
from satsim.cosim.run import load_config, run_one
from satsim.cosim.trace import Trace

METRICS = [
    "good_long_frac",
    "goodput_tok_s",
    "waste_ratio",
    "long_p95_all_s",
    "link_GB_hops",
    "zero_power_frac",
    "energy_kj_total",
    "wakeup_count",
    "min_thermal_margin_C",
    "migrations",
]

POLICIES = {
    "place_once": {"place": "orbit_thermal", "migrate": "none"},
    "deadline_rc": {"policy": "thermal_rebalance", "placement": {"policy": "orbit_load"}, "thermal_aware": "rc"},
    "discrete_rh_greedy": {"policy": "discrete_rh_greedy", "placement": {"policy": "orbit_load"}},
    "discrete_rh_tier_v4": {"policy": "discrete_rh_tier_v4"},
    "discrete_rh_tier_v5": {"policy": "discrete_rh_tier_v5"},
}


def _worker(task: dict) -> dict:
    cfg = copy.deepcopy(task["cfg"])
    trace = Trace.load(Path(task["trace_path"]))
    t0 = time.perf_counter()
    w, _ = run_one(cfg, trace, POLICIES[task["policy"]], task["seed"], record=False)
    s = summarize(w)
    row = {"scenario": task["scenario"], "policy": task["policy"], "seed": task["seed"], "wall_s": round(time.perf_counter() - t0, 3)}
    for m in METRICS:
        row[m] = s.get(m)
    return row


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = load_config(repo_root / "configs/cosim_discrete_power_w53_v1.yaml")
    trace_path = str((Path(base_cfg["_dir"]) / base_cfg["trace"]).resolve())

    nhw_cfg = copy.deepcopy(base_cfg)
    nhw_cfg["power"]["P_dark"] = 80.0
    nhw_cfg["power"]["wakeup_energy_j"] = 2500.0
    nhw_cfg["power"]["wakeup_latency_s"] = 3.0
    nhw_cfg["power"]["tiers"] = [
        {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
        {"name": "low", "power_w": 75.0, "max_batch": 4, "clock_f": 0.50, "prefill_clock_f": 0.30},
        {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
    ]
    nhw_cfg["thermal"]["tau_s"] = 60.0
    nhw_cfg["thermal"]["T_max_C"] = 76.0

    tasks = [{"scenario": "w53_nonheat_wakeup", "policy": p, "seed": 11, "cfg": nhw_cfg, "trace_path": trace_path} for p in POLICIES]
    out_dir = repo_root / "out" / "exp_v5_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        for f in as_completed([ex.submit(_worker, t) for t in tasks]):
            r = f.result()
            rows.append(r)
            print(
                f"[{r['scenario']}] {r['policy']:22s} SLO={r['good_long_frac']*100:6.2f}% "
                f"waste={r['waste_ratio']*100:5.2f}% E={r['energy_kj_total']:8.1f}kJ "
                f"mig={r['migrations']:4d} wake={r['wakeup_count']:4d} ({r['wall_s']}s)",
                flush=True,
            )

    order = list(POLICIES.keys())
    rows.sort(key=lambda r: order.index(r["policy"]))
    with open(out_dir / "runs.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["scenario", "policy", "seed", "wall_s", *METRICS])
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(rows, fp, indent=2)
    print(f"Done {len(rows)} runs in {time.perf_counter() - t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
