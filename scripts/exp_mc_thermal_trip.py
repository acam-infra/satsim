"""Mixed-Criticality Slack + Thermal Trip + P_dark/Wakeup benchmark on coop-w53 (seed=11, 5s/24/4)."""
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
    "good_urgent_frac",
    "good_normal_long_frac",
    "goodput_tok_s",
    "waste_ratio",
    "long_p95_all_s",
    "link_GB_hops",
    "zero_power_frac",
    "energy_kj_total",
    "wakeup_count",
    "min_thermal_margin_C",
    "throttle_s",
    "migrations",
]

POLICIES = {
    "place_once": {"place": "orbit_thermal", "migrate": "none"},
    "least_loaded": {"place": "least_loaded", "migrate": "none"},
    "cool_aware": {"place": "orbit_thermal", "migrate": "cool_aware"},
    "deadline_rc": {"policy": "thermal_rebalance", "placement": {"policy": "orbit_load"}, "thermal_aware": "rc"},
    "llumnix_style_mig_only": {"policy": "llumnix_style_mig_only"},
    "tapas_reactive_thermal": {"policy": "tapas_reactive_thermal"},
    "mcs_slack_dvfs_no_mig": {"policy": "mcs_slack_dvfs_no_mig"},
    "discrete_rh_greedy": {"policy": "discrete_rh_greedy", "placement": {"policy": "orbit_load"}},
    "discrete_rh_tier_v3": {"policy": "discrete_rh_tier_v3"},
    "discrete_rh_tier_v4": {"policy": "discrete_rh_tier_v4"},
    "discrete_rh_tier_v5": {"policy": "discrete_rh_tier_v5"},
}


def _worker(task: dict) -> dict:
    cfg = copy.deepcopy(task["cfg"])
    trace = Trace.load(Path(task["trace_path"]))
    t0 = time.perf_counter()
    w, _ = run_one(cfg, trace, POLICIES[task["policy"]], task["seed"], record=False)
    s = summarize(w)
    row = {
        "scenario": task["scenario"],
        "policy": task["policy"],
        "seed": task["seed"],
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    for m in METRICS:
        row[m] = s.get(m)
    return row


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = load_config(repo_root / "configs/cosim_discrete_power_w53_v1.yaml")
    trace_path = str((Path(base_cfg["_dir"]) / base_cfg["trace"]).resolve())

    mc_cfg = copy.deepcopy(base_cfg)
    mc_cfg["power"]["P_dark"] = 80.0
    mc_cfg["power"]["wakeup_energy_j"] = 2500.0
    mc_cfg["power"]["wakeup_latency_s"] = 3.0
    mc_cfg["power"]["tiers"] = [
        {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
        {"name": "low", "power_w": 75.0, "max_batch": 4, "clock_f": 0.50, "prefill_clock_f": 0.30},
        {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
    ]
    mc_cfg["thermal"]["tau_s"] = 60.0
    mc_cfg["thermal"]["T_max_C"] = 76.0
    mc_cfg["slo"]["urgent_frac"] = 0.25
    mc_cfg["slo"]["urgent_deadline_factor_long"] = 1.55
    mc_cfg["slo"]["deadline_factor_long"] = 2.60

    tasks = [
        {"scenario": "w53_mc_thermal_trip", "policy": pol, "seed": 11, "cfg": mc_cfg, "trace_path": trace_path}
        for pol in POLICIES
    ]

    out_dir = repo_root / "out" / "exp_mc_thermal_trip"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(10, len(tasks))) as ex:
        for f in as_completed([ex.submit(_worker, t) for t in tasks]):
            r = f.result()
            rows.append(r)
            print(
                f"[{r['scenario']}] {r['policy']:24s} SLO={r['good_long_frac']*100:6.2f}% "
                f"URG={((r.get('good_urgent_frac') or 0.0)*100):6.2f}% "
                f"E={r['energy_kj_total']:8.1f}kJ mig={r['migrations']:4d} "
                f"wake={r['wakeup_count']:4d} thr={r['throttle_s']:5.1f}s ({r['wall_s']}s)",
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
