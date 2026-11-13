"""Experiment: H1 (Prefill/Decode asymmetry) + H2 (min_dwell_s=20s) on coop-w53 (seed=11)."""
from __future__ import annotations

import copy
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from satsim.cosim.run import load_config, run_one, summarize
from satsim.cosim.trace import Trace

METRICS = [
    "good_long_frac",
    "good_urgent_frac",
    "waste_ratio",
    "goodput_tok_s",
    "energy_kj_total",
    "zero_power_frac",
    "link_GB_hops",
    "migrations",
    "wakeup_count",
    "throttle_s",
]

POLICIES = {
    "place_once": {"place": "orbit_thermal", "migrate": "none"},
    "least_loaded": {"policy": "least_loaded"},
    "cool_aware": {"place": "orbit_thermal", "migrate": "cool_aware"},
    "deadline_rc": {"policy": "thermal_rebalance", "placement": {"policy": "orbit_load"}, "thermal_aware": "rc"},
    "llumnix_style_mig_only": {"policy": "llumnix_style_mig_only"},
    "tapas_reactive_thermal": {"policy": "tapas_reactive_thermal"},
    "mcs_slack_dvfs_no_mig": {"policy": "mcs_slack_dvfs_no_mig"},
    "discrete_rh_greedy": {"policy": "discrete_rh_greedy"},
    "discrete_rh_tier_v6": {"policy": "discrete_rh_tier_v6"},
    "discrete_rh_tier_v7": {"policy": "discrete_rh_tier_v7"},
    "discrete_rh_tier_v8": {"policy": "discrete_rh_tier_v8"},
    "discrete_rh_tier_v9": {"policy": "discrete_rh_tier_v9"},
    "discrete_rh_tier_v10": {"policy": "discrete_rh_tier_v10"},
    "discrete_rh_tier_v11": {"policy": "discrete_rh_tier_v11"},
    "discrete_rh_tier_v12": {"policy": "discrete_rh_tier_v12"},
    "discrete_rh_tier_v13": {"policy": "discrete_rh_tier_v13"},
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

    cfg = copy.deepcopy(base_cfg)
    # H1: A100-grounded Prefill (compute-bound, 0.30x at 75W) vs Decode (bw-bound, 0.78x at 75W)
    cfg["gpu"]["t_prefill_tok_s"] = 0.005  # 20s solo prefill at 300W vs 66.7s at 75W for 4000 tok
    cfg["power"]["P_dark"] = 80.0
    cfg["power"]["wakeup_energy_j"] = 2500.0
    cfg["power"]["wakeup_latency_s"] = 3.0
    cfg["power"]["min_dwell_s"] = 20.0  # H2: anti-free-PWM minimum tier dwell
    cfg["power"]["tiers"] = [
        {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
        {"name": "low", "power_w": 75.0, "max_batch": 4, "clock_f": 0.78, "prefill_clock_f": 0.30},
        {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
    ]
    cfg["thermal"]["tau_s"] = 60.0
    cfg["thermal"]["T_max_C"] = 76.0
    cfg["thermal"]["hysteresis_C"] = 4.0
    cfg["slo"]["urgent_frac"] = 0.25
    cfg["slo"]["urgent_deadline_factor_long"] = 1.55
    cfg["slo"]["deadline_factor_long"] = 2.40

    tasks = [
        {"scenario": "w53_h1_h2_prefill_dwell", "policy": pol, "seed": 11, "cfg": cfg, "trace_path": trace_path}
        for pol in POLICIES
    ]

    out_dir = repo_root / "out" / "exp_h1_h2_prefill_dwell"
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
                f"thr={r['throttle_s']:6.1f}s ({r['wall_s']}s)",
                flush=True,
            )

    order = {p: i for i, p in enumerate(POLICIES)}
    rows.sort(key=lambda r: order[r["policy"]])
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(rows, fp, indent=2)
    with open(out_dir / "runs.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Done {len(rows)} runs in {time.perf_counter() - t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
