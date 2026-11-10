"""Paired stress study of buffered remote-sensing releases at a fixed SLO."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cosim_placement_sweep import run_case
from satsim.cosim.run import load_config, provenance


PROFILES = {
    "smooth": (150, 90, None),
    "staggered": (150, 90, None),
    "aligned150": (150, 0, None),
    "aligned60": (60, 0, None),
    "regional120": (120, 0, 2),
}
BASELINES = ("round_robin", "nearest", "least_loaded", "greedy", "orbit_load", "orbit_thermal")
RUNTIME = ("speed", "deadline_rc", "torch_lp")


def profile_config(cfg, rate, profile, seed):
    cfg = copy.deepcopy(cfg)
    cfg["workload"]["rate_per_rs_per_min"] = rate
    duration, phase, source_count = PROFILES[profile]
    burst = {"period_s": 900, "duration_s": duration, "source_phase_s": phase}
    if source_count:
        burst["sources"] = [(seed - 1 + i) % 3 for i in range(source_count)]
    cfg["workload"]["release_burst"] = burst
    return cfg, burst


def evaluate(spec):
    cfg, rate, profile, seed, policy = spec
    cfg, burst = profile_config(cfg, rate, profile, seed)
    row, _ = run_case((cfg, seed, policy, rate, "smooth" if profile == "smooth" else "burst",
                       cfg["thermal"]["R_th"], "mixed", False))
    row.update(release_profile=profile, release_duration_s=burst["duration_s"] if profile != "smooth" else None,
               release_sources="/".join(map(str, burst.get("sources", (0, 1, 2)))) if profile != "smooth" else None,
               long_deadline_s=cfg["slo"]["long_deadline_s"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cosim_placement_thermal.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--stage", choices=("pilot", "confirm", "focus"), required=True)
    parser.add_argument("--rate", type=int)
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--secondary-profile", choices=("aligned150", "aligned60", "regional120"))
    parser.add_argument("--rs-gbps", type=float, help="override remote-sensing uplink capacity")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.rs_gbps is not None:
        if args.rs_gbps <= 0:
            parser.error("--rs-gbps must be positive")
        cfg["net"]["link_gbps"]["rs"] = args.rs_gbps
    metadata = provenance(cfg)
    placement = cfg["policies"]["orbit_thermal"]
    cfg["policies"].update(
        speed={"policy": "thermal_rebalance", "placement": placement,
               "deadline_aware": False, "externality": False},
        deadline_rc={"policy": "thermal_rebalance", "placement": placement, "thermal_aware": "rc"},
        torch_lp={"policy": "torch_rebalance", "placement": placement},
    )
    if args.stage == "pilot":
        seeds = (1,)
        cells = [(rate, profile, ("orbit_thermal", "deadline_rc"))
                 for rate in (12, 13, 14, 15) for profile in PROFILES if profile != "smooth"]
    else:
        if args.rate is None or args.profile in (None, "smooth", "staggered"):
            parser.error("confirmation requires a selected burst --profile and --rate")
        seeds = (11, 12, 13, 14, 15) if args.stage == "confirm" else (21, 22, 23, 24, 25)
        cells = [
            (args.rate, "smooth", ("orbit_thermal", "deadline_rc")),
        ]
        if args.stage == "confirm":
            cells.extend([(args.rate, "staggered", ("orbit_thermal", "deadline_rc")),
                          (args.rate, args.profile, BASELINES + RUNTIME)])
        else:
            cells.append((args.rate, args.profile, ("orbit_thermal", *RUNTIME)))
        if args.secondary_profile:
            if args.secondary_profile == args.profile:
                parser.error("the secondary release profile must differ from the primary profile")
            cells.append((args.rate, args.secondary_profile, ("orbit_thermal", *RUNTIME)))
    cases = [(cfg, rate, profile, seed, policy) for rate, profile, policies in cells
             for seed in seeds for policy in policies]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "provenance.json").write_text(json.dumps({**metadata, "stage": args.stage, "seeds": seeds,
                                                         "cells": cells, "profiles": PROFILES,
                                                         "rs_gbps": cfg["net"]["link_gbps"]["rs"]}, indent=2) + "\n")
    with (args.out / "runs.csv").open("w", newline="") as output, ProcessPoolExecutor(args.jobs) as pool:
        writer = None
        for i, row in enumerate(pool.map(evaluate, cases), 1):
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            output.flush()
            print(f"{i}/{len(cases)} {row['release_profile']} {row['rate_per_rs_min']} {row['policy']} "
                  f"seed={row['seed']} long_slo={row['good_long_frac']:.3f}", flush=True)


if __name__ == "__main__":
    main()
