"""Paired placement-only experiment on the existing contact trace.

Run from the repository root: python scripts/cosim_placement_sweep.py --out runs/placement
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import simpy

from satsim.cosim.policies import OrbitLoad, make_policy, path_lifetime, transfer_estimate
from satsim.cosim.record import summarize
from satsim.cosim.run import load_config, provenance
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, deadline_for, make_requests


class Probe:
    def __init__(self, world):
        self.w = world
        self.eta = OrbitLoad()
        self.windows = {}
        self.samples = []
        self.w.env.process(self.sample())

    def sample(self):
        w = self.w
        while True:
            yield w.env.timeout(30)
            now = w.env.now
            idle = [j for j in w.tr.compute if not w.nodes[j] and w.reserved[j] == 0]
            hot = [j for j in w.tr.compute if w.nodes[j] and w.thermal_f[j] < 1 and not w.tr.is_dark(j, now)]
            self.samples.append([now, len(hot), len(idle), max(map(len, w.nodes.values()))])
            if not idle:
                continue
            for src in hot:
                for r in w.nodes[src]:
                    if not r.long or r.i in self.windows or now >= r.deadline:
                        continue
                    stay = now + r.remaining * w.iter_of(src)
                    if stay <= r.deadline:
                        continue
                    for dst in idle:
                        if not w.fits(dst, r):
                            continue
                        path = w.tr.path(src, dst, now)
                        if path is None:
                            continue
                        dt = transfer_estimate(w, path, r.kv * w.bpt)
                        if path_lifetime(w, path) < now + dt:
                            continue
                        # Hindsight remaining length, frozen batch and no source progress in flight.
                        move = self.eta._delivery(w, r, r.remaining, dst, dt)
                        if move <= r.deadline:
                            self.windows[r.i] = [now, src, dst, stay, move]
                            break


def run_case(case):
    cfg, seed, policy, rate, arrival, resistance, length_mode, incident = case
    cfg = copy.deepcopy(cfg)
    wl = cfg["workload"]
    wl["rate_per_rs_per_min"] = rate
    if incident:
        source = (seed - 1) % 3  # rotate which physical access trace carries the incident
        wl["source_rates_per_min"] = [rate / 2] * 3
        wl["source_rates_per_min"][source] = 2 * rate
        wl["release_burst"] = {"period_s": 900, "duration_s": 240, "source_phase_s": 0, "sources": [source]}
    if arrival == "smooth":
        del wl["release_burst"]
    if resistance:
        thermal = cfg["thermal"]
        thermal["R_th"] = resistance
        thermal["T0_C"] = thermal["T_amb_C"] + resistance * cfg["power"]["P_idle"]
    else:
        del cfg["thermal"]
    trace = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    env = simpy.Environment()
    w = CoWorld(env, trace, cfg, make_policy(cfg["policies"][policy]))
    reqs = make_requests(cfg, trace, seed)
    if length_mode == "equal_long":
        # Preserve every source's total output work and its number of long requests.
        for origin in trace.rs:
            group = [r for r in reqs if r.origin == origin and r.long]
            if group:
                base, extra = divmod(sum(r.pout for r in group), len(group))
                for i, r in enumerate(group):
                    r.pout = base + int(i < extra)
                    r.deadline = deadline_for(cfg, r.t, r.pin, r.pout, r.long)
    assert max((r.deadline for r in reqs), default=0) < trace.horizon
    for r in reqs:
        w.submit(r)
    probe = Probe(w)
    env.run(until=trace.horizon)
    metrics = summarize(w)
    window_min = wl["arrival_window_s"] / 60
    placement = Counter(r.trail[0] for r in reqs if r.trail)
    missed_long = [r for r in reqs if r.long and not w.good(r)]
    row = {
        "rate_per_rs_min": rate, "arrival": arrival, "resistance": resistance, "length_mode": length_mode,
        "scenario": "incident" if incident else "distributed",
        "policy": policy, "seed": seed, **metrics,
        "good_req_min": metrics["good"] / window_min,
        "good_long_req_min": metrics["good_long"] / window_min,
        "output_tokens": sum(r.pout for r in reqs),
        "top_quartile_placements": sum(x[1] for x in placement.most_common(len(trace.compute) // 4))
        / max(1, placement.total()),
        "thermal_busy_node_min": sum(x[1] for x in probe.samples) / 2,
        "hot_and_idle_min": sum(bool(x[1] and x[2]) for x in probe.samples) / 2,
        "missed_long_with_window": sum(r.i in probe.windows for r in missed_long),
    }
    detail = None
    if rate == 12 and arrival == "burst" and resistance == 0.30 and seed == 1 and length_mode == "mixed":
        detail = {"policy": policy, "samples": probe.samples, "windows": probe.windows, "reqs": [
            [r.long, (r.t_delivered - r.t) / (r.deadline - r.t) if r.t_delivered is not None else None,
             r.trail, r.t, r.t_delivered, r.deadline]
            for r in reqs
        ]}
    return row, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/cosim_placement_thermal.yaml"))
    ap.add_argument("--out", type=Path, default=Path("runs/placement"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--rebalance", action="store_true", help="compare migration with a common initial policy")
    ap.add_argument("--incident", action="store_true", help="one bursty incident source plus two routine sources")
    ap.add_argument("--speed-control", action="store_true", help="run completion-time-only migration control")
    ap.add_argument("--no-heat-forecast", action="store_true", help="ablate temperature information for migration")
    ap.add_argument("--rc-forecast", action="store_true", help="forecast temperature transitions and recovery")
    ap.add_argument("--sensitivity-only", action="store_true", help="run cooling and output-length controls")
    ap.add_argument("--high-load", action="store_true", help="run the saturated 24 requests/RS/min setting")
    args = ap.parse_args()
    cfg = load_config(args.config)
    inputs = provenance(cfg)
    seeds = cfg["seeds"][:1] if args.pilot else cfg["seeds"]
    policies = cfg["policies"]
    base_cases = [(12, a, r, "mixed", policies) for a in ("smooth", "burst") for r in (0, 0.30)]
    extra_cases = [] if args.pilot else [
        *((rate, "burst", 0.30, "mixed", policies) for rate in (8, 16)),
        *((12, "burst", r, "mixed", ("nearest", "greedy", "orbit_thermal")) for r in (0.24, 0.36)),
        (12, "burst", 0.30, "equal_long", policies),
    ]
    if args.rebalance:
        base_cases, extra_cases = [], []
        for placement in ("nearest", "orbit_thermal"):
            variants = [placement]
            for tag, externality in (("self", False), ("joint", True)):
                name = f"{placement}_{tag}"
                policies[name] = {"policy": "thermal_rebalance", "placement": policies[placement],
                                  "externality": externality}
                variants.append(name)
            for rate in ((12,) if args.pilot else (8, 12, 16)):
                base_cases.append((rate, "burst", 0.30, "mixed", variants))
            if not args.pilot:
                base_cases.append((12, "burst", 0, "mixed", variants))
    if args.incident:
        if args.rebalance:
            base_cases = [case for case in base_cases if case[0] == 12 and case[2] > 0]
        else:
            comparators = ("nearest", "round_robin", "greedy", "orbit_thermal")
            base_cases = [(12, a, r, "mixed", comparators) for a in ("smooth", "burst") for r in (0, 0.30)]
            extra_cases = []
    if args.speed_control:
        policies["orbit_thermal_speed"] = {"policy": "thermal_rebalance", "placement": policies["orbit_thermal"],
                                           "externality": False, "deadline_aware": False}
        base_cases = [(rate, "burst", 0.30, "mixed", ("orbit_thermal_speed",)) for rate in (8, 12, 16)]
        extra_cases = []
    if args.no_heat_forecast:
        policies["orbit_thermal_no_forecast"] = {"policy": "thermal_rebalance", "placement": policies["orbit_thermal"],
                                                 "thermal_aware": False}
        base_cases = [(rate, "burst", 0.30, "mixed", ("orbit_thermal_no_forecast",)) for rate in (8, 12, 16)]
        extra_cases = []
    if args.rc_forecast:
        policies["orbit_thermal_rc"] = {"policy": "thermal_rebalance", "placement": policies["orbit_thermal"],
                                        "thermal_aware": "rc"}
        base_cases = [(rate, "burst", 0.30, "mixed", ("orbit_thermal_rc",)) for rate in (8, 12, 16)]
        extra_cases = []
    if args.sensitivity_only:
        base_cases = []
        extra_cases = [case for case in extra_cases if case[0] == 12]
    if args.high_load:
        base_cases = [(24, "burst", 0.30, "mixed", policies)]
        extra_cases = []
    cases = [(cfg, seed, p, rate, arrival, resistance, length, args.incident) for rate, arrival, resistance, length, ps
             in base_cases + extra_cases for seed in seeds for p in ps]
    args.out.mkdir(parents=True, exist_ok=True)
    details = []
    with (args.out / "runs.csv").open("w", newline="") as output, ProcessPoolExecutor(args.jobs) as pool:
        writer = None
        for i, (row, detail) in enumerate(pool.map(run_case, cases), 1):
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            output.flush()
            if detail:
                details.append(detail)
            print(f"{i}/{len(cases)} {row['policy']} seed={row['seed']} good/min={row['good_req_min']:.2f}", flush=True)
    (args.out / "case_seed1.json").write_text(json.dumps(details), encoding="utf-8")
    (args.out / "provenance.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
