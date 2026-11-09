"""Upper bounds on deadline completions under a zero-transfer, pooled-GPU relaxation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simpy
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix

from cosim_burst_stress import PROFILES, profile_config
from satsim.cosim.policies import make_policy
from satsim.cosim.run import load_config, provenance
from satsim.cosim.trace import Trace
from satsim.cosim.world import CHUNK_S, CoWorld, make_requests
from satsim.method.engine.timing import iter_time


@dataclass(frozen=True)
class Job:
    release: float
    deadline: float
    work: float
    serial_work: float
    work_rate: float
    home: int | None = None
    home_until: float = 0.0


class WorkModel:
    def __init__(self, world):
        self.gpu = world.gpu
        self.bpt = world.bpt
        self.fixed = self.gpu["t_fixed_s"] + self.gpu["t_weights_s"]
        self.kv = self.gpu["t_kv_tok_s"]
        self.seq = self.gpu["t_seq_s"]
        long_reserve = min(r.reserved_tokens for r in world.reqs if r.long)
        self.batch_limit = math.floor(world.kv_cap / (long_reserve * world.bpt))
        if self.batch_limit < 1:
            raise ValueError("a long request must fit on an empty satellite")

    def job(self, request, release, generated, *, kv_copy_bps=None):
        # CoWorld holds a rate for up to CHUNK_S. Another resident's committed
        # progress can already be a chunk old when that rate is sampled.
        lag = CHUNK_S / iter_time(1, request.pin, 1.0, self.gpu)
        end = request.pout
        serial_base = self.fixed + self.kv * request.pin + self.seq
        pooled_base = self.fixed / self.batch_limit + self.kv * request.pin + self.seq

        def work(base, stale):
            growth = (max(0, end - stale)**2 - max(0, generated - stale)**2) / 2
            return (end - generated) * base + self.kv * growth

        # The work-per-token / serial-seconds-per-token ratio is monotone
        # between the lag breakpoints, so checking their endpoints suffices.
        endpoints = (generated, end, min(end, max(generated, lag)), min(end, max(generated, 2 * lag)))
        rate = max((pooled_base + self.kv * max(0, g - 2 * lag)) /
                   (serial_base + self.kv * max(0, g - lag)) for g in endpoints)
        home = request.node if kv_copy_bps else None
        home_until = release
        if home is not None:
            # A copy may read committed progress, which can be one chunk old.
            min_tokens = request.pin + max(0, generated - lag)
            home_until += min_tokens * self.bpt / kv_copy_bps
        return Job(release, request.deadline, work(pooled_base, 2 * lag), work(serial_base, lag), rate,
                   home, home_until)


class Capacity:
    def __init__(self, world):
        trace, gpu, f = world.tr, world.gpu, world.f_dark
        # Normalize into sunlit GPU-seconds. Take the least clock-induced
        # slowdown among terms so the resulting rate always rounds upward.
        fixed = gpu["t_fixed_s"] + gpu["t_weights_s"]
        dark_rate = 1 / min((gpu["t_fixed_s"] + gpu["t_weights_s"] / f**gpu["mem_freq_exp"]) / fixed,
                            1 / f**gpu["mem_freq_exp"], 1 / f)
        self.edges = np.array(sorted({0.0, trace.horizon, *trace.change_times()}))
        duration = np.diff(self.edges)
        midpoints = self.edges[:-1] + duration / 2
        rates = {j: np.array([dark_rate if trace.is_dark(j, t) else 1.0 for t in midpoints])
                 for j in trace.compute}
        values = np.array(list(rates.values()))
        self.pool_rate = values.sum(axis=0)
        self.pool = np.r_[0.0, np.cumsum(self.pool_rate * duration)]
        self.solo = np.r_[0.0, np.cumsum(values.max(axis=0) * duration)]
        self.nodes = {j: np.r_[0.0, np.cumsum(node_rates * duration)] for j, node_rates in rates.items()}

    def serial(self, time):
        return np.interp(time, self.edges, self.solo)

    def serial_between(self, job, start, end):
        available = self.serial(end) - self.serial(start)
        if job.home is not None and start < job.home_until:
            split = min(end, job.home_until)
            local = self.nodes[job.home]
            available -= self.serial(split) - self.serial(start)
            available += np.interp(split, self.edges, local) - np.interp(start, self.edges, local)
        return available


def pooled_bound(jobs, capacity, slot, time_limit):
    candidates = [j for j in jobs if j.release < j.deadline and
                  j.serial_work <= capacity.serial_between(j, j.release, j.deadline) + 1e-7]
    n = len(candidates)
    if not n:
        return 0, 0, 0, "empty"
    start = min(j.release for j in candidates)
    end = max(j.deadline for j in candidates)
    edges = np.r_[start, np.arange(start + slot, end, slot), end]
    slots = len(edges) - 1
    supply = np.diff(np.interp(edges, capacity.edges, capacity.pool))
    row, col, vals, limits = [], [], [], []
    for i, j in enumerate(candidates):
        first = max(0, np.searchsorted(edges, j.release, side="right") - 1)
        last = min(slots, np.searchsorted(edges, j.deadline, side="left"))
        for k in range(first, last):
            a, b = max(edges[k], j.release), min(edges[k + 1], j.deadline)
            limit = min(j.work, j.work_rate * capacity.serial_between(j, a, b))
            if limit <= 0:
                continue
            idx = len(limits)
            row.extend((k, slots + i))
            col.extend((idx, idx))
            vals.extend((1, -1))
            limits.append(limit)
    flows = len(limits)
    # Negative job rows require enough work if the integer completion is 1.
    work = np.array([j.work for j in candidates])
    matrix = coo_matrix((vals, (row, col)), shape=(slots + n, flows)).tocsc()
    all_selected = linprog(np.zeros(flows), A_ub=matrix, b_ub=np.r_[supply, -work],
                           bounds=np.c_[np.zeros(flows), limits], method="highs")
    if all_selected.success:
        return n, n, n, "all_serial_feasible"

    row.extend(range(slots, slots + n))
    col.extend(range(flows, flows + n))
    vals.extend(work)
    matrix = coo_matrix((vals, (row, col)), shape=(slots + n, flows + n)).tocsc()
    result = milp(np.r_[np.zeros(flows), -np.ones(n)],
                  integrality=np.r_[np.zeros(flows), np.ones(n)],
                  bounds=Bounds(0, np.r_[limits, np.ones(n)]),
                  constraints=LinearConstraint(matrix, -np.inf, np.r_[supply, np.zeros(n)]),
                  options={"time_limit": time_limit, "mip_rel_gap": 0})
    feasible = round(-result.fun) if result.fun is not None else 0
    dual = getattr(result, "mip_dual_bound", None)
    upper = min(n, math.floor(-dual + 1e-6)) if dual is not None and math.isfinite(dual) else n
    return feasible, upper, n, "optimal" if result.success else "mip_bound"


def observe(world, times, samples):
    for t in times:
        yield world.env.timeout(t - world.env.now)
        samples[t] = [(r, r.progress_at(t)) for rs in world.nodes.values() for r in rs if r.long]


def run_case(args):
    base, rate, profile, seed, slot, limit = args
    cfg, burst = profile_config(base, rate, profile, seed)
    trace = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    env = simpy.Environment()
    world = CoWorld(env, trace, cfg, make_policy(cfg["policies"]["orbit_thermal"]), seed=seed)
    reqs = make_requests(cfg, trace, seed)
    for req in reqs:
        world.submit(req)
    sample_times = [block + burst["duration_s"] + offset
                    for block in range(0, int(cfg["workload"]["arrival_window_s"]), burst["period_s"])
                    for offset in (0, 120, 300)]
    samples = {}
    env.process(observe(world, sample_times, samples))
    env.run(until=trace.horizon)
    capacity, work = Capacity(world), WorkModel(world)
    longs = [r for r in reqs if r.long]
    common = {"profile": profile, "rate_per_rs_min": rate, "seed": seed}

    def row(scope, eligible, jobs, when=None, tokens=None):
        best, upper, solo, status = pooled_bound(jobs, capacity, slot, limit)
        baseline = sum(world.good(r) for r in eligible)
        assert upper >= baseline, (common, scope, when, upper, baseline)
        return {**common, "scope": scope, "snapshot_s": when, "long_count": len(eligible),
                "baseline_good": baseline, "solo_eligible": solo, "pool_feasible": best,
                "pool_upper": upper, "solver": status, "remaining_tokens": tokens,
                "pool_work_gpu_s": sum(j.work for j in jobs)}

    # Release after the fastest possible upload with zero queue and a dedicated
    # first hop. All remaining network and placement restrictions are relaxed.
    first_hop_bps = max(float(cfg["net"]["link_gbps"][trace.kind[i]]) * 1e9 / 8
                       for i, (a, b) in enumerate(trace.edges) if a in trace.rs or b in trace.rs)
    isl_bps = max(world.net.cap[i] for i, (a, b) in enumerate(trace.edges)
                  if a in trace.compute and b in trace.compute)
    online_release = [work.job(r, r.t + r.upload_bytes / first_hop_bps, 0) for r in longs]
    frozen_release = [work.job(r, r.t_admit, 0) for r in longs if r.t_admit is not None]
    frozen_copy = [work.job(r, r.t_admit, 0, kv_copy_bps=isl_bps) for r in longs if r.t_admit is not None]
    rows = [row("all_ideal_admission", longs, online_release), row("all_fixed_admission", longs, frozen_release),
            row("all_fixed_admission_dedicated_copy", longs, frozen_copy)]
    for when, residents in samples.items():
        eligible = [r for r, _ in residents]
        jobs = [work.job(r, when, gen) for r, gen in residents]
        rows.append(row("inflight", eligible, jobs, when, sum(r.pout - gen for r, gen in residents)))
        jobs = [work.job(r, when, gen, kv_copy_bps=isl_bps) for r, gen in residents]
        rows.append(row("inflight_dedicated_copy", eligible, jobs, when, sum(r.pout - gen for r, gen in residents)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/cosim_placement_thermal.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--slot", type=float, default=30)
    ap.add_argument("--time-limit", type=float, default=10)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if min(args.jobs, args.slot, args.time_limit) <= 0:
        ap.error("jobs, slot and time limit must be positive")
    cfg = load_config(args.config)
    scenarios = ((14, "aligned150", range(11, 16)), (12, "aligned60", range(21, 26)))
    cases = [(cfg, rate, profile, seed, args.slot, args.time_limit)
             for rate, profile, seeds in scenarios for seed in (seeds[:1] if args.pilot else seeds)]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "bounds.csv").open("w", newline="") as file, ProcessPoolExecutor(args.jobs) as pool:
        writer = None
        for rows in pool.map(run_case, cases):
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=rows[0])
                writer.writeheader()
            writer.writerows(rows)
            file.flush()
            print(rows[0]["profile"], rows[0]["seed"],
                  [(r["scope"], r["baseline_good"], r["pool_upper"], r["long_count"]) for r in rows[:3]], flush=True)
    metadata = {**provenance(cfg), "slot_s": args.slot, "mip_time_limit_s": args.time_limit,
                "cases": [(rate, p, seed) for _, rate, p, seed, _, _ in cases]}
    (args.out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
