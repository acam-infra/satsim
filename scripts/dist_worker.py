"""Persistent single-host worker for distributed satsim-suite execution.

Reads 0-based task indices (one integer per line) from stdin, executes them in a
local ProcessPoolExecutor, and writes one JSON line per completed cell to stdout.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time

from satsim.cosim.suite import _worker_task, build_suite_plan


def _exec_indexed(args):
    idx, task = args
    t0 = time.perf_counter()
    axis_comb, pol_name, seed, metrics = _worker_task(task)
    return idx, {**axis_comb, "policy": pol_name, "seed": seed, **metrics}, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Distributed satsim-suite worker")
    ap.add_argument("suite", help="Path to suite YAML file")
    ap.add_argument("--jobs", "-j", type=int, default=0, help="Local worker processes (0 = cpu_count)")
    ap.add_argument("--set", action="append", default=[], help="Suite overrides")
    args = ap.parse_args(argv)

    plan = build_suite_plan(Path(args.suite), overrides=args.set)
    tasks = plan["tasks"]
    n_workers = (os.cpu_count() or 1) if args.jobs <= 0 else args.jobs

    print(json.dumps({"ready": True, "cpus": os.cpu_count() or 1, "workers": n_workers, "total": len(tasks)}),
          flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        in_flight: set[concurrent.futures.Future] = set()

        def _drain_done(block: bool = False) -> None:
            if not in_flight:
                return
            timeout = None if block else 0.0
            done, _ = concurrent.futures.wait(
                in_flight, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                in_flight.remove(fut)
                idx, row, wall_s = fut.result()
                print(json.dumps({"idx": idx, "wall_s": round(wall_s, 3), "row": row}), flush=True)

        for line in sys.stdin:
            s = line.strip()
            if not s:
                continue
            if s == "QUIT":
                break
            idx = int(s)
            in_flight.add(pool.submit(_exec_indexed, (idx, tasks[idx])))
            if len(in_flight) >= n_workers:
                _drain_done(block=True)
            else:
                _drain_done(block=False)

        while in_flight:
            _drain_done(block=True)


if __name__ == "__main__":
    main()
