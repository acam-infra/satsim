"""Multi-machine dynamic work-stealing coordinator for satsim-suite.

Example:
  uv run --frozen python scripts/dist_suite.py configs/suites/e4_fair_quota.yaml \\
    --hosts local@16,cnb-compute@16,tyy-237@4,host-192-168-9-68@4 --sync
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time

from satsim.cosim.suite import (
    _cell_key,
    _load_existing_rows,
    build_suite_plan,
    write_suite_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_host(spec: str) -> tuple[str, int]:
    host, sep, jobs_s = spec.strip().partition("@")
    return host, (int(jobs_s) if sep and jobs_s else 0)


def _sync_remote(host: str, remote_dir: str) -> None:
    """Tar-stream repo (without .venv/runs/.git) to remote_dir and ensure Python env."""
    excludes = ["--exclude=.git", "--exclude=.venv", "--exclude=runs", "--exclude=__pycache__",
                "--exclude=.pytest_cache", "--exclude=*.pyc"]
    tar_proc = subprocess.Popen(
        ["tar", "-czf", "-", *excludes, "-C", str(REPO_ROOT), "."],
        stdout=subprocess.PIPE,
    )
    remote_cmd = (
        f"mkdir -p {shlex.quote(remote_dir)} && "
        f"tar -xzf - -C {shlex.quote(remote_dir)} && "
        f"cd {shlex.quote(remote_dir)} && "
        f"export PATH=$HOME/.local/bin:/usr/local/bin:$PATH && "
        f"if command -v uv >/dev/null 2>&1; then "
        f"  (.venv/bin/python -c 'import simpy, numpy, scipy, yaml' 2>/dev/null) || "
        f"  uv sync --frozen --no-dev -q || "
        f"  uv sync --frozen --no-dev --no-config -q; "
        f"else "
        f"  if [ ! -d .venv ]; then python3 -m venv .venv; fi; "
        f"  .venv/bin/python -c 'import simpy, numpy, scipy, yaml' 2>/dev/null || "
        f"  .venv/bin/pip install -q --timeout 15 -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pyyaml scipy simpy || "
        f"  .venv/bin/pip install -q numpy pyyaml scipy simpy; "
        f"fi"
    )
    ssh_proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote_cmd],
        stdin=tar_proc.stdout,
        capture_output=True,
        text=True,
    )
    if tar_proc.stdout:
        tar_proc.stdout.close()
    tar_rc = tar_proc.wait()
    if tar_rc != 0 or ssh_proc.returncode != 0:
        raise RuntimeError(f"sync failed on {host} (tar={tar_rc}, ssh={ssh_proc.returncode}): {ssh_proc.stderr}")


def _launch_worker(host: str, jobs: int, suite_rel: str, overrides: list[str], remote_dir: str):
    worker_args = [suite_rel, "--jobs", str(jobs)]
    for ov in overrides:
        worker_args.extend(["--set", ov])
    quoted_args = " ".join(shlex.quote(a) for a in worker_args)
    if host in ("local", "localhost"):
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "dist_worker.py"), *worker_args]
        return subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    remote_cmd = (
        f"cd {shlex.quote(remote_dir)} && "
        f"export PATH=$HOME/.local/bin:/usr/local/bin:$PATH && "
        f"export PYTHONPATH=$PWD/src:${{PYTHONPATH:-}} && "
        f"if [ -x .venv/bin/python ]; then "
        f"  .venv/bin/python scripts/dist_worker.py {quoted_args}; "
        f"else "
        f"  uv run --frozen python scripts/dist_worker.py {quoted_args}; "
        f"fi"
    )
    return subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote_cmd],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Distributed satsim-suite coordinator")
    ap.add_argument("suite", help="Path to suite YAML file (relative to satsim repo)")
    ap.add_argument("--hosts", default="local@0",
                    help="Comma-separated host[@jobs] list, e.g. local@16,cnb-compute@16,tyy-237@4")
    ap.add_argument("--remote-dir", default="satsim_dist", help="Remote working directory")
    ap.add_argument("--sync", action="store_true", help="Sync code + deps to remote hosts before running")
    ap.add_argument("--out", default="", help="Output directory (default: runs/<name>)")
    ap.add_argument("--resume", action="store_true", help="Skip cells already present in runs.csv")
    ap.add_argument("--set", action="append", default=[], help="Suite overrides")
    args = ap.parse_args(argv)

    suite_path = Path(args.suite).resolve()
    try:
        suite_rel = str(suite_path.relative_to(REPO_ROOT))
    except ValueError:
        suite_rel = args.suite

    plan = build_suite_plan(suite_path, overrides=args.set)
    name = plan["name"]
    axis_names = plan["axis_names"]
    combinations = plan["combinations"]
    policies = plan["policies"]
    seeds = plan["seeds"]
    tasks = plan["tasks"]
    report_block = plan["report_block"]

    out_dir = Path(args.out) if args.out else REPO_ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "runs.csv"

    rows, done_keys = _load_existing_rows(csv_path, axis_names) if args.resume else ([], set())
    pending_indices = [
        idx for idx, t in enumerate(tasks)
        if _cell_key(axis_names, t[5], t[2], t[4]) not in done_keys
    ]

    host_specs = [_parse_host(h) for h in args.hosts.split(",") if h.strip()]
    if args.sync:
        remotes = [h for h, _ in host_specs if h not in ("local", "localhost")]
        if remotes:
            print(f"Syncing repo to {remotes}...", flush=True)
            with threading.BoundedSemaphore(len(remotes)):
                threads = []
                errs: list[tuple[str, Exception]] = []

                def _do_sync(h: str):
                    try:
                        _sync_remote(h, args.remote_dir)
                        print(f"  synced {h}", flush=True)
                    except Exception as e:
                        errs.append((h, e))

                for h in remotes:
                    th = threading.Thread(target=_do_sync, args=(h,))
                    th.start()
                    threads.append(th)
                for th in threads:
                    th.join()
                if errs:
                    for h, e in errs:
                        print(f"ERROR syncing {h}: {e}", file=sys.stderr)
                    sys.exit(1)

    if not pending_indices:
        print(f"All {len(tasks)} cells already completed in {csv_path}.", flush=True)
        summary_path = write_suite_summary(
            out_dir, name, axis_names, combinations, policies, seeds, rows, report_block
        )
        print(f"Summary: {summary_path}")
        return

    task_q: queue.Queue[int] = queue.Queue()
    for idx in pending_indices:
        task_q.put(idx)

    write_lock = threading.Lock()
    fieldnames: list[str] | None = list(rows[0].keys()) if rows else None
    mode = "a" if (args.resume and csv_path.is_file() and rows) else "w"
    t_start = time.perf_counter()
    completed_now = 0
    total_pending = len(pending_indices)

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer: csv.DictWriter | None = csv.DictWriter(f, fieldnames=fieldnames) if fieldnames else None
        if mode == "w" and writer is not None:
            writer.writeheader()
            f.flush()

        def _worker_thread(host: str, jobs: int):
            nonlocal fieldnames, writer, completed_now
            proc = _launch_worker(host, jobs, suite_rel, args.set, args.remote_dir)
            assert proc.stdin is not None and proc.stdout is not None
            ready_line = proc.stdout.readline()
            if not ready_line:
                err = proc.stderr.read() if proc.stderr else ""
                print(f"[{host}] failed to start: {err}", file=sys.stderr, flush=True)
                return
            info = json.loads(ready_line)
            slots = int(info["workers"])
            print(f"[{host}] ready ({slots} workers)", flush=True)

            in_flight = 0
            eof_sent = False
            try:
                while True:
                    while in_flight < slots and not eof_sent:
                        try:
                            idx = task_q.get_nowait()
                        except queue.Empty:
                            proc.stdin.write("QUIT\n")
                            proc.stdin.flush()
                            eof_sent = True
                            break
                        proc.stdin.write(f"{idx}\n")
                        proc.stdin.flush()
                        in_flight += 1
                    if in_flight == 0:
                        break
                    line = proc.stdout.readline()
                    if not line:
                        err = proc.stderr.read() if proc.stderr else ""
                        print(f"[{host}] exited unexpectedly with {in_flight} in-flight: {err}",
                              file=sys.stderr, flush=True)
                        break
                    msg = json.loads(line)
                    in_flight -= 1
                    row = msg["row"]
                    with write_lock:
                        rows.append(row)
                        if writer is None:
                            fieldnames = list(row.keys())
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                        writer.writerow(row)
                        f.flush()
                        completed_now += 1
                        elapsed = time.perf_counter() - t_start
                        print(f"  [{completed_now}/{total_pending}] {host:18s} "
                              f"pol={row['policy']:22s} seed={row['seed']} ({msg['wall_s']:.1f}s) "
                              f"[elapsed {elapsed:.1f}s]", flush=True)
            finally:
                try:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                    proc.stdin.close()
                except OSError:
                    pass
                proc.wait(timeout=15)

        threads = [
            threading.Thread(target=_worker_thread, args=(h, j), daemon=True)
            for h, j in host_specs
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    summary_path = write_suite_summary(
        out_dir, name, axis_names, combinations, policies, seeds, rows, report_block
    )
    total_wall = time.perf_counter() - t_start
    print(f"\nDistributed suite completed: {completed_now}/{total_pending} new runs in {total_wall:.1f}s "
          f"(total {len(rows)} rows).")
    print(f"  CSV:     {csv_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
