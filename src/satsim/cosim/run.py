"""Run a co-simulation experiment and write the summary + the viewer playback.

  satsim-cosim configs/cosim_coop_w53_v1.yaml [--out runs/<name>] [--viz <dir>] [--set k=v ...]

The YAML names an orbitsim trace and scenario (``orbitsim/export_cosim.py``), the hardware, the
workload and the policies to compare. Every policy sees the same arrivals for a given seed.
``--viz`` writes ``<dir>/<name>.json`` + updates ``<dir>/index.json`` for viz/play.html.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics as st
import sys
import time
from pathlib import Path

import simpy
import yaml

from satsim.cosim.policies import make_policy
from satsim.cosim.record import Recorder, cumulative_good, request_table, summarize
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, make_requests

REQUIRED = {"name", "trace", "scenario", "model", "gpu", "power", "net", "migration", "workload",
            "slo", "delivery", "control_period_s", "policies", "seeds"}
REBALANCE_INVARIANT_KEYS = ("interval_s", "max_inflight", "max_mig_per_req")
_INVARIANT_TARGETS = {"thermal_rebalance", "torch_rebalance", "torch_lp"}
_INVARIANT_MIGRATORS = {"reactive", "contact_aware"}


def apply_rebalance_invariants(cfg: dict, spec: dict) -> dict:
    """Inject the optional ``rebalance_invariants`` block into one policy spec.

    The block belongs to the fair-comparison harness, not to a strategy: it is applied to every
    strategy that owns migration quota. An explicit same-named value in the policy spec wins
    only when it is already equal; a different value is an error rather than a silent override.
    """
    inv = cfg.get("rebalance_invariants")
    if inv is None:
        return spec
    if not isinstance(inv, dict):
        raise ValueError("rebalance_invariants must be a mapping")
    out = copy.deepcopy(spec)
    kind = out.get("policy")
    if kind in _INVARIANT_TARGETS:
        accepts = True
    elif "place" in out:
        accepts = out.get("migrate", "none") in _INVARIANT_MIGRATORS
    else:
        accepts = False
    if not accepts:
        return out
    for key in REBALANCE_INVARIANT_KEYS:
        if key not in inv:
            continue
        if key in out and out[key] != inv[key]:
            raise ValueError(
                f"rebalance_invariants.{key}={inv[key]!r} conflicts with policy spec value "
                f"{out[key]!r} for {kind or out.get('place')}"
            )
        out[key] = inv[key]
    return out


def load_config(path: Path, overrides: list[str] = ()) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = REQUIRED - set(cfg)
    if missing:
        raise SystemExit(f"cosim config missing {sorted(missing)}")
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        *head, last = key.split(".")
        for k in head:
            node = node[k]
        node[last] = yaml.safe_load(raw)
    if cfg["delivery"] not in ("final", "stream"):
        raise SystemExit("delivery must be final | stream")
    cfg["_dir"] = str(path.parent.resolve())
    long_deadline_s = cfg["slo"].get("long_deadline_s")
    if long_deadline_s is not None:
        trace_path = (Path(cfg["_dir"]) / cfg["trace"]).resolve()
        horizon_s = float(json.loads(trace_path.read_text(encoding="utf-8"))["horizon_s"])
        arrival_window_s = float(cfg["workload"]["arrival_window_s"])
        if horizon_s - arrival_window_s < float(long_deadline_s):
            print(
                "warning: trace horizon minus arrival_window_s is shorter than long_deadline_s; "
                "long requests may not have time to finish and the all-request P95 may be "
                "dominated by censoring",
                file=sys.stderr,
            )
    return cfg


def _resolve(cfg: dict, key: str) -> Path:
    return (Path(cfg["_dir"]) / cfg[key]).resolve()


def run_one(cfg: dict, trace: Trace, pol_spec: dict, seed: int, record: bool = False):
    pol_spec = apply_rebalance_invariants(cfg, pol_spec)
    env = simpy.Environment()
    rec = Recorder(float(cfg.get("frame_s", 10.0))) if record else None
    w = CoWorld(env, trace, cfg, make_policy(pol_spec), recorder=rec, seed=seed)
    if rec is not None:
        rec.attach(w)
    for r in make_requests(cfg, trace, seed):
        w.submit(r)
    env.run(until=trace.horizon)
    return w, rec


def run(cfg: dict, viz_dir: Path | None = None, quiet: bool = False) -> dict:
    trace = Trace.load(_resolve(cfg, "trace"))
    seeds = [int(s) for s in cfg["seeds"]]
    rec_seed = int(cfg.get("record_seed", seeds[0]))
    rows, playback_runs = [], {}
    for name, spec in cfg["policies"].items():
        for seed in seeds:
            t0 = time.perf_counter()
            w, rec = run_one(cfg, trace, spec, seed, record=(seed == rec_seed and viz_dir is not None))
            row = {"policy": name, "seed": seed, **summarize(w), "wall_s": time.perf_counter() - t0}
            rows.append(row)
            if not quiet:
                print(f"  {name:14} seed={seed}  good={row['good']}/{row['n_req']}"
                      f"  good_long={row['good_long']}/{row['n_long']}  mig={row['migrations']}"
                      f"  abort={row['mig_abort']}  dark_tok={row['dark_tok_share']:.2f}"
                      f"  ({row['wall_s']:.1f}s)")
            if rec is not None:
                playback_runs[name] = {
                    "summary": row,
                    "frames": rec.frames,
                    # arrive/admit are implied by upload; the viewer draws paths, not every state
                    "events": [e for e in rec.events if e[1] not in ("arrive", "admit")],
                    "good_curve": cumulative_good(w, float(cfg.get("frame_s", 10.0)) * 3),
                }
                if cfg.get("playback_requests", False):
                    playback_runs[name]["requests"] = request_table(w)
    table = {}
    for name in cfg["policies"]:
        rs = [r for r in rows if r["policy"] == name]
        table[name] = {k: st.fmean([r[k] for r in rs if r[k] is not None]) if any(r[k] is not None for r in rs) else None
                       for k in rs[0] if k not in ("policy", "seed")}
    payload = {"name": cfg["name"], "trace": trace.name, "delivery": cfg["delivery"], "rows": rows, "mean": table}
    if viz_dir is not None:
        write_playback(cfg, trace, playback_runs, table, viz_dir)
    return payload


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(repo: Path) -> str | None:
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=repo, capture_output=True,
                             text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], cwd=repo,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (out.stdout.strip() + ("+dirty" if dirty else "")) or None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(cfg: dict) -> dict:
    """What produced a playback: lets satcosim `cosim.py check` flag playbacks older than their inputs."""
    import hashlib

    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return {
        "satsim_commit": _git_head(Path(__file__).resolve().parents[3]),
        "trace_sha256": _sha256(_resolve(cfg, "trace")),
        "scenario_sha256": _sha256(_resolve(cfg, "scenario")),
        "config_sha256": hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest(),
        "seeds": cfg["seeds"],
        "record_seed": cfg.get("record_seed", cfg["seeds"][0]),
    }


def write_playback(cfg: dict, trace: Trace, runs: dict, table: dict, viz_dir: Path) -> Path:
    scenario = json.loads(_resolve(cfg, "scenario").read_text(encoding="utf-8"))
    trace_raw = json.loads(_resolve(cfg, "trace").read_text(encoding="utf-8"))
    body = {
        "schema": "satsim.cosim_playback/v1",
        "name": cfg["name"],
        "provenance": provenance(cfg),
        "title": cfg.get("title", cfg["name"]),
        "description": cfg.get("description", ""),
        "takeaway": cfg.get("takeaway", ""),
        "delivery": cfg["delivery"],
        "frame_s": float(cfg.get("frame_s", 10.0)),
        "link_gbps": cfg["net"]["link_gbps"],
        "scenario": scenario,
        "trace": {k: trace_raw[k] for k in ("nodes", "edges", "dark", "horizon_s", "stats")},
        "policies": list(runs),
        "mean": table,
        "runs": runs,
    }
    viz_dir.mkdir(parents=True, exist_ok=True)
    out = viz_dir / f"{cfg['name']}.json"
    out.write_text(json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    idx_path = viz_dir / "index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else []
    idx = [e for e in idx if e["name"] != cfg["name"]]
    idx.append({"name": cfg["name"], "title": body["title"], "policies": body["policies"],
                "trace_sha256": body["provenance"]["trace_sha256"],
                "satsim_commit": body["provenance"]["satsim_commit"]})
    idx.sort(key=lambda e: e["name"])
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="satsim × orbitsim co-simulation")
    ap.add_argument("config")
    ap.add_argument("--out", default="")
    ap.add_argument("--viz", default="", help="directory for the viewer playback JSON")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. workload.rate_per_rs_per_min=8")
    ap.add_argument("--provenance-only", action="store_true",
                    help="print the provenance a fresh run would record, as JSON, and exit")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config), args.set)
    if args.provenance_only:
        print(json.dumps(provenance(cfg)))
        return
    print(f"{cfg['name']}  delivery={cfg['delivery']}")
    payload = run(cfg, Path(args.viz) if args.viz else None)
    dest = Path(args.out) if args.out else Path("runs") / cfg["name"]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    keys = ["good_frac", "good_long_frac", "goodput_tok_s", "lat_long_p50_s", "lat_long_p90_s",
            "dark_tok_share", "migrations", "mig_abort", "upload_fail"]
    print("mean over seeds:")
    print("  " + " ".join(f"{k:>15}" for k in ["policy"] + keys))
    for name, row in payload["mean"].items():
        print("  " + " ".join(f"{x:>15}" for x in [name] + [_fmt(row[k]) for k in keys]))
    print(f"wrote {dest / 'summary.json'}")


if __name__ == "__main__":
    main()
