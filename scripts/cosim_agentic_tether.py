"""Agentic Remote-Sensing Multi-Class Data-Anchor Tethering & Burst Benchmark.

Evaluates two core paper claims:
1. Standard satsim burst profiles (aligned60, aligned150, regional120 across 5 seeds each):
   comparing initial placement (orbit_thermal), 1-to-1 speed migration (speed),
   1-to-1 RC-forecast migration (deadline_rc), and Ours (torch_lp, plus ablations).
2. Agentic Multi-Class Data-Anchor Tethering (3 heterogeneous remote-sensing business classes):
   - Type 1 (scan_high_reaccess): wide-area scan with repeated high-res raw-image tile pulls
     (m_c = 5 pulls x 125 MB) from the initial ingress compute satellite u_0 (data anchor).
   - Type 2 (pinpoint_deep): pinpoint deep chain-of-thought reasoning (m_c = 0 pulls), pure
     long decode with high thermal load.
   - Type 3 (phase_transition): wide-area scan in phase 1 (pulls tiles) transitioning to pure
     deep reasoning in phase 2 (m_c -> 0).
   Demonstrates the concentric distance-tether sorting law (average hop distance from u_0,
   tether ISL overhead, and per-class SLO attainment).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import simpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cosim_burst_stress import profile_config
from satsim.cosim.policies import make_policy
from satsim.cosim.record import summarize
from satsim.cosim.run import load_config
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, make_requests


def run_single(spec: dict) -> dict:
    cfg = copy.deepcopy(spec["cfg"])
    profile = spec["profile"]
    rate = spec["rate"]
    seed = spec["seed"]
    pol_name = spec["policy_name"]
    pol_spec = spec["policy_spec"]
    agentic = spec.get("agentic", False)
    mode_tag = spec.get("mode_tag", "agentic" if agentic else "standard")
    isl_gbps = spec.get("isl_gbps")

    cfg, _ = profile_config(cfg, rate, profile, seed)
    if isl_gbps is not None:
        cfg["net"]["link_gbps"]["intra"] = float(isl_gbps)
        cfg["net"]["link_gbps"]["inter"] = float(isl_gbps)
    if agentic:
        cfg["workload"]["agentic_classes"] = {
            "p_scan": 0.35,
            "p_pinpoint": 0.40,
            "tile_MB": 250.0,
            "scan_pulls": 5,
            "pinpoint_pulls": 0,
            "phase_pulls": 4,
        }

    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    env = simpy.Environment()
    pol = make_policy(pol_spec)
    w = CoWorld(env, tr, cfg, pol, seed=seed)
    reqs = make_requests(cfg, tr, seed)
    for r in reqs:
        w.submit(r)
    env.run(until=tr.horizon)

    s = summarize(w)
    th_diag = w.thermal.diagnostics() if w.thermal else {"throttle_time_s_sum": 0.0}

    # Compute per-class metrics and average final hop distance from anchor_node
    longs = [r for r in reqs if r.long]
    class_stats = {}
    for cls_name in ("scan_high_reaccess", "pinpoint_deep", "phase_transition"):
        cls_reqs = [r for r in longs if r.task_class == cls_name]
        if cls_reqs:
            good_c = sum(1 for r in cls_reqs if w.good(r)) / len(cls_reqs)
            hops_c = []
            for r in cls_reqs:
                if r.anchor_node >= 0 and r.node >= 0:
                    if r.anchor_node == r.node:
                        hops_c.append(0)
                    else:
                        p = tr.path(r.anchor_node, r.node, min(tr.horizon - 1.0, r.t_decoded or tr.horizon - 1.0))
                        hops_c.append(len(p[1]) if (p and p[1]) else 3)
            class_stats[f"good_{cls_name}"] = good_c
            class_stats[f"hops_{cls_name}"] = st.fmean(hops_c) if hops_c else 0.0
            class_stats[f"mig_frac_{cls_name}"] = sum(1 for r in cls_reqs if r.n_mig > 0) / len(cls_reqs)
        else:
            class_stats[f"good_{cls_name}"] = 0.0
            class_stats[f"hops_{cls_name}"] = 0.0
            class_stats[f"mig_frac_{cls_name}"] = 0.0

    tether_gb = w.acc.get("tether_bytes", 0.0) / 1e9
    tether_s_mean = st.fmean([r.tether_s for r in longs]) if longs else 0.0

    return {
        "mode": mode_tag,
        "profile": profile,
        "rate": rate,
        "seed": seed,
        "policy": pol_name,
        "good_long": s["good_long"],
        "n_long": s["n_long"],
        "good_long_frac": s["good_long_frac"],
        "good_frac": s["good_frac"],
        "goodput_tok_s": s["goodput_tok_s"],
        "lat_long_p50_s": s["lat_long_p50_s"],
        "lat_long_p90_s": s["lat_long_p90_s"],
        "migrations": s["migrations"],
        "mig_abort": s["mig_abort"],
        "mig_GB": s["mig_GB"],
        "tether_GB": tether_gb,
        "tether_s_mean": tether_s_mean,
        "link_GB_hops": s["link_GB_hops"],
        "dark_tok_share": s["dark_tok_share"],
        "throttle_s": th_diag["throttle_time_s_sum"],
        **class_stats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/cosim_placement_thermal.yaml"))
    ap.add_argument("--out", type=Path, default=Path("runs/torch_paper_eval"))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    base = load_config(args.config)
    placement = base["policies"]["orbit_thermal"]

    std_policies = {
        "round_robin": base["policies"]["round_robin"],
        "nearest": base["policies"]["nearest"],
        "orbit_thermal": placement,
        "speed": {
            "policy": "thermal_rebalance",
            "placement": placement,
            "deadline_aware": False,
            "externality": False,
        },
        "deadline_rc": {
            "policy": "thermal_rebalance",
            "placement": placement,
            "thermal_aware": "rc",
        },
        "torch_no_relay": {
            "policy": "torch_rebalance",
            "placement": placement,
            "pre_admission_relay": False,
            "viable_slack_s": 0.0,
        },
        "torch_lp": {
            "policy": "torch_rebalance",
            "placement": placement,
            "pre_admission_relay": True,
            "viable_slack_s": 485.0,
        },
    }

    agentic_policies = {
        "orbit_thermal": placement,
        "deadline_rc": {
            "policy": "thermal_rebalance",
            "placement": placement,
            "thermal_aware": "rc",
        },
        "torch_tether_blind": {
            "policy": "torch_rebalance",
            "placement": placement,
            "tether_aware": False,
            "viable_slack_s": 485.0,
        },
        "torch_lp": {
            "policy": "torch_rebalance",
            "placement": placement,
            "tether_aware": True,
            "viable_slack_s": 485.0,
        },
    }

    tasks = []
    # 1. Standard Burst Benchmarks (5 seeds each)
    for profile, rate, seeds in [
        ("aligned150", 14, (11, 12, 13, 14, 15)),
        ("regional120", 14, (11, 12, 13, 14, 15)),
        ("aligned60", 12, (21, 22, 23, 24, 25)),
    ]:
        for seed in seeds:
            for pname, pspec in std_policies.items():
                tasks.append({
                    "cfg": base,
                    "profile": profile,
                    "rate": rate,
                    "seed": seed,
                    "policy_name": pname,
                    "policy_spec": pspec,
                    "agentic": False,
                    "mode_tag": "standard",
                })

    # 2. Agentic Multi-Class Data-Anchor Tethering Benchmarks (10 Gbps ISL & 3 Gbps Constrained ISL)
    for mode_tag, isl_gbps in [("agentic", 10.0), ("agentic_isl3g", 3.0)]:
        for profile, rate, seeds in [
            ("aligned150", 14, (11, 12, 13, 14, 15)),
            ("aligned60", 12, (21, 22, 23, 24, 25)),
        ]:
            for seed in seeds:
                for pname, pspec in agentic_policies.items():
                    tasks.append({
                        "cfg": base,
                        "profile": profile,
                        "rate": rate,
                        "seed": seed,
                        "policy_name": pname,
                        "policy_spec": pspec,
                        "agentic": True,
                        "mode_tag": mode_tag,
                        "isl_gbps": isl_gbps,
                    })

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "runs.csv"
    existing_rows = []
    done_keys = set()
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Convert numeric fields back
                parsed = {}
                for k, v in r.items():
                    if k in ("mode", "profile", "policy"):
                        parsed[k] = v
                    elif k in ("rate", "seed", "good_long", "n_long", "migrations", "mig_abort"):
                        parsed[k] = int(float(v))
                    else:
                        parsed[k] = float(v)
                existing_rows.append(parsed)
                done_keys.add((parsed["mode"], parsed["profile"], parsed["rate"], parsed["seed"], parsed["policy"]))

    rem_tasks = [
        t for t in tasks
        if (t["mode_tag"], t["profile"], t["rate"], t["seed"], t["policy_name"])
        not in done_keys
    ]
    print(f"Loaded {len(existing_rows)} existing runs; {len(rem_tasks)} remaining tasks to run.", flush=True)

    rows = list(existing_rows)
    if rem_tasks:
        write_header = not existing_rows
        with csv_path.open("a", newline="") as f, ProcessPoolExecutor(args.jobs) as pool:
            writer = csv.DictWriter(f, fieldnames=list(existing_rows[0].keys())) if existing_rows else None
            for idx, row in enumerate(pool.map(run_single, rem_tasks), 1):
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)
                f.flush()
                rows.append(row)
                print(
                    f"[{len(existing_rows)+idx}/{len(tasks)}] {row['mode']:8s} {row['profile']:11s} s={row['seed']} "
                    f"{row['policy']:18s} -> good_long={row['good_long_frac']:.3f} migs={row['migrations']}",
                    flush=True,
                )

    # Aggregate summary
    summary = {}
    groups = {}
    for r in rows:
        key = (r["mode"], r["profile"], r["rate"], r["policy"])
        groups.setdefault(key, []).append(r)

    print("\n================ AGGREGATED MULTI-SEED SUMMARY ================")
    for (mode, prof, rate, pol), grp in sorted(groups.items()):
        gl = st.fmean(x["good_long_frac"] for x in grp)
        ga = st.fmean(x["good_frac"] for x in grp)
        gp = st.fmean(x["goodput_tok_s"] for x in grp)
        p90 = st.fmean(x["lat_long_p90_s"] for x in grp)
        migs = st.fmean(x["migrations"] for x in grp)
        thr = st.fmean(x["throttle_s"] for x in grp)
        dsh = st.fmean(x["dark_tok_share"] for x in grp)
        tgb = st.fmean(x["tether_GB"] for x in grp)
        summary[f"{mode}:{prof}:{pol}"] = {
            "good_long_frac": round(gl, 4),
            "good_frac": round(ga, 4),
            "goodput_tok_s": round(gp, 1),
            "lat_long_p90_s": round(p90, 1),
            "migrations": round(migs, 1),
            "throttle_s": round(thr, 0),
            "dark_tok_share": round(dsh, 4),
            "tether_GB": round(tgb, 1),
            "good_scan": round(st.fmean(x["good_scan_high_reaccess"] for x in grp), 4),
            "good_pinpoint": round(st.fmean(x["good_pinpoint_deep"] for x in grp), 4),
            "good_phase": round(st.fmean(x["good_phase_transition"] for x in grp), 4),
            "hops_scan": round(st.fmean(x["hops_scan_high_reaccess"] for x in grp), 2),
            "hops_pinpoint": round(st.fmean(x["hops_pinpoint_deep"] for x in grp), 2),
            "hops_phase": round(st.fmean(x["hops_phase_transition"] for x in grp), 2),
        }
        if mode == "standard":
            print(
                f"[STD] {prof:11s} r={rate} {pol:18s}: "
                f"good_long={gl*100:5.1f}% good_all={ga*100:5.1f}% p90={p90:5.1f}s "
                f"migs={migs:5.1f} thr_s={thr:6.0f} dark={dsh*100:4.1f}%"
            )
        else:
            s_obj = summary[f"{mode}:{prof}:{pol}"]
            print(
                f"[AGT] {prof:11s} r={rate} {pol:18s}: "
                f"good_long={gl*100:5.1f}% (Scan={s_obj['good_scan']*100:4.1f}% "
                f"Pin={s_obj['good_pinpoint']*100:4.1f}% Phase={s_obj['good_phase']*100:4.1f}%) "
                f"Hops(Scan/Pin/Ph)={s_obj['hops_scan']:.2f}/{s_obj['hops_pinpoint']:.2f}/{s_obj['hops_phase']:.2f} "
                f"tether_GB={tgb:5.1f}"
            )

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
