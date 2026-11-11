#!/usr/bin/env python3
"""Research Motivation Experiment: Task Dispatch & Compute Load on Heterogeneous Multi-SSO Orbits.

Synthesizes a Multi-SSO compute constellation (eclipsed LTAN 10:30/13:30 planes + 100%-sunlit
dawn-dusk LTAN 06:00/18:00 planes) and evaluates standard task-dispatch baselines without migration:
  - nearest      : shortest-hop (access-local) dispatch from the origin RS satellite
  - rr           : cyclic Round-Robin across all reachable compute satellites
  - random       : uniform random dispatch across all reachable compute satellites
  - least_loaded : Join-Shortest-Queue (concurrency-aware, umbra-blind)
compared against orbit/power-aware schedulers:
  - greedy       : instantaneous speed-greedy placement (current load + current umbra, no future)
  - ephemeris    : future-ephemeris placement (no runtime migration)
  - contact_aware: ephemeris placement + externality-aware runtime migration

Usage:
  PYTHONPATH=satsim/src:orbitsim/src python3 satsim/scripts/exp_motivation_multi_sso.py
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from satsim.cosim.policies import make_policy
from satsim.cosim.probe import LoadProbe, analyze_run
from satsim.cosim.synth import synth_multi_sso
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, make_requests


def default_cfg() -> dict:
    return {
        "name": "exp_motivation_multi_sso",
        "delivery": "final",
        "control_period_s": 2.0,
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": 24.0,
            "t_fixed_s": 0.003,
            "t_weights_s": 0.01,
            "t_kv_tok_s": 1.0e-7,
            "t_seq_s": 5.0e-5,
            "mem_freq_exp": 0.6,
        },
        "power": {"P_idle": 60, "P_max": 300, "gamma": 2.4, "f_min": 0.35, "P_dark": 80},
        "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
        "migration": {"timeout_s": 120.0, "switch_s": 0.02},
        "workload": {
            "arrival_window_s": 5400,
            "rate_per_rs_per_min": 12,
            "p_long": 0.5,
            "long_out": [16000, 32000],
            "short_out": [400, 1200],
            "long_pin": 4000,
            "short_pin": 500,
            "long_upload_MB": 200,
            "short_upload_MB": 20,
        },
        "slo": {
            "deadline_factor_long": 2.0,
            "deadline_factor_short": 4.0,
            "min_deadline_s": 120,
            "max_stall_frac": 0.1,
        },
        "policy": {"stall_weight": 1.0},
    }


POLICIES = {
    "nearest": {"place": "nearest", "migrate": "none"},
    "rr": {"place": "rr", "migrate": "none"},
    "random": {"place": "random", "migrate": "none"},
    "least_loaded": {"place": "least_loaded", "migrate": "none"},
    "greedy": {"place": "greedy", "migrate": "none"},
    "ephemeris": {"place": "ephemeris", "migrate": "none"},
    "contact_aware": {"policy": "contact_aware"},
}


def run_policy_seeds(trace: Trace, cfg: dict, pol_spec: dict, seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        env = simpy.Environment()
        probe = LoadProbe(sample_s=10.0)
        w = CoWorld(env, trace, cfg, make_policy(pol_spec), seed=seed)
        probe.attach(w)
        for r in make_requests(cfg, trace, seed):
            w.submit(r)
        env.run(until=trace.horizon)
        rows.append(analyze_run(w, probe))

    mean_row: dict = {}
    for k in rows[0]:
        if k == "plane_shares":
            all_planes = sorted({p for r in rows for p in r["plane_shares"]})
            mean_row["plane_shares"] = {
                p: st.fmean([r["plane_shares"].get(p, 0.0) for r in rows]) for p in all_planes
            }
        elif isinstance(rows[0][k], (int, float)):
            vals = [r[k] for r in rows if r[k] is not None]
            mean_row[k] = st.fmean(vals) if vals else 0.0
    return mean_row


def print_main_table(results: dict[str, dict]) -> None:
    print("\n=== 表 1: 多轨道面 SSO 异构星座下的计算负载分布与明暗错配诊断 ===")
    header = (
        f"{'Policy':<15} | {'影面分配%':>8} | {'亮面分配%':>8} | {'直投暗星%':>8} | "
        f"{'暗星队列':>8} | {'亮星队列':>8} | {'暗/亮队列比':>10} | {'Jain均衡度':>9} | {'影内Token%':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<15} | "
            f"{100 * r['eclipsed_plane_disp_frac']:>7.1f}% | "
            f"{100 * r['sunlit_plane_disp_frac']:>7.1f}% | "
            f"{100 * r['admit_dark_frac']:>7.1f}% | "
            f"{r['q_dark_instant_mean']:>8.2f} | "
            f"{r['q_lit_instant_mean']:>8.2f} | "
            f"{r['q_dark_over_lit']:>9.2f}x | "
            f"{r['load_jains_index']:>9.3f} | "
            f"{100 * r['dark_tok_share']:>8.1f}%"
        )

    print("\n=== 表 2: 解码耗时膨胀与 SLO 达标性能对比 ===")
    header2 = (
        f"{'Policy':<15} | {'长任务达标↑':>10} | {'总达标率↑':>9} | {'有效吞吐(tok/s)↑':>16} | "
        f"{'解码膨胀p90↓':>12} | {'长任务p90(min)↓':>15} | {'迁移次数':>8}"
    )
    print(header2)
    print("-" * len(header2))
    for name, r in results.items():
        p90_min = (r["lat_long_p90_s"] or 0.0) / 60.0
        print(
            f"{name:<15} | "
            f"{100 * r['good_long_frac']:>9.1f}% | "
            f"{100 * r['good_frac']:>8.1f}% | "
            f"{r['goodput_tok_s']:>16.1f} | "
            f"{r['decode_stretch_p90']:>11.2f}x | "
            f"{p90_min:>14.1f}m | "
            f"{r['migrations']:>8.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--rate", type=float, default=12.0, help="requests per RS per min for main table")
    ap.add_argument("--sweep-rates", default="6,10,14,18", help="comma-separated rates for load sweep")
    ap.add_argument("--out", default="runs/motivation_multi_sso/summary.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.perf_counter()
    print(
        "1. 正在内存即时合成 Multi-SSO 异构星座轨迹 (LTAN 10:30, 13:30 影轨 + LTAN 06:00 晨昏全亮轨, 3×10 星)...",
        flush=True,
    )
    trace = synth_multi_sso(
        compute_ltans=(10.5, 13.5, 6.0),
        sats_per_plane=10,
        compute_alt_km=550.0,
        inter_mode="grid",
        inter_range_km=5000.0,
        rs_range_km=4000.0,
        horizon_s=7200.0,
        dt_s=5.0,
    )
    st_info = trace.raw_stats
    print(
        f"   合成完成 ({time.perf_counter() - t0:.2f}s): {len(trace.rs)} 颗遥感星 + {len(trace.compute)} 颗算力星 (3×10), "
        f"全网整体影时占比={100 * st_info['compute_dark_share']:.1f}% "
        f"(其中 LTAN 10:30/13:30 约 36% 进影, LTAN 06:00 晨昏轨 0% 进影)",
        flush=True,
    )

    cfg = default_cfg()
    cfg["workload"]["rate_per_rs_per_min"] = args.rate
    print(f"\n2. 运行主实验 (rate={args.rate}/min/RS, seeds={seeds})...", flush=True)
    main_results: dict[str, dict] = {}
    for name, spec in POLICIES.items():
        ts = time.perf_counter()
        main_results[name] = run_policy_seeds(trace, cfg, spec, seeds)
        r = main_results[name]
        print(
            f"   [{name:<13}] good_long={100*r['good_long_frac']:5.1f}%  "
            f"goodput={r['goodput_tok_s']:6.1f} tok/s  "
            f"dark_admit={100*r['admit_dark_frac']:4.1f}%  "
            f"q_dark/lit={r['q_dark_over_lit']:4.2f}x  ({time.perf_counter()-ts:.1f}s)",
            flush=True,
        )

    print_main_table(main_results)

    sweep_rates = [float(x) for x in args.sweep_rates.split(",") if x.strip()]
    sweep_results: dict[str, dict[str, dict]] = {}
    if sweep_rates:
        print(f"\n=== 表 3: 不同计算负载强度下的长任务达标率扫描 (seeds={seeds[:2]}) ===", flush=True)
        pol_names = list(POLICIES.keys())
        hdr = f"{'Rate(/min/RS)':<14} | " + " | ".join(f"{p:>13}" for p in pol_names)
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)
        for rate in sweep_rates:
            scfg = copy.deepcopy(cfg)
            scfg["workload"]["rate_per_rs_per_min"] = rate
            sweep_results[str(rate)] = {}
            cells = []
            for p_name, spec in POLICIES.items():
                res = run_policy_seeds(trace, scfg, spec, seeds[:2])
                sweep_results[str(rate)][p_name] = res
                cells.append(f"{100 * res['good_long_frac']:>12.1f}%")
            print(f"{rate:<14.1f} | " + " | ".join(cells), flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"main": main_results, "sweep": sweep_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
