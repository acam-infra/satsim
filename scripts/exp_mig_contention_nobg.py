#!/usr/bin/env python3
"""Experiment: multi-migration ISL contention with background=0.

Without background, default 10Gbps finishes each KV move before the next
control tick — concurrency never builds. We (1) raise caps / speed up control
and (2) slow the link so overlapping migrations share bandwidth and can abort.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simpy

import satsim.method.engine.world as world_mod
import satsim.method.policies.schedulers._select as select_mod
import satsim.method.policies.schedulers.rebalance_random as random_mod
from satsim.__main__ import resolve_t_swap_s
from satsim.input.generate.dispatch import generate_arrivals
from satsim.input.load import apply_t_swap, resolve_spec
from satsim.input.schema import setp
from satsim.method.engine.loop import control_proc, decode_proc, swap_proc
from satsim.method.engine.world import World, net_proc
from satsim.method.registry import bundle_from_policy
from satsim.output.metrics import row_from_run

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "input" / "scenarios" / "stac_s1_cost_claims_v3.yaml"
OUT = ROOT / "runs"
SLO = 0.032
SEEDS = [1, 2, 3, 4, 5]
DEFAULT_INF, DEFAULT_OUT = 8, 2

# Axis A: concurrency caps @ slow link (so moves overlap)
# Axis B: link speed @ high concurrency + fast control
SWEEP_CONC = [
    # label, inflight, out, period, link_gbps
    ("slow_cap1", 1, 1, 0.25, 1.0),
    ("slow_cap2", 2, 2, 0.25, 1.0),
    ("slow_cap4", 4, 2, 0.25, 1.0),
    ("slow_cap8", 8, 4, 0.25, 1.0),
    ("slow_cap16", 16, 8, 0.25, 1.0),
    ("slow_cap32", 32, 8, 0.25, 1.0),
]
SWEEP_LINK = [
    ("link10_hi", 32, 8, 0.25, 10.0),
    ("link2_hi", 32, 8, 0.25, 2.0),
    ("link1_hi", 32, 8, 0.25, 1.0),
    ("link05_hi", 32, 8, 0.25, 0.5),
]
# Baselines @ production-like period, nobg, default caps, default link
TIMEOUT_SWEEP = [
    # label, inflight, out, period, link_gbps, timeout_s
    ("t30_cap1", 1, 1, 0.25, 1.0, 30.0),
    ("t30_cap2", 2, 2, 0.25, 1.0, 30.0),
    ("t30_cap4", 4, 2, 0.25, 1.0, 30.0),
    ("t30_cap8", 8, 4, 0.25, 1.0, 30.0),
    ("t30_cap16", 16, 8, 0.25, 1.0, 30.0),
    ("t30_cap32", 32, 8, 0.25, 1.0, 30.0),
    ("t60_cap32", 32, 8, 0.25, 1.0, 60.0),
    ("t120_cap32", 32, 8, 0.25, 1.0, 120.0),
]

BASELINES = [
    ("base10_def", 8, 2, 1.0, 10.0),
    ("base1_def", 8, 2, 1.0, 1.0),
]


def _set_caps(inflight: int, out: int) -> None:
    for mod in (world_mod, random_mod, select_mod):
        mod.MAX_INFLIGHT = int(inflight)
        mod.MAX_OUT = int(out)


def _latencies(reqs) -> list[float]:
    return [float(r.t_done) - float(r.t) for r in reqs if r.t_done is not None]


def _row(reqs, world, cfg, t_swap_s, slo, anchor, cluster_n) -> dict:
    arrival_window = float(cfg["arrival_window_s"])
    horizon = arrival_window + float(cfg["drain_s"])
    good_long = n_long = completed_long = n_completed = 0
    tok_slo_out = tok_total = 0
    for r in reqs:
        if r.long:
            n_long += 1
        if r.t_done is not None:
            n_completed += 1
            tok_total += int(r.pin) + int(r.pout)
            if r.long:
                completed_long += 1
        post = (r.post_time / r.post_tok) if r.post_tok >= 1 else 0.0
        if r.t_done is not None and post <= slo:
            tok_slo_out += r.pout
            if r.long:
                good_long += 1
    return row_from_run(
        good_long=good_long,
        n_long=n_long,
        n_completed=n_completed,
        completed_long=completed_long,
        request_throughput_rps=n_completed / horizon,
        total_token_throughput_tok_s=tok_total / horizon,
        goodput_tok_s=tok_slo_out / arrival_window,
        migrations=world.acc["mig"],
        migration_abort=world.acc["abort"],
        migrated_GB=world.acc["mig_bytes"] / 1e9,
        link_traffic_GB=world.net.bytehop / 1e9,
        link_cap_gbps=float(cfg["net"]["link_gbps"]),
        max_flows_on_link=world.acc["max_flows_on_link"],
        T_SWAP=t_swap_s,
        max_cluster_t=anchor,
        max_cluster_n=cluster_n,
        n_req=len(reqs),
        horizon_s=horizon,
    )


def run_one(cfg, pol, seed, trace, t_swap_s, *, track: bool):
    reqs, anchor, cluster_n = generate_arrivals(trace, float(cfg["workload"]["p_long"]), seed)
    env = simpy.Environment()
    world = World(env, cfg, pol, t_swap_s, bundle=bundle_from_policy(pol), seed=seed)
    conc_peak = 0
    events: list[dict] = []

    if track:
        orig = world.link_cost.transfer

        def wrapped(world_, req, src, dst, nbytes: float):
            nonlocal conc_peak
            t0 = world_.env.now
            inflight = sum(1 for rs in world_.nodes for r in rs if r.migrating)
            conc_peak = max(conc_peak, inflight)
            nmig_before = req.nmig
            yield from orig(world_, req, src, dst, nbytes)
            aborted = req.nmig == nmig_before and req.node == src
            events.append(
                dict(
                    t_start=round(t0, 3),
                    t_end=round(world_.env.now, 3),
                    dur_s=round(world_.env.now - t0, 3),
                    req=req.i,
                    long=bool(req.long),
                    src=src,
                    dst=dst,
                    nbytes_GB=round(nbytes / 1e9, 4),
                    inflight_at_start=inflight,
                    aborted=bool(aborted),
                    committed=req.nmig > nmig_before,
                )
            )

        world.link_cost.transfer = wrapped  # type: ignore

    env.process(net_proc(env, world.net))
    env.process(swap_proc(env, world))
    env.process(control_proc(env, world))
    for req in reqs:
        req.proc = env.process(decode_proc(env, world, req))
    env.run(until=float(cfg["arrival_window_s"]) + float(cfg["drain_s"]))
    row = _row(reqs, world, cfg, t_swap_s, SLO, anchor, cluster_n)
    lats = _latencies(reqs)
    extra = dict(
        conc_peak=int(conc_peak),
        lat_p50=float(np.percentile(lats, 50)) if lats else float("nan"),
        lat_p99=float(np.percentile(lats, 99)) if lats else float("nan"),
        lat_mean=float(np.mean(lats)) if lats else float("nan"),
        incomplete=len(reqs) - len(lats),
        abort_at_conc_ge2=sum(1 for e in events if e["aborted"] and e["inflight_at_start"] >= 2),
        mean_inflight_at_start=float(np.mean([e["inflight_at_start"] for e in events])) if events else 0.0,
        mean_mig_dur=float(np.mean([e["dur_s"] for e in events])) if events else 0.0,
    )
    return row, extra, events


def mean_std(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return dict(mean=float(np.mean(a)), std=float(np.std(a)), n=int(len(a)))


def pack(row, extra, arm, label, inf, out, period, link, seed, n_events):
    return dict(
        arm=arm,
        sweep=label,
        max_inflight=inf,
        max_out=out,
        control_period_s=period,
        link_gbps=link,
        seed=seed,
        good_long=row["good_long"],
        n_long=row["n_long"],
        goodput_tok_s=round(row["goodput_tok_s"], 4),
        migrations=row["migrations"],
        migration_abort=row["migration_abort"],
        link_traffic_GB=round(row["link_traffic_GB"], 4),
        migrated_GB=round(row["migrated_GB"], 4),
        max_flows_on_link=row["max_flows_on_link"],
        n_events=n_events,
        **extra,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec, cfg0, jobs = resolve_spec(SPEC)
    t_swap_s, anchor, cluster_n = resolve_t_swap_s(spec, cfg0)
    p_cap = float(spec["t_swap"]["P_cap"])
    pols = {}
    for name, pol, *_ in jobs:
        if name in {"place_once", "rebalance_free", "rebalance_random"} and name not in pols:
            pols[name] = copy.deepcopy(pol)

    base = copy.deepcopy(cfg0)
    setp(base, "background.gbps_per_node", 0)

    results: list[dict] = []

    def run_cfg(label, inf, out, period, link, arms, timeout=None):
        _set_caps(inf, out)
        for arm in arms:
            for seed in SEEDS:
                cfg = copy.deepcopy(base)
                cfg["control_period_s"] = float(period)
                cfg["net"]["link_gbps"] = float(link)
                if timeout is not None:
                    cfg["cost"]["timeout_s"] = float(timeout)
                apply_t_swap(cfg, t_swap_s, p_cap)
                row, extra, events = run_one(
                    cfg, pols[arm], seed, spec["trace"], t_swap_s, track=(arm == "rebalance_random")
                )
                rec = pack(row, extra, arm, label, inf, out, period, link, seed, len(events))
                results.append(rec)
                print(
                    f"{label:12} {arm:16} s={seed} gl={rec['good_long']:2d} "
                    f"ab={rec['migration_abort']:2d} mig={rec['migrations']:2d} "
                    f"p99={rec['lat_p99']:.0f} peak={rec['conc_peak']} "
                    f"flows={rec['max_flows_on_link']} dur={rec['mean_mig_dur']:.1f}s",
                    flush=True,
                )

    # baselines: all three arms
    for item in BASELINES:
        run_cfg(*item, arms=("place_once", "rebalance_free", "rebalance_random"))
    # concurrency sweep @ 1Gbps: random only (+ free/place once at ends)
    for i, item in enumerate(SWEEP_CONC):
        arms = ("rebalance_random",)
        if i in (0, len(SWEEP_CONC) - 1):
            arms = ("place_once", "rebalance_free", "rebalance_random")
        run_cfg(*item, arms=arms)
    # link sweep @ hi conc
    for item in SWEEP_LINK:
        run_cfg(*item, arms=("rebalance_random",))
    for item in TIMEOUT_SWEEP:
        label, inf, out, period, link, timeout = item
        arms = ("rebalance_random",)
        if label in ("t30_cap1", "t30_cap32"):
            arms = ("place_once", "rebalance_free", "rebalance_random")
        run_cfg(label, inf, out, period, link, arms=arms, timeout=timeout)

    _set_caps(DEFAULT_INF, DEFAULT_OUT)

    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        groups[(r["arm"], r["sweep"])].append(r)

    def agg(arm, sweep):
        rs = groups.get((arm, sweep), [])
        if not rs:
            return None
        return dict(
            arm=arm,
            sweep=sweep,
            link_gbps=rs[0]["link_gbps"],
            max_inflight=rs[0]["max_inflight"],
            max_out=rs[0]["max_out"],
            control_period_s=rs[0]["control_period_s"],
            good_long=mean_std([r["good_long"] for r in rs]),
            abort=mean_std([r["migration_abort"] for r in rs]),
            migrations=mean_std([r["migrations"] for r in rs]),
            link_GB=mean_std([r["link_traffic_GB"] for r in rs]),
            lat_p99=mean_std([r["lat_p99"] for r in rs]),
            conc_peak=mean_std([r["conc_peak"] for r in rs]),
            max_flows=mean_std([float(r["max_flows_on_link"]) for r in rs]),
            mean_mig_dur=mean_std([r["mean_mig_dur"] for r in rs]),
            abort_at_conc_ge2=mean_std([r["abort_at_conc_ge2"] for r in rs]),
        )

    summary = [agg(a, s) for (a, s) in sorted(groups) if agg(a, s)]

    # showcase: max abort among random
    rand_rows = [r for r in results if r["arm"] == "rebalance_random"]
    show = max(rand_rows, key=lambda r: (r["migration_abort"], r["conc_peak"], r["max_flows_on_link"]))
    _set_caps(show["max_inflight"], show["max_out"])
    cfg = copy.deepcopy(base)
    cfg["control_period_s"] = float(show["control_period_s"])
    cfg["net"]["link_gbps"] = float(show["link_gbps"])
    apply_t_swap(cfg, t_swap_s, p_cap)
    _row2, _ex, ev = run_one(cfg, pols["rebalance_random"], show["seed"], spec["trace"], t_swap_s, track=True)
    _set_caps(DEFAULT_INF, DEFAULT_OUT)
    aborts = [e for e in ev if e["aborted"]]
    concurrent = [e for e in ev if e["inflight_at_start"] >= 2]
    timeline = dict(
        seed=show["seed"],
        sweep=show["sweep"],
        link_gbps=show["link_gbps"],
        metrics={k: show[k] for k in (
            "good_long", "migration_abort", "migrations", "lat_p99", "conc_peak",
            "max_flows_on_link", "link_traffic_GB", "mean_mig_dur",
        )},
        abort_sample=aborts[:10],
        concurrent_sample=concurrent[:10],
    )

    # ---- figures ----
    # 1) concurrency vs abort/p99 @ 1Gbps
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=140)
    conc_labels = [x[0] for x in SWEEP_CONC]
    xs = [groups[("rebalance_random", lab)][0]["max_inflight"] for lab in conc_labels]
    ab = [float(np.mean([r["migration_abort"] for r in groups[("rebalance_random", lab)]])) for lab in conc_labels]
    p99 = [float(np.mean([r["lat_p99"] for r in groups[("rebalance_random", lab)]])) for lab in conc_labels]
    peak = [float(np.mean([r["conc_peak"] for r in groups[("rebalance_random", lab)]])) for lab in conc_labels]
    gl = [float(np.mean([r["good_long"] for r in groups[("rebalance_random", lab)]])) for lab in conc_labels]

    ax = axes[0]
    ax.plot(xs, ab, "o-", color="#E45756", lw=2, label="abort")
    ax.plot(xs, peak, "s--", color="#F58518", lw=1.5, label="conc_peak")
    ax.set_xlabel("MAX_INFLIGHT cap")
    ax.set_ylabel("count (mean over seeds)")
    ax.set_title("nobg @ 1Gbps: abort & peak concurrency vs cap")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(xs, p99, "o-", color="#4C78A8", lw=2, label="lat p99")
    ax2 = ax.twinx()
    ax2.plot(xs, gl, "^-", color="#54A24B", lw=2, label="good_long")
    ax.set_xlabel("MAX_INFLIGHT cap")
    ax.set_ylabel("latency p99 (s)")
    ax2.set_ylabel("good_long")
    ax.set_title("nobg @ 1Gbps: p99 & good_long vs cap")
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    fig.suptitle("Multi-migration contention (bg=0, control=0.25s, link=1Gbps)", fontsize=11)
    fig.tight_layout()
    png1 = OUT / "exp_mig_contention_nobg_sweep.png"
    fig.savefig(png1)
    plt.close(fig)

    # 2) link speed vs abort at hi concurrency
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=140)
    link_labs = [x[0] for x in SWEEP_LINK]
    lx = [groups[("rebalance_random", lab)][0]["link_gbps"] for lab in link_labs]
    labort = [float(np.mean([r["migration_abort"] for r in groups[("rebalance_random", lab)]])) for lab in link_labs]
    lp99 = [float(np.mean([r["lat_p99"] for r in groups[("rebalance_random", lab)]])) for lab in link_labs]
    lpeak = [float(np.mean([r["conc_peak"] for r in groups[("rebalance_random", lab)]])) for lab in link_labs]
    lgl = [float(np.mean([r["good_long"] for r in groups[("rebalance_random", lab)]])) for lab in link_labs]
    lflows = [float(np.mean([r["max_flows_on_link"] for r in groups[("rebalance_random", lab)]])) for lab in link_labs]

    ax = axes[0]
    ax.plot(lx, labort, "o-", color="#E45756", lw=2, label="abort")
    ax.plot(lx, lflows, "s--", color="#F58518", lw=1.5, label="max_flows/link")
    ax.set_xscale("log")
    ax.set_xlabel("link_gbps")
    ax.set_ylabel("mean count")
    ax.set_title("hi-conc: abort vs link speed (bg=0)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    # arm compare at base1_def and slow_cap32
    compare = [("base10_def", "10G def"), ("base1_def", "1G def"), ("slow_cap1", "1G cap1"), ("slow_cap32", "1G cap32")]
    arms = ["place_once", "rebalance_free", "rebalance_random"]
    colors = {"place_once": "#4C78A8", "rebalance_free": "#54A24B", "rebalance_random": "#E45756"}
    x = np.arange(len(compare))
    width = 0.25
    for i, arm in enumerate(arms):
        vals = []
        for sw, _ in compare:
            rs = groups.get((arm, sw), [])
            vals.append(float(np.mean([r["good_long"] for r in rs])) if rs else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, color=colors[arm], label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in compare])
    ax.set_ylabel("good_long mean")
    ax.set_title("Arms @ nobg (wrong dest + optional mig fight)")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    png2 = OUT / "exp_mig_contention_nobg_arms.png"
    fig.savefig(png2)
    plt.close(fig)

    # 3) timeline of aborts for showcase
    png3 = None
    if aborts:
        fig, ax = plt.subplots(figsize=(9.5, 3.4), dpi=140)
        for e in aborts[:10]:
            ax.barh(e["req"], e["dur_s"], left=e["t_start"], height=0.7, color="#E45756", alpha=0.9)
            ax.text(
                e["t_start"] + min(e["dur_s"], 30) / 2,
                e["req"],
                f"{e['src']}→{e['dst']} inf={e['inflight_at_start']}",
                va="center",
                ha="center",
                fontsize=7,
                color="white",
            )
        ax.set_xlabel("sim time (s)")
        ax.set_ylabel("req id")
        ax.set_title(
            f"ABORT timeline seed={timeline['seed']} {timeline['sweep']} "
            f"link={timeline['link_gbps']}G peak={timeline['metrics']['conc_peak']}"
        )
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        png3 = OUT / f"exp_mig_contention_nobg_timeline_seed{timeline['seed']}.png"
        fig.savefig(png3)
        plt.close(fig)

    def m(arm, sweep, field):
        rs = groups.get((arm, sweep), [])
        return float(np.mean([r[field] for r in rs])) if rs else float("nan")

    readout = {
        "nobg_default_10G": dict(
            place_gl=m("place_once", "base10_def", "good_long"),
            free_gl=m("rebalance_free", "base10_def", "good_long"),
            rand_gl=m("rebalance_random", "base10_def", "good_long"),
            rand_abort=m("rebalance_random", "base10_def", "migration_abort"),
            rand_peak=m("rebalance_random", "base10_def", "conc_peak"),
            rand_p99=m("rebalance_random", "base10_def", "lat_p99"),
        ),
        "nobg_1G_serial_cap1": dict(
            rand_gl=m("rebalance_random", "slow_cap1", "good_long"),
            rand_abort=m("rebalance_random", "slow_cap1", "migration_abort"),
            rand_peak=m("rebalance_random", "slow_cap1", "conc_peak"),
            rand_p99=m("rebalance_random", "slow_cap1", "lat_p99"),
            rand_flows=m("rebalance_random", "slow_cap1", "max_flows_on_link"),
            free_gl=m("rebalance_free", "slow_cap1", "good_long"),
        ),
        "nobg_1G_hi_cap32": dict(
            rand_gl=m("rebalance_random", "slow_cap32", "good_long"),
            rand_abort=m("rebalance_random", "slow_cap32", "migration_abort"),
            rand_peak=m("rebalance_random", "slow_cap32", "conc_peak"),
            rand_p99=m("rebalance_random", "slow_cap32", "lat_p99"),
            rand_flows=m("rebalance_random", "slow_cap32", "max_flows_on_link"),
            free_gl=m("rebalance_free", "slow_cap32", "good_long"),
            place_gl=m("place_once", "slow_cap32", "good_long"),
        ),
        "link_sweep_hi_conc_abort": {
            lab: m("rebalance_random", lab, "migration_abort") for lab in link_labs
        },
        "link_sweep_hi_conc_peak": {
            lab: m("rebalance_random", lab, "conc_peak") for lab in link_labs
        },
    }

    payload = dict(
        setting=dict(
            background_gbps_per_node=0,
            timeout_s=float(base["cost"]["timeout_s"]),
            seeds=SEEDS,
            t_swap_s=t_swap_s,
            note=(
                "bg=0 removes background contention. Default 10G rarely overlaps migrations; "
                "slow link (1G/0.5G) + fast control (0.25s) + raised MAX_INFLIGHT/OUT creates "
                "multi-mig sharing on ISL. Caps patched in-process only."
            ),
            sweep_conc=SWEEP_CONC,
            sweep_link=SWEEP_LINK,
            baselines=BASELINES,
        ),
        summary=summary,
        showcase=timeline,
        readout=readout,
        png_sweep=str(png1),
        png_arms=str(png2),
        png_timeline=str(png3) if png3 else None,
        per_seed=results,
    )
    out_json = OUT / "exp_mig_contention_nobg.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(readout, indent=2))
    print("png:", png1)
    print("png:", png2)
    print("png:", png3)
    print("json:", out_json)


if __name__ == "__main__":
    main()
