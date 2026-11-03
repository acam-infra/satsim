#!/usr/bin/env python3
"""Gamma CV sweep: latency CDF at high CV + p99 vs CV (Llumnix-style)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simpy

from satsim.input.generate.dispatch import generate_arrivals
from satsim.input.load import apply_t_swap, load_yaml
from satsim.input.schema import setp, validate_base, validate_policy
from satsim.method.engine.loop import control_proc, decode_proc, swap_proc
from satsim.method.engine.world import World, net_proc
from satsim.method.registry import bundle_from_policy

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs"
CVS = [1.0, 2.0, 4.0, 8.0]
ARMS = ["place_once", "rebalance_free", "rebalance_random"]
SEEDS = [1, 2, 3]
HIGH_CV = 8.0

TOKEN_PAD = dict(mode="floor", long_out=16000, long_pin=8000, short_out=200, short_pin=500)
WINDOW_S = 180.0
RATE = 0.4
ARRIVAL_WINDOW = 300.0
DRAIN = 600.0
T_SWAP = 90.0
P_CAP = 80.0
P_LONG = 0.2
BG = 2.75

POLICIES = {
    "place_once": dict(scheduler="place_once", link_cost="off"),
    "rebalance_free": dict(scheduler="rebalance_free", link_cost="off", hysteresis=0.15),
    "rebalance_random": dict(scheduler="rebalance_random", link_cost="full"),
}


def make_base_cfg() -> dict:
    cfg = load_yaml(ROOT / "input" / "scenarios" / "construct_s1.yaml")
    validate_base(cfg)
    cfg["arrival_window_s"] = ARRIVAL_WINDOW
    cfg["drain_s"] = DRAIN
    cfg["background"]["gbps_per_node"] = BG
    cfg["background"]["strict_priority"] = True
    cfg["net"]["link_gbps"] = 10.0
    cfg["gpu"]["kv_gib"] = 24.0
    cfg["workload"]["p_long"] = P_LONG
    return cfg


def run_latencies(cv: float, arm: str, seed: int, base_cfg: dict) -> list[float]:
    pol = copy.deepcopy(POLICIES[arm])
    validate_policy(arm, pol)
    cfg = copy.deepcopy(base_cfg)
    apply_t_swap(cfg, T_SWAP, P_CAP)
    for k, v in pol.get("knobs", {}).items():
        setp(cfg, k, v)
    trace = dict(
        kind="gamma",
        window_s=WINDOW_S,
        rate_per_s=RATE,
        cv=cv,
        jitter_s=0.0,
        token_pad=TOKEN_PAD,
    )
    reqs, _, _ = generate_arrivals(trace, P_LONG, seed)
    env = simpy.Environment()
    world = World(env, cfg, pol, T_SWAP, bundle=bundle_from_policy(pol), seed=seed)
    env.process(net_proc(env, world.net))
    env.process(swap_proc(env, world))
    env.process(control_proc(env, world))
    for req in reqs:
        req.proc = env.process(decode_proc(env, world, req))
    env.run(until=ARRIVAL_WINDOW + DRAIN)
    out: list[float] = []
    for r in reqs:
        if r.t_done is not None:
            out.append(float(r.t_done) - float(r.t))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = make_base_cfg()
    latencies: dict[float, dict[str, list[float]]] = {cv: {a: [] for a in ARMS} for cv in CVS}
    for cv in CVS:
        for arm in ARMS:
            for seed in SEEDS:
                latencies[cv][arm].extend(run_latencies(cv, arm, seed, base_cfg))
            print(f"cv={cv} arm={arm} n={len(latencies[cv][arm])}", flush=True)

    colors = {"place_once": "#4C78A8", "rebalance_free": "#54A24B", "rebalance_random": "#E45756"}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    table: dict = {}
    for arm in ARMS:
        xs = np.sort(np.asarray(latencies[HIGH_CV][arm], dtype=float))
        if len(xs) == 0:
            raise SystemExit(f"no completed requests for {arm} at CV={HIGH_CV}")
        ys = np.arange(1, len(xs) + 1) / len(xs)
        p50, p95, p99 = np.percentile(xs, [50, 95, 99])
        table[arm] = dict(
            n=int(len(xs)),
            p50=float(p50),
            p95=float(p95),
            p99=float(p99),
            mean=float(np.mean(xs)),
        )
        ax.plot(xs, ys, lw=2, color=colors[arm], label=f"{arm} (p99={p99:.0f}s)")
    ax.set_xlabel("Latency = t_done − t_arrive (s)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Gamma arrivals CV={HIGH_CV:g}: request latency CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.text(
        0.01,
        0.01,
        f"IAT~Gamma(k=1/cv²,θ=cv²/rate); rate={RATE}/s window={WINDOW_S}s "
        f"seeds={SEEDS}; incomplete excluded; bg={BG} Gbps/node",
        fontsize=7,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    cdf_path = OUT_DIR / f"latency_cdf_gamma_cv{int(HIGH_CV)}.png"
    fig.savefig(cdf_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    p99_by = {a: [] for a in ARMS}
    for cv in CVS:
        for arm in ARMS:
            xs = np.asarray(latencies[cv][arm], dtype=float)
            p99_by[arm].append(float(np.percentile(xs, 99)))
    for arm in ARMS:
        ax.plot(CVS, p99_by[arm], marker="o", lw=2, color=colors[arm], label=arm)
    ax.set_xlabel("Arrival interval CV")
    ax.set_ylabel("Latency p99 (s)")
    ax.set_title("Latency p99 vs Gamma CV")
    ax.set_xticks(CVS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p99_path = OUT_DIR / "latency_p99_vs_gamma_cv.png"
    fig.savefig(p99_path)
    plt.close(fig)

    summary = {
        "formula": "interarrival ~ Gamma(k=1/cv^2, scale=cv^2/rate); mean=1/rate; cv=1 => Exp/Poisson",
        "params": dict(
            window_s=WINDOW_S,
            rate_per_s=RATE,
            cvs=CVS,
            seeds=SEEDS,
            t_swap=T_SWAP,
            bg_gbps_per_node=BG,
            arrival_window_s=ARRIVAL_WINDOW,
            drain_s=DRAIN,
        ),
        "cdf_cv": HIGH_CV,
        "cdf_percentiles": table,
        "p99_vs_cv": {a: {str(c): p99_by[a][i] for i, c in enumerate(CVS)} for a in ARMS},
        "png_cdf": str(cdf_path),
        "png_p99": str(p99_path),
    }
    (OUT_DIR / "gamma_cv_latency_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
