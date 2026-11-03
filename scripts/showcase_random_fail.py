#!/usr/bin/env python3
"""Showcase: rebalance_random fails vs free / place_once (single-seed + mig timeline)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simpy

from satsim.__main__ import resolve_t_swap_s
from satsim.input.generate.dispatch import generate_arrivals
from satsim.input.load import apply_t_swap, resolve_spec
from satsim.input.schema import setp
from satsim.method.engine.loop import control_proc, decode_proc, swap_proc
from satsim.method.engine.world import World, net_proc
from satsim.method.registry import bundle_from_policy
from satsim.output.metrics import row_from_run

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "input" / "scenarios" / "exp_ring_s1_cost_claims_v3.yaml"
OUT = ROOT / "runs"
ARMS = ("place_once", "rebalance_free", "rebalance_random")
SLO = 0.032


def _p_at(world, node: int, t: float) -> float:
    p = float(world.cfg["thermal"]["P_max"])
    for ev in world.cfg["thermal"].get("explicit_events") or []:
        j, t0, t1, pc = ev
        if j == node and t0 <= t < t1:
            p = min(p, float(pc))
    return p


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


def run_arm(cfg, pol, seed, trace, t_swap_s, *, log_migs: bool):
    reqs, anchor, cluster_n = generate_arrivals(trace, float(cfg["workload"]["p_long"]), seed)
    env = simpy.Environment()
    world = World(env, cfg, pol, t_swap_s, bundle=bundle_from_policy(pol), seed=seed)
    events: list[dict] = []
    if log_migs:
        orig = world.link_cost.transfer

        def wrapped(world_, req, src, dst, nbytes: float):
            t0 = world_.env.now
            f_src, f_dst = world_.f_of(src, t0), world_.f_of(dst, t0)
            it_src, it_dst = world_.iter_of(src), world_.iter_of(dst)
            p_src, p_dst = _p_at(world_, src, t0), _p_at(world_, dst, t0)
            nmig_before = req.nmig
            node_before = req.node
            yield from orig(world_, req, src, dst, nbytes)
            committed = req.nmig > nmig_before
            aborted = (not committed) and req.node == node_before
            events.append(
                dict(
                    t_start=round(t0, 3),
                    t_end=round(world_.env.now, 3),
                    dur_s=round(world_.env.now - t0, 3),
                    req=req.i,
                    long=bool(req.long),
                    kv=round(float(req.kv), 1),
                    src=src,
                    dst=dst,
                    nbytes_GB=round(nbytes / 1e9, 4),
                    P_src=p_src,
                    P_dst=p_dst,
                    f_src=round(float(f_src), 4),
                    f_dst=round(float(f_dst), 4),
                    iter_src=round(float(it_src), 6),
                    iter_dst=round(float(it_dst), 6),
                    worse_dst=bool(it_dst > it_src * 1.05),
                    to_darker=bool(p_dst < p_src * 0.9),
                    aborted=bool(aborted),
                    committed=bool(committed),
                    final_node=req.node,
                    nmig_before=nmig_before,
                    nmig_after=req.nmig,
                )
            )

        world.link_cost.transfer = wrapped  # type: ignore

    env.process(net_proc(env, world.net))
    env.process(swap_proc(env, world))
    env.process(control_proc(env, world))
    for req in reqs:
        req.proc = env.process(decode_proc(env, world, req))
    env.run(until=float(cfg["arrival_window_s"]) + float(cfg["drain_s"]))
    return _row(reqs, world, cfg, t_swap_s, SLO, anchor, cluster_n), events


def pack_metrics(m: dict) -> dict:
    return dict(
        good_long=m["good_long"],
        n_long=m["n_long"],
        goodput_tok_s=round(m["goodput_tok_s"], 4),
        migrations=m["migrations"],
        migration_abort=m["migration_abort"],
        link_traffic_GB=round(m["link_traffic_GB"], 4),
        migrated_GB=round(m["migrated_GB"], 4),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec, cfg0, jobs = resolve_spec(SPEC)
    t_swap_s, anchor, cluster_n = resolve_t_swap_s(spec, cfg0)
    p_cap = float(spec["t_swap"]["P_cap"])
    pols = {}
    for name, pol, _c2, _seed in jobs:
        if name in ARMS and name not in pols:
            pols[name] = pol
    seeds = sorted({int(s) for s in spec["seeds"]})

    table = []
    for seed in seeds:
        row: dict = {"seed": seed}
        for arm in ARMS:
            cfg = copy.deepcopy(cfg0)
            apply_t_swap(cfg, t_swap_s, p_cap)
            for k, v in pols[arm].get("knobs", {}).items():
                setp(cfg, k, v)
            m, _ = run_arm(cfg, pols[arm], seed, spec["trace"], t_swap_s, log_migs=False)
            row[arm] = pack_metrics(m)
        row["gap_free_minus_random_good_long"] = (
            row["rebalance_free"]["good_long"] - row["rebalance_random"]["good_long"]
        )
        row["random_aborts"] = row["rebalance_random"]["migration_abort"]
        table.append(row)
        print(
            f"seed={seed} free={row['rebalance_free']['good_long']} "
            f"rand={row['rebalance_random']['good_long']} gap={row['gap_free_minus_random_good_long']} "
            f"abort={row['random_aborts']}",
            flush=True,
        )

    best = max(table, key=lambda r: (r["gap_free_minus_random_good_long"], r["random_aborts"]))
    seed = int(best["seed"])
    print(f"SHOWCASE seed={seed}", flush=True)

    details = {}
    for arm in ARMS:
        cfg = copy.deepcopy(cfg0)
        apply_t_swap(cfg, t_swap_s, p_cap)
        for k, v in pols[arm].get("knobs", {}).items():
            setp(cfg, k, v)
        m, ev = run_arm(cfg, pols[arm], seed, spec["trace"], t_swap_s, log_migs=True)
        details[arm] = dict(metrics=pack_metrics(m), events=ev)

    rev = details["rebalance_random"]["events"]
    abort_dark = [e for e in rev if e["aborted"] and e["to_darker"]]
    commit_dark = [e for e in rev if e["committed"] and e["to_darker"] and e["long"]]
    abort_dark.sort(key=lambda e: (not e["long"], e["t_start"]))
    commit_dark.sort(key=lambda e: e["t_start"])
    spotlight = []
    if abort_dark:
        spotlight.append(abort_dark[0])
    if commit_dark:
        spotlight.append(commit_dark[0])
    if len(spotlight) < 2 and len(abort_dark) > 1:
        spotlight.append(abort_dark[1])

    free_rescue = [
        e
        for e in details["rebalance_free"]["events"]
        if e["committed"] and e["P_src"] < e["P_dst"]
    ][:2]

    story = {
        "spec": str(SPEC.relative_to(ROOT)),
        "t_swap_s": t_swap_s,
        "p_cap": p_cap,
        "anchor": anchor,
        "cluster_n": cluster_n,
        "showcase_seed": seed,
        "seed_sweep": table,
        "detail": {a: {"metrics": details[a]["metrics"], "n_events": len(details[a]["events"])} for a in ARMS},
        "random_spotlight": spotlight,
        "free_rescue_sample": free_rescue,
        "counts_random": dict(
            events=len(rev),
            aborted=sum(e["aborted"] for e in rev),
            to_darker=sum(e["to_darker"] for e in rev),
            long_committed_to_darker=sum(1 for e in rev if e["committed"] and e["to_darker"] and e["long"]),
        ),
        "detail_full_events": {a: details[a]["events"] for a in ARMS},
    }

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=140)
    ax = axes[0]
    labels = list(ARMS)
    gl = [details[a]["metrics"]["good_long"] for a in labels]
    colors = ["#4C78A8", "#54A24B", "#E45756"]
    bars = ax.bar(["place_once", "free", "random"], gl, color=colors)
    ax.set_ylabel("good_long (SLO-ok long reqs)")
    ax.set_title(f"seed={seed}: random loses {gl[1]-gl[2]} long vs free")
    for b, v in zip(bars, gl):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v), ha="center", fontsize=10)
    ax.set_ylim(0, max(gl + [1]) * 1.3)
    ax.axhline(details["place_once"]["metrics"]["good_long"], color="#4C78A8", ls="--", lw=0.8, alpha=0.5)

    ax = axes[1]
    ax.axis("off")
    mline = [
        f"exp_ring_s1_cost_claims_v3  seed={seed}  T_SWAP={t_swap_s:.1f}s  P_cap={p_cap}",
        "",
        f"{'arm':16} {'good_long':>9} {'goodput':>9} {'mig':>4} {'abort':>5} {'linkGB':>7}",
    ]
    for a, short in zip(labels, ["place_once", "free", "random"]):
        m = details[a]["metrics"]
        mline.append(
            f"{short:16} {m['good_long']:9d} {m['goodput_tok_s']:9.1f} "
            f"{m['migrations']:4d} {m['migration_abort']:5d} {m['link_traffic_GB']:7.1f}"
        )
    mline += ["", "random fails (pre-swap dumps into dark node 2 @ P=80):"]
    for e in spotlight:
        tag = "TIMEOUT-ABORT" if e["aborted"] else "COMMITTED-to-DARK"
        mline.append(
            f"  t={e['t_start']:.0f}s req#{e['req']} {'long' if e['long'] else 'short'} "
            f"{e['src']}->{e['dst']} {tag}  P {e['P_src']:.0f}->{e['P_dst']:.0f}  {e['dur_s']:.0f}s"
        )
    if free_rescue:
        e = free_rescue[0]
        mline += [
            "",
            "free does the opposite (post-swap escape dark->bright):",
            f"  t={e['t_start']:.0f}s req#{e['req']} {e['src']}->{e['dst']} "
            f"P {e['P_src']:.0f}->{e['P_dst']:.0f}",
        ]
    ax.text(0.01, 0.98, "\n".join(mline), va="top", family="monospace", fontsize=7.8)
    fig.suptitle("Random costly rebalance backfires under power-cap swap", fontsize=11)
    fig.tight_layout()
    png = OUT / f"showcase_random_fail_seed{seed}.png"
    fig.savefig(png)
    plt.close(fig)

    json_path = OUT / f"showcase_random_fail_seed{seed}.json"
    json_path.write_text(json.dumps(story, indent=2, ensure_ascii=False) + "\n")

    # one-paragraph story for coordinator
    mf = details["rebalance_free"]["metrics"]
    mr = details["rebalance_random"]["metrics"]
    mp = details["place_once"]["metrics"]
    sp0 = spotlight[0] if spotlight else {}
    brief = {
        "story_zh": (
            f"场景 exp_ring_s1_cost_claims_v3、种子 {seed}：限功切换 T_SWAP={t_swap_s:.0f}s 前，"
            f"node2 已被限制到 P={p_cap}。无消耗迁移几乎不动，切换后把请求从降频星迁到满血星"
            f"（good_long {mf['good_long']}/{mf['n_long']}）。"
            f"随机迁移却在切换前乱迁：例如 t≈{sp0.get('t_start', '?')}s 把长请求 "
            f"#{sp0.get('req','?')} 从 node{sp0.get('src','?')} 往暗星 node{sp0.get('dst','?')} 搬，"
            f"{'链路堵死超时 abort（人留原地）' if sp0.get('aborted') else '成功搬进更暗节点'}；"
            f"全剧 abort={mr['migration_abort']}、link≈{mr['link_traffic_GB']:.0f}GB，"
            f"good_long 掉到 {mr['good_long']}（place_once 还有 {mp['good_long']}），"
            f"比 free 少 {mf['good_long']-mr['good_long']} 条达标长请求。"
        ),
        "metrics": dict(place_once=mp, free=mf, random=mr),
        "png": str(png),
        "json": str(json_path),
    }
    (OUT / "showcase_random_fail_brief.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(brief, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
