#!/usr/bin/env python3
"""orbit_planes 跨轨 E★ 加严争用：有代价臂被共享瓶颈压下去。

构造：
1. 2×2 grid，E★=(0,2)。
2. 默认换位 2→0；再额外压暗 node1 → 0/1 都要疏散；place_once 仍可靠 2/3 存活。
3. 有代价臂目标收束到亮算力星 2，控制拍内 sync_burst 填满并发。
4. 主轴 link_gbps=0.5；辅 timeout=30、control=0.25、高 caps；bg=0。
"""
from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import simpy
import yaml

import satsim.method.engine.world as world_mod
import satsim.method.policies.schedulers._select as select_mod
import satsim.method.policies.schedulers.rebalance_random as random_mod
from satsim.__main__ import resolve_t_swap_s
from satsim.input.generate.dispatch import generate_arrivals
from satsim.input.load import apply_t_swap, load_yaml
from satsim.input.schema import setp, validate_base, validate_trace
from satsim.method.engine.loop import control_proc, decode_proc, swap_proc
from satsim.method.engine.world import MAX_INFLIGHT, MAX_OUT, World, net_proc
from satsim.method.policies.schedulers._placement import place_by_iter_time
from satsim.method.registry import PolicyBundle, bundle_from_policy
from satsim.output.metrics import row_from_run

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "configs" / "orbit_planes_estar_contention_v1.yaml"
OUT = ROOT / "runs"
DEFAULT_INF, DEFAULT_OUT = 8, 2
ESTAR = (0, 2)


class CorridorBurstRandom:
    """有消耗：目标固定为跨轨亮星；每拍尽量多发迁（sync_burst）；每请求只尝试一次。"""

    id = "rebalance_random"

    def __init__(self, dest: int, *, sync_burst: bool = True):
        self.dest = int(dest)
        self.sync_burst = bool(sync_burst)
        self._attempted: set[int] = set()

    def place(self, world, req) -> int:
        return place_by_iter_time(world, req)

    def maybe_rebalance(self, world) -> None:
        launched = 0
        while True:
            inflight = sum(1 for rs in world.nodes for r in rs if r.migrating)
            if inflight >= MAX_INFLIGHT:
                return
            now = world.env.now
            dark = [
                r
                for rs in world.nodes
                for r in rs
                if (not r.migrating)
                and r.nmig == 0
                and r.i not in self._attempted
                and r.t <= now
                and r.node != self.dest
                and world.f_of(self.dest, now) > world.f_of(r.node, now)
            ]
            if not dark:
                return
            by_src: dict[int, list] = defaultdict(list)
            for r in dark:
                by_src[r.node].append(r)
            eligible_src = [
                s
                for s, rs in by_src.items()
                if sum(1 for x in world.nodes[s] if x.migrating) < MAX_OUT
            ]
            if not eligible_src:
                return
            # 大 KV 优先：更容易在 E★ 上重叠
            src = max(eligible_src, key=lambda s: max(r.kv for r in by_src[s]))
            req = max(by_src[src], key=lambda r: r.kv)
            cap = float(world.cfg["gpu"]["kv_gib"]) * (2**30)
            used = sum(r.kv for r in world.nodes[self.dest]) * world.bpt
            if used + req.kv * world.bpt > cap:
                self._attempted.add(req.i)
                launched += 1
                if not self.sync_burst or launched > 64:
                    return
                continue
            self._attempted.add(req.i)
            world.env.process(world.migrate_proc(req, src, self.dest))
            launched += 1
            if not self.sync_burst or launched > 64:
                return


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _set_caps(inflight: int, out: int) -> None:
    for mod in (world_mod, random_mod, select_mod):
        mod.MAX_INFLIGHT = int(inflight)
        mod.MAX_OUT = int(out)


def _apply_thermal(cfg: dict, t_swap_s: float, p_cap: float, extra_dark: list[int]) -> None:
    apply_t_swap(cfg, t_swap_s, p_cap)
    # apply_t_swap: [[2,0,t,cap],[0,t,inf,cap]]；追加换位后额外暗星
    events = list(cfg["thermal"]["explicit_events"])
    for j in extra_dark:
        j = int(j)
        if j == 0:
            continue  # already dark after swap
        events.append([j, float(t_swap_s), 1.0e9, float(p_cap)])
    cfg["thermal"]["explicit_events"] = events


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


def run_one(cfg, pol, seed, trace, t_swap_s, slo, *, track: bool, scheduler=None):
    reqs, anchor, cluster_n = generate_arrivals(trace, float(cfg["workload"]["p_long"]), seed)
    env = simpy.Environment()
    bundle = bundle_from_policy(pol)
    if scheduler is not None:
        bundle = PolicyBundle(scheduler=scheduler, link_cost=bundle.link_cost)
    world = World(env, cfg, pol, t_swap_s, bundle=bundle, seed=seed)
    conc_peak = 0
    estar_peak = 0
    events: list[dict] = []

    if track:
        orig = world.link_cost.transfer

        def wrapped(world_, req, src, dst, nbytes: float):
            nonlocal conc_peak, estar_peak
            t0 = world_.env.now
            inflight = sum(1 for rs in world_.nodes for r in rs if r.migrating)
            conc_peak = max(conc_peak, inflight)
            path = world_.pick_path(src, dst)
            via_estar = any(_edge_key(a, b) == _edge_key(*ESTAR) for a, b in path)
            if via_estar:
                on = world_.net.flows_on(ESTAR) + world_.net.flows_on((ESTAR[1], ESTAR[0])) + 1
                estar_peak = max(estar_peak, on)
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
                    path=[list(h) for h in path],
                    via_estar=bool(via_estar),
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
    row = _row(reqs, world, cfg, t_swap_s, slo, anchor, cluster_n)
    lats = _latencies(reqs)
    via = [e for e in events if e["via_estar"]]
    extra = dict(
        conc_peak=int(conc_peak),
        estar_peak=int(estar_peak),
        n_mig_events=len(events),
        n_via_estar=len(via),
        abort_via_estar=sum(1 for e in via if e["aborted"]),
        lat_p50=float(np.percentile(lats, 50)) if lats else float("nan"),
        lat_p99=float(np.percentile(lats, 99)) if lats else float("nan"),
        mean_mig_dur=float(np.mean([e["dur_s"] for e in events])) if events else 0.0,
        mean_estar_dur=float(np.mean([e["dur_s"] for e in via])) if via else 0.0,
    )
    return row, extra, events


def mean_std(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return dict(mean=float(np.mean(a)), std=float(np.std(a)), n=int(len(a)))


def pct(val: float, base: float) -> str:
    if abs(base) < 1e-12:
        return "n/a"
    return f"{(val - base) / base * 100:+.1f}%"


def fmt_cell(val: float, base: float, *, is_base: bool, best: bool, nd: int = 1) -> str:
    body = f"{val:.{nd}f}"
    if not is_base:
        body = f"{body} ({pct(val, base)})"
    return f"**{body}**" if best else body


def main() -> None:
    exp = yaml.safe_load(EXP.read_text())
    base_path = (EXP.parent / exp["base"]).resolve()
    cfg0 = load_yaml(base_path)
    validate_base(cfg0)
    validate_trace(exp["trace"])
    t_swap_s, anchor, cluster_n = resolve_t_swap_s(exp, cfg0)
    p_cap = float(exp["t_swap"]["P_cap"])
    kn = exp["contention"]
    slo = float(exp["slo"]["tpot_slo"])
    seeds = [int(s) for s in exp["seeds"]]
    dest = int(kn["corridor_bright_node"])
    extra_dark = list(kn.get("extra_dark_after_swap") or [])
    sync_burst = bool(kn.get("sync_burst", True))

    pols = {
        "place_once": {"scheduler": "place_once", "link_cost": "off"},
        "rebalance_free": {"scheduler": "rebalance_free", "link_cost": "off", "hysteresis": 0.15},
        "rebalance_random": {"scheduler": "rebalance_random", "link_cost": "full"},
    }

    _set_caps(int(kn["max_inflight"]), int(kn["max_out"]))
    results: list[dict] = []
    try:
        for arm in exp["arms"]:
            for seed in seeds:
                cfg = copy.deepcopy(cfg0)
                setp(cfg, "background.gbps_per_node", float(kn["background_gbps_per_node"]))
                setp(cfg, "net.link_gbps", float(kn["link_gbps"]))
                setp(cfg, "cost.timeout_s", float(kn["timeout_s"]))
                cfg["control_period_s"] = float(kn["control_period_s"])
                _apply_thermal(cfg, t_swap_s, p_cap, extra_dark)
                sched = (
                    CorridorBurstRandom(dest, sync_burst=sync_burst)
                    if arm == "rebalance_random"
                    else None
                )
                row, extra, _ = run_one(
                    cfg,
                    pols[arm],
                    seed,
                    exp["trace"],
                    t_swap_s,
                    slo,
                    track=(arm == "rebalance_random"),
                    scheduler=sched,
                )
                rec = dict(
                    arm=arm,
                    seed=seed,
                    good_long=row["good_long"],
                    goodput_tok_s=round(row["goodput_tok_s"], 4),
                    migrations=row["migrations"],
                    migration_abort=row["migration_abort"],
                    link_traffic_GB=round(row["link_traffic_GB"], 4),
                    max_flows_on_link=row["max_flows_on_link"],
                    T_SWAP=t_swap_s,
                    **extra,
                )
                results.append(rec)
                print(
                    f"{arm:18} s={seed} gl={rec['good_long']:2d} gp={rec['goodput_tok_s']:7.1f} "
                    f"mig={rec['migrations']:2d} ab={rec['migration_abort']:2d} "
                    f"linkGB={rec['link_traffic_GB']:5.1f} peak={rec['conc_peak']} "
                    f"estar={rec['estar_peak']} via={rec['n_via_estar']} p99={rec['lat_p99']:.0f}",
                    flush=True,
                )
    finally:
        _set_caps(DEFAULT_INF, DEFAULT_OUT)

    by_arm: dict[str, list] = defaultdict(list)
    for r in results:
        by_arm[r["arm"]].append(r)

    def agg(arm: str) -> dict:
        rs = by_arm[arm]
        return dict(
            arm=arm,
            good_long=mean_std([r["good_long"] for r in rs]),
            goodput=mean_std([r["goodput_tok_s"] for r in rs]),
            migrations=mean_std([r["migrations"] for r in rs]),
            abort=mean_std([r["migration_abort"] for r in rs]),
            link_GB=mean_std([r["link_traffic_GB"] for r in rs]),
            max_flows=mean_std([float(r["max_flows_on_link"]) for r in rs]),
            conc_peak=mean_std([r["conc_peak"] for r in rs]),
            estar_peak=mean_std([r["estar_peak"] for r in rs]),
            lat_p99=mean_std([r["lat_p99"] for r in rs]),
            n_via_estar=mean_std([r["n_via_estar"] for r in rs]),
            abort_via_estar=mean_std([r["abort_via_estar"] for r in rs]),
        )

    summary = {a: agg(a) for a in exp["arms"]}
    base_gl = summary["place_once"]["good_long"]["mean"]
    base_gp = summary["place_once"]["goodput"]["mean"]
    order = [
        ("place_once", "首次分配"),
        ("rebalance_free", "无消耗再分配"),
        ("rebalance_random", "有消耗·E★汇聚"),
    ]
    best_gl = max(order, key=lambda x: summary[x[0]]["good_long"]["mean"])[0]
    best_gp = max(order, key=lambda x: summary[x[0]]["goodput"]["mean"])[0]

    lines = [
        "| 臂 | good_long ↑ | goodput ↑ | link_GB ↓ | abort ↓ | E★峰值流 |",
        "|---|---|---|---|---|---|",
    ]
    for arm, label in order:
        s = summary[arm]
        is_base = arm == "place_once"
        gl = fmt_cell(s["good_long"]["mean"], base_gl, is_base=is_base, best=(arm == best_gl))
        gp = fmt_cell(s["goodput"]["mean"], base_gp, is_base=is_base, best=(arm == best_gp))
        link = f"{s['link_GB']['mean']:.1f}"
        ab = f"{s['abort']['mean']:.1f}"
        ep = f"{s['estar_peak']['mean']:.1f}"
        if arm == best_gl:
            label = f"**{label}**"
        lines.append(f"| {label} | {gl} | {gp} | {link} | {ab} | {ep} |")
    table = "\n".join(lines)
    print("\n" + table + "\n")

    readout = dict(
        knobs=kn,
        estar=list(ESTAR),
        note=(
            "换位后暗 0+1、亮 2+3；有代价臂 sync_burst 迁向 node2，共享 E★=(0,2)。"
            "主轴 link_gbps；timeout/control/caps 为派生；bg=0。"
        ),
        table_md=table,
        vs_place_once={
            arm: dict(
                good_long_pct=pct(summary[arm]["good_long"]["mean"], base_gl),
                goodput_pct=pct(summary[arm]["goodput"]["mean"], base_gp),
            )
            for arm in exp["arms"]
        },
        summary=summary,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "exp_orbit_estar_contention.json"
    out_json.write_text(
        json.dumps(dict(readout=readout, per_seed=results, t_swap_s=t_swap_s), indent=2, ensure_ascii=False)
        + "\n"
    )
    print("json:", out_json)


if __name__ == "__main__":
    main()
