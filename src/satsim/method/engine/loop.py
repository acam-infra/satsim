"""SimPy 事件进程与单次 run。"""
from __future__ import annotations

import simpy

from satsim.input.generate.dispatch import generate_arrivals
from satsim.method.engine.world import Req, World, net_proc
from satsim.method.registry import bundle_from_policy
from satsim.output.metrics import row_from_run


def decode_proc(env: simpy.Environment, world: World, req: Req):
    if req.t > env.now:
        yield env.timeout(req.t - env.now)
    world.admit(req)
    period = float(world.cfg["control_period_s"])
    while req.gen < req.pout:
        it = world.iter_of(req.node)
        remain = req.pout - req.gen
        step = min(period, it * remain)
        t0, g0 = env.now, req.gen
        try:
            yield env.timeout(max(step, 1e-9))
            req.gen = min(req.pout, g0 + step / it)
        except simpy.Interrupt:
            dt = env.now - t0
            if dt > 0 and it > 0:
                req.gen = min(req.pout, g0 + dt / it)
        t1, g1 = env.now, req.gen
        if req.t_first is None and g1 >= 1:
            req.t_first = t0 + it * (1 - g0) if g0 < 1 else t0
        a = max(t0, world.t_swap)
        if t1 > a and t1 > t0:
            frac = (t1 - a) / (t1 - t0)
            req.post_tok += (g1 - g0) * frac
            req.post_time += t1 - a
    req.t_done = env.now
    world.leave(req)


def swap_proc(env: simpy.Environment, world: World):
    if world.t_swap > env.now:
        yield env.timeout(world.t_swap - env.now)
    for rs in world.nodes:
        for r in rs:
            world._poke(r)


def control_proc(env: simpy.Environment, world: World):
    period = float(world.cfg["control_period_s"])
    tend = float(world.cfg["arrival_window_s"]) + float(world.cfg["drain_s"])
    while env.now < tend:
        yield env.timeout(period)
        world.thermal_tick()
        world.try_move()


def run_one(cfg: dict, policy: dict, seed: int, trace: dict, t_swap_s: float, slo: float) -> dict:
    reqs, anchor, cluster_n = generate_arrivals(trace, float(cfg["workload"]["p_long"]), seed)
    env = simpy.Environment()
    bundle = bundle_from_policy(policy)
    world = World(env, cfg, policy, t_swap_s, bundle=bundle, seed=seed)
    env.process(net_proc(env, world.net))
    env.process(swap_proc(env, world))
    env.process(control_proc(env, world))
    for req in reqs:
        req.proc = env.process(decode_proc(env, world, req))
    arrival_window = float(cfg["arrival_window_s"])
    horizon = arrival_window + float(cfg["drain_s"])  # 仿真使役时间 = 到达窗 + drain
    env.run(until=horizon)
    good_long = n_long = completed_long = n_completed = 0
    tok_slo_out = 0
    tok_total = 0  # prompt + output for completed
    for r in reqs:
        if r.long:
            n_long += 1
        if r.t_done is not None:
            n_completed += 1
            tok_total += int(r.pin) + int(r.pout)
            if r.long:
                completed_long += 1
        post = (r.post_time / r.post_tok) if r.post_tok >= 1 else 0.0
        ok = r.t_done is not None and post <= slo
        if ok:
            tok_slo_out += r.pout
            if r.long:
                good_long += 1
    row = row_from_run(
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
    diag = getattr(world.thermal, "diagnostics", None)
    if callable(diag):
        d = diag()
        row["thermal_peak_T_C_max"] = d.get("peak_T_C_max")
        row["thermal_throttle_time_s_sum"] = d.get("throttle_time_s_sum")
        row["thermal_nodes_ever_throttled"] = d.get("nodes_ever_throttled")
    return row
