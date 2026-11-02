"""SimPy event loop for the orbit-pod claims.

Processes: request decode, power-cap swap, placement controller,
permanent background flows, KV migration. Link rates are recomputed
when a flow starts or finishes — there is no fixed timestep.

C1 uses free re-placement (the request changes node, KV stays off the wire).
C2 uses casual migration: the whole current KV is sent on a shortest ring
path with no byte budget. On a 4-ring the opposite node has two equal
paths; the one with more residual is used, and byte·hop follows that path.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import simpy

from satsim.spec import SpecError

ROUND_S = 0.1
MAX_OUT = 2
MAX_INFLIGHT = 8


@dataclass
class Req:
    i: int
    t: float
    long: bool
    pin: int
    pout: int
    node: int = -1
    gen: float = 0.0
    t_first: float | None = None
    t_done: float | None = None
    migrating: bool = False
    nmig: int = 0
    proc: object | None = None
    post_tok: float = 0.0
    post_time: float = 0.0

    @property
    def kv(self) -> float:
        return self.pin + self.gen


@dataclass
class Flow:
    path: list[tuple[int, int]]
    remaining: float
    demand: float
    priority: int
    sent: float = 0.0
    rate: float = 0.0
    done: object = None
    hops: int = 0


def kv_bytes_per_tok(model: dict) -> int:
    return 2 * model["layers"] * model["kv_heads"] * model["head_dim"] * model["bytes_per_elem"]


def clock_factor(p_cap: float, thermal: dict) -> float:
    dyn = max(p_cap - thermal["P_idle"], 1e-9) / (thermal["P_max"] - thermal["P_idle"])
    f = dyn ** (1.0 / thermal["gamma"])
    return min(max(f, thermal["f_min"]), 1.0)


def iter_time(nd: int, ctx: float, f: float, gpu: dict) -> float:
    beta = gpu["mem_freq_exp"]
    mem = gpu["t_weights_s"] + gpu["t_kv_tok_s"] * ctx
    return gpu["t_fixed_s"] + mem / (f ** beta) + (gpu["t_seq_s"] * nd) / f


def cluster_anchor(times: list[float]) -> tuple[float, int]:
    """Densest 0.1s slot. A flat pass ties at 1; anchor the last slot so the whole pass is resident."""
    if not times:
        raise SpecError("空到达")
    slots = [round(t / ROUND_S) * ROUND_S for t in times]
    counts = Counter(slots)
    best = max(counts.values())
    winners = sorted(t for t, c in counts.items() if c == best)
    t_star = winners[-1] if best == 1 and len(winners) > 1 else winners[0]
    return float(t_star), int(best)


def make_pass(trace: dict, p_long: float, seed: int) -> tuple[list[Req], float, int]:
    n = int(trace["target_n"])
    horizon = float(trace["compress_horizon_s"])
    jitter = float(trace["jitter_s"])
    pad = trace["token_pad"]
    rng = random.Random(seed)
    times = []
    for i in range(n):
        t = 0.0 if n == 1 else i * horizon / (n - 1)
        if jitter:
            t += rng.uniform(-jitter, jitter)
        times.append(min(horizon, max(0.0, t)))
    anchor, count = cluster_anchor(times)
    reqs = []
    for i, t in enumerate(times):
        long = rng.random() < p_long
        reqs.append(
            Req(
                i=i,
                t=t,
                long=long,
                pin=int(pad["long_pin"] if long else pad["short_pin"]),
                pout=int(pad["long_out"] if long else pad["short_out"]),
            )
        )
    return reqs, anchor, count


def ring_routes(n: int, s: int, d: int) -> list[list[tuple[int, int]]]:
    if s == d:
        return [[]]
    routes = []
    for step in (1, -1):
        path = []
        x = s
        while x != d:
            y = (x + step) % n
            path.append((x, y))
            x = y
        routes.append(path)
    hop = min(len(p) for p in routes)
    return [p for p in routes if len(p) == hop]


def _bfs_path(n: int, s: int, d: int) -> list[tuple[int, int]]:
    """Low-id BFS, used by the background pairs (same tie-break as the lab1616 ring)."""
    adj = [sorted(((i + 1) % n, (i - 1) % n)) for i in range(n)]
    prev: dict[int, int | None] = {s: None}
    q = [s]
    while q:
        u = q.pop(0)
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    hop = []
    x = d
    while prev[x] is not None:
        hop.append((prev[x], x))
        x = prev[x]
    return list(reversed(hop))


class Net:
    def __init__(self, env: simpy.Environment, cfg: dict):
        self.env = env
        self.cap = float(cfg["net"]["link_gbps"]) * 1e9 / 8
        self.flows: list[Flow] = []
        self.t = 0.0
        self.wake = env.event()
        self.bytehop = 0.0

    def kick(self) -> None:
        if not self.wake.triggered:
            self.wake.succeed()

    def add(self, path: list[tuple[int, int]], nbytes: float, priority: int, demand: float) -> Flow:
        flow = Flow(
            path=path,
            remaining=nbytes,
            demand=demand,
            priority=priority,
            hops=len(path),
            done=self.env.event(),
        )
        self.flows.append(flow)
        self.kick()
        return flow

    def cancel(self, flow: Flow) -> None:
        self.advance(self.env.now)
        if flow in self.flows:
            self.flows.remove(flow)
        flow.rate = 0.0
        self.kick()

    def advance(self, now: float) -> None:
        dt = now - self.t
        if dt <= 0:
            self.t = now
            return
        done = []
        for f in self.flows:
            if f.remaining == float("inf") or f.rate <= 0:
                continue
            step = min(f.remaining, f.rate * dt)
            f.remaining -= step
            f.sent += step
            self.bytehop += step * f.hops
            if f.remaining <= 1.0:
                f.remaining = 0.0
                done.append(f)
        self.t = now
        for f in done:
            self.flows.remove(f)
            f.rate = 0.0
            if not f.done.triggered:
                f.done.succeed()

    def allocate(self, now: float) -> None:
        self.advance(now)
        for f in self.flows:
            f.rate = 0.0
        self._assign(0, self.cap)
        used = defaultdict(float)
        for f in self.flows:
            if f.priority == 0:
                for e in f.path:
                    used[e] += f.rate
        # residual capacities live in the assign via a second pass
        self._assign(1, None, used)

    def _assign(self, priority: int, cap: float | None, used: dict | None = None) -> None:
        active = [f for f in self.flows if f.priority == priority and f.remaining > 0]
        if not active:
            return
        edges = {e for f in active for e in f.path}
        if cap is None:
            rem = {e: max(0.0, self.cap - used.get(e, 0.0)) for e in edges}
        else:
            rem = {e: cap for e in edges}
        rate = {id(f): 0.0 for f in active}
        pending = set(id(f) for f in active)
        by_id = {id(f): f for f in active}
        guard = 0
        while pending and guard < 10000:
            guard += 1
            cnt = Counter()
            for i in pending:
                for e in by_id[i].path:
                    if e in rem:
                        cnt[e] += 1
            if not cnt:
                break
            sh, edge = min((rem[e] / c, e) for e, c in cnt.items() if c)
            room = []
            for i in list(pending):
                f = by_id[i]
                need = f.demand - rate[i]
                if need <= sh + 1e-6:
                    room.append(i)
            takers = room or [i for i in pending if edge in by_id[i].path]
            give = sh
            if room:
                for i in takers:
                    got = max(0.0, by_id[i].demand - rate[i])
                    rate[i] += got
                    for e in by_id[i].path:
                        rem[e] = max(0.0, rem.get(e, 0.0) - got)
                    pending.discard(i)
            else:
                for i in takers:
                    rate[i] += give
                    for e in by_id[i].path:
                        rem[e] = max(0.0, rem.get(e, 0.0) - give)
                    pending.discard(i)
        for f in active:
            f.rate = rate[id(f)]

    def next_dt(self) -> float | None:
        dts = [f.remaining / f.rate for f in self.flows if f.rate > 1e-9 and f.remaining < float("inf")]
        return min(dts) if dts else None

    def residual(self, edge: tuple[int, int]) -> float:
        used = 0.0
        for f in self.flows:
            if f.priority == 0 and edge in f.path:
                used += f.demand
        return max(0.0, self.cap - used)


def net_proc(env: simpy.Environment, net: Net):
    while True:
        net.allocate(env.now)
        if net.wake.triggered:
            net.wake = env.event()
            continue
        dt = net.next_dt()
        wake = net.wake
        if dt is None:
            yield wake
            continue
        yield env.timeout(dt) | wake


class World:
    def __init__(self, env: simpy.Environment, cfg: dict, policy: dict, t_swap: float):
        self.env = env
        self.cfg = cfg
        self.policy = policy
        self.t_swap = t_swap
        self.n = cfg["nodes"]
        self.migrate = policy.get("migrate", "none") not in ("none", None)
        self.hyst = float(policy.get("hysteresis") or 0.15)
        self.net = Net(env, cfg)
        self.nodes: list[list[Req]] = [[] for _ in range(self.n)]
        self.acc = dict(mig=0, abort=0, mig_bytes=0.0)
        self.bpt = kv_bytes_per_tok(cfg["model"])
        self._install_background()

    def _install_background(self) -> None:
        bg = self.cfg["background"]
        total = float(bg["gbps_per_node"]) * self.n * 1e9 / 8
        pairs = [tuple(p) for p in bg["pairs"]]
        if total <= 0 or not pairs:
            return
        per = total / len(pairs)
        for s, d in pairs:
            path = _bfs_path(self.n, s, d)
            self.net.add(path, float("inf"), priority=0, demand=per)

    def f_of(self, node: int, t: float) -> float:
        p = self.cfg["thermal"]["P_max"]
        for ev in self.cfg["thermal"].get("explicit_events") or []:
            j, t0, t1, pc = ev
            if j == node and t0 <= t < t1:
                p = min(p, pc)
        return clock_factor(p, self.cfg["thermal"])

    def _stats(self):
        nd = [len(rs) for rs in self.nodes]
        ctx = [sum(r.kv for r in rs) for rs in self.nodes]
        return nd, ctx

    def iter_of(self, node: int) -> float:
        nd, ctx = self._stats()
        return iter_time(nd[node], ctx[node], self.f_of(node, self.env.now), self.cfg["gpu"])

    def _poke(self, req: Req) -> None:
        proc = req.proc
        if proc is None or not proc.is_alive:
            return
        try:
            proc.interrupt()
        except RuntimeError:
            pass

    def poke_node(self, node: int) -> None:
        for r in list(self.nodes[node]):
            self._poke(r)

    def admit(self, req: Req) -> None:
        cap = float(self.cfg["gpu"]["kv_gib"]) * (2 ** 30)
        scores = []
        for j in range(self.n):
            used = sum(r.kv for r in self.nodes[j]) * self.bpt
            if used + req.kv * self.bpt > cap:
                scores.append(float("inf"))
                continue
            nd, ctx = self._stats()
            scores.append(
                iter_time(nd[j] + 1, ctx[j] + req.kv, self.f_of(j, self.env.now), self.cfg["gpu"])
            )
        if min(scores) == float("inf"):
            raise SpecError(f"请求 {req.i} 在所有节点都放不下 KV")
        j = scores.index(min(scores))
        req.node = j
        self.nodes[j].append(req)
        self.poke_node(j)

    def leave(self, req: Req) -> None:
        if req in self.nodes[req.node]:
            self.nodes[req.node].remove(req)
            self.poke_node(req.node)

    def pick_path(self, s: int, d: int) -> list[tuple[int, int]]:
        routes = ring_routes(self.n, s, d)
        return max(routes, key=lambda p: min((self.net.residual(e) for e in p), default=0.0))

    def try_move(self) -> None:
        if not self.migrate:
            return
        nd, ctx = self._stats()
        f = [self.f_of(j, self.env.now) for j in range(self.n)]
        gpu = self.cfg["gpu"]
        sc = [iter_time(nd[j], ctx[j], f[j], gpu) for j in range(self.n)]
        src = max(range(self.n), key=lambda j: sc[j])
        dst = min(range(self.n), key=lambda j: sc[j])
        if src == dst or sc[src] <= sc[dst] * (1.0 + self.hyst):
            return
        inflight = [r for rs in self.nodes for r in rs if r.migrating]
        if len(inflight) >= MAX_INFLIGHT:
            return
        if sum(1 for r in self.nodes[src] if r.migrating) >= MAX_OUT:
            return
        cands = [r for r in self.nodes[src] if not r.migrating and r.nmig == 0 and r.t <= self.env.now]
        if not cands:
            return
        req = max(cands, key=lambda r: r.kv)
        cap = float(self.cfg["gpu"]["kv_gib"]) * (2 ** 30)
        used = sum(r.kv for r in self.nodes[dst]) * self.bpt
        if used + req.kv * self.bpt > cap:
            return
        self.env.process(self.migrate_proc(req, src, dst))

    def migrate_proc(self, req: Req, src: int, dst: int):
        req.migrating = True
        mode = self.cfg["cost"]["mode"]
        nbytes = req.kv * self.bpt
        if mode == "free":
            yield self.env.timeout(0)
            self._commit(req, src, dst, 0.0)
            return
        path = self.pick_path(src, dst)
        flow = self.net.add(path, nbytes, priority=1, demand=float("inf"))
        timeout = float(self.cfg["cost"]["timeout_s"])
        yield flow.done | self.env.timeout(timeout)
        if not flow.done.triggered:
            self.net.cancel(flow)
            self.acc["abort"] += 1
            req.migrating = False
            return
        tail = len(path) * float(self.cfg["net"]["hop_latency_s"]) + float(self.cfg["cost"]["switch_s"])
        if tail > 0:
            yield self.env.timeout(tail)
        self._commit(req, src, dst, nbytes)

    def _commit(self, req: Req, src: int, dst: int, nbytes: float) -> None:
        if req in self.nodes[src]:
            self.nodes[src].remove(req)
        req.node = dst
        req.nmig += 1
        req.migrating = False
        if req.t_done is None:
            self.nodes[dst].append(req)
        self.acc["mig"] += 1
        self.acc["mig_bytes"] += nbytes
        self.poke_node(src)
        self.poke_node(dst)


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
        world.try_move()


def run_one(cfg: dict, policy: dict, seed: int, trace: dict, t_swap_s: float, slo: float) -> dict:
    reqs, anchor, cluster_n = make_pass(trace, float(cfg["workload"]["p_long"]), seed)
    env = simpy.Environment()
    world = World(env, cfg, policy, t_swap_s)
    env.process(net_proc(env, world.net))
    env.process(swap_proc(env, world))
    env.process(control_proc(env, world))
    for req in reqs:
        req.proc = env.process(decode_proc(env, world, req))
    tend = float(cfg["arrival_window_s"]) + float(cfg["drain_s"])
    env.run(until=tend)
    good_long = n_long = 0
    tok = 0
    for r in reqs:
        if r.long:
            n_long += 1
        post = (r.post_time / r.post_tok) if r.post_tok >= 1 else 0.0
        ok = r.t_done is not None and post <= slo
        if ok:
            tok += r.pout
            if r.long:
                good_long += 1
    return dict(
        status="ok",
        good_long=good_long,
        n_long=n_long,
        goodput_tok_s=tok / float(cfg["arrival_window_s"]),
        mig=world.acc["mig"],
        mig_abort=world.acc["abort"],
        mig_GB=world.acc["mig_bytes"] / 1e9,
        mig_byteHop_GB=world.net.bytehop / 1e9,
        T_SWAP=t_swap_s,
        max_cluster_t=anchor,
        max_cluster_n=cluster_n,
        n_req=len(reqs),
    )
