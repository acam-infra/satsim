"""世界状态：节点驻留、链路、背景流；通过接口做放置/迁移。"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import simpy

from satsim.domain import Request
from satsim.method.engine.timing import clock_factor, iter_time, kv_bytes_per_tok

MAX_OUT = 2
MAX_INFLIGHT = 8

Req = Request  # 兼容别名


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


def _bfs_path(n: int, s: int, d: int) -> list[tuple[int, int]]:
    """Low-id BFS on the ring; background pairs use the same tie-break."""
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

    def flows_on(self, edge: tuple[int, int]) -> int:
        return sum(1 for f in self.flows if edge in f.path)


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
    def __init__(self, env: simpy.Environment, cfg: dict, policy: dict, t_swap: float, bundle=None, seed: int = 0):
        self.env = env
        self.cfg = cfg
        self.policy = policy
        self.t_swap = t_swap
        self.n = cfg["nodes"]
        self.net = Net(env, cfg)
        self.nodes: list[list[Req]] = [[] for _ in range(self.n)]
        self.acc = dict(mig=0, abort=0, mig_bytes=0.0, max_flows_on_link=0)
        self.bpt = kv_bytes_per_tok(cfg["model"])
        if bundle is None:
            from satsim.method.registry import bundle_from_policy

            bundle = bundle_from_policy(policy)
        self.scheduler = bundle.scheduler
        self.link_cost = bundle.link_cost
        self.rng = random.Random(int(seed) ^ 0xC0FFEE)
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
        j = self.scheduler.place(self, req)
        req.node = j
        self.nodes[j].append(req)
        self.poke_node(j)

    def leave(self, req: Req) -> None:
        if req in self.nodes[req.node]:
            self.nodes[req.node].remove(req)
            self.poke_node(req.node)

    def pick_path(self, s: int, d: int) -> list[tuple[int, int]]:
        return _bfs_path(self.n, s, d)

    def try_move(self) -> None:
        self.scheduler.maybe_rebalance(self)

    def migrate_proc(self, req: Req, src: int, dst: int):
        req.migrating = True
        nbytes = req.kv * self.bpt
        yield from self.link_cost.transfer(self, req, src, dst, nbytes)

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
