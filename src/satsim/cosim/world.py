"""Event loop of the co-simulation: requests from RS satellites served by a moving compute fleet.

Life of a request (one SimPy process):

1. **admit**  wait until some compute node is reachable from the origin RS and has KV room; the
   policy picks one. The prompt payload (e.g. an image tile) is uploaded along the current path.
   A link dropping mid-upload fails the upload; the request goes back to step 1.
2. **decode** iterate on the resident node: iteration time from ``timing.iter_time`` with the
   node's clock factor — 1 in sunlight, ``clock_factor(P_dark)`` in the umbra. Every control
   period the policy may migrate it: its KV crosses the ISLs as a fluid flow while it keeps
   decoding at the source; a link dropping or a timeout aborts the migration.
3. **deliver** wait until the node can reach the origin again, plus hop latency.

Outputs per request: arrival / admitted / decoded / delivered times, stalled seconds (decoding
while the origin is unreachable), tokens decoded in the dark, migrations and aborts.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import simpy

from satsim.cosim.net import FluidNet
from satsim.cosim.trace import INF, Trace
from satsim.method.engine.timing import clock_factor, iter_time, kv_bytes_per_tok

CHUNK_S = 30.0  # longest decode step between re-evaluations when nothing pokes the request


@dataclass(eq=False)
class CRequest:
    i: int
    origin: int
    t: float
    long: bool
    pin: int
    pout: int
    upload_bytes: float
    deadline: float
    node: int = -1
    state: str = "wait"  # wait | upload | decode | deliver | done
    gen: float = 0.0
    t_admit: float | None = None
    t_first: float | None = None
    t_decoded: float | None = None
    t_delivered: float | None = None
    stall_s: float = 0.0
    dark_tok: float = 0.0
    upload_fail: int = 0
    n_mig: int = 0
    mig_abort: int = 0
    migrating: bool = False
    proc: object = None
    trail: list = field(default_factory=list)  # nodes visited

    @property
    def kv(self) -> float:
        return self.pin + self.gen

    @property
    def remaining(self) -> float:
        return max(0.0, self.pout - self.gen)


class CoWorld:
    def __init__(self, env: simpy.Environment, trace: Trace, cfg: dict, policy, recorder=None, seed: int = 0):
        self.env = env
        self.tr = trace
        self.cfg = cfg
        self.policy = policy
        self.rec = recorder
        self.rng = random.Random(seed ^ 0x5EED)
        self.gpu = cfg["gpu"]
        self.bpt = kv_bytes_per_tok(cfg["model"])
        self.kv_cap = float(self.gpu["kv_gib"]) * 2**30
        self.f_dark = clock_factor(float(cfg["power"]["P_dark"]), cfg["power"])
        caps = cfg["net"]["link_gbps"]
        self.net = FluidNet(env, [float(caps[k]) * 1e9 / 8 for k in trace.kind])
        self.hop_lat = float(cfg["net"]["hop_latency_s"])
        self.mig_timeout = float(cfg["migration"]["timeout_s"])
        self.switch_s = float(cfg["migration"]["switch_s"])
        self.period = float(cfg["control_period_s"])
        self.stream = cfg["delivery"] == "stream"
        self.nodes: dict[int, list[CRequest]] = {j: [] for j in trace.compute}
        self.reserved = {j: 0.0 for j in trace.compute}  # KV bytes of inbound uploads / migrations
        self.reqs: list[CRequest] = []
        self.topo_event = env.event()
        self.acc = dict(mig=0, mig_abort=0, mig_bytes=0.0, upload_fail=0)
        env.process(self._changes())
        env.process(self._control())

    # -- node state --------------------------------------------------------------------------
    def f_of(self, j: int, t: float | None = None) -> float:
        return self.f_dark if self.tr.is_dark(j, self.env.now if t is None else t) else 1.0

    def kv_used(self, j: int) -> float:
        """Bytes committed on j: each resident request's KV at completion, plus inbound reservations."""
        return sum(r.pin + r.pout for r in self.nodes[j]) * self.bpt + self.reserved[j]

    def fits(self, j: int, r: CRequest) -> bool:
        return self.kv_used(j) + (r.pin + r.pout) * self.bpt <= self.kv_cap

    def iter_of(self, j: int, extra: CRequest | None = None, f: float | None = None) -> float:
        rs = self.nodes[j]
        nd = len(rs) + (1 if extra is not None else 0)
        ctx = sum(r.kv for r in rs) + (extra.kv if extra is not None else 0.0)
        return iter_time(max(nd, 1), ctx, self.f_of(j) if f is None else f, self.gpu)

    def ideal_iter(self, r: CRequest) -> float:
        return iter_time(1, r.pin + r.pout / 2, 1.0, self.gpu)

    def poke(self, j: int) -> None:
        for r in list(self.nodes[j]):
            if r.state == "decode" and r.proc is not None and r.proc.is_alive:
                try:
                    r.proc.interrupt()
                except RuntimeError:
                    pass

    def log(self, kind: str, **kw) -> None:
        if self.rec is not None:
            self.rec.event(self.env.now, kind, **kw)

    # -- background processes ----------------------------------------------------------------
    def _changes(self):
        for t in self.tr.change_times():
            if t > self.env.now:
                yield self.env.timeout(t - self.env.now)
            now = self.env.now
            for f in self.net.abort_dead(lambda e: self.tr.edge_up(e, now)):
                self.log("cut", tag=f.tag, req=f.meta.get("req"))
            for j in self.tr.compute:
                self.poke(j)
            ev, self.topo_event = self.topo_event, self.env.event()
            ev.succeed()

    def _control(self):
        while True:
            yield self.env.timeout(self.period)
            self.policy.rebalance(self)

    # -- request lifecycle -------------------------------------------------------------------
    def submit(self, r: CRequest) -> None:
        self.reqs.append(r)
        r.proc = self.env.process(self._life(r))

    def _life(self, r: CRequest):
        if r.t > self.env.now:
            yield self.env.timeout(r.t - self.env.now)
        self.log("arrive", req=r.i, o=r.origin, long=r.long)
        while True:  # admission + upload
            j = self.policy.place(self, r)
            if j is None:
                yield self.topo_event | self.env.timeout(self.period)
                continue
            p = self.tr.path(r.origin, j, self.env.now)
            if p is None:
                yield self.env.timeout(self.period)
                continue
            need = (r.kv + r.remaining) * self.bpt
            self.reserved[j] += need
            r.state = "upload"
            f = self.net.add("up", p[0], p[1], r.upload_bytes, req=r.i)
            self.log("upload", req=r.i, path=p[0])
            yield f.done
            self.reserved[j] -= need
            if f.failed:
                r.upload_fail += 1
                self.acc["upload_fail"] += 1
                r.state = "wait"
                continue
            yield self.env.timeout(self.hop_lat * len(p[1]))
            break
        r.node = j
        r.trail.append(j)
        r.t_admit = self.env.now
        r.state = "decode"
        self.nodes[j].append(r)
        self.poke(j)
        self.log("admit", req=r.i, node=j)

        while r.remaining > 1e-9:  # decode
            j = r.node
            dark = self.tr.is_dark(j, self.env.now)
            it = self.iter_of(j)
            step = min(CHUNK_S, it * r.remaining)
            t0, g0 = self.env.now, r.gen
            try:
                yield self.env.timeout(max(step, 1e-9))
            except simpy.Interrupt:
                pass
            dt = self.env.now - t0
            got = min(r.remaining, dt / it) if it > 0 else 0.0
            r.gen = g0 + got
            if dark:
                r.dark_tok += got
            if self.stream and dt > 0:
                r.stall_s += self.tr.disconnected_time(r.origin, j, t0, self.env.now)
            if r.t_first is None and r.gen >= 1:
                r.t_first = self.env.now
        r.state = "deliver"
        r.t_decoded = self.env.now
        j = r.node
        self.nodes[j].remove(r)
        self.poke(j)
        t_conn = self.tr.next_conn(r.origin, j, self.env.now)
        if t_conn == INF:
            r.state = "done"
            self.log("lost", req=r.i, node=j)
            return
        if t_conn > self.env.now:
            yield self.env.timeout(t_conn - self.env.now)
        p = self.tr.path(j, r.origin, self.env.now)
        hops = len(p[1]) if p else 1
        yield self.env.timeout(self.hop_lat * hops)
        r.t_delivered = self.env.now
        r.state = "done"
        self.log("deliver", req=r.i, node=j, o=r.origin, path=p[0] if p else [j, r.origin])

    # -- migration ---------------------------------------------------------------------------
    def migrate(self, r: CRequest, dst: int) -> bool:
        if r.migrating or r.state != "decode" or dst == r.node:
            return False
        p = self.tr.path(r.node, dst, self.env.now)
        if p is None or not self.fits(dst, r):
            return False
        self.env.process(self._migrate(r, r.node, dst, p))
        return True

    def _migrate(self, r: CRequest, src: int, dst: int, p):
        r.migrating = True
        need = (r.kv + r.remaining) * self.bpt
        self.reserved[dst] += need
        nbytes = r.kv * self.bpt
        f = self.net.add("mig", p[0], p[1], nbytes, req=r.i)
        self.log("mig", req=r.i, src=src, dst=dst, path=p[0], GB=nbytes / 1e9)
        yield f.done | self.env.timeout(self.mig_timeout)
        ok = f.done.triggered and not f.failed
        if not ok:
            self.net.cancel(f)
        else:
            yield self.env.timeout(self.hop_lat * len(p[1]) + self.switch_s)
        self.reserved[dst] -= need
        r.migrating = False
        if not ok:
            r.mig_abort += 1
            self.acc["mig_abort"] += 1
            self.log("mig_abort", req=r.i, src=src, dst=dst)
            return
        if r.state != "decode" or r.node != src:  # finished at the source meanwhile
            return
        self.nodes[src].remove(r)
        r.node = dst
        r.trail.append(dst)
        r.n_mig += 1
        self.nodes[dst].append(r)
        self.acc["mig"] += 1
        self.acc["mig_bytes"] += nbytes
        self.log("mig_done", req=r.i, src=src, dst=dst)
        self.poke(src)
        self.poke(dst)

    # -- results -----------------------------------------------------------------------------
    def good(self, r: CRequest) -> bool:
        if r.t_delivered is None or r.t_delivered > r.deadline:
            return False
        if self.stream:
            dur = (r.t_decoded or r.t_delivered) - (r.t_admit or r.t)
            if dur > 0 and r.stall_s / dur > float(self.cfg["slo"]["max_stall_frac"]):
                return False
        return True


def deadline_for(cfg: dict, t: float, pin: int, pout: int, long: bool) -> float:
    """Arrival + factor × the request's decode time alone on a sunlit node."""
    ideal = pout * iter_time(1, pin + pout / 2, 1.0, cfg["gpu"])
    k = float(cfg["slo"]["deadline_factor_long" if long else "deadline_factor_short"])
    return t + max(k * ideal, float(cfg["slo"]["min_deadline_s"]))


def make_requests(cfg: dict, trace: Trace, seed: int) -> list[CRequest]:
    """Poisson arrivals at every RS satellite over the arrival window."""
    wl = cfg["workload"]
    rng = random.Random(seed)
    out: list[CRequest] = []
    window = float(wl["arrival_window_s"])
    rate = float(wl["rate_per_rs_per_min"]) / 60.0
    for o in trace.rs:
        t = float(wl.get("start_s", 0.0))
        while True:
            t += rng.expovariate(rate)
            if t >= window:
                break
            long = rng.random() < float(wl["p_long"])
            lo, hi = wl["long_out" if long else "short_out"]
            pout = rng.randint(int(lo), int(hi))
            pin = int(wl["long_pin" if long else "short_pin"])
            up = float(wl["long_upload_MB" if long else "short_upload_MB"]) * 1e6
            out.append(CRequest(0, o, t, long, pin, pout, up, deadline_for(cfg, t, pin, pout, long)))
    out.sort(key=lambda r: (r.t, r.origin))
    for i, r in enumerate(out):
        r.i = i
    return out


def finite(x: float) -> float | None:
    return None if x is None or math.isinf(x) or math.isnan(x) else x
