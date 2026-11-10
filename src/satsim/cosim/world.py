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
from satsim.method.policies.thermal.dynamic import DynamicThermal

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
    output_budget: int | None = None  # declared cap, when provided by the workload
    decode_step: tuple[float, float] | None = None  # start time and seconds per token
    task_class: str = "standard"  # standard | scan_high_reaccess | pinpoint_deep | phase_transition
    anchor_node: int = -1  # initial compute node holding the raw source image (data anchor)
    reaccess_total: int = 0  # total mid-decode raw-image tile re-access pulls required
    reaccess_done: int = 0  # completed raw-image tile pulls
    reaccess_tile_bytes: float = 0.0  # bytes per raw-image tile pull from anchor_node
    tether_s: float = 0.0  # cumulative seconds spent fetching raw-image tiles across ISLs
    tether_bytes: float = 0.0  # cumulative bytes fetched from anchor_node across ISLs

    @property
    def reaccess_remaining(self) -> int:
        return max(0, self.reaccess_total - self.reaccess_done)

    @property
    def reserved_tokens(self) -> int:
        return self.pin + (self.pout if self.output_budget is None else self.output_budget)

    @property
    def kv(self) -> float:
        return self.pin + self.gen

    @property
    def remaining(self) -> float:
        return max(0.0, self.pout - self.gen)

    def progress_at(self, now: float) -> float:
        """Read partial decode progress without waking the SimPy process."""
        if self.state != "decode" or self.decode_step is None:
            return self.gen
        start, iteration = self.decode_step
        return self.gen + min(self.remaining, max(0.0, now - start) / iteration)


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
        thermal_cfg = cfg.get("thermal")
        self.thermal = DynamicThermal({**cfg["power"], **thermal_cfg}, n=trace.n) if thermal_cfg else None
        self.thermal_f = [self.thermal.clock_factor(j, 0.0) if self.thermal else 1.0 for j in range(trace.n)]
        self.thermal_power_w = [float(cfg["power"]["P_idle"])] * trace.n
        env.process(self._changes())
        env.process(self._control())

    # -- node state --------------------------------------------------------------------------
    def f_of(self, j: int, t: float | None = None) -> float:
        orbital_f = self.f_dark if self.tr.is_dark(j, self.env.now if t is None else t) else 1.0
        return min(orbital_f, self.thermal_f[j])

    def device_power(self, batch: int, f: float) -> float:
        """Idle plus a saturating batch-dependent fraction of the DVFS dynamic power."""
        power = self.cfg["power"]
        if batch == 0:
            return power["P_idle"]
        thermal = self.cfg["thermal"]
        utilization = 1 - (1 - thermal["busy_util_floor"]) * math.exp(-batch / thermal["batch_scale"])
        return power["P_idle"] + (power["P_max"] - power["P_idle"]) * utilization * f**power["gamma"]

    def _thermal_tick(self) -> None:
        now = self.env.now
        self.thermal.advance_with_power(now, self.thermal_power_w)
        for j in self.tr.compute:
            previous = self.f_of(j)
            self.thermal_f[j] = self.thermal.clock_factor(j, now)
            current = self.f_of(j)
            if current != previous:
                self.poke(j)
            self.thermal_power_w[j] = self.device_power(len(self.nodes[j]), current)

    def kv_used(self, j: int) -> float:
        """Bytes committed on j: each resident request's KV at completion, plus inbound reservations."""
        return sum(r.reserved_tokens for r in self.nodes[j]) * self.bpt + self.reserved[j]

    def fits(self, j: int, r: CRequest) -> bool:
        return self.kv_used(j) + r.reserved_tokens * self.bpt <= self.kv_cap

    def iter_of(self, j: int, extra: CRequest | None = None, f: float | None = None) -> float:
        rs = self.nodes[j]
        nd = len(rs) + (1 if extra is not None else 0)
        ctx = sum(r.kv for r in rs) + (extra.kv if extra is not None else 0.0)
        return iter_time(max(nd, 1), ctx, self.f_of(j) if f is None else f, self.gpu)

    def ideal_iter(self, r: CRequest) -> float:
        return iter_time(1, r.pin + r.pout / 2, 1.0, self.gpu)

    def poke(self, j: int) -> None:
        for r in list(self.nodes[j]):
            if r.state == "decode" and r.decode_step is not None and r.proc is not None and r.proc.is_alive:
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
            if self.thermal is not None:
                self._thermal_tick()
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
            need = r.reserved_tokens * self.bpt
            self.reserved[j] += need
            r.state = "upload"
            f = self.net.add("up", p[0], p[1], r.upload_bytes, req=r.i)
            self.log("upload", req=r.i, path=p[0])
            yield f.done
            if f.failed:
                self.reserved[j] -= need
                r.upload_fail += 1
                self.acc["upload_fail"] += 1
                r.state = "wait"
                continue
            yield self.env.timeout(self.hop_lat * len(p[1]))
            self.reserved[j] -= need
            break
        r.node = j
        if r.anchor_node < 0:
            r.anchor_node = j
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
            rem_to_next_pull = r.remaining
            if r.reaccess_total > 0 and r.reaccess_done < r.reaccess_total:
                next_pull_tok = (r.reaccess_done + 1) * (r.pout / (r.reaccess_total + 1))
                rem_to_next_pull = max(1.0, next_pull_tok - r.gen)
            step = min(CHUNK_S, it * min(r.remaining, rem_to_next_pull))
            t0, g0 = self.env.now, r.gen
            r.decode_step = (t0, it)
            try:
                yield self.env.timeout(max(step, 1e-9))
            except simpy.Interrupt:
                pass
            r.decode_step = None
            dt = self.env.now - t0
            got = min(r.remaining, dt / it) if it > 0 else 0.0
            r.gen = g0 + got
            if dark:
                r.dark_tok += got
            if self.stream and dt > 0:
                r.stall_s += self.tr.disconnected_time(r.origin, j, t0, self.env.now)
            if r.t_first is None and r.gen >= 1:
                r.t_first = self.env.now
            # Agentic raw-image data-anchor re-access: when reaching a tool-call milestone,
            # if migrated away from anchor_node, pull the high-res image tile across ISLs.
            if r.reaccess_total > 0 and r.reaccess_done < r.reaccess_total:
                next_pull_tok = (r.reaccess_done + 1) * (r.pout / (r.reaccess_total + 1))
                if r.gen + 1e-6 >= next_pull_tok:
                    r.reaccess_done += 1
                    # Phase-transition tasks (Type 3) finish wide-area scanning midway and enter pure reasoning
                    if r.task_class == "phase_transition" and r.reaccess_done >= max(1, r.reaccess_total // 2):
                        r.reaccess_done = r.reaccess_total
                    if r.node != r.anchor_node and r.reaccess_tile_bytes > 0:
                        t_pull0 = self.env.now
                        p_pull = self.tr.path(r.anchor_node, r.node, self.env.now)
                        if p_pull is not None and p_pull[1]:
                            f_pull = self.net.add("tether", p_pull[0], p_pull[1], r.reaccess_tile_bytes, req=r.i)
                            while not f_pull.done.triggered and self.env.now - t_pull0 < self.mig_timeout:
                                try:
                                    yield f_pull.done | self.env.timeout(
                                        max(1e-6, self.mig_timeout - (self.env.now - t_pull0))
                                    )
                                except simpy.Interrupt:
                                    pass
                            if not f_pull.done.triggered:
                                self.net.cancel(f_pull)
                            else:
                                try:
                                    yield self.env.timeout(self.hop_lat * len(p_pull[1]))
                                except simpy.Interrupt:
                                    pass
                            r.tether_bytes += r.reaccess_tile_bytes
                            self.acc["tether_bytes"] = self.acc.get("tether_bytes", 0.0) + r.reaccess_tile_bytes
                        r.tether_s += self.env.now - t_pull0
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
    def migrate(self, r: CRequest, dst: int, path: tuple[list[int], list[int]] | None = None) -> bool:
        if r.migrating or r.state != "decode" or dst == r.node:
            return False
        p = path if path is not None else self.tr.path(r.node, dst, self.env.now)
        if p is None or not self.fits(dst, r):
            return False
        self.env.process(self._migrate(r, r.node, dst, p))
        return True

    def _migrate(self, r: CRequest, src: int, dst: int, p):
        r.migrating = True
        need = r.reserved_tokens * self.bpt
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
    """Use a class deadline when configured, otherwise scale isolated decode time."""
    fixed = cfg["slo"].get("long_deadline_s" if long else "short_deadline_s")
    if fixed is not None:
        return t + float(fixed)
    ideal = pout * iter_time(1, pin + pout / 2, 1.0, cfg["gpu"])
    k = float(cfg["slo"]["deadline_factor_long" if long else "deadline_factor_short"])
    return t + max(k * ideal, float(cfg["slo"]["min_deadline_s"]))


def make_requests(cfg: dict, trace: Trace, seed: int) -> list[CRequest]:
    """Poisson arrivals at every RS satellite over the arrival window."""
    wl = cfg["workload"]
    rng = random.Random(seed)
    out: list[CRequest] = []
    window = float(wl["arrival_window_s"])
    rates = wl.get("source_rates_per_min")
    if rates is not None and len(rates) != len(trace.rs):
        raise ValueError("source_rates_per_min must cover each remote-sensing satellite")
    for source_index, o in enumerate(trace.rs):
        rate = float(rates[source_index] if rates is not None else wl["rate_per_rs_per_min"]) / 60.0
        if rate == 0:
            continue
        if rate < 0:
            raise ValueError("arrival rate must be nonnegative")
        source = []
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
            budget = hi if wl.get("reserve_output_budget") else None
            source.append(CRequest(0, o, t, long, pin, pout, up, deadline_for(cfg, t, pin, pout, long),
                                   output_budget=budget))
        if burst := wl.get("release_burst"):
            if source_index in burst.get("sources", range(len(trace.rs))):
                _compress_releases(source, source_index, window, burst)
        out.extend(source)
    out.sort(key=lambda r: (r.t, r.origin))
    agentic_cfg = wl.get("agentic_classes")
    rng_cls = random.Random(seed ^ 0xA6E4) if agentic_cfg else None
    for i, r in enumerate(out):
        r.i = i
        if agentic_cfg and r.long and rng_cls is not None:
            u = rng_cls.random()
            p_scan = float(agentic_cfg.get("p_scan", 0.35))
            p_pinpoint = float(agentic_cfg.get("p_pinpoint", 0.40))
            tile_mb = float(agentic_cfg.get("tile_MB", 125.0))
            if u < p_scan:
                r.task_class = "scan_high_reaccess"
                r.reaccess_total = int(agentic_cfg.get("scan_pulls", 5))
                r.reaccess_tile_bytes = tile_mb * 1e6
            elif u < p_scan + p_pinpoint:
                r.task_class = "pinpoint_deep"
                r.reaccess_total = int(agentic_cfg.get("pinpoint_pulls", 0))
                r.reaccess_tile_bytes = 0.0
            else:
                r.task_class = "phase_transition"
                r.reaccess_total = int(agentic_cfg.get("phase_pulls", 4))
                r.reaccess_tile_bytes = tile_mb * 1e6
    return out


def _compress_releases(requests: list[CRequest], source_index: int, window: float, burst: dict) -> None:
    """Compress each source's releases into its assigned window without changing its jobs."""
    period = float(burst["period_s"])
    duration = float(burst["duration_s"])
    phase = source_index * float(burst["source_phase_s"])
    if period <= 0 or duration <= 0 or phase < 0 or phase + duration > period:
        raise ValueError("source release window must fit inside the period")
    for r in requests:
        block = math.floor(r.t / period) * period
        scale = min(period, window - block) / period
        release = block + (phase + (r.t - block) / period * duration) * scale
        r.deadline += release - r.t
        r.t = release


def finite(x: float) -> float | None:
    return None if x is None or math.isinf(x) or math.isnan(x) else x
