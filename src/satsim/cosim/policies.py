"""Placement / migration policies for the co-simulation, from orbit-blind to ephemeris-driven.

| id            | places on                                   | migrates when                               |
|---------------|---------------------------------------------|---------------------------------------------|
| greedy        | reachable node with the fastest iteration    | never                                       |
| reactive      | same as greedy                               | its node is in the umbra (or, in stream     |
|               |                                             | mode, has lost the origin) — to the fastest |
|               |                                             | sunlit reachable node                       |
| ephemeris     | reachable node with the earliest predicted  | never                                       |
|               | delivery (umbra + reachability from the     |                                             |
|               | trace, i.e. the orbit is known in advance)  |                                             |
| contact_aware | same as ephemeris                            | the predicted delivery improves by more     |
|               |                                             | than the margin after paying the KV transfer|
|               |                                             | on a path that stays up for the transfer    |

The orbit is deterministic, so using the trace for prediction is legitimate: it is the
"ephemeris" a real scheduler would hold. ``ephemeris`` isolates what placement alone buys;
``contact_aware`` - ``ephemeris`` is what runtime migration adds on top.
"""
from __future__ import annotations

import math
from collections import Counter

from satsim.cosim.trace import INF
from satsim.method.engine.timing import iter_time

MAX_INFLIGHT = 8
MAX_MIG_PER_REQ = 2


def candidates(w, r) -> list[int]:
    """Compute nodes reachable from the request's origin now, with KV room."""
    t = w.env.now
    return [j for j in w.tr.compute if w.tr.connected(r.origin, j, t) and w.fits(j, r)]


_HOP_CACHE: dict[tuple[int, int, int], list[int]] = {}


def hop_count(w, origin: int, j: int, t: float | None = None) -> int:
    """ISL hop count from origin to j at time t (cached single-source BFS per epoch)."""
    now = w.env.now if t is None else t
    ep = w.tr.epoch_index(now)
    key = (id(w.tr), origin, ep)
    dists = _HOP_CACHE.get(key)
    if dists is None:
        adj = w.tr.adjacency(now)
        dists = [999999] * w.tr.n
        dists[origin] = 0
        q = [origin]
        idx = 0
        while idx < len(q):
            u = q[idx]
            idx += 1
            if u != origin and w.tr.role[u] == "rs":
                continue
            du = dists[u] + 1
            for v, _ in adj[u]:
                if dists[v] == 999999:
                    dists[v] = du
                    q.append(v)
        _HOP_CACHE[key] = dists
    return dists[j]


def predict_decode_end(w, j: int, r, t0: float, remaining: float, *, joining: bool) -> float:
    """When r would finish decoding on j starting at t0, with j's current load held fixed."""
    extra = r if joining else None
    it_lit = w.iter_of(j, extra, f=1.0)
    it_dark = w.iter_of(j, extra, f=w.f_dark)
    left = remaining
    for a, b, dark in w.tr.power_segments(j, t0):
        it = it_dark if dark else it_lit
        span = b - a
        if left * it <= span:
            return a + left * it
        left -= span / it
    return INF


def predicted_cost(w, j: int, r, t0: float, remaining: float, *, joining: bool) -> float:
    """Predicted delivery time (+ weighted stall in stream mode)."""
    end = predict_decode_end(w, j, r, t0, remaining, joining=joining)
    if end == INF:
        return INF
    deliver = w.tr.next_conn(r.origin, j, end)
    if deliver == INF:
        return INF
    cost = deliver
    if w.stream:
        cost += float(w.cfg["policy"]["stall_weight"]) * w.tr.disconnected_time(r.origin, j, t0, end)
    return cost


def transfer_estimate(w, p, nbytes: float) -> float:
    """Seconds to move nbytes along path p, sharing each link with the flows already on it."""
    if not p[1]:
        return 0.0
    users: dict[int, int] = {}
    for f in w.net.flows:
        for _, _, e in f.hops:
            users[e] = users.get(e, 0) + 1
    bw = min(w.net.cap[e] / (users.get(e, 0) + 1) for e in p[1])
    return nbytes / bw + w.hop_lat * len(p[1]) + w.switch_s


def path_lifetime(w, p) -> float:
    t = w.env.now
    return min((w.tr.edge_up_until(e, t) for e in p[1]), default=INF)


def inflight(w) -> int:
    return sum(1 for rs in w.nodes.values() for r in rs if r.migrating)


def iter_time_of(w, j: int, residents) -> float:
    """Iteration time of node j if only ``residents`` were on it (current power state)."""
    from satsim.method.engine.timing import iter_time

    return iter_time(max(len(residents), 1), sum(x.kv for x in residents), w.f_of(j), w.gpu)


class Greedy:
    id = "greedy"

    def place(self, w, r):
        cs = candidates(w, r)
        return min(cs, key=lambda j: (w.iter_of(j, r), j)) if cs else None

    def rebalance(self, w) -> None:
        return


class Ephemeris(Greedy):
    id = "ephemeris"

    def place(self, w, r):
        cs = candidates(w, r)
        if not cs:
            return None
        t = w.env.now
        return min(cs, key=lambda j: (predicted_cost(w, j, r, t, r.remaining, joining=True), j))


class ReachableRoundRobin(Greedy):
    def __init__(self):
        self.next_index = 0

    def place(self, w, r):
        eligible = set(candidates(w, r))
        if not eligible:
            return None
        for offset in range(len(w.tr.compute)):
            index = (self.next_index + offset) % len(w.tr.compute)
            node = w.tr.compute[index]
            if node in eligible:
                self.next_index = (index + 1) % len(w.tr.compute)
                return node


class LeastReservedKV(Greedy):
    def place(self, w, r):
        return min(candidates(w, r), key=lambda j: (w.kv_used(j), j), default=None)


def upload_options(w, r):
    """Currently feasible nodes and their upload ETA under a snapshot of link sharing."""
    now = w.env.now
    users = Counter(e for flow in w.net.flows for _, _, e in flow.hops)
    for j in candidates(w, r):
        path = w.tr.path(r.origin, j, now)
        if path is None:
            continue
        edges = path[1]
        bw = min(w.net.cap[e] / (users[e] + 1) for e in edges)
        duration = r.upload_bytes / bw + w.hop_lat * len(edges)
        if path_lifetime(w, path) >= now + duration:
            yield j, duration



class RoundRobinPlacer:
    """Cyclic round-robin over reachable compute nodes; blind to queue load and umbra state."""

    id = "rr"

    def __init__(self):
        self._last = -1

    def place(self, w, r):
        cs = candidates(w, r)
        if not cs:
            return None
        higher = [j for j in cs if j > self._last]
        pick = higher[0] if higher else cs[0]
        self._last = pick
        return pick

    def rebalance(self, w) -> None:
        return


class Nearest:
    """Shortest-hop (access-local) placement from the origin RS satellite; RR tie-breaking."""

    id = "nearest"

    def __init__(self, tie: str = "rr"):
        self.tie = tie
        self._last = -1

    def place(self, w, r):
        cs = candidates(w, r)
        if not cs:
            return None
        t = w.env.now
        hops = {j: hop_count(w, r.origin, j, t) for j in cs}
        min_h = min(hops.values())
        best = [j for j in cs if hops[j] == min_h]
        if self.tie == "id" or len(best) == 1:
            pick = best[0]
        else:
            higher = [j for j in best if j > self._last]
            pick = higher[0] if higher else best[0]
        self._last = pick
        return pick

    def rebalance(self, w) -> None:
        return


class RandomPlace:
    """Uniform random placement across reachable compute nodes."""

    id = "random"

    def place(self, w, r):
        cs = candidates(w, r)
        return w.rng.choice(cs) if cs else None

    def rebalance(self, w) -> None:
        return


class LeastLoaded:
    """Join-Shortest-Queue (JSQ): fewest active requests, blind to umbra power throttling."""

    id = "least_loaded"

    def place(self, w, r):
        cs = candidates(w, r)
        return min(cs, key=lambda j: (len(w.nodes[j]), w.kv_used(j), j)) if cs else None

    def rebalance(self, w) -> None:
        return


class NearestUpload(Greedy):
    def place(self, w, r):
        best = min(upload_options(w, r), key=lambda x: (x[1], x[0]), default=None)
        return best[0] if best else None


def thermal_service_clock(w, j, batch, tokens, context):
    """Estimate the lit decode clock over a batch's service interval from its heat budget."""
    thermal = w.thermal
    if thermal is None or w.thermal_f[j] < 1:
        return w.thermal_f[j]
    power = w.device_power(batch, 1.0)
    equilibrium = thermal.T_amb_C + thermal.R_th * power
    if equilibrium <= thermal.T_throttle_C:
        return 1.0
    temp = min(thermal.temperature(j), thermal.T_throttle_C)
    time_to_hot = thermal.tau_s * math.log((equilibrium - temp) / (equilibrium - thermal.T_throttle_C))
    full_speed_time = tokens * iter_time(batch, context, 1.0, w.gpu)
    if full_speed_time <= time_to_hot:
        return 1.0
    idle = w.cfg["power"]["P_idle"]
    sustainable_w = (thermal.T_throttle_C - thermal.T_amb_C) / thermal.R_th
    ratio = max(0.0, (sustainable_w - idle) / (power - idle))
    sustainable_f = max(thermal.f_throttle, ratio ** (1 / thermal.gamma))
    slowed = (full_speed_time - time_to_hot) / sustainable_f
    return full_speed_time / (time_to_hot + slowed)


class OrbitLoad(Greedy):
    """One-shot placement with upload, an output-length prior, and a load/heat forecast."""

    def __init__(self, thermal_aware: bool = True):
        self.thermal_aware = thermal_aware

    def place(self, w, r):
        bounds = w.cfg["workload"]["long_out" if r.long else "short_out"]
        tokens = sum(bounds) / len(bounds)
        best = min(upload_options(w, r), key=lambda x: (self._delivery(w, r, tokens, *x), x[1], x[0]),
                   default=None)
        return best[0] if best else None

    def _delivery(self, w, r, tokens, j, upload_s):
        batch = len(w.nodes[j]) + 1 + int(w.reserved[j] > 0)
        context = sum(x.kv for x in w.nodes[j]) + r.pin + tokens / 2 + w.reserved[j] / w.bpt / 2
        start = w.env.now + upload_s
        clock = thermal_service_clock(w, j, batch, tokens, context) if self.thermal_aware else 1.0
        for a, b, dark in w.tr.power_segments(j, start):
            f = min(clock, w.f_dark) if dark else clock
            step = iter_time(batch, context, f, w.gpu)
            if tokens * step <= b - a:
                return w.tr.next_conn(r.origin, j, a + tokens * step)
            tokens -= (b - a) / step
        return INF


class Reactive(Greedy):
    id = "reactive"

    def __init__(self, speedup: float = 1.3, max_inflight: int | None = None,
                 max_mig_per_req: int | None = None, interval_s: float | None = None):
        self.speedup = speedup
        self.max_inflight = MAX_INFLIGHT if max_inflight is None else int(max_inflight)
        self.max_mig_per_req = MAX_MIG_PER_REQ if max_mig_per_req is None else int(max_mig_per_req)
        self.interval_s = interval_s
        self.next_plan = float(interval_s) if interval_s is not None else 0.0

    def rebalance(self, w) -> None:
        now = w.env.now
        if self.interval_s is not None:
            if now < self.next_plan:
                return
            self.next_plan = now + float(self.interval_s)
        t = now
        budget = self.max_inflight - inflight(w)
        for j in w.tr.compute:
            if budget <= 0:
                return
            dark = w.tr.is_dark(j, t)
            for r in list(w.nodes[j]):
                if budget <= 0:
                    return
                lost = w.stream and not w.tr.connected(r.origin, j, t)
                if not (dark or lost) or r.migrating or r.n_mig >= self.max_mig_per_req or r.remaining < 200:
                    continue
                here = w.iter_of(j)
                best, best_it = None, here / self.speedup
                for k in w.tr.compute:
                    if k == j or w.tr.is_dark(k, t) or not w.fits(k, r):
                        continue
                    if w.stream and not w.tr.connected(r.origin, k, t):
                        continue
                    it = w.iter_of(k, r)
                    if it < best_it:
                        best, best_it = k, it
                if best is not None and w.migrate(r, best):
                    budget -= 1


class ContactAware(Ephemeris):
    """Ephemeris placement + migration scored on the whole node pair, not just the mover.

    net gain = mover's predicted delivery gain
             + speed-up of the requests left behind on the source
             - slow-down imposed on the requests already on the destination
    (each side: Δ iteration time × their remaining tokens). One inbound move per destination per
    tick, and destinations with an inbound migration in flight are skipped, so a lit satellite is
    not flooded by every request that sees it at once. ``externality=False`` gives the v1 policy
    that scores the mover alone (it over-migrates under load; see docs/notes).
    """

    id = "contact_aware"

    def __init__(self, margin_s: float = 20.0, margin_frac: float = 0.1, every: int = 2, max_hops: int = 6,
                 externality: bool = True, max_inflight: int | None = None,
                 max_mig_per_req: int | None = None, interval_s: float | None = None):
        self.margin_s = margin_s
        self.margin_frac = margin_frac
        self.every = every
        self.max_hops = max_hops
        self.externality = externality
        self._tick = 0
        self.max_inflight = MAX_INFLIGHT if max_inflight is None else int(max_inflight)
        self.max_mig_per_req = MAX_MIG_PER_REQ if max_mig_per_req is None else int(max_mig_per_req)
        self.interval_s = interval_s
        self.next_plan = float(interval_s) if interval_s is not None else 0.0

    @staticmethod
    def _others_delay(w, j: int, exclude, d_it: float) -> float:
        return d_it * sum(x.remaining for x in w.nodes[j] if x is not exclude)

    def rebalance(self, w) -> None:
        self._tick += 1
        if self.every and self._tick % self.every:
            return
        now = w.env.now
        if self.interval_s is not None:
            if now < self.next_plan:
                return
            self.next_plan = now + float(self.interval_s)
        t = now
        budget = self.max_inflight - inflight(w)
        if budget <= 0:
            return
        inbound = {j for j in w.tr.compute if w.reserved[j] > 0}
        moves = []
        for j in w.tr.compute:
            for r in w.nodes[j]:
                if r.migrating or r.n_mig >= self.max_mig_per_req or r.remaining < 200:
                    continue
                stay = predicted_cost(w, j, r, t, r.remaining, joining=False)
                if not self._at_risk(w, j, r, t, stay):
                    continue
                it_here = w.iter_of(j)
                nbytes = r.kv * w.bpt
                relief = 0.0
                if self.externality and len(w.nodes[j]) > 1:
                    rest = [x for x in w.nodes[j] if x is not r]
                    it_without = iter_time_of(w, j, rest)
                    relief = self._others_delay(w, j, r, it_here - it_without)
                best = None
                for k in w.tr.compute:
                    if k == j or not w.fits(k, r) or (self.externality and k in inbound):
                        continue
                    p = w.tr.path(j, k, t)
                    if p is None or len(p[1]) > self.max_hops:
                        continue
                    dt = transfer_estimate(w, p, nbytes)
                    if path_lifetime(w, p) < t + 1.5 * dt:
                        continue  # the path would drop mid-transfer
                    left = max(0.0, r.remaining - dt / it_here)
                    move = predicted_cost(w, k, r, t + dt, left, joining=True)
                    gain = stay - move
                    if self.externality:
                        burden = self._others_delay(w, k, None, w.iter_of(k, r) - w.iter_of(k))
                        gain += relief - burden
                    if gain > max(self.margin_s, self.margin_frac * (stay - t)) and (best is None or gain > best[0]):
                        best = (gain, k)
                if best is not None:
                    moves.append((best[0], r, best[1]))
        moves.sort(key=lambda m: -m[0])
        taken: set[int] = set()
        for _, r, k in moves:
            if budget <= 0:
                break
            if self.externality and k in taken:
                continue
            if w.migrate(r, k):
                taken.add(k)
                budget -= 1

    @staticmethod
    def _at_risk(w, j, r, t, stay) -> bool:
        """Only re-plan requests whose plan is hurt by the orbit: umbra ahead, or a delivery gap."""
        if stay == INF:
            return True
        end_alone = t + r.remaining * w.iter_of(j, f=1.0)
        if w.tr.is_dark(j, t):
            return True
        for a, b, dark in w.tr.power_segments(j, t):
            if a >= end_alone:
                break
            if dark:
                return True
        return stay > end_alone + 30.0


class RoundRobin(Greedy):
    """Ground-style naive first placement: no orbit/load-future awareness.

    ``place`` skips nodes currently in the umbra or KV-full, then hands requests
    out round-robin over the reachable nodes in arrival order. Variants:
    ``random`` = uniform over nodes that fit; ``least_loaded`` = fewest residents
    right now. None of them look at the future, so whatever locality / thermal
    residual they leave is what migration must pay for.
    """

    id = "round_robin"

    def __init__(self, mode: str = "rr"):
        self.mode = mode
        self._rr = 0

    def place(self, w, r):
        cs = candidates(w, r)
        if not cs:
            return None
        if self.mode == "random":
            return w.rng.choice(cs)
        if self.mode == "least_loaded":
            return min(cs, key=lambda j: (len(w.nodes[j]), j))
        lit = [j for j in cs if not w.tr.is_dark(j, w.env.now)]
        pool = lit or cs
        self._rr += 1
        return pool[self._rr % len(pool)]

    def rebalance(self, w) -> None:
        return



class CompositePolicy:
    """Orthogonal composition of any Placer (place) and any Migrator (rebalance)."""

    def __init__(self, placer, migrator=None):
        self.placer = placer
        self.migrator = migrator
        self.id = f"{placer.id}+{migrator.id}" if migrator is not None else placer.id

    def place(self, w, r):
        return self.placer.place(w, r)

    def rebalance(self, w) -> None:
        if self.migrator is not None:
            self.migrator.rebalance(w)


PLACERS = {
    "rr": RoundRobinPlacer,
    "round_robin": RoundRobinPlacer,
    "nearest": Nearest, "nearest_upload": NearestUpload, "rr": RoundRobinPlacer,
    "random": RandomPlace,
    "least_loaded": LeastLoaded,
    "jsq": LeastLoaded,
    "greedy": Greedy,
    "ephemeris": Ephemeris,
}

MIGRATORS = {
    "none": None,
    "reactive": Reactive,
    "contact_aware": ContactAware,
}

POLICIES = {
    **PLACERS,
    "reactive": Reactive,
    "contact_aware": ContactAware,
}

def make_policy(spec: dict):
    if "place" in spec:
        p_kind = spec["place"]
        m_kind = spec.get("migrate", "none")
        if p_kind not in PLACERS:
            raise ValueError(f"unknown placer {p_kind!r}; one of {sorted(PLACERS)}")
        if m_kind not in MIGRATORS:
            raise ValueError(f"unknown migrator {m_kind!r}; one of {sorted(MIGRATORS)}")
        p_kw = spec.get("place_kwargs", {})
        m_kw = {k: v for k, v in spec.items() if k not in ("policy", "place", "migrate", "place_kwargs")}
        placer = PLACERS[p_kind](**p_kw)
        migrator = MIGRATORS[m_kind](**m_kw) if MIGRATORS[m_kind] is not None else None
        return CompositePolicy(placer, migrator)
    kind = spec["policy"]
    kw = {k: v for k, v in spec.items() if k != "policy"}
    if kind == "thermal_rebalance":
        from satsim.cosim.thermal_rebalance import ThermalRebalance

        return ThermalRebalance(**kw)
    if kind in ("torch_rebalance", "torch_lp"):
        from satsim.cosim.thermal_rebalance import TorchRebalance

        return TorchRebalance(**kw)
    table = {"greedy": Greedy, "ephemeris": Ephemeris, "reactive": Reactive, "contact_aware": ContactAware,
             "round_robin": RoundRobin,
             "random": lambda **_kw: RoundRobin(mode="random"),
             "least_loaded": lambda **_kw: RoundRobin(mode="least_loaded"),
             "round_robin_all": ReachableRoundRobin, "least_reserved_kv": LeastReservedKV,
             "nearest": Nearest, "orbit_load": OrbitLoad}
    if kind not in table:
        raise ValueError(f"unknown cosim policy {kind!r}; one of {sorted(table)}")
    return table[kind](**kw)
