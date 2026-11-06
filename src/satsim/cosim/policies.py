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

from satsim.cosim.trace import INF

MAX_INFLIGHT = 8
MAX_MIG_PER_REQ = 2


def candidates(w, r) -> list[int]:
    """Compute nodes reachable from the request's origin now, with KV room."""
    t = w.env.now
    return [j for j in w.tr.compute if w.tr.connected(r.origin, j, t) and w.fits(j, r)]


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


class Reactive(Greedy):
    id = "reactive"

    def __init__(self, speedup: float = 1.3):
        self.speedup = speedup

    def rebalance(self, w) -> None:
        t = w.env.now
        budget = MAX_INFLIGHT - inflight(w)
        for j in w.tr.compute:
            if budget <= 0:
                return
            dark = w.tr.is_dark(j, t)
            for r in list(w.nodes[j]):
                if budget <= 0:
                    return
                lost = w.stream and not w.tr.connected(r.origin, j, t)
                if not (dark or lost) or r.migrating or r.n_mig >= MAX_MIG_PER_REQ or r.remaining < 200:
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
                 externality: bool = True):
        self.margin_s = margin_s
        self.margin_frac = margin_frac
        self.every = every
        self.max_hops = max_hops
        self.externality = externality
        self._tick = 0

    @staticmethod
    def _others_delay(w, j: int, exclude, d_it: float) -> float:
        return d_it * sum(x.remaining for x in w.nodes[j] if x is not exclude)

    def rebalance(self, w) -> None:
        self._tick += 1
        if self._tick % self.every:
            return
        t = w.env.now
        budget = MAX_INFLIGHT - inflight(w)
        if budget <= 0:
            return
        inbound = {j for j in w.tr.compute if w.reserved[j] > 0}
        moves = []
        for j in w.tr.compute:
            for r in w.nodes[j]:
                if r.migrating or r.n_mig >= MAX_MIG_PER_REQ or r.remaining < 200:
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


def make_policy(spec: dict):
    kind = spec["policy"]
    kw = {k: v for k, v in spec.items() if k != "policy"}
    table = {"greedy": Greedy, "ephemeris": Ephemeris, "reactive": Reactive, "contact_aware": ContactAware}
    if kind not in table:
        raise ValueError(f"unknown cosim policy {kind!r}; one of {sorted(table)}")
    return table[kind](**kw)


__all__ = ["make_policy", "predicted_cost", "predict_decode_end", "candidates", "math"]
