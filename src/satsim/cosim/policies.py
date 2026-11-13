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
    id = "round_robin_all"

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
    id = "least_reserved_kv"

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
    id = "nearest_upload"

    def place(self, w, r):
        best = min(upload_options(w, r), key=lambda x: (x[1], x[0]), default=None)
        return best[0] if best else None


def _prior_remaining_s(w, r) -> float:
    """Prior on a resident's remaining output tokens (mean of the length band, net of progress)."""
    lo, hi = w.cfg["workload"]["long_out" if r.long else "short_out"]
    return max(1.0, (max(float(lo), r.gen) + float(hi)) / 2.0 - r.gen)


def thermal_service_clock(w, j, batch, tokens, context):
    """Estimate the lit decode clock over a batch's service interval from its heat budget."""
    thermal = w.thermal
    if thermal is None:
        pm = getattr(w, "power_model", None)
        if getattr(pm, "mode", "") == "discrete_tiers":
            from satsim.cosim.power_tiers import select_tier_for_batch

            if tokens <= 0 or batch <= 0:
                return pm.operating_point(j, batch, is_dark=False).clock_f
            cur_op = pm.operating_point(j, batch, is_dark=False)
            high_tier = select_tier_for_batch(pm.spec, max(batch, 1), is_eclipse=False)
            if cur_op.clock_f < high_tier.clock_f - 1e-9:
                return cur_op.clock_f
            eq_temp = pm.t_amb_c + pm.r_th * high_tier.power_w
            if eq_temp <= pm.spec.t_max_c:
                return high_tier.clock_f
            temp = min(pm.temperature(j), pm.spec.t_max_c)
            time_to_cap = pm.tau_s * math.log((eq_temp - temp) / (eq_temp - pm.spec.t_max_c))
            full_speed_time = tokens * iter_time(max(batch, 1), context, high_tier.clock_f, w.gpu)
            if full_speed_time <= time_to_cap:
                return high_tier.clock_f
            low_tiers = [
                t for t in pm.spec.tiers[1:] if pm.t_amb_c + pm.r_th * t.power_w <= pm.spec.t_max_c + 1e-6
            ]
            sustain_f = max((t.clock_f for t in low_tiers), default=pm.spec.tiers[1].clock_f)
            slowed = (full_speed_time - time_to_cap) * (high_tier.clock_f / max(1e-6, sustain_f))
            return high_tier.clock_f * full_speed_time / (time_to_cap + slowed)
        return w.thermal_f[j]
    if w.thermal_f[j] < 1:
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
    """One-shot placement with upload, an output-length prior, and a load/heat forecast.

    ``thermal_mode`` selects how much of the heat model is used. It only changes the clock the
    *candidate* is priced at; the orbit/eclipse segments, upload ETA, prior length and inbound
    ``reserved`` KV are identical in every mode, so the modes form a single-variable ladder from
    "star-blind" to "same RC integration as the simulator":

    | mode         | id                       | sunlit clock                                                  |
    |--------------|--------------------------|---------------------------------------------------------------|
    | ``"off"``    | ``orbit_load``           | ``1.0`` always                                                |
    | ``"hot_only"`` | ``orbit_hot_only``     | current ``thermal_f`` if already throttled, else ``1.0``       |
    | ``"dvfs"``   | ``orbit_thermal``        | ``thermal_service_clock`` (closed-form DVFS average, default)  |
    | ``"rc_const"`` | ``orbit_thermal_rc_const`` | RC projection, arrival batch held constant for the whole request |
    | ``"rc_staged"`` | ``orbit_thermal_rc``  | RC projection, residents drain in a staircase, batch shrinks   |

    ``thermal_aware`` is kept for callers that still pass a bool (``True`` -> ``"dvfs"``,
    ``False`` -> ``"off"``). ``thermal_mode`` wins when both are given, and unknown modes raise
    instead of silently falling through to a default.
    """

    id = "orbit_load"
    _IDS = {
        "off": "orbit_load",
        "hot_only": "orbit_hot_only",
        "dvfs": "orbit_thermal",
        "rc_const": "orbit_thermal_rc_const",
        "rc_staged": "orbit_thermal_rc",
    }
    _ALIASES = {"rc": "rc_staged", True: "dvfs", False: "off"}

    def __init__(self, thermal_mode: str | None = None, thermal_aware: bool | str = True):
        if thermal_mode is None:
            thermal_mode = thermal_aware
        mode = self._ALIASES.get(thermal_mode, thermal_mode)
        if mode not in self._IDS:
            raise ValueError(f"unknown thermal mode {thermal_mode!r}; one of {sorted(self._IDS)}")
        self.thermal_mode = mode
        self.id = self._IDS[mode]

    @property
    def thermal_aware(self):
        """Back-compat view of ``thermal_mode``: True for dvfs, False for off, else the mode name."""
        return {"dvfs": True, "off": False}.get(self.thermal_mode, self.thermal_mode)

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
        is_disc = getattr(getattr(w, "power_model", None), "mode", "") == "discrete_tiers"
        if self.thermal_mode in ("rc_const", "rc_staged") and (w.thermal is not None or is_disc):
            return self._delivery_rc(w, r, tokens, j, upload_s, batch, context, start)
        if self.thermal_mode == "off":
            clock = 1.0
        elif self.thermal_mode == "hot_only":
            clock = w.thermal_f[j] if (w.thermal is not None or is_disc) else 1.0
        else:
            clock = thermal_service_clock(w, j, batch, tokens, context)
        for a, b, dark in w.tr.power_segments(j, start):
            f = min(clock, w.f_dark) if dark else clock
            step = iter_time(batch, context, f, w.gpu)
            if tokens * step <= b - a:
                return w.tr.next_conn(r.origin, j, a + tokens * step)
            tokens -= (b - a) / step
        return INF

    def _delivery_rc(self, w, r, tokens, j, upload_s, batch, context, start):
        """RC projection shared by ``rc_const`` and ``rc_staged``.

        Both advance the forecast across the upload window (the request is not decoding yet, but
        the node keeps heating from its residents). They differ in what happens next: ``rc_const``
        holds the arrival batch for the whole request, ``rc_staged`` drains residents.
        """
        from satsim.cosim.thermal_rebalance import ThermalForecast

        now = w.env.now
        forecast = ThermalForecast(w, j)
        # The node heats over the upload window from whatever is decoding on it right now. An idle
        # node sits at P_idle (device_power(0) == P_idle), so pass the true count, not max(..., 1).
        forecast.advance(now, upload_s, len(w.nodes[j]) + int(w.reserved[j] > 0))
        if self.thermal_mode == "rc_const":
            end = forecast.finish(start, batch, context, tokens)
            return INF if end == math.inf else w.tr.next_conn(r.origin, j, end)

        # Residents plus any inbound upload still holding a slot; both drain, the reserved one at
        # its own (unknown) rate, so give it the same prior treatment.
        residents = [(_prior_remaining_s(w, x), x.kv) for x in w.nodes[j]]
        if w.reserved[j] > 0:
            # ``kv`` below is token-denominated everywhere else, so convert the byte reservation.
            reserved_tok = w.reserved[j] / w.bpt
            residents.append((reserved_tok, reserved_tok))
        # Residents keep decoding while the new request uploads: discount that progress so the
        # staircase starts from what is actually left on arrival.
        if residents:
            clock0 = w.thermal_f[j] if w.thermal is not None else 1.0
            if w.tr.is_dark(j, now):
                clock0 = min(clock0, w.f_dark)
            iter0 = iter_time(max(len(residents), 1), sum(kv for _, kv in residents) + r.pin,
                              clock0, w.gpu)
            done = upload_s / iter0
            residents = [(max(1.0, rt - done), kv) for rt, kv in residents]
        kv_left = sorted(residents)
        t = start
        left = tokens
        while left > 0:
            batch = 1 + len(kv_left)
            ctx = sum(kv for _, kv in kv_left) + r.pin + left / 2
            step_tok = min(left, kv_left[0][0]) if kv_left else left
            end = forecast.finish(t, batch, ctx, step_tok)
            if end == math.inf:
                return INF
            left -= step_tok
            kv_left = [(rt - step_tok, kv) for rt, kv in kv_left if rt - step_tok > 1e-9]
            t = end
        return w.tr.next_conn(r.origin, j, t)


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

    ``interval_s``, when supplied by ``rebalance_invariants``, is the single cadence gate and
    supersedes the legacy ``every`` tick gate so all migration strategies replan on the same clock.
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
        if self.interval_s is None and self.every and self._tick % self.every:
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



class CoolAware:
    """Stage 2 cool-aware migrator: moves decoding requests hurt by eclipse or heat to cool lit nodes."""

    id = "cool_aware"

    def __init__(
        self,
        lead_s: float = 60.0,
        cool_margin_C: float = 10.0,
        min_gain_s: float = 30.0,
        max_inflight: int | None = None,
        max_mig_per_req: int | None = None,
        interval_s: float | None = None,
    ):
        self.lead_s = float(lead_s)
        self.cool_margin_C = float(cool_margin_C)
        self.min_gain_s = float(min_gain_s)
        self.max_inflight = MAX_INFLIGHT if max_inflight is None else int(max_inflight)
        self.max_mig_per_req = MAX_MIG_PER_REQ if max_mig_per_req is None else int(max_mig_per_req)
        self.interval_s = float(interval_s) if interval_s is not None else None
        self.next_plan = float(interval_s) if interval_s is not None else 0.0

    def place(self, w, r):
        return None

    def _enters_eclipse_soon(self, w, j: int, now: float) -> bool:
        for a, b, dark in w.tr.power_segments(j, now):
            if a >= now + self.lead_s:
                break
            if dark:
                return True
        return False

    @staticmethod
    def _will_throttle(w, j: int) -> bool:
        """True when resident batch heat load will push node j into thermal throttling before decode ends."""
        is_disc = getattr(getattr(w, "power_model", None), "mode", "") == "discrete_tiers"
        if (getattr(w, "thermal", None) is None and not is_disc) or not w.nodes[j]:
            return False
        rem = max((r.remaining for r in w.nodes[j]), default=0.0)
        if rem <= 0.0:
            return False
        ctx = sum(r.kv for r in w.nodes[j])
        return thermal_service_clock(w, j, len(w.nodes[j]), rem, ctx) < 1.0 - 1e-6

    def _decoded_on_src(self, w, src: int, now: float, dt: float) -> float:
        """Tokens decoded at the source while KV streams across ISLs."""
        if dt <= 0.0:
            return 0.0
        f_lit = w.thermal_f[src] if getattr(w, "thermal_f", None) is not None else 1.0
        f_dark = min(w.f_dark, f_lit)
        it_lit = w.iter_of(src, None, f=f_lit)
        it_dark = w.iter_of(src, None, f=f_dark)
        toks = 0.0
        t_end = now + dt
        for a, b, dark in w.tr.power_segments(src, now):
            if a >= t_end:
                break
            seg = min(b, t_end) - a
            it = it_dark if dark else it_lit
            if it > 0.0:
                toks += seg / it
        return toks

    def _est_finish(
        self,
        w,
        node: int,
        r,
        now: float,
        t_transfer: float,
        joining: bool,
        remaining: float | None = None,
    ) -> float:
        t_start = now + t_transfer
        if t_start >= w.tr.horizon:
            return INF

        left = r.remaining if remaining is None else remaining
        if left <= 1e-9:
            return_time = w.tr.next_conn(r.origin, node, t_start)
            return INF if return_time == INF else t_transfer + (return_time - t_start)

        extra = r if joining else None
        batch = max(len(w.nodes[node]) + (1 if joining else 0), 1)
        ctx = sum(x.kv for x in w.nodes[node]) + (r.kv if joining else 0.0)
        is_disc = getattr(getattr(w, "power_model", None), "mode", "") == "discrete_tiers"
        if getattr(w, "thermal", None) is not None or is_disc:
            f_lit = thermal_service_clock(w, node, batch, left, ctx)
        else:
            f_lit = w.thermal_f[node] if getattr(w, "thermal_f", None) is not None else 1.0
        f_dark = min(w.f_dark, f_lit)
        it_lit = w.iter_of(node, extra, f=f_lit)
        it_dark = w.iter_of(node, extra, f=f_dark)

        decode_end = INF
        for a, b, dark in w.tr.power_segments(node, t_start):
            it = it_dark if dark else it_lit
            span = b - a
            if left * it <= span:
                decode_end = a + left * it
                break
            left -= span / it

        if decode_end == INF or decode_end >= w.tr.horizon:
            return INF

        return_time = w.tr.next_conn(r.origin, node, decode_end)
        if return_time == INF:
            return INF

        t_decode_remaining = decode_end - t_start
        t_return = return_time - decode_end
        return t_transfer + t_decode_remaining + t_return

    def rebalance(self, w) -> None:
        now = w.env.now
        if self.interval_s is not None:
            if now < self.next_plan:
                return
            self.next_plan = now + self.interval_s

        budget = self.max_inflight - inflight(w)
        if budget <= 0:
            return

        candidate_reqs = []
        for j in w.tr.compute:
            if not (
                w.is_eclipse(j, now)
                or self._enters_eclipse_soon(w, j, now)
                or w.is_hot(j)
                or self._will_throttle(w, j)
            ):
                continue
            for r in w.nodes[j]:
                if r.state != "decode" or r.migrating or r.n_mig >= self.max_mig_per_req or r.remaining <= 0:
                    continue
                if now >= r.deadline:
                    continue
                candidate_reqs.append(r)

        if not candidate_reqs:
            return

        edge_users: dict[int, int] = {}
        claimed_dst: set[int] = set()
        for f in w.net.flows:
            for _, _, e in f.hops:
                edge_users[e] = edge_users.get(e, 0) + 1
            if f.tag in ("mig", "mig_delta") and f.hops:
                claimed_dst.add(f.hops[-1][1])

        proposals = []
        for r in candidate_reqs:
            src = r.node
            est_finish_src = self._est_finish(w, src, r, now, t_transfer=0.0, joining=False)
            slack = r.deadline - now
            at_risk = est_finish_src > slack - 30.0

            best_dst = None
            best_p = None
            best_gain = self.min_gain_s - 1e-9

            for k in w.tr.compute:
                if k == src or k in claimed_dst:
                    continue
                if w.is_eclipse(k, now) or not w.is_cool(k, self.cool_margin_C) or not w.fits(k, r):
                    continue
                if not at_risk and len(w.nodes[k]) > 0:
                    continue

                p = w.tr.path(src, k, now)
                if p is None or not p[1]:
                    continue
                bw = min(w.net.cap[e] / (edge_users.get(e, 0) + 1.0) for e in p[1])
                if bw <= 0:
                    continue
                t_net = (r.kv * w.bpt) / bw
                if getattr(w, "delta_kv", False):
                    it_src = w.iter_of(src)
                    ratio = (w.bpt / it_src) / bw if it_src > 0 else 0.0
                    if ratio >= 0.9:
                        continue
                    t_net *= 1.0 + ratio + ratio * ratio
                t_transfer = t_net + w.hop_lat * len(p[1]) + w.switch_s
                if t_transfer >= 0.85 * w.mig_timeout or path_lifetime(w, p) < now + 1.15 * t_transfer:
                    continue

                left = r.remaining - self._decoded_on_src(w, src, now, t_transfer)
                if left <= 0.0:
                    continue

                est_finish_k = self._est_finish(w, k, r, now, t_transfer=t_transfer, joining=True, remaining=left)
                if est_finish_k == INF or est_finish_k > slack:
                    continue

                gain = est_finish_src - est_finish_k
                if gain >= self.min_gain_s and gain > best_gain:
                    best_gain = gain
                    best_dst = k
                    best_p = p

            if best_dst is not None:
                proposals.append(((0 if at_risk else 1, -round(best_gain, 6), slack, r.i), r, best_dst, best_p))

        proposals.sort(key=lambda x: x[0])
        for _, r, best_dst, best_p in proposals:
            if budget <= 0:
                break
            if best_dst in claimed_dst:
                continue
            if path_lifetime(w, best_p) < now + 2.0:
                continue
            if w.migrate(r, best_dst):
                claimed_dst.add(best_dst)
                budget -= 1


class LowTierPack:
    """Pack slack-safe requests onto already-awake low-tier nodes before waking 0W nodes."""

    id = "low_tier_pack"

    def __init__(self, fallback: dict | None = None, tier_slack_margin_s: float = 10.0):
        self.fallback = make_policy(fallback) if fallback is not None else OrbitLoad(thermal_mode="rc_staged")
        self.tier_slack_margin_s = float(tier_slack_margin_s)

    def place(self, w, r):
        from satsim.cosim.discrete_milp import DiscreteThermalMigrator

        helper = getattr(self, "_helper", None)
        if helper is None:
            helper = DiscreteThermalMigrator(
                explicit_tier=True,
                on_arrival_tier=True,
                low_tier_pack=True,
                tier_slack_margin_s=self.tier_slack_margin_s,
            )
            self._helper = helper
        packed = helper._low_tier_pack_place(w, r)
        if packed is not None:
            return packed
        return self.fallback.place(w, r)


class CompositePolicy:
    """Orthogonal composition of any Placer (place) and any Migrator (rebalance)."""

    def __init__(self, placer, migrator=None):
        self.placer = placer
        self.migrator = migrator
        self.id = f"{placer.id}+{migrator.id}" if migrator is not None else placer.id

    def place(self, w, r):
        return self.placer.place(w, r)

    def on_arrival(self, w, node: int, r) -> None:
        if hasattr(self.placer, "on_arrival"):
            self.placer.on_arrival(w, node, r)
        if self.migrator is not None and hasattr(self.migrator, "on_arrival"):
            self.migrator.on_arrival(w, node, r)

    def on_departure(self, w, node: int, r) -> None:
        if hasattr(self.placer, "on_departure"):
            self.placer.on_departure(w, node, r)
        if self.migrator is not None and hasattr(self.migrator, "on_departure"):
            self.migrator.on_departure(w, node, r)

    def rebalance(self, w) -> None:
        if self.migrator is not None:
            self.migrator.rebalance(w)


def _make_thermal_rebalance(**kw):
    from satsim.cosim.thermal_rebalance import ThermalRebalance

    return ThermalRebalance(**kw)


def _make_torch_rebalance(**kw):
    from satsim.cosim.thermal_rebalance import TorchRebalance

    return TorchRebalance(**kw)


def _make_discrete_rh_greedy(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v2(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", True)
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v3(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", True)
    kw.setdefault("tier_slack_ratio", 0.25)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v4(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", False)
    kw.setdefault("tier_slack_ratio", 0.45)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    kw.setdefault("eclipse_lookahead_s", 30.0)
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v5(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", False)
    kw.setdefault("low_tier_pack", False)
    kw.setdefault("tier_slack_ratio", 0.35)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    kw.setdefault("eclipse_lookahead_s", 30.0)
    kw.setdefault("thermal_precool_horizon_s", 30.0)
    kw.setdefault("urgent_isolation", True)
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v6(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", False)
    kw.setdefault("tier_slack_ratio", 0.45)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    kw.setdefault("eclipse_lookahead_s", 30.0)
    kw.setdefault("thermal_precool_horizon_s", 30.0)
    kw.setdefault("soft_thermal_gate", True)
    kw.setdefault("urgent_isolation", True)
    kw.setdefault("strict_urgent_isolation", True)
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v7(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", False)
    kw.setdefault("tier_slack_ratio", 0.35)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    kw.setdefault("eclipse_lookahead_s", 30.0)
    kw.setdefault("thermal_precool_horizon_s", 30.0)
    kw.setdefault("soft_thermal_gate", True)
    kw.setdefault("urgent_isolation", True)
    kw.setdefault("strict_urgent_isolation", True)
    kw.setdefault("delivery_aware_tier", True)
    kw.setdefault("prefill_burst", True)
    kw.setdefault("id_override", "discrete_rh_tier_v7")
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v8(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", False)
    kw.setdefault("tier_slack_ratio", 0.35)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    kw.setdefault("eclipse_lookahead_s", 30.0)
    kw.setdefault("thermal_precool_horizon_s", 30.0)
    kw.setdefault("soft_thermal_gate", True)
    kw.setdefault("urgent_isolation", True)
    kw.setdefault("strict_urgent_isolation", True)
    kw.setdefault("delivery_aware_tier", True)
    kw.setdefault("prefill_burst", True)
    kw.setdefault("recompute_aware", True)
    kw.setdefault("id_override", "discrete_rh_tier_v8")
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v9(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", False)
    kw.setdefault("tier_slack_ratio", 0.45)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    kw.setdefault("eclipse_lookahead_s", 30.0)
    kw.setdefault("thermal_precool_horizon_s", 30.0)
    kw.setdefault("soft_thermal_gate", True)
    kw.setdefault("urgent_isolation", True)
    kw.setdefault("strict_urgent_isolation", True)
    kw.setdefault("delivery_aware_tier", True)
    kw.setdefault("prefill_burst", True)
    kw.setdefault("recompute_aware", True)
    kw.setdefault("prefill_dwell_place", True)
    kw.setdefault("dwell_aware_tier", True)
    kw.setdefault("prefill_aware_mig", True)
    kw.setdefault("id_override", "discrete_rh_tier_v9")
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v10(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", False)
    kw.setdefault("tier_slack_ratio", 0.45)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    kw.setdefault("eclipse_lookahead_s", 30.0)
    kw.setdefault("thermal_precool_horizon_s", 30.0)
    kw.setdefault("soft_thermal_gate", True)
    kw.setdefault("urgent_isolation", True)
    kw.setdefault("strict_urgent_isolation", True)
    kw.setdefault("delivery_aware_tier", True)
    kw.setdefault("prefill_burst", True)
    kw.setdefault("recompute_aware", True)
    kw.setdefault("prefill_dwell_place", True)
    kw.setdefault("dwell_aware_tier", True)
    kw.setdefault("selective_prefill_rescue", True)
    kw.setdefault("prefill_traj_aware", True)
    kw.setdefault("safe_urg_mix_order", False)
    kw.setdefault("id_override", "discrete_rh_tier_v10")
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v11(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", False)
    kw.setdefault("tier_slack_ratio", 0.45)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    kw.setdefault("eclipse_lookahead_s", 30.0)
    kw.setdefault("thermal_precool_horizon_s", 30.0)
    kw.setdefault("soft_thermal_gate", True)
    kw.setdefault("urgent_isolation", True)
    kw.setdefault("strict_urgent_isolation", True)
    kw.setdefault("delivery_aware_tier", True)
    kw.setdefault("prefill_burst", True)
    kw.setdefault("recompute_aware", True)
    kw.setdefault("prefill_dwell_place", True)
    kw.setdefault("dwell_aware_tier", True)
    kw.setdefault("selective_prefill_rescue", True)
    kw.setdefault("prefill_traj_aware", True)
    kw.setdefault("safe_urg_mix_order", False)
    kw.setdefault("dwell_thermal_gate", True)
    kw.setdefault("isolate_lo_evict", True)
    kw.setdefault("id_override", "discrete_rh_tier_v11")
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v12(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", False)
    kw.setdefault("tier_slack_ratio", 0.45)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    kw.setdefault("eclipse_lookahead_s", 30.0)
    kw.setdefault("thermal_precool_horizon_s", 30.0)
    kw.setdefault("soft_thermal_gate", True)
    kw.setdefault("urgent_isolation", True)
    kw.setdefault("strict_urgent_isolation", True)
    kw.setdefault("delivery_aware_tier", True)
    kw.setdefault("prefill_burst", True)
    kw.setdefault("recompute_aware", True)
    kw.setdefault("prefill_dwell_place", True)
    kw.setdefault("dwell_aware_tier", True)
    kw.setdefault("selective_prefill_rescue", True)
    kw.setdefault("prefill_traj_aware", True)
    kw.setdefault("safe_urg_mix_order", False)
    kw.setdefault("dwell_thermal_gate", True)
    kw.setdefault("isolate_lo_evict", True)
    kw.setdefault("strict_evict_urg", True)
    kw.setdefault("decode_only_low_tier", True)
    kw.setdefault("id_override", "discrete_rh_tier_v12")
    return DiscreteThermalMigrator(**kw)


def _make_discrete_rh_tier_v13(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, IncrementalGreedySolver

    kw.setdefault("solver", IncrementalGreedySolver())
    kw.setdefault("explicit_tier", True)
    kw.setdefault("on_arrival_tier", True)
    kw.setdefault("low_tier_pack", False)
    kw.setdefault("tier_slack_ratio", 0.45)
    kw.setdefault("suppress_low_tier_self_mig", True)
    kw.setdefault("urgent_tier_guard", True)
    kw.setdefault("eclipse_lookahead_s", 30.0)
    kw.setdefault("thermal_precool_horizon_s", 30.0)
    kw.setdefault("soft_thermal_gate", True)
    kw.setdefault("urgent_isolation", True)
    kw.setdefault("strict_urgent_isolation", True)
    kw.setdefault("delivery_aware_tier", True)
    kw.setdefault("prefill_burst", True)
    kw.setdefault("recompute_aware", True)
    kw.setdefault("prefill_dwell_place", True)
    kw.setdefault("dwell_aware_tier", True)
    kw.setdefault("selective_prefill_rescue", True)
    kw.setdefault("prefill_traj_aware", True)
    kw.setdefault("safe_urg_mix_order", False)
    kw.setdefault("dwell_thermal_gate", True)
    kw.setdefault("isolate_lo_evict", True)
    kw.setdefault("strict_evict_urg", True)
    kw.setdefault("decode_only_low_tier", True)
    kw.setdefault("wakeup_aware_mig", True)
    kw.setdefault("hopeless_mig_guard", True)
    kw.setdefault("id_override", "discrete_rh_tier_v13")
    return DiscreteThermalMigrator(**kw)


def _make_discrete_step_milp(**kw):
    from satsim.cosim.discrete_milp import DiscreteThermalMigrator, StepMilpSolver

    kw.setdefault("solver", StepMilpSolver())
    return DiscreteThermalMigrator(**kw)


def _make_llumnix_style_mig_only(**kw):
    from satsim.cosim.topconf_baselines import LlumnixStyleMigOnly

    return LlumnixStyleMigOnly(**kw)


def _make_tapas_reactive_thermal(**kw):
    from satsim.cosim.topconf_baselines import TapasReactiveThermal

    return TapasReactiveThermal(**kw)


def _make_mcs_slack_dvfs_no_mig(**kw):
    from satsim.cosim.topconf_baselines import McsSlackDvfsNoMig

    return McsSlackDvfsNoMig(**kw)


PLACERS = {
    "rr": RoundRobinPlacer,
    "round_robin": RoundRobinPlacer,
    "round_robin_all": ReachableRoundRobin,
    "round_robin_lit": RoundRobin,
    "nearest": Nearest,
    "nearest_upload": NearestUpload,
    "random": RandomPlace,
    "least_loaded": LeastLoaded,
    "least_reserved_kv": LeastReservedKV,
    "jsq": LeastLoaded,
    "greedy": Greedy,
    "ephemeris": Ephemeris,
    "orbit_load": OrbitLoad,
    "orbit_thermal": OrbitLoad,
    "low_tier_pack": LowTierPack,
    "llumnix_style_mig_only": _make_llumnix_style_mig_only,
    "tapas_reactive_thermal": _make_tapas_reactive_thermal,
    "mcs_slack_dvfs_no_mig": _make_mcs_slack_dvfs_no_mig,
}

MIGRATORS = {
    "none": None,
    "reactive": Reactive,
    "contact_aware": ContactAware,
    "cool_aware": CoolAware,
    "thermal_rebalance": _make_thermal_rebalance,
    "torch_rebalance": _make_torch_rebalance,
    "torch_lp": _make_torch_rebalance,
    "discrete_rh_greedy": _make_discrete_rh_greedy,
    "discrete_rh_tier": _make_discrete_rh_tier,
    "discrete_rh_tier_v2": _make_discrete_rh_tier_v2,
    "discrete_rh_tier_v3": _make_discrete_rh_tier_v3,
    "discrete_rh_tier_v4": _make_discrete_rh_tier_v4,
    "discrete_rh_tier_v5": _make_discrete_rh_tier_v5,
    "discrete_rh_tier_v6": _make_discrete_rh_tier_v6,
    "discrete_rh_tier_v7": _make_discrete_rh_tier_v7,
    "discrete_rh_tier_v8": _make_discrete_rh_tier_v8,
    "discrete_rh_tier_v9": _make_discrete_rh_tier_v9,
    "discrete_rh_tier_v10": _make_discrete_rh_tier_v10,
    "discrete_rh_tier_v11": _make_discrete_rh_tier_v11,
    "discrete_rh_tier_v12": _make_discrete_rh_tier_v12,
    "discrete_rh_tier_v13": _make_discrete_rh_tier_v13,
    "discrete_step_milp": _make_discrete_step_milp,
    "llumnix_style_mig_only": _make_llumnix_style_mig_only,
    "tapas_reactive_thermal": _make_tapas_reactive_thermal,
    "mcs_slack_dvfs_no_mig": _make_mcs_slack_dvfs_no_mig,
}

POLICIES = {
    **PLACERS,
    **{k: v for k, v in MIGRATORS.items() if v is not None},
    # Legacy standalone policy aliases preserved for backward compatibility
    "round_robin": RoundRobin,
    "random": lambda **_kw: RoundRobin(mode="random"),
    "least_loaded": lambda **_kw: RoundRobin(mode="least_loaded"),
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
    if kind not in POLICIES:
        raise ValueError(f"unknown cosim policy {kind!r}; one of {sorted(POLICIES)}")
    return POLICIES[kind](**kw)
