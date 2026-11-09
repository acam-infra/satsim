"""Deadline-aware migration for long requests on thermally constrained nodes."""
from __future__ import annotations

import math

from satsim.cosim.policies import (
    MAX_INFLIGHT, inflight, make_policy, path_lifetime, thermal_service_clock, transfer_estimate,
)
from satsim.method.engine.timing import iter_time


class ThermalRebalance:
    def __init__(self, placement: dict, externality: bool = True, interval_s: float = 30.0,
                 deadline_aware: bool = True, thermal_aware: bool | str = True):
        self.placement = make_policy(placement)
        self.externality = externality
        self.deadline_aware = deadline_aware
        self.thermal_aware = thermal_aware
        self.rc_forecast = thermal_aware == "rc"
        self.interval_s = interval_s
        self.next_plan = interval_s

    def place(self, w, r):
        return self.placement.place(w, r)

    @staticmethod
    def _remaining(w, r):
        lo, hi = w.cfg["workload"]["long_out" if r.long else "short_out"]
        # Condition the workload prior on this request still being alive at r.gen.
        return max(1.0, (max(lo, r.gen) + hi) / 2 - r.gen)

    def _batch_eta(self, w, node, jobs, start):
        active = sorted(((self._remaining(w, r), r.i, r) for r in jobs))
        previous = 0.0
        result = {}
        forecast = ThermalForecast(w, node) if self.rc_forecast and w.thermal is not None else None
        for i, (target, _, r) in enumerate(active):
            step = target - previous
            rest = active[i:]
            batch = len(rest)
            if step > 0:
                ctx = sum(x.kv + (target + previous) / 2 for _, _, x in rest)
                if forecast is not None:
                    start = forecast.finish(start, batch, ctx, step)
                else:
                    lit_clock = thermal_service_clock(w, node, batch, step, ctx) if self.thermal_aware else 1.0
                    for a, b, dark in w.tr.power_segments(node, start):
                        clock = min(lit_clock, w.f_dark) if dark else lit_clock
                        it = iter_time(batch, ctx, clock, w.gpu)
                        if step * it <= b - a:
                            start = a + step * it
                            break
                        step -= (b - a) / it
            result[r] = w.tr.next_conn(r.origin, node, start)
            previous = target
        return result

    @staticmethod
    def _value(r, finish):
        # A soft deadline avoids a discontinuous score under length forecast error.
        spread = min(60.0, (r.deadline - r.t) / 6)
        return min(1.0, max(0.0, 0.5 + (r.deadline - finish) / (2 * spread)))

    def rebalance(self, w):
        now = w.env.now
        if now < self.next_plan:
            return
        self.next_plan = now + self.interval_s
        budget = MAX_INFLIGHT - inflight(w)
        if budget <= 0:
            return
        ready = [j for j in w.tr.compute if w.reserved[j] == 0]
        ready.sort(key=lambda j: (w.iter_of(j), w.kv_used(j), j))
        before = {j: self._batch_eta(w, j, w.nodes[j], now) for j in w.tr.compute}
        moves = []
        for src, batch in w.nodes.items():
            for r in batch:
                if not r.long or r.migrating or r.n_mig or now >= r.deadline:
                    continue
                stay = before[src][r]
                if self.deadline_aware and stay < r.deadline - 60:
                    continue
                relief = 0.0
                if self.externality:
                    without = self._batch_eta(w, src, [x for x in batch if x is not r], now)
                    relief = sum(self._value(x, t) - self._value(x, before[src][x]) for x, t in without.items())
                feasible = 0
                for dst in ready:
                    if src == dst or not w.fits(dst, r):
                        continue
                    path = w.tr.path(src, dst, now)
                    if path is None:
                        continue
                    delay = transfer_estimate(w, path, r.kv * w.bpt)
                    if path_lifetime(w, path) < now + 1.25 * delay or delay >= w.mig_timeout:
                        continue
                    feasible += 1
                    after = self._batch_eta(w, dst, [*w.nodes[dst], r], now + delay)
                    mover = after.pop(r)
                    gain = (self._value(r, mover) - self._value(r, stay) if self.deadline_aware
                            else (stay - mover) / 120.0)
                    if self.externality:
                        gain += relief + sum(self._value(x, t) - self._value(x, before[dst][x]) for x, t in after.items())
                    if gain > 0.10:
                        moves.append((gain, -delay, r.i, src, dst, r))
                    if feasible == 8:
                        break
        used = set()
        for _, _, _, src, dst, r in sorted(moves, key=lambda x: x[:-1], reverse=True):
            if src in used or dst in used:
                continue
            if w.migrate(r, dst):
                used.update((src, dst))
                budget -= 1
                if budget == 0:
                    break


class ThermalForecast:
    """Project the existing RC model until each predicted batch departure or thermal transition."""

    def __init__(self, world, node):
        self.w = world
        self.node = node
        self.model = world.thermal
        self.temp = self.model.temperature(node)
        self.throttled = world.thermal_f[node] < 1

    def finish(self, start, batch, context, tokens):
        w, model = self.w, self.model
        for now, end, dark in w.tr.power_segments(self.node, start):
            orbit_clock = w.f_dark if dark else 1.0
            while now < end:
                clock = min(orbit_clock, max(model.f_min, model.f_throttle)) if self.throttled else orbit_clock
                power = w.device_power(batch, clock)
                equilibrium = model.T_amb_C + model.R_th * power
                boundary = model.T_release_C if self.throttled else model.T_throttle_C
                can_cross = equilibrium < boundary if self.throttled else equilibrium > boundary
                until_cross = math.inf
                if can_cross:
                    ratio = (self.temp - equilibrium) / (boundary - equilibrium)
                    until_cross = model.tau_s * math.log(max(1.0, ratio))
                iteration = iter_time(batch, context, clock, w.gpu)
                service = tokens * iteration
                dt = min(service, end - now, until_cross)
                if dt == until_cross:
                    self.temp = boundary
                    self.throttled = not self.throttled
                else:
                    self.temp = equilibrium + (self.temp - equilibrium) * math.exp(-dt / model.tau_s)
                now += dt
                if dt == service:
                    return now
                tokens -= dt / iteration
        return math.inf
