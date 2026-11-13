"""Pluggable PowerThermalModel adapters for CoWorld and planners (LAB-1874 / LAB-1881).

Implements the ``satsim.cosim.interfaces.PowerThermalModel`` seam with three adapters:
- ``OrbitOnlyPowerModel``: no RC thermal model; sunlit 1.0x vs eclipse ``f_dark``.
- ``ContinuousDVFSModel``: legacy continuous ``device_power`` + ``DynamicThermal`` 84C/76C hysteresis.
- ``DiscreteTierModel``: strict ``0.0 W`` idle tier + discrete working tiers + linear RC cap.
"""
from __future__ import annotations

import math
from typing import Callable

from satsim.cosim.interfaces import OperatingPoint, PowerThermalModel
from satsim.cosim.power_tiers import (
    ACC_ENERGY_J,
    ACC_TIER_TIME_S,
    ACC_TIER_UNSAT_S,
    ACC_WAKEUP_COUNT,
    ACC_WAKEUP_ENERGY_J,
    ACC_ZERO_POWER_S,
    DiscretePowerSpec,
    effective_clock_f,
    is_discrete_power,
    parse_discrete_power_spec,
    rc_linear_coeffs,
    select_tier_for_batch,
)
from satsim.method.engine.timing import clock_factor, iter_time
from satsim.method.policies.thermal.dynamic import DynamicThermal


def _iter_batches(batches: dict[int, int] | list[int]):
    if isinstance(batches, dict):
        return batches.items()
    return enumerate(batches)


class OrbitOnlyPowerModel:
    """Power/clock adapter when no ``thermal`` block is configured."""

    mode = "orbit_only"

    def __init__(self, cfg: dict, n: int) -> None:
        self.n = int(n)
        power = cfg["power"]
        self.p_idle = float(power.get("P_idle", 0.0))
        self.p_max = float(power.get("P_max", self.p_idle))
        self.gamma = float(power.get("gamma", 2.4))
        self.f_dark = clock_factor(float(power["P_dark"]), power)
        self.t_amb_c = float((cfg.get("thermal") or {}).get("T_amb_C", 20.0))

    def operating_point(
        self,
        node: int,
        batch: int,
        *,
        is_dark: bool,
        t: float | None = None,
    ) -> OperatingPoint:
        f = self.f_dark if is_dark else 1.0
        if batch <= 0:
            p_w = max(0.0, self.p_idle)
        else:
            p_w = max(0.0, self.p_idle + (self.p_max - self.p_idle) * (f ** self.gamma))
        return OperatingPoint(power_w=p_w, clock_f=f)

    def step(
        self,
        now: float,
        batches: dict[int, int] | list[int],
        is_dark_fn: Callable[[int, float], bool],
    ) -> list[int]:
        return []

    def project_finish(
        self,
        node: int,
        start: float,
        batch: int,
        context: float,
        tokens: float,
        *,
        trace: object,
        gpu: dict,
        initial_temp_c: float | None = None,
    ) -> tuple[float, float]:
        temp = self.t_amb_c if initial_temp_c is None else float(initial_temp_c)
        rem = float(tokens)
        if rem <= 0.0:
            return (float(start), temp)
        eval_batch = max(int(batch), 1)
        for now, end, dark in trace.power_segments(node, start):
            f = self.f_dark if dark else 1.0
            it = iter_time(eval_batch, context, f, gpu)
            service = rem * it
            if now + service <= end:
                return (now + service, temp)
            rem -= (end - now) / it
        return (math.inf, temp)

    def advance_temp(
        self,
        node: int,
        start: float,
        dt: float,
        batch: int,
        *,
        trace: object,
        initial_temp_c: float | None = None,
    ) -> float:
        return self.t_amb_c if initial_temp_c is None else float(initial_temp_c)

    def set_temperature(self, node: int, temp_c: float) -> None:
        return None

    def temperature(self, node: int) -> float:
        return self.t_amb_c

    def headroom_c(self, node: int) -> float:
        return math.inf

    def diagnostics(self) -> dict:
        return {"mode": self.mode}


class ContinuousDVFSModel:
    """Legacy continuous DVFS power + ``DynamicThermal`` 84C/76C hysteresis adapter."""

    mode = "continuous"

    def __init__(self, cfg: dict, n: int) -> None:
        self.n = int(n)
        self._power_cfg = cfg["power"]
        self._thermal_cfg = cfg["thermal"]
        self.thermal = DynamicThermal({**self._power_cfg, **self._thermal_cfg}, n=self.n)
        self.f_dark = clock_factor(float(self._power_cfg["P_dark"]), self._power_cfg)
        self._thermal_f = [self.thermal.clock_factor(j, 0.0) for j in range(self.n)]
        self._power_w = [float(self._power_cfg["P_idle"])] * self.n

    def device_power(self, batch: int, f: float) -> float:
        power = self._power_cfg
        if batch == 0:
            return float(power["P_idle"])
        thermal = self._thermal_cfg
        utilization = 1.0 - (1.0 - thermal["busy_util_floor"]) * math.exp(-batch / thermal["batch_scale"])
        return float(power["P_idle"]) + (float(power["P_max"]) - float(power["P_idle"])) * utilization * (
            f ** float(power["gamma"])
        )

    def operating_point(
        self,
        node: int,
        batch: int,
        *,
        is_dark: bool,
        t: float | None = None,
    ) -> OperatingPoint:
        orbit_f = self.f_dark if is_dark else 1.0
        th_f = self.thermal._f[node]
        f = min(orbit_f, th_f)
        p_w = self.device_power(batch, f)
        return OperatingPoint(power_w=p_w, clock_f=f, throttled=(th_f < 1.0))

    def step(
        self,
        now: float,
        batches: dict[int, int] | list[int],
        is_dark_fn: Callable[[int, float], bool],
    ) -> list[int]:
        self.thermal.advance_with_power(now, self._power_w)
        changed: list[int] = []
        for j, batch in _iter_batches(batches):
            orbit_f = self.f_dark if is_dark_fn(j, now) else 1.0
            previous = min(orbit_f, self._thermal_f[j])
            self._thermal_f[j] = self.thermal.clock_factor(j, now)
            current = min(orbit_f, self._thermal_f[j])
            if current != previous:
                changed.append(j)
            self._power_w[j] = self.device_power(int(batch), current)
        return changed

    def project_finish(
        self,
        node: int,
        start: float,
        batch: int,
        context: float,
        tokens: float,
        *,
        trace: object,
        gpu: dict,
        initial_temp_c: float | None = None,
        initial_throttled: bool | None = None,
    ) -> tuple[float, float]:
        model = self.thermal
        temp = model.temperature(node) if initial_temp_c is None else float(initial_temp_c)
        throttled = (model._f[node] < 1.0) if initial_throttled is None else bool(initial_throttled)
        rem = float(tokens)
        if rem <= 0.0:
            return (float(start), temp)
        eval_batch = max(int(batch), 1)
        for now, end, dark in trace.power_segments(node, start):
            orbit_clock = self.f_dark if dark else 1.0
            while now < end:
                clock = min(orbit_clock, max(model.f_min, model.f_throttle)) if throttled else orbit_clock
                power = self.device_power(batch, clock)
                equilibrium = model.T_amb_C + model.R_th * power
                boundary = model.T_release_C if throttled else model.T_throttle_C
                can_cross = equilibrium < boundary if throttled else equilibrium > boundary
                until_cross = math.inf
                if can_cross:
                    ratio = (temp - equilibrium) / (boundary - equilibrium)
                    until_cross = model.tau_s * math.log(max(1.0, ratio))
                iteration = iter_time(eval_batch, context, clock, gpu)
                service = rem * iteration
                dt = min(service, end - now, until_cross)
                if dt == until_cross:
                    temp = boundary
                    throttled = not throttled
                else:
                    temp = equilibrium + (temp - equilibrium) * math.exp(-dt / model.tau_s)
                now += dt
                if dt == service:
                    return (now, temp)
                rem -= dt / iteration
        return (math.inf, temp)

    def set_temperature(self, node: int, temp_c: float) -> None:
        self.thermal._T[node] = float(temp_c)
        if temp_c > self.thermal.peak_T[node]:
            self.thermal.peak_T[node] = float(temp_c)

    def temperature(self, node: int) -> float:
        return self.thermal.temperature(node)

    def headroom_c(self, node: int) -> float:
        return self.thermal.T_throttle_C - self.thermal.temperature(node)

    def diagnostics(self) -> dict:
        return self.thermal.diagnostics()


class DiscreteTierModel:
    """Discrete power-tier adapter with strict ``0.0 W`` idle, RC cap, and energy accumulators."""

    mode = "discrete_tiers"

    def __init__(self, cfg: dict, n: int) -> None:
        spec = parse_discrete_power_spec(cfg)
        if spec is None:
            raise ValueError("DiscreteTierModel requires power.mode == 'discrete_tiers'")
        self.spec: DiscretePowerSpec = spec
        self.n = int(n)
        th = cfg.get("thermal") or {}
        self.tau_s = float(th.get("tau_s", 240.0))
        self.r_th = float(th.get("R_th", 0.30))
        self.t_amb_c = float(th.get("T_amb_C", 20.0))
        t0_c = float(th.get("T0_C", self.t_amb_c))
        self.control_period_s = float(cfg.get("control_period_s", 2.0))
        self._rc_probe_coeffs = rc_linear_coeffs(
            self.tau_s, self.r_th, self.t_amb_c, self.control_period_s
        )

        self._node_t = [0.0] * self.n
        self._T = [t0_c] * self.n
        self._peak_T = [t0_c] * self.n
        self._tier_idx = [0] * self.n
        self._power_w = [0.0] * self.n
        self._clock_f = [self.spec.tiers[-1].clock_f] * self.n
        self._explicit_tier_idx: list[int | None] = [None] * self.n

        self.zero_power_s = 0.0
        self.tier_time_s: dict[int, float] = {t.index: 0.0 for t in self.spec.tiers}
        self.energy_j = 0.0
        self.tier_unsat_s = 0.0
        self.wakeup_count = 0
        self.wakeup_energy_j_total = 0.0
        self._wakeup_ready_t: list[float] = [0.0] * self.n
        self._thermal_tripped: list[bool] = [False] * self.n
        self.thermal_trip_count = 0
        self.thermal_trip_s = 0.0

    def wakeup_wait_s(self, node: int, now: float) -> float:
        """Remaining cold-start wakeup latency (seconds) on ``node`` at ``now``."""
        return max(0.0, self._wakeup_ready_t[node] - float(now))

    def set_explicit_tier(self, node: int, tier_idx: int | None) -> None:
        self._explicit_tier_idx[node] = tier_idx

    def _is_throttled(self, node: int, tier_index: int, batch: int, is_dark: bool) -> bool:
        if self._thermal_tripped[node] or self._T[node] >= self.spec.t_max_c:
            return True
        if batch > 0 and not is_dark and tier_index < self.spec.tiers[-1].index:
            return True
        return False

    def operating_point(
        self,
        node: int,
        batch: int,
        *,
        is_dark: bool,
        t: float | None = None,
    ) -> OperatingPoint:
        tripped = self._thermal_tripped[node]
        tier = select_tier_for_batch(
            self.spec,
            int(batch),
            is_eclipse=is_dark,
            current_temp_c=self._T[node],
            rc_step_coeffs=self._rc_probe_coeffs,
            explicit_tier_idx=self._explicit_tier_idx[node],
            thermal_trip_lockout=tripped,
        )
        eval_f = effective_clock_f(
            self.spec,
            int(batch),
            is_eclipse=is_dark,
            current_temp_c=self._T[node],
            rc_step_coeffs=self._rc_probe_coeffs,
            explicit_tier_idx=self._explicit_tier_idx[node],
            thermal_trip_lockout=tripped,
        )
        throttled = self._is_throttled(node, tier.index, int(batch), is_dark)
        return OperatingPoint(
            power_w=tier.power_w,
            clock_f=eval_f,
            prefill_clock_f=tier.prefill_clock_f if batch > 0 else None,
            tier_index=tier.index,
            throttled=throttled,
        )

    def _advance_node(self, node: int, now: float, batch_during_interval: int) -> None:
        dt = float(now) - self._node_t[node]
        if dt <= 0.0:
            return
        self._node_t[node] = float(now)
        p_w = self._power_w[node]
        tier_idx = self._tier_idx[node]

        self.energy_j += p_w * dt
        self.tier_time_s[tier_idx] = self.tier_time_s.get(tier_idx, 0.0) + dt
        if tier_idx == 0:
            self.zero_power_s += dt
        if batch_during_interval > self.spec.tiers[tier_idx].max_batch:
            self.tier_unsat_s += dt

        if self._thermal_tripped[node]:
            self.thermal_trip_s += dt
        alpha, offset, gain = rc_linear_coeffs(self.tau_s, self.r_th, self.t_amb_c, dt)
        t_next = alpha * self._T[node] + offset + gain * p_w
        self._T[node] = t_next
        if t_next > self._peak_T[node]:
            self._peak_T[node] = t_next
        hyst = self.spec.thermal_hysteresis_c
        if hyst > 0.0:
            if not self._thermal_tripped[node]:
                # Trip when node hits T_max_C (or next control step at high tier would breach T_max_C).
                alpha_p, offset_p, gain_p = self._rc_probe_coeffs
                max_p = self.spec.tiers[-1].power_w
                would_breach = (
                    batch_during_interval > 0
                    and (alpha_p * t_next + offset_p + gain_p * max_p) > self.spec.t_max_c + 1e-9
                )
                if t_next >= self.spec.t_max_c - 1e-6 or (would_breach and self._explicit_tier_idx[node] is None):
                    self._thermal_tripped[node] = True
                    self.thermal_trip_count += 1
            elif t_next <= self.spec.t_max_c - hyst:
                self._thermal_tripped[node] = False

    def sync_node(
        self,
        node: int,
        now: float,
        new_batch: int,
        is_dark: bool,
        *,
        prev_batch: int | None = None,
    ) -> bool:
        """Advance ``node`` to ``now`` and update its tier/power/clock for ``new_batch``."""
        interval_batch = new_batch if prev_batch is None else prev_batch
        self._advance_node(node, now, int(interval_batch))
        prev_f = self._clock_f[node]
        prev_tier = self._tier_idx[node]
        op = self.operating_point(node, int(new_batch), is_dark=is_dark, t=now)
        new_tier = int(op.tier_index if op.tier_index is not None else 0)
        if prev_tier == 0 and new_tier > 0:
            self.wakeup_count += 1
            if self.spec.wakeup_energy_j > 0.0:
                self.energy_j += self.spec.wakeup_energy_j
                self.wakeup_energy_j_total += self.spec.wakeup_energy_j
            if self.spec.wakeup_latency_s > 0.0:
                self._wakeup_ready_t[node] = float(now) + self.spec.wakeup_latency_s
        self._tier_idx[node] = new_tier
        self._power_w[node] = op.power_w
        self._clock_f[node] = op.clock_f
        return op.clock_f != prev_f

    def step(
        self,
        now: float,
        batches: dict[int, int] | list[int],
        is_dark_fn: Callable[[int, float], bool],
    ) -> list[int]:
        changed: list[int] = []
        t_now = float(now)
        for j, batch in _iter_batches(batches):
            if self.sync_node(j, t_now, int(batch), bool(is_dark_fn(j, t_now)), prev_batch=int(batch)):
                changed.append(j)
        return changed

    def _until_tier_switch(self, temp: float, eq: float, cur_tier_power_w: float, dark: bool) -> float:
        """Exact analytical RC time until ``select_tier_for_batch`` could change tier."""
        alpha_p, offset_p, gain_p = self._rc_probe_coeffs
        if alpha_p <= 0.0:
            return math.inf
        t_max = self.spec.t_max_c
        if eq > temp + 1e-6:
            t_crit = (t_max - offset_p - gain_p * cur_tier_power_w) / alpha_p
            if temp < t_crit < eq - 1e-6:
                return max(self.control_period_s, self.tau_s * math.log((temp - eq) / (t_crit - eq)) + 1e-3)
            return math.inf
        if eq < temp - 1e-6:
            best_cross = math.inf
            for tk in self.spec.tiers[1:]:
                if dark and tk.power_w > self.spec.p_dark_w + 1e-9:
                    continue
                if tk.power_w <= cur_tier_power_w + 1e-9:
                    continue
                t_crit = (t_max - offset_p - gain_p * tk.power_w) / alpha_p
                if eq + 1e-6 < t_crit < temp:
                    dt_c = self.tau_s * math.log((temp - eq) / (t_crit - eq)) + 1e-3
                    if dt_c < best_cross:
                        best_cross = dt_c
            return max(self.control_period_s, best_cross) if best_cross < math.inf else math.inf
        return math.inf

    def advance_temp(
        self,
        node: int,
        start: float,
        dt: float,
        batch: int,
        *,
        trace: object,
        initial_temp_c: float | None = None,
    ) -> float:
        temp = self._T[node] if initial_temp_c is None else float(initial_temp_c)
        if dt <= 0.0:
            return temp
        target = float(start) + float(dt)
        for now, end, dark in trace.power_segments(node, float(start)):
            seg_end = min(end, target)
            while now < seg_end:
                if batch <= 0:
                    p_w = 0.0
                    until_sw = math.inf
                else:
                    tier = select_tier_for_batch(
                        self.spec,
                        int(batch),
                        is_eclipse=dark,
                        current_temp_c=temp,
                        rc_step_coeffs=self._rc_probe_coeffs,
                        explicit_tier_idx=self._explicit_tier_idx[node],
                    )
                    p_w = tier.power_w
                    eq = self.t_amb_c + self.r_th * p_w
                    until_sw = self._until_tier_switch(temp, eq, p_w, dark)
                step = min(seg_end - now, until_sw)
                eq = self.t_amb_c + self.r_th * p_w
                temp = eq + (temp - eq) * math.exp(-step / self.tau_s)
                now += step
            if now >= target:
                break
        return temp

    def project_finish(
        self,
        node: int,
        start: float,
        batch: int,
        context: float,
        tokens: float,
        *,
        trace: object,
        gpu: dict,
        initial_temp_c: float | None = None,
    ) -> tuple[float, float]:
        temp = self._T[node] if initial_temp_c is None else float(initial_temp_c)
        rem = float(tokens)
        if rem <= 0.0:
            return (float(start), temp)
        eval_batch = max(int(batch), 1)
        for now, end, dark in trace.power_segments(node, start):
            while now < end:
                tier = select_tier_for_batch(
                    self.spec,
                    eval_batch,
                    is_eclipse=dark,
                    current_temp_c=temp,
                    rc_step_coeffs=self._rc_probe_coeffs,
                    explicit_tier_idx=self._explicit_tier_idx[node],
                )
                it = iter_time(eval_batch, context, tier.clock_f, gpu)
                service = rem * it
                p_w = 0.0 if batch <= 0 else tier.power_w
                eq = self.t_amb_c + self.r_th * p_w
                until_sw = self._until_tier_switch(temp, eq, p_w, dark) if batch > 0 else math.inf
                dt = min(service, end - now, until_sw)
                temp = eq + (temp - eq) * math.exp(-dt / self.tau_s)
                now += dt
                if dt == service:
                    return (now, temp)
                rem -= dt / it
        return (math.inf, temp)

    def set_temperature(self, node: int, temp_c: float) -> None:
        val = float(temp_c)
        self._T[node] = val
        if val > self._peak_T[node]:
            self._peak_T[node] = val

    def temperature(self, node: int) -> float:
        return self._T[node]

    def headroom_c(self, node: int) -> float:
        return self.spec.t_max_c - self._T[node]

    def diagnostics(self) -> dict:
        return {
            "mode": self.mode,
            "peak_T_C": list(self._peak_T),
            "peak_T_C_max": max(self._peak_T) if self._peak_T else None,
            "tier_idx": list(self._tier_idx),
            "power_w": list(self._power_w),
            ACC_ZERO_POWER_S: float(self.zero_power_s),
            ACC_TIER_TIME_S: dict(self.tier_time_s),
            ACC_ENERGY_J: float(self.energy_j),
            ACC_TIER_UNSAT_S: float(self.tier_unsat_s),
            ACC_WAKEUP_COUNT: int(self.wakeup_count),
            ACC_WAKEUP_ENERGY_J: float(self.wakeup_energy_j_total),
            "thermal_trip_count": int(self.thermal_trip_count),
            "thermal_trip_s": float(self.thermal_trip_s),
        }


def make_power_thermal_model(cfg: dict, n: int) -> PowerThermalModel:
    """Construct the appropriate ``PowerThermalModel`` adapter for ``cfg``."""
    if is_discrete_power(cfg):
        return DiscreteTierModel(cfg, n)
    if cfg.get("thermal"):
        return ContinuousDVFSModel(cfg, n)
    return OrbitOnlyPowerModel(cfg, n)
