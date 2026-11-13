"""Discrete satellite power-tier contract and linear RC thermal projection helpers.

Frozen in Stage 0 (LAB-1867 / LAB-1881 / ADR-0007) as the narrow seam shared by:
- ``world.py`` & ``dynamic.py`` & ``record.py`` (DP-1: physical discrete power, RC evolution & metrics),
- ``discrete_milp.py`` & ``policies.py`` & ``run.py`` (DP-2: 120s snapshot Time-ILP + online multi-hop 0W-drain RH-Greedy),
- ``configs/*`` & ``scripts/verify_discrete_milp.py`` (DP-3: calibrated hotspot configs & cnb-compute ablation).

Key safety & fairness invariants (LAB-1881 review fixes):
1. **Zero-request 0 W power vs. strictly positive step-time frequency**:
   ``select_tier_for_batch(spec, 0, ...)`` returns ``idle_tier`` (``power_w == 0.0``) for physical
   thermal integration, whereas ``effective_clock_f(spec, batch, ...)`` evaluates at
   ``max(batch, 1)`` so ``CoWorld.f_of`` / ``CoWorld.iter_of`` NEVER divide by zero on empty nodes.
2. **Symmetric baseline fairness**:
   By default (``prefer_high_clock=True``), sunlit nodes with safe thermal margin select the fastest
   feasible working tier rather than locking light-load baselines (``batch = 1..4``) into a slow tier;
   entering eclipse (``P_k > P_dark``) or approaching ``T_max_C`` automatically steps down to a lower
   power tier, while planners may also proactively set ``explicit_tier_idx`` (e.g. pre-cooling).
3. **Zero legacy regression**:
   When ``cfg["power"].get("mode", "continuous") != "discrete_tiers"``, callers MUST preserve
   legacy continuous ``device_power`` and ``84°C/76°C`` hysteresis with ``diffs=0``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

POWER_MODE_CONTINUOUS = "continuous"
POWER_MODE_DISCRETE = "discrete_tiers"

# Canonical CoWorld.acc keys populated when discrete_tiers mode is active.
ACC_ZERO_POWER_S = "zero_power_s"      # float: cumulative compute-node seconds spent at 0 W (tier 0)
ACC_TIER_TIME_S = "tier_time_s"        # dict[int, float]: cumulative compute-node seconds per tier index
ACC_ENERGY_J = "energy_j"              # float: cumulative Joules across compute nodes
ACC_TIER_UNSAT_S = "tier_unsat_s"      # float: cumulative node-seconds where batch > max tier capacity
ACC_WAKEUP_COUNT = "wakeup_count"      # int: cumulative 0W -> working tier transitions
ACC_WAKEUP_ENERGY_J = "wakeup_energy_j_total"  # float: cumulative 0W cold-start wakeup Joules


@dataclass(frozen=True)
class PowerTier:
    """One discrete satellite operating state ``(P_k, B^max_k, f_k)``."""

    index: int
    name: str
    power_w: float
    max_batch: int
    clock_f: float
    prefill_clock_f: float | None = None

    @property
    def effective_prefill_f(self) -> float:
        return self.clock_f if self.prefill_clock_f is None else self.prefill_clock_f


@dataclass(frozen=True)
class DiscretePowerSpec:
    """Validated discrete power & linear thermal-cap configuration for a co-simulation run."""

    tiers: tuple[PowerTier, ...]
    p_dark_w: float
    t_max_c: float
    thermal_horizon_s: float
    prefer_high_clock: bool = True
    wakeup_energy_j: float = 0.0
    wakeup_latency_s: float = 0.0
    thermal_hysteresis_c: float = 0.0

    @property
    def idle_tier(self) -> PowerTier:
        return self.tiers[0]

    @property
    def work_tiers(self) -> tuple[PowerTier, ...]:
        return self.tiers[1:]

    @property
    def max_capacity(self) -> int:
        return self.tiers[-1].max_batch


def is_discrete_power(cfg: dict) -> bool:
    """Return True iff ``cfg["power"]["mode"] == "discrete_tiers"``."""
    power = cfg.get("power") or {}
    return power.get("mode", POWER_MODE_CONTINUOUS) == POWER_MODE_DISCRETE


def parse_discrete_power_spec(cfg: dict) -> DiscretePowerSpec | None:
    """Parse and validate ``DiscretePowerSpec`` from ``cfg``, or return ``None`` in legacy mode.

    Expected YAML schema under ``power:`` when ``mode: discrete_tiers``::

        power:
          mode: discrete_tiers
          P_idle: 0
          P_max: 300
          gamma: 2.4
          f_min: 0.35
          P_dark: 150
          prefer_high_clock: true   # optional (default true: sunlit cool nodes use fastest safe tier)
          tiers:
            - {name: idle_0w, power_w: 0.0, max_batch: 0, clock_f: 0.0}   # optional; auto-inserted
            - {name: low,     power_w: 150.0, max_batch: 4, clock_f: 0.65, prefill_clock_f: 0.35}
            - {name: high,    power_w: 300.0, max_batch: 12, clock_f: 1.0, prefill_clock_f: 1.0}
    """
    if not is_discrete_power(cfg):
        return None
    power = cfg["power"]
    raw_tiers = list(power.get("tiers") or [])
    if not raw_tiers:
        raise ValueError("power.mode='discrete_tiers' requires a non-empty power.tiers list")

    parsed: list[PowerTier] = []
    has_zero = False
    for item in raw_tiers:
        pw = float(item["power_w"])
        mb = int(item["max_batch"])
        cf = float(item["clock_f"])
        pcf_raw = item.get("prefill_clock_f")
        pcf = float(pcf_raw) if pcf_raw is not None else None
        nm = str(item.get("name", f"tier_{len(parsed)}"))
        if mb == 0:
            if pw != 0.0 or cf != 0.0 or (pcf is not None and pcf != 0.0):
                raise ValueError("zero-request tier (max_batch=0) must have power_w=0.0 and clock_f=0.0")
            has_zero = True
        else:
            if mb < 0 or pw <= 0.0 or cf <= 0.0 or (pcf is not None and pcf <= 0.0):
                raise ValueError(f"working tier {nm!r} must have positive power_w, max_batch, and clock_f")
        parsed.append(PowerTier(index=0, name=nm, power_w=pw, max_batch=mb, clock_f=cf, prefill_clock_f=pcf))

    if not has_zero:
        parsed.insert(0, PowerTier(index=0, name="idle_0w", power_w=0.0, max_batch=0, clock_f=0.0))

    parsed.sort(key=lambda t: (t.max_batch, t.power_w))
    if parsed[0].max_batch != 0 or parsed[0].power_w != 0.0:
        raise ValueError("tier 0 must be the zero-request 0 W idle tier")

    reindexed: list[PowerTier] = []
    seen_batches: set[int] = set()
    for idx, t in enumerate(parsed):
        if t.max_batch in seen_batches:
            raise ValueError(f"duplicate max_batch={t.max_batch} in power.tiers")
        seen_batches.add(t.max_batch)
        reindexed.append(
            PowerTier(
                index=idx,
                name=t.name,
                power_w=t.power_w,
                max_batch=t.max_batch,
                clock_f=t.clock_f,
                prefill_clock_f=t.prefill_clock_f,
            )
        )
    if len(reindexed) < 2:
        raise ValueError("discrete_tiers requires at least one working tier (k >= 1)")

    thermal = cfg.get("thermal") or {}
    t_max_c = float(thermal.get("T_max_C", thermal.get("T_throttle_C", 84.0)))
    horizon_s = float(thermal.get("thermal_horizon_s", 30.0))
    if horizon_s <= 0.0:
        raise ValueError("thermal.thermal_horizon_s must be positive")
    p_dark_w = float(power["P_dark"])
    prefer_high_clock = bool(power.get("prefer_high_clock", True))
    wakeup_energy_j = float(power.get("wakeup_energy_j", 0.0))
    wakeup_latency_s = float(power.get("wakeup_latency_s", 0.0))
    if wakeup_energy_j < 0.0 or wakeup_latency_s < 0.0:
        raise ValueError("power.wakeup_energy_j and power.wakeup_latency_s must be non-negative")
    hyst_c = float(thermal.get("hysteresis_C", 0.0))
    if hyst_c < 0.0:
        raise ValueError("thermal.hysteresis_C must be non-negative")

    return DiscretePowerSpec(
        tiers=tuple(reindexed),
        p_dark_w=p_dark_w,
        t_max_c=t_max_c,
        thermal_horizon_s=horizon_s,
        prefer_high_clock=prefer_high_clock,
        wakeup_energy_j=wakeup_energy_j,
        wakeup_latency_s=wakeup_latency_s,
        thermal_hysteresis_c=hyst_c,
    )


def rc_linear_coeffs(tau_s: float, r_th: float, t_amb_c: float, dt_s: float) -> tuple[float, float, float]:
    """Return ``(alpha, const_offset, power_gain)`` for linear 1st-order RC step ``dt_s``.

    Exact discrete-time step under constant tier power ``P`` over ``dt_s``::

        T(t + dt_s) = alpha * T(t) + const_offset + power_gain * P

    where ``alpha = exp(-dt_s / tau_s)``, ``const_offset = (1 - alpha) * t_amb_c``,
    and ``power_gain = (1 - alpha) * r_th``.
    """
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive")
    if dt_s <= 0.0:
        return (1.0, 0.0, 0.0)
    alpha = math.exp(-dt_s / tau_s)
    one_minus = 1.0 - alpha
    return (alpha, one_minus * t_amb_c, one_minus * r_th)


def rollout_rc_temperatures(
    t0_c: float,
    powers_w: list[float] | tuple[float, ...],
    *,
    tau_s: float,
    r_th: float,
    t_amb_c: float,
    dt_s: float,
) -> list[float]:
    """Roll out ``H`` multi-slot RC temperatures ``[T_1, ..., T_H]`` under piecewise-constant ``powers_w``.

    Used by both multi-slot time-indexed ILP verification and rolling-horizon thermal-aware greedy
    (``discrete_rh_greedy``) to guarantee identical discrete-time thermal trajectories.
    """
    alpha, offset, gain = rc_linear_coeffs(tau_s, r_th, t_amb_c, dt_s)
    cur = float(t0_c)
    traj: list[float] = []
    for p_w in powers_w:
        cur = alpha * cur + offset + gain * float(p_w)
        traj.append(cur)
    return traj


def allowed_tier_indices(spec: DiscretePowerSpec, is_eclipse: bool) -> list[int]:
    """Return tier indices allowed under the current orbital illumination envelope (D3)."""
    if not is_eclipse:
        return [t.index for t in spec.tiers]
    return [t.index for t in spec.tiers if t.power_w <= spec.p_dark_w + 1e-9]


def select_tier_for_batch(
    spec: DiscretePowerSpec,
    batch: int,
    *,
    is_eclipse: bool = False,
    current_temp_c: float | None = None,
    rc_step_coeffs: tuple[float, float, float] | None = None,
    explicit_tier_idx: int | None = None,
    prefer_high_clock: bool | None = None,
    thermal_trip_lockout: bool = False,
) -> PowerTier:
    """Deterministic runtime governor mapping resident ``batch`` to an active ``PowerTier``.

    - ``batch <= 0`` ALWAYS selects ``spec.idle_tier`` (``0.0 W, clock_f = 0.0``).
    - ``batch >= 1`` filters working tiers allowed by ``is_eclipse`` (D3) and, when ``current_temp_c``
      and ``rc_step_coeffs`` are supplied, tiers keeping ``T_next <= spec.t_max_c`` (D4).
    - If ``explicit_tier_idx`` names a feasible working tier whose ``max_batch >= batch``, that
      planner-chosen tier (e.g. proactive pre-cooling at ``low`` tier) is honored directly.
    - Otherwise, among feasible tiers with ``max_batch >= batch``:
      - when ``prefer_high_clock`` (defaults to ``spec.prefer_high_clock``, ``True``), picks the
        highest-clock tier (tie-broken by lower power) so sunlit cool nodes are not artificially
        throttled at light load;
      - when ``prefer_high_clock=False``, picks the minimum-power fitting tier.
    - If ``batch`` exceeds all feasible tiers' ``max_batch``, clamps to the highest-capacity
      feasible working tier so ongoing decode never divides by 0.
    """
    if batch <= 0:
        return spec.idle_tier

    allowed_set = set(allowed_tier_indices(spec, is_eclipse))
    work = [t for t in spec.work_tiers if t.index in allowed_set]
    if not work:
        work = [min(spec.work_tiers, key=lambda t: t.power_w)]

    if thermal_trip_lockout:
        work = [min(work, key=lambda t: t.power_w)]
    elif current_temp_c is not None and rc_step_coeffs is not None:
        alpha, offset, gain = rc_step_coeffs
        thermally_safe = [
            t for t in work if alpha * current_temp_c + offset + gain * t.power_w <= spec.t_max_c + 1e-9
        ]
        if thermally_safe:
            work = thermally_safe
        else:
            work = [min(work, key=lambda t: t.power_w)]

    if explicit_tier_idx is not None and explicit_tier_idx >= 1:
        for t in work:
            if t.index == explicit_tier_idx and t.max_batch >= batch:
                return t

    fitting = [t for t in work if t.max_batch >= batch]
    if fitting:
        use_high = spec.prefer_high_clock if prefer_high_clock is None else prefer_high_clock
        if use_high:
            return max(fitting, key=lambda t: (t.clock_f, -t.power_w, -t.max_batch))
        return min(fitting, key=lambda t: (t.power_w, t.max_batch))
    return max(work, key=lambda t: t.max_batch)


def effective_clock_f(
    spec: DiscretePowerSpec,
    batch: int,
    *,
    is_eclipse: bool = False,
    current_temp_c: float | None = None,
    rc_step_coeffs: tuple[float, float, float] | None = None,
    explicit_tier_idx: int | None = None,
    prefer_high_clock: bool | None = None,
    thermal_trip_lockout: bool = False,
) -> float:
    """Return strictly positive decode ``clock_f > 0`` for resident or hypothetical ``batch``.

    Decouples empty-node ``0.0 W`` thermal dissipation from candidate step-time estimation
    (``CoWorld.f_of`` / ``CoWorld.iter_of``): when ``batch <= 0`` (an idle node probed by placement
    or rebalance, or a request newly admitted before the next ``_thermal_tick``), evaluates at
    ``eval_batch = 1`` so ``timing.iter_time`` never raises ``ZeroDivisionError``.
    """
    eval_batch = max(int(batch), 1)
    tier = select_tier_for_batch(
        spec,
        eval_batch,
        is_eclipse=is_eclipse,
        current_temp_c=current_temp_c,
        rc_step_coeffs=rc_step_coeffs,
        explicit_tier_idx=explicit_tier_idx,
        prefer_high_clock=prefer_high_clock,
        thermal_trip_lockout=thermal_trip_lockout,
    )
    return tier.clock_f
