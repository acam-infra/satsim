"""Narrow pluggable seams for co-simulation physics and scheduling (LAB-1881 / ADR-0007).

Following ``.agents/skills/codebase-design`` ("one adapter = hypothetical seam, two = real"):
only subsystems with >= 2 genuine implementations get a Protocol seam here:

1. ``PowerThermalModel`` — unifies simulation thermal evolution and planner forward projection
   across ``ContinuousDVFSModel`` (legacy continuous power + 84/76C hysteresis),
   ``DiscreteTierModel`` (``0W`` idle + discrete working tiers + linear RC cap), and
   ``OrbitOnlyPowerModel`` (no thermal RC; sunlit 1.0x vs eclipse ``f_dark``).
2. ``Placer`` / ``Migrator`` / ``AssignmentSolver`` — orthogonalizes initial placement,
   runtime migration (with protocol-driven ``rebalance_invariants`` quota enforcement rather
   than hardcoded policy-name whitelists), and combinatorial move selection (incremental greedy
   vs single-step MILP vs offline 120s ``Time-ILP`` oracle).

Subsystems with a single physical implementation (``FluidNet``, ``SimPy`` loop) or an existing
data-schema seam (``Trace`` via ``orbitsim.contact_trace/v1``) intentionally have NO wrapper here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

REBALANCE_QUOTA_ATTRS: tuple[str, str, str] = ("interval_s", "max_inflight", "max_mig_per_req")


@dataclass(frozen=True)
class OperatingPoint:
    """Instantaneous node operating state returned by ``PowerThermalModel.operating_point``.

    Invariants:
    - ``power_w >= 0.0`` (strictly ``0.0`` when ``batch == 0`` in ``discrete_tiers`` mode).
    - ``clock_f > 0.0`` ALWAYS (even when ``batch == 0``), so ``iter_time`` / ``CoWorld.iter_of``
      never divide by zero when probing an empty candidate node or right upon admission.
    """

    power_w: float
    clock_f: float
    prefill_clock_f: float | None = None
    tier_index: int | None = None
    throttled: bool = False

    def __post_init__(self) -> None:
        if self.power_w < 0.0:
            raise ValueError(f"power_w must be >= 0.0, got {self.power_w}")
        if self.clock_f <= 0.0:
            raise ValueError(f"clock_f must be > 0.0 (no division by zero), got {self.clock_f}")
        if self.prefill_clock_f is not None and self.prefill_clock_f <= 0.0:
            raise ValueError(f"prefill_clock_f must be > 0.0 when set, got {self.prefill_clock_f}")

    @property
    def effective_prefill_f(self) -> float:
        return self.clock_f if self.prefill_clock_f is None else self.prefill_clock_f


@runtime_checkable
class PowerThermalModel(Protocol):
    """Unified seam shared by ``CoWorld`` physics stepping and policy forward trajectory forecasts."""

    mode: str

    def operating_point(
        self,
        node: int,
        batch: int,
        *,
        is_dark: bool,
        t: float | None = None,
    ) -> OperatingPoint:
        """Return ``(power_w, clock_f, ...)`` for ``node`` at ``batch`` and eclipse state."""

    def step(
        self,
        now: float,
        batches: dict[int, int] | list[int],
        is_dark_fn: Callable[[int, float], bool],
    ) -> list[int]:
        """Advance node temperatures/tiers to ``now`` and return compute node IDs whose clock changed."""

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
    ) -> tuple[float, float]:
        """Project ``(finish_time_s, end_temp_c)`` for ``tokens`` of decode starting at ``start``."""

    def temperature(self, node: int) -> float:
        """Current junction temperature in Celsius (or ambient baseline when unmodeled)."""

    def headroom_c(self, node: int) -> float:
        """Remaining Celsius margin below thermal throttle / ``T_max_C`` cap (``inf`` if unmodeled)."""

    def diagnostics(self) -> dict:
        """Return summary diagnostics (e.g. ``peak_T_C``, ``tier_time_s``, ``energy_j``)."""


@dataclass(frozen=True)
class CandidateMove:
    """Scored runtime migration candidate passed from evaluator to ``AssignmentSolver``."""

    req_id: int
    req: object
    src: int
    dst: int
    path: tuple[list[int], list[int]]
    nbytes: float
    delay_s: float
    gain: float
    delta_power_w: float = 0.0
    empties_src_to_0w: bool = False
    wakes_dst_from_0w: bool = False
    meta: dict = field(default_factory=dict)


@runtime_checkable
class AssignmentSolver(Protocol):
    """Combinatorial selector over scored ``CandidateMove`` items under node/ISL/quota capacities."""

    id: str

    def solve(
        self,
        w: object,
        candidates: list[CandidateMove],
        *,
        budget: int,
        now: float,
    ) -> list[CandidateMove]:
        """Select a mutually feasible subset of at most ``budget`` moves to execute at ``now``."""


@runtime_checkable
class Placer(Protocol):
    """Initial admission placement seam."""

    id: str

    def place(self, w: object, r: object) -> int | None:
        """Pick a reachable compute node with KV capacity for newly arrived request ``r``."""


@runtime_checkable
class Migrator(Protocol):
    """Runtime stateful KV migration seam."""

    id: str

    def rebalance(self, w: object) -> None:
        """Inspect ``w`` at a control tick and issue ``w.migrate(r, dst, path=...)`` calls."""


def accepts_rebalance_quota(policy_or_migrator: object) -> bool:
    """Protocol check replacing hardcoded policy-name whitelists for ``rebalance_invariants``.

    Returns True if ``policy_or_migrator`` (or ``policy.migrator`` on a ``CompositePolicy``)
    exposes ``rebalance`` and either sets ``accepts_rebalance_invariants = True`` or carries the
    canonical quota attributes ``(interval_s, max_inflight, max_mig_per_req)``.
    """
    target = getattr(policy_or_migrator, "migrator", None)
    if target is None:
        if hasattr(policy_or_migrator, "migrator"):
            # CompositePolicy with migrator=None (place_once)
            return False
        target = policy_or_migrator
    explicit = getattr(target, "accepts_rebalance_invariants", None)
    if explicit is not None:
        return bool(explicit)
    return callable(getattr(target, "rebalance", None)) and all(
        hasattr(target, attr) for attr in REBALANCE_QUOTA_ATTRS
    )
