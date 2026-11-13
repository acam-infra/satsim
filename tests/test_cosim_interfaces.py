"""Contract tests for satsim.cosim.interfaces (LAB-1881 pluggable seams)."""
from __future__ import annotations

import pytest

from satsim.cosim.interfaces import (
     AssignmentSolver,
    CandidateMove,
    Migrator,
    OperatingPoint,
    Placer,
    PowerThermalModel,
    accepts_rebalance_quota,
)


def test_operating_point_rejects_zero_clock_or_negative_power() -> None:
    pt = OperatingPoint(power_w=0.0, clock_f=0.65, prefill_clock_f=0.35, tier_index=0)
    assert pt.power_w == 0.0
    assert pt.clock_f == 0.65
    assert pt.effective_prefill_f == 0.35

    with pytest.raises(ValueError, match="clock_f must be > 0.0"):
        OperatingPoint(power_w=0.0, clock_f=0.0)

    with pytest.raises(ValueError, match="power_w must be >= 0.0"):
        OperatingPoint(power_w=-1.0, clock_f=1.0)


def test_accepts_rebalance_quota_protocol_vs_place_once() -> None:
    class PlaceOnly:
        id = "greedy"

        def place(self, w, r):
            return 0

        def rebalance(self, w) -> None:
            return None

    class QuotaMigrator:
        id = "discrete_rh_greedy"
        interval_s = 5.0
        max_inflight = 24
        max_mig_per_req = 4

        def rebalance(self, w) -> None:
            return None

    class CompositeStub:
        def __init__(self, migrator):
            self.migrator = migrator

    assert not accepts_rebalance_quota(PlaceOnly())
    assert not accepts_rebalance_quota(CompositeStub(None))
    assert accepts_rebalance_quota(QuotaMigrator())
    assert accepts_rebalance_quota(CompositeStub(QuotaMigrator()))
    assert isinstance(PlaceOnly(), Placer)
    assert isinstance(QuotaMigrator(), Migrator)


def test_runtime_checkable_power_and_solver_protocols() -> None:
    class DummyPower:
        mode = "discrete_tiers"

        def operating_point(self, node: int, batch: int, *, is_dark: bool, t: float | None = None) -> OperatingPoint:
            return OperatingPoint(power_w=0.0 if batch == 0 else 150.0, clock_f=1.0, tier_index=0 if batch == 0 else 1)

        def step(self, now, batches, is_dark_fn):
            return []

        def project_finish(self, node, start, batch, context, tokens, *, trace, gpu):
            return (start + 1.0, 50.0)

        def temperature(self, node: int) -> float:
            return 45.0

        def headroom_c(self, node: int) -> float:
            return 39.0

        def diagnostics(self) -> dict:
            return {}

    class DummySolver:
        id = "greedy"

        def solve(self, w, candidates: list[CandidateMove], *, budget: int, now: float) -> list[CandidateMove]:
            return candidates[:budget]

    assert isinstance(DummyPower(), PowerThermalModel)
    assert isinstance(DummySolver(), AssignmentSolver)
