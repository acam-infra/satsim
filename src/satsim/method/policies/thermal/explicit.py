"""定时限功事件 → DVFS（默认；与历史行为逐位一致）。"""
from __future__ import annotations

from typing import Sequence

from satsim.method.engine.timing import clock_factor as dvfs_clock_factor


class ExplicitThermal:
    mode = "explicit"

    def __init__(self, thermal: dict):
        self._th = thermal

    def clock_factor(self, node: int, t: float) -> float:
        p = self._th["P_max"]
        for ev in self._th.get("explicit_events") or []:
            j, t0, t1, pc = ev
            if j == node and t0 <= t < t1:
                p = min(p, pc)
        return dvfs_clock_factor(p, self._th)

    def advance(self, t: float, node_loads: Sequence[int]) -> bool:
        return False
