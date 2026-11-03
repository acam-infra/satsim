"""只做首次分配，限功切换后不再迁移。"""
from __future__ import annotations

from satsim.method.policies.schedulers._placement import place_by_iter_time


class PlaceOnceScheduler:
    id = "place_once"

    def place(self, world, req) -> int:
        return place_by_iter_time(world, req)

    def maybe_rebalance(self, world) -> None:
        return
