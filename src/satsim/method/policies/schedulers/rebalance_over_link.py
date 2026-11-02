"""初次分配 + 换位后再分配；配合 link_cost=full|net（整段 KV 上链路）。"""
from __future__ import annotations

from satsim.method.policies.schedulers._placement import place_by_iter_time
from satsim.method.policies.schedulers._select import pick_migration


class RebalanceOverLinkScheduler:
    id = "rebalance_over_link"

    def __init__(self, hysteresis: float = 0.15):
        self.hysteresis = float(hysteresis)

    def place(self, world, req) -> int:
        return place_by_iter_time(world, req)

    def maybe_rebalance(self, world) -> None:
        picked = pick_migration(world, self.hysteresis)
        if picked is None:
            return
        req, src, dst = picked
        world.env.process(world.migrate_proc(req, src, dst))
