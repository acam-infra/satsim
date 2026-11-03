"""无消耗迁移：限功切换后按负载迁到更快的节点；须配 link_cost=off。"""
from __future__ import annotations

from satsim.method.policies.schedulers._placement import place_by_iter_time
from satsim.method.policies.schedulers._select import pick_migration


class RebalanceFreeScheduler:
    """不占星间带宽的运行中迁移（free / no-cost）。"""

    id = "rebalance_free"

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
