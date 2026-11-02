"""运行时重分配：换位后再放；KV 不上链路（配合 link_cost=off）。"""
from __future__ import annotations

from satsim.method.policies.rebalance._select import pick_migration


class LocalRebalance:
    def maybe_migrate(self, world) -> None:
        picked = pick_migration(world)
        if picked is None:
            return
        req, src, dst = picked
        world.env.process(world.migrate_proc(req, src, dst))
