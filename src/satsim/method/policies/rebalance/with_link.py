"""运行时重分配且整段 KV 上链路（配合 link_cost=full|net）。"""
from __future__ import annotations

from satsim.method.policies.rebalance._select import pick_migration


class WithLinkRebalance:
    def maybe_migrate(self, world) -> None:
        picked = pick_migration(world)
        if picked is None:
            return
        req, src, dst = picked
        world.env.process(world.migrate_proc(req, src, dst))
