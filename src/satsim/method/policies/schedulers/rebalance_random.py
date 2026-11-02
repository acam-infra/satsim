"""有消耗再分配 · random 基线：随机选请求与目标节点，迁移走链路。"""
from __future__ import annotations

from satsim.method.engine.world import MAX_INFLIGHT, MAX_OUT
from satsim.method.policies.schedulers._placement import place_by_iter_time


class RebalanceRandomScheduler:
    """有链路代价下的随机再分配基线。"""

    id = "rebalance_random"

    def place(self, world, req) -> int:
        return place_by_iter_time(world, req)

    def maybe_rebalance(self, world) -> None:
        inflight = [r for rs in world.nodes for r in rs if r.migrating]
        if len(inflight) >= MAX_INFLIGHT:
            return
        cands = [
            r
            for rs in world.nodes
            for r in rs
            if not r.migrating and r.nmig == 0 and r.t <= world.env.now
        ]
        if not cands:
            return
        req = world.rng.choice(cands)
        src = req.node
        if sum(1 for r in world.nodes[src] if r.migrating) >= MAX_OUT:
            return
        cap = float(world.cfg["gpu"]["kv_gib"]) * (2**30)
        dests = []
        for dst in range(world.n):
            if dst == src:
                continue
            used = sum(r.kv for r in world.nodes[dst]) * world.bpt
            if used + req.kv * world.bpt <= cap:
                dests.append(dst)
        if not dests:
            return
        dst = world.rng.choice(dests)
        world.env.process(world.migrate_proc(req, src, dst))
