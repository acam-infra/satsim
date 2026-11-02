"""再分配候选选择。"""
from __future__ import annotations

from satsim.method.engine.timing import iter_time
from satsim.method.engine.world import MAX_INFLIGHT, MAX_OUT


def pick_migration(world, hysteresis: float):
    """若应发起迁移，返回 (req, src, dst)，否则 None。"""
    nd, ctx = world._stats()
    f = [world.f_of(j, world.env.now) for j in range(world.n)]
    gpu = world.cfg["gpu"]
    sc = [iter_time(nd[j], ctx[j], f[j], gpu) for j in range(world.n)]
    src = max(range(world.n), key=lambda j: sc[j])
    dst = min(range(world.n), key=lambda j: sc[j])
    if src == dst or sc[src] <= sc[dst] * (1.0 + hysteresis):
        return None
    inflight = [r for rs in world.nodes for r in rs if r.migrating]
    if len(inflight) >= MAX_INFLIGHT:
        return None
    if sum(1 for r in world.nodes[src] if r.migrating) >= MAX_OUT:
        return None
    cands = [r for r in world.nodes[src] if not r.migrating and r.nmig == 0 and r.t <= world.env.now]
    if not cands:
        return None
    req = max(cands, key=lambda r: r.kv)
    cap = float(world.cfg["gpu"]["kv_gib"]) * (2**30)
    used = sum(r.kv for r in world.nodes[dst]) * world.bpt
    if used + req.kv * world.bpt > cap:
        return None
    return req, src, dst
