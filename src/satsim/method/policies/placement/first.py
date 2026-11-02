"""首次分配：到来时按迭代时间代价选节点放一次。"""
from __future__ import annotations

from satsim.input.schema import SpecError
from satsim.method.engine.timing import iter_time


class FirstPlacement:
    def place(self, world, req) -> int:
        cap = float(world.cfg["gpu"]["kv_gib"]) * (2**30)
        scores = []
        for j in range(world.n):
            used = sum(r.kv for r in world.nodes[j]) * world.bpt
            if used + req.kv * world.bpt > cap:
                scores.append(float("inf"))
                continue
            nd, ctx = world._stats()
            scores.append(
                iter_time(
                    nd[j] + 1,
                    ctx[j] + req.kv,
                    world.f_of(j, world.env.now),
                    world.cfg["gpu"],
                )
            )
        if min(scores) == float("inf"):
            raise SpecError(f"请求 {req.i} 在所有节点都放不下 KV")
        return scores.index(min(scores))
