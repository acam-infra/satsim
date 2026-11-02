"""free：瞬时搬走，KV 不上链路。"""
from __future__ import annotations


class OffLinkCost:
    mode = "free"

    def transfer(self, world, req, src, dst, nbytes: float):
        yield world.env.timeout(0)
        world._commit(req, src, dst, 0.0)
