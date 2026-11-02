"""首次放置接口。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from satsim.domain import Request
    from satsim.method.engine.world import World


class PlacementPolicy(Protocol):
    def place(self, world: World, req: Request) -> int:
        """返回节点 id；所有节点放不下时应抛错。"""
        ...
