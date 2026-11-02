"""迁移链路代价接口。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Protocol

if TYPE_CHECKING:
    from satsim.domain import Request
    from satsim.method.engine.world import World


class LinkCostModel(Protocol):
    mode: str  # free | full | net —— 与 cfg cost.mode 对齐

    def transfer(
        self,
        world: World,
        req: Request,
        src: int,
        dst: int,
        nbytes: float,
    ) -> Generator:
        """付完代价后 commit（或 abort）。由 engine 以 SimPy process 跑。"""
        ...
