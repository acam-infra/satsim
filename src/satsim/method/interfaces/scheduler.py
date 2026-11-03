"""全局调度器：首次分配和运行中迁移（可提前干预）合成一个入口。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from satsim.domain import Request
    from satsim.method.engine.world import World


class Scheduler(Protocol):
    """可看全图的全局策略。engine 只调这两个钩子。"""

    id: str

    def place(self, world: World, req: Request) -> int:
        """首次分配：返回节点 id。"""
        ...

    def maybe_rebalance(self, world: World) -> None:
        """控制周期触发的迁移 / 提前干预；无动作可空实现。"""
        ...
