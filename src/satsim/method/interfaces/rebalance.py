"""换位后重分配接口。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from satsim.method.engine.world import World


class RebalancePolicy(Protocol):
    def maybe_migrate(self, world: World) -> None:
        """控制周期触发：可选发起一次迁移。none 为空操作。"""
        ...
