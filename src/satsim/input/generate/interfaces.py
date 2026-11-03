"""到达过程接口。默认线下：一次 generate 产出整窗请求。"""
from __future__ import annotations

from typing import Protocol

from satsim.domain import Request


class ArrivalProcess(Protocol):
    def generate(
        self,
        trace: dict,
        *,
        p_long: float,
        seed: int,
    ) -> tuple[list[Request], float, int]:
        """返回 (请求列表, 限功切换锚点时刻, 到达簇计数)。"""
        ...
