"""节点时钟倍率：显式功率帽时间线，或动态温度→节流。"""
from __future__ import annotations

from typing import Protocol, Sequence


class ThermalModel(Protocol):
    """World.f_of / iter_of 只认这个接口；decode 与调度臂不感知后端。"""

    mode: str  # explicit | dynamic

    def clock_factor(self, node: int, t: float) -> float:
        """节点在仿真时刻 t 的 DVFS 倍率，落在 [f_min, 1]。"""
        ...

    def advance(self, t: float, node_loads: Sequence[int]) -> bool:
        """把热状态推到 t（控制周期边界调用）。返回是否有节点倍率变化（需 poke）。"""
        ...
