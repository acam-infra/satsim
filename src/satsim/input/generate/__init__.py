"""线下请求生成：仿真开始前产出完整序列。"""

from satsim.input.generate.dispatch import generate_arrivals
from satsim.input.generate.synthetic_pass import cluster_anchor

__all__ = ["ArrivalProcess", "cluster_anchor", "generate_arrivals"]


def __getattr__(name: str):
    if name == "ArrivalProcess":
        from satsim.input.generate.interfaces import ArrivalProcess

        return ArrivalProcess
    raise AttributeError(name)
