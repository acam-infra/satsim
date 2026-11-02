"""通用仿真内核：世界状态与事件循环。不含具体策略名。"""

from satsim.method.engine.world import Req, World

__all__ = ["Req", "World", "run_one"]


def __getattr__(name: str):
    if name == "run_one":
        from satsim.method.engine.loop import run_one

        return run_one
    raise AttributeError(name)
