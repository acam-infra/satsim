"""窄接口：全局 Scheduler + 链路代价。"""

from satsim.method.interfaces.link_cost import LinkCostModel
from satsim.method.interfaces.scheduler import Scheduler

__all__ = ["Scheduler", "LinkCostModel"]
