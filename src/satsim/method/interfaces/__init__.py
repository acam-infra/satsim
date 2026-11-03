"""窄接口：全局 Scheduler + 链路代价 + 热模型。"""

from satsim.method.interfaces.link_cost import LinkCostModel
from satsim.method.interfaces.scheduler import Scheduler
from satsim.method.interfaces.thermal import ThermalModel

__all__ = ["Scheduler", "LinkCostModel", "ThermalModel"]
