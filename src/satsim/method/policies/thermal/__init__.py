"""可插拔热模型后端。"""

from __future__ import annotations

from satsim.method.policies.thermal.dynamic import DynamicThermal
from satsim.method.policies.thermal.explicit import ExplicitThermal

__all__ = ["ExplicitThermal", "DynamicThermal", "thermal_from_cfg"]


def thermal_from_cfg(cfg: dict):
    """按 thermal.mode 装配；缺省 / explicit → 显式事件，与旧 golden 同路径。"""
    th = cfg["thermal"]
    mode = th.get("mode", "explicit")
    if mode == "explicit":
        return ExplicitThermal(th)
    if mode == "dynamic":
        return DynamicThermal(th, n=int(cfg["nodes"]))
    raise ValueError(f"未知 thermal.mode: {mode}")
