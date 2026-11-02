"""net：与 full 同过程，用于核对代价是否来自链路（claim 对照）。"""
from __future__ import annotations

from satsim.method.policies.link_cost.full import FullLinkCost


class NetLinkCost(FullLinkCost):
    mode = "net"
