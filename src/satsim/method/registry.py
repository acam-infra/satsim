"""scheduler id + link_cost → 实现。"""
from __future__ import annotations

from dataclasses import dataclass

from satsim.method.policies.link_cost.full import FullLinkCost
from satsim.method.policies.link_cost.net import NetLinkCost
from satsim.method.policies.link_cost.off import OffLinkCost
from satsim.method.policies.schedulers.place_once import PlaceOnceScheduler
from satsim.method.policies.schedulers.rebalance_local import RebalanceLocalScheduler
from satsim.method.policies.schedulers.rebalance_over_link import RebalanceOverLinkScheduler


@dataclass
class PolicyBundle:
    scheduler: object
    link_cost: object


def _scheduler(kind: str, *, hysteresis: float | None):
    if kind == "place_once":
        return PlaceOnceScheduler()
    if kind == "rebalance_local":
        return RebalanceLocalScheduler(hysteresis=float(hysteresis if hysteresis is not None else 0.15))
    if kind == "rebalance_over_link":
        return RebalanceOverLinkScheduler(hysteresis=float(hysteresis if hysteresis is not None else 0.15))
    raise ValueError(f"未知 scheduler: {kind}")


def _link_cost(kind: str):
    if kind == "off":
        return OffLinkCost()
    if kind == "full":
        return FullLinkCost()
    if kind == "net":
        return NetLinkCost()
    raise ValueError(f"未知 link_cost: {kind}")


def bundle_from_policy(pol: dict, cfg: dict | None = None) -> PolicyBundle:
    return PolicyBundle(
        scheduler=_scheduler(pol["scheduler"], hysteresis=pol.get("hysteresis")),
        link_cost=_link_cost(pol["link_cost"]),
    )
