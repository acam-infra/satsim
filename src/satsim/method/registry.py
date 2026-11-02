"""显式 placement / rebalance / link_cost → 接口实现。"""
from __future__ import annotations

from dataclasses import dataclass

from satsim.method.policies.link_cost.full import FullLinkCost
from satsim.method.policies.link_cost.net import NetLinkCost
from satsim.method.policies.link_cost.off import OffLinkCost
from satsim.method.policies.placement.first import FirstPlacement
from satsim.method.policies.rebalance.local import LocalRebalance
from satsim.method.policies.rebalance.none import NoRebalance
from satsim.method.policies.rebalance.with_link import WithLinkRebalance


@dataclass
class PolicyBundle:
    placement: object
    rebalance: object
    link_cost: object


def _placement(kind: str):
    if kind == "first":
        return FirstPlacement()
    raise ValueError(f"未知 placement: {kind}")


def _rebalance(kind: str):
    if kind == "none":
        return NoRebalance()
    if kind == "local":
        return LocalRebalance()
    if kind == "with_link":
        return WithLinkRebalance()
    raise ValueError(f"未知 rebalance: {kind}")


def _link_cost(kind: str):
    if kind == "off":
        return OffLinkCost()
    if kind == "full":
        return FullLinkCost()
    if kind == "net":
        return NetLinkCost()
    raise ValueError(f"未知 link_cost: {kind}")


def bundle_from_ids(placement: str, rebalance: str, link_cost: str) -> PolicyBundle:
    return PolicyBundle(
        placement=_placement(placement),
        rebalance=_rebalance(rebalance),
        link_cost=_link_cost(link_cost),
    )


def bundle_from_policy(pol: dict, cfg: dict | None = None) -> PolicyBundle:
    return bundle_from_ids(pol["placement"], pol["rebalance"], pol["link_cost"])
