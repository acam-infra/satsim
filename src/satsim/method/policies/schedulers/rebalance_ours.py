"""有消耗再分配 · ours：待解决问题的骨架，勿当作已实现最优策略。"""
from __future__ import annotations

from satsim.method.policies.schedulers._placement import place_by_iter_time


class RebalanceOursScheduler:
    """TODO(paper): 有链路代价约束下的再分配策略（ours）— 尚未实现。

    当前 maybe_rebalance 为空操作，仅保证接口可挂、实验可加载。
    """

    id = "rebalance_ours"
    TODO = "有消耗再分配 ours：待解决"

    def place(self, world, req) -> int:
        return place_by_iter_time(world, req)

    def maybe_rebalance(self, world) -> None:
        # TODO(paper): 在 link_cost=full|net 下选择迁谁、迁去哪
        return
