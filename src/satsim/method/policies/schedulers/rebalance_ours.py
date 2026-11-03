"""占带宽迁移 · 自研策略：待实现的骨架，不要当成已完成的最优策略。"""
from __future__ import annotations

from satsim.method.policies.schedulers._placement import place_by_iter_time


class RebalanceOursScheduler:
    """TODO(paper): 占星间带宽时的自研迁移策略（ours）— 尚未实现。

    当前 maybe_rebalance 为空操作，仅保证接口可挂、实验可加载。
    """

    id = "rebalance_ours"
    TODO = "占带宽迁移的自研策略：待实现"

    def place(self, world, req) -> int:
        return place_by_iter_time(world, req)

    def maybe_rebalance(self, world) -> None:
        # TODO(paper): 在 link_cost=full|net 下选择迁谁、迁去哪
        return
