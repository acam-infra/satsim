"""换位后不重分配（baseline：仅首次放置）。"""
from __future__ import annotations


class NoRebalance:
    def maybe_migrate(self, world) -> None:
        return
