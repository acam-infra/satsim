"""薄物理辅助：轨道位置 → ISL 邻接快照。不做星座引擎。"""

from satsim.physics.orbit_links import (
    adjacency_from_edges,
    build_orbit_edges,
    node_roles,
    positions_eci_km,
)

__all__ = [
    "adjacency_from_edges",
    "build_orbit_edges",
    "node_roles",
    "positions_eci_km",
]
