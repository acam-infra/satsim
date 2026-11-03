"""解析圆轨位置 → 距离门限 ISL 边（启动快照，非时变）。

对照常见「轨道传播 / link matcher」分层，只保留：壳层圆轨 + 门限连边。
不做 Walker 专用逻辑、不做逐 tick 重连。
"""
from __future__ import annotations

import math
from typing import Any


def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def positions_eci_km(orbit: dict[str, Any]) -> dict[int, tuple[float, float, float]]:
    """壳层圆轨 → 近似 ECI 直角坐标 (km)。

    每平面：altitude_km、inclination_deg、raan_deg；节点给 true_anomaly_deg 与全局 id。
    """
    re = float(orbit.get("earth_radius_km", 6371.0))
    out: dict[int, tuple[float, float, float]] = {}
    for plane in orbit["planes"]:
        r = re + float(plane["altitude_km"])
        inc = _deg2rad(float(plane.get("inclination_deg", 0.0)))
        raan = _deg2rad(float(plane.get("raan_deg", 0.0)))
        cos_o, sin_o = math.cos(raan), math.sin(raan)
        cos_i, sin_i = math.cos(inc), math.sin(inc)
        for node in plane["nodes"]:
            nid = int(node["id"])
            nu = _deg2rad(float(node["true_anomaly_deg"]))
            cos_u, sin_u = math.cos(nu), math.sin(nu)
            x = r * (cos_o * cos_u - sin_o * sin_u * cos_i)
            y = r * (sin_o * cos_u + cos_o * sin_u * cos_i)
            z = r * (sin_u * sin_i)
            out[nid] = (x, y, z)
    return out


def _dist_km(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def build_orbit_edges(orbit: dict[str, Any], n_nodes: int) -> list[tuple[int, int]]:
    """生成无向边 (u, v) 且 u < v。

    - 任意两点距离 ≤ distance_threshold_km → 连边（含跨平面）。
    - same_plane_neighbors=true（默认）时，平面内按真近点角排序再串邻接；
      plane_ring=true（默认）时平面内成环。
    """
    if n_nodes < 2:
        raise ValueError("n_nodes 至少为 2")
    pos = positions_eci_km(orbit)
    missing = [i for i in range(n_nodes) if i not in pos]
    if missing:
        raise ValueError(f"orbit 未覆盖节点 id {missing}")
    extra = sorted(set(pos) - set(range(n_nodes)))
    if extra:
        raise ValueError(f"orbit 含超出 nodes 的 id {extra}")

    edges: set[tuple[int, int]] = set()
    thr = float(orbit["distance_threshold_km"])
    ids = list(range(n_nodes))
    for i, u in enumerate(ids):
        for v in ids[i + 1 :]:
            if _dist_km(pos[u], pos[v]) <= thr + 1e-9:
                edges.add((u, v) if u < v else (v, u))

    if orbit.get("same_plane_neighbors", True):
        ring = bool(orbit.get("plane_ring", True))
        for plane in orbit["planes"]:
            ordered = sorted(
                (float(nd["true_anomaly_deg"]), int(nd["id"])) for nd in plane["nodes"]
            )
            seq = [nid for _, nid in ordered]
            if len(seq) < 2:
                continue
            for a, b in zip(seq, seq[1:]):
                edges.add((a, b) if a < b else (b, a))
            if ring and len(seq) >= 3:
                a, b = seq[-1], seq[0]
                edges.add((a, b) if a < b else (b, a))
            elif ring and len(seq) == 2:
                a, b = seq[0], seq[1]
                edges.add((a, b) if a < b else (b, a))

    return sorted(edges)


def adjacency_from_edges(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """无向边 → 每节点排好序的邻接表（供 BFS）。"""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError(f"非法边 {(u, v)}")
        adj[u].append(v)
        adj[v].append(u)
    return [sorted(set(nbrs)) for nbrs in adj]


def node_roles(orbit: dict[str, Any]) -> dict[int, str]:
    """节点 id → sensing|compute（元数据；连边不读角色）。"""
    roles: dict[int, str] = {}
    for plane in orbit["planes"]:
        role = str(plane["role"])
        for node in plane["nodes"]:
            roles[int(node["id"])] = role
    return roles


def graph_is_connected(n: int, adj: list[list[int]]) -> bool:
    if n == 0:
        return True
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n
