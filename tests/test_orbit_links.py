"""轨道连边快照：边生成、图 BFS、冒烟跑通。"""
from __future__ import annotations

from pathlib import Path

import pytest

from satsim.__main__ import run_spec
from satsim.input.load import resolve_spec
from satsim.input.schema import SpecError, validate_base
from satsim.method.engine.world import _bfs_path, build_adjacency, ring_adjacency
from satsim.physics.orbit_links import (
    adjacency_from_edges,
    build_orbit_edges,
    graph_is_connected,
    node_roles,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "input" / "scenarios" / "orbit_planes_smoke_v1.yaml"
BASE_ORBIT = ROOT / "input" / "scenarios" / "construct_orbit_planes_s1.yaml"
RING_SMOKE = ROOT / "input" / "scenarios" / "poisson_smoke_v1.yaml"


def _orbit_fixture() -> dict:
    return {
        "earth_radius_km": 6371.0,
        "distance_threshold_km": 1500.0,
        "same_plane_neighbors": True,
        "plane_ring": True,
        "planes": [
            {
                "id": "sensing",
                "role": "sensing",
                "altitude_km": 500,
                "inclination_deg": 97.0,
                "raan_deg": 0.0,
                "nodes": [
                    {"id": 0, "true_anomaly_deg": 0},
                    {"id": 1, "true_anomaly_deg": 40},
                ],
            },
            {
                "id": "compute",
                "role": "compute",
                "altitude_km": 550,
                "inclination_deg": 97.0,
                "raan_deg": 0.0,
                "nodes": [
                    {"id": 2, "true_anomaly_deg": 5},
                    {"id": 3, "true_anomaly_deg": 45},
                ],
            },
        ],
    }


def test_build_orbit_edges_grid_like():
    edges = build_orbit_edges(_orbit_fixture(), 4)
    assert (0, 1) in edges  # sensing 邻接
    assert (2, 3) in edges  # compute 邻接
    assert (0, 2) in edges and (1, 3) in edges  # 跨壳近距
    adj = adjacency_from_edges(4, edges)
    assert graph_is_connected(4, adj)
    roles = node_roles(_orbit_fixture())
    assert roles[0] == "sensing" and roles[2] == "compute"


def test_bfs_on_orbit_graph_and_ring_unchanged():
    adj = adjacency_from_edges(4, build_orbit_edges(_orbit_fixture(), 4))
    path = _bfs_path(adj, 0, 3)
    assert path[0][0] == 0 and path[-1][1] == 3
    # 环：4 节点 0→2 仍是两跳之一（低 id 邻接优先）
    ring = ring_adjacency(4)
    rpath = _bfs_path(ring, 0, 2)
    assert rpath == [(0, 1), (1, 2)] or rpath == [(0, 3), (3, 2)]


def test_build_adjacency_from_base_yaml():
    import yaml

    cfg = yaml.safe_load(BASE_ORBIT.read_text())
    validate_base(cfg)
    adj = build_adjacency(cfg)
    assert graph_is_connected(4, adj)
    assert _bfs_path(adj, 0, 3)


def test_ring_rejects_orbit_block():
    import copy
    import yaml

    cfg = yaml.safe_load((ROOT / "input" / "scenarios" / "construct_s1.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["orbit"] = _orbit_fixture()
    with pytest.raises(SpecError, match="ring 时不准写 orbit"):
        validate_base(cfg)


def test_orbit_planes_smoke_runs():
    spec, cfg, jobs = resolve_spec(SMOKE)
    assert cfg["topology"] == "orbit_planes"
    assert len(jobs) >= 1
    payload = run_spec(SMOKE, outdir=None)
    assert payload["T_SWAP"] == pytest.approx(40.0)
    assert "place_once" in payload["claims"]["policies"]
    assert "rebalance_free" in payload["claims"]["policies"]


def test_ring_smoke_still_works():
    payload = run_spec(RING_SMOKE, outdir=None)
    assert payload["T_SWAP"] == pytest.approx(40.0)
