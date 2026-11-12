from types import SimpleNamespace
import pytest
from satsim.cosim.trace import Trace
from satsim.cosim.thermal_rebalance import TorchRebalance


def test_p0_1_tether_3_hop_formula():
    """Verify tether time does not multiply transmission time by hop count (FluidNet alignment)."""
    tr = Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "test_3hop",
        "horizon_s": 1000.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
            {"id": 4, "role": "compute"},
        ],
        "edges": [
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 1000]]},
            {"u": 2, "v": 3, "kind": "intra", "up": [[0, 1000]]},
            {"u": 3, "v": 4, "kind": "intra", "up": [[0, 1000]]},
        ],
        "dark": [[], [], [], [], []],
    })

    # 3 hops between node 1 and node 4
    p = tr.path(1, 4, 10.0)
    assert p is not None
    assert len(p[1]) == 3

    bw = 1.25e9  # bytes/s (10 Gbps)
    net = SimpleNamespace(cap=[bw, bw, bw])
    w = SimpleNamespace(tr=tr, net=net, hop_lat=0.005)

    r = SimpleNamespace(
        anchor_node=1,
        reaccess_remaining=3,
        reaccess_tile_bytes=125_000_000.0,  # 125 MB -> 0.1s transmission
    )

    rebalancer = TorchRebalance(placement={"policy": "nearest"}, tether_aware=True)
    pull_s, edges = rebalancer._tether_extra_s(w, node=4, r=r, now=10.0)

    # 3 pulls * (125MB / 1.25GBps + 0.005s * 3 hops) = 3 * (0.1 + 0.015) = 0.345s
    # Note: old buggy formula would give 3 * (0.1 * 3 + 0.015) = 0.945s
    expected_pull_s = 3 * (125_000_000.0 / bw + 0.005 * 3)
    assert pull_s == pytest.approx(expected_pull_s)
    assert pull_s == pytest.approx(0.345)
    assert edges == tuple(p[1])

    # Also test no-path fallback: 4 hops estimate
    pull_s_nopath, edges_nopath = rebalancer._tether_extra_s(w, node=0, r=r, now=10.0)
    expected_nopath = 3 * (125_000_000.0 / bw + 0.005 * 4)
    assert pull_s_nopath == pytest.approx(expected_nopath)
    assert pull_s_nopath == pytest.approx(0.360)
    assert edges_nopath == ()
