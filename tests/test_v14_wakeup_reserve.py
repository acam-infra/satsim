"""Unit tests for discrete_rh_tier_v14: wakeup_aware_place, warm_high_reserve, and urg_fast_place."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from satsim.cosim.discrete_milp import DiscreteThermalMigrator
from satsim.cosim.policies import make_policy
from satsim.cosim.power_model import DiscreteTierModel
from satsim.cosim.run import load_config


class _DummyEnv:
    def __init__(self, now: float = 0.0):
        self.now = float(now)


class _DummyTr:
    def __init__(self, compute=(0, 1, 2)):
        self.compute = list(compute)
        self.dark = set()

    def is_dark(self, node: int, t: float) -> bool:
        return node in self.dark

    def power_segments(self, node: int, t0: float):
        return [(t0, t0 + 1000.0, node in self.dark)]

    def next_conn(self, origin: int, node: int, t_dec: float) -> float:
        return t_dec

    def connected(self, origin: int, node: int, t: float) -> bool:
        return True


class _Req:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _make_cfg():
    base = load_config(Path(__file__).resolve().parents[1] / "configs/cosim_discrete_power_w53_v1.yaml")
    gpu = dict(base["gpu"])
    gpu["t_prefill_tok_s"] = 0.005
    return {
        "workload": {"long_pin": 1000, "long_out": (100, 100), "short_out": (20, 20)},
        "gpu": gpu,
        "power": {
            "mode": "discrete_tiers",
            "P_dark": 80.0,
            "wakeup_energy_j": 2500.0,
            "wakeup_latency_s": 3.0,
            "min_dwell_s": 20.0,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
                {"name": "low", "power_w": 75.0, "max_batch": 4, "clock_f": 0.78, "prefill_clock_f": 0.30},
                {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
            ],
        },
        "thermal": {"T_amb_C": 25.0, "T_max_C": 76.0, "R_th": 0.16, "tau_s": 60.0, "hysteresis_C": 4.0},
    }


def test_v14_factory_and_id():
    pol = make_policy({"policy": "discrete_rh_tier_v15"})
    assert isinstance(pol, DiscreteThermalMigrator)
    assert pol.id == "discrete_rh_tier_v15"
    assert pol.wakeup_aware_place is True
    assert pol.warm_high_reserve == 2
    assert pol.urg_fast_place is True


def test_wakeup_aware_place_prefers_warm_active_over_0w():
    cfg = _make_cfg()
    tr = _DummyTr(compute=(0, 1))
    pm = DiscreteTierModel(cfg, n=2)
    # Node 0 is active at high tier with 1 prefilling non-urgent request; Node 1 is 0W idle.
    pm._tier_idx[0] = 2
    pm._tier_idx[1] = 0
    r_active = _Req(i=1, origin=0, t=0.0, pin=500, kv=500, gen=10, long=True, urgent=False, deadline=500.0, state="prefill", prefill_rem=100.0)
    r_new = _Req(i=2, origin=0, t=10.0, pin=500, kv=0, gen=0, long=True, urgent=False, deadline=500.0)
    w = SimpleNamespace(
        env=_DummyEnv(10.0),
        tr=tr,
        cfg=cfg,
        gpu=cfg["gpu"],
        power_model=pm,
        nodes={0: [r_active], 1: []},
        iter_of=lambda j, req: 0.05,
        prefill_f_of=lambda j, t: 1.0,
    )
    pol = make_policy({"policy": "discrete_rh_tier_v15"})
    pol.placement = SimpleNamespace(_delivery=lambda _w, _r, tok, j, pre: 10.0 + pre + tok * 0.05)
    # Monkeypatch upload_options via _prefill_dwell_place: origin=0, nodes 0 and 1 both reachable
    import satsim.cosim.policies as pol_mod
    orig_upload = pol_mod.upload_options
    pol_mod.upload_options = lambda _w, _r: [(0, 0.0), (1, 0.0)]
    try:
        dst = pol._prefill_dwell_place(w, r_new)
    finally:
        pol_mod.upload_options = orig_upload
    assert dst == 0, f"Expected warm node 0 to be preferred over 0W idle node 1, got {dst}"


def test_warm_high_reserve_keeps_singleton_cool_node_high():
    cfg = _make_cfg()
    tr = _DummyTr(compute=(0, 1, 2))
    pm = DiscreteTierModel(cfg, n=3)
    pm._tier_idx[0] = 2
    pm._tier_idx[1] = 2
    pm._tier_idx[2] = 0
    r0 = _Req(i=1, origin=0, t=0.0, pin=500, kv=600, gen=10, long=True, urgent=False, deadline=500.0, state="decode", prefill_rem=0.0)
    r1 = _Req(i=2, origin=1, t=0.0, pin=500, kv=600, gen=10, long=True, urgent=False, deadline=500.0, state="decode", prefill_rem=0.0)
    w = SimpleNamespace(
        env=_DummyEnv(10.0),
        tr=tr,
        cfg=cfg,
        gpu=cfg["gpu"],
        power_model=pm,
        nodes={0: [r0], 1: [r1], 2: []},
    )
    pol = make_policy({"policy": "discrete_rh_tier_v15"})
    # With 2 active sunlit non-low nodes <= warm_high_reserve(2), node 0 stays high (None)
    assert pol._select_explicit_tier_for_node(w, 0) is None
    # With batch=2 on node 0, warm_high_reserve only protects batch <= 1, so node 0 can drop to low (tier 1)
    r0b = _Req(i=3, origin=0, t=0.0, pin=500, kv=600, gen=10, long=True, urgent=False, deadline=500.0, state="decode", prefill_rem=0.0)
    w.nodes[0].append(r0b)
    assert pol._select_explicit_tier_for_node(w, 0) == 1
