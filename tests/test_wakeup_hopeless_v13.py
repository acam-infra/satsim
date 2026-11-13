"""Unit tests for discrete_rh_tier_v13: wakeup-aware migration and hopeless-request guard."""
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
    def __init__(self, compute=(0, 1)):
        self.compute = list(compute)
        self.dark = set()

    def is_dark(self, node: int, t: float) -> bool:
        return node in self.dark

    def power_segments(self, node: int, t0: float):
        return [(t0, t0 + 1000.0, node in self.dark)]

    def next_conn(self, origin: int, node: int, t_dec: float) -> float:
        return t_dec

    def path(self, src: int, dst: int, t: float):
        if src == dst:
            return ([src], [])
        return ([src, dst], [0])

    def adjacency(self, t: float):
        return {0: [(1, 0), (2, 0)], 1: [(0, 0), (2, 0)], 2: [(0, 0), (1, 0)]}

    def edge_up_until(self, e: int, t: float) -> float:
        return t + 1000.0

    def connected(self, origin: int, node: int, t: float) -> bool:
        return True


class _DummyNet:
    def __init__(self, cap_bps: float = 1.5e9):
        self.cap = [cap_bps]
        self.flows = []


class _Req:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _make_cfg():
    base = load_config(Path(__file__).resolve().parents[1] / "configs/cosim_discrete_power_w53_v1.yaml")
    gpu = dict(base["gpu"])
    gpu["t_prefill_tok_s"] = 0.008
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


def _make_req(
    i: int,
    *,
    origin: int = 0,
    pin: int = 1000,
    prefill_rem: float = 1000.0,
    remaining: float = 100.0,
    deadline: float = 40.0,
    urgent: bool = False,
):
    return _Req(
        i=i,
        t=0.0,
        origin=origin,
        long=True,
        pin=pin,
        gen=0,
        kv=pin,
        remaining=remaining,
        prefill_rem=prefill_rem,
        state="prefill" if prefill_rem > 0 else "decode",
        deadline=deadline,
        urgent=urgent,
        migrating=False,
        n_mig=0,
        anchor_node=-1,
        reaccess_remaining=0,
        reaccess_tile_bytes=0.0,
        upload_bytes=1024.0,
    )


def _make_world(cfg, nodes_map, dark=(), tier_idx=(2, 0)):
    tr = _DummyTr(compute=tuple(sorted(nodes_map.keys())))
    tr.dark = set(dark)
    pm = DiscreteTierModel(cfg, n=len(nodes_map))
    for j, ti in enumerate(tier_idx):
        pm._tier_idx[j] = ti
        pm._power_w[j] = pm.spec.tiers[ti].power_w
        pm._clock_f[j] = pm.spec.tiers[ti].clock_f
        pm._prefill_clock_f[j] = pm.spec.tiers[ti].effective_prefill_f
    return SimpleNamespace(
        env=_DummyEnv(10.0),
        cfg=cfg,
        gpu=cfg["gpu"],
        tr=tr,
        net=_DummyNet(),
        nodes=nodes_map,
        reserved={j: 0.0 for j in nodes_map},
        kv_cap=100_000_000,
        bpt=1024,
        thermal=None,
        thermal_f={j: 1.0 for j in nodes_map},
        power_model=pm,
        hop_lat=0.005,
        switch_s=0.1,
        mig_timeout=30.0,
        f_dark=0.3,
        kv_used=lambda j: sum(x.kv for x in nodes_map[j]),
        fits=lambda j, r: True,
        prefill_f_of=lambda j, t=None: 0.3 if tr.is_dark(j, t or 10.0) else 1.0,
        iter_of=lambda j, r=None: 0.05,
        upload_s=lambda j, r: 0.0,
        prefill_clock_f=lambda j: pm._prefill_clock_f[j],
    )


def test_v13_policy_factory_and_id():
    pol = make_policy({"policy": "discrete_rh_tier_v13"})
    assert isinstance(pol, DiscreteThermalMigrator)
    assert pol.id == "discrete_rh_tier_v13"
    assert pol.wakeup_aware_mig is True
    assert pol.hopeless_mig_guard is True


def test_v13_hopeless_guard_blocks_solo_urgent_prefill_and_hopeless_mig():
    cfg = _make_cfg()
    r_urg = _make_req(1, pin=1000, prefill_rem=400.0, deadline=22.0, urgent=True)
    w = _make_world(cfg, {0: [r_urg], 1: []}, tier_idx=(2, 0))
    pol_v13 = make_policy({"policy": "discrete_rh_tier_v13"})
    assert pol_v13._collect_candidates(w) == []

    r_dead = _make_req(2, pin=1000, prefill_rem=0.0, deadline=5.0, urgent=False)
    r_other = _make_req(3, pin=1000, prefill_rem=0.0, deadline=40.0, urgent=False)
    w2 = _make_world(cfg, {0: [r_dead, r_other], 1: []}, tier_idx=(2, 0))
    cands = pol_v13._collect_candidates(w2)
    assert all(c.req is not r_dead for c in cands)


def test_v13_wakeup_aware_mig_penalizes_0w_cold_start():
    cfg = _make_cfg()
    cfg["workload"]["long_out"] = (2000, 2000)
    cfg["power"]["wakeup_latency_s"] = 8.0
    r_tight = _make_req(1, pin=1000, prefill_rem=0.0, deadline=38.5, urgent=True)
    bgs = [_make_req(i + 2, pin=4000, prefill_rem=0.0, deadline=60.0, urgent=False) for i in range(4)]
    w = _make_world(cfg, {0: [r_tight, *bgs], 1: []}, tier_idx=(2, 0))
    pol_v12 = make_policy({"policy": "discrete_rh_tier_v12"})
    pol_v13 = make_policy({"policy": "discrete_rh_tier_v13"})
    cands_v12 = [c for c in pol_v12._collect_candidates(w) if c.req is r_tight]
    cands_v13 = [c for c in pol_v13._collect_candidates(w) if c.req is r_tight]
    assert len(cands_v12) >= 1
    assert len(cands_v13) == 0
