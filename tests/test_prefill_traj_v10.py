"""Unit tests for discrete_rh_tier_v10: prefill trajectory awareness, selective prefill rescue, and safe urg_mix ordering."""
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
        return {0: [(1, 0)], 1: [(0, 0)]}

    def edge_up_until(self, e: int, t: float) -> float:
        return t + 1000.0


class _DummyNet:
    def __init__(self, cap_bps: float = 1.5e9):
        self.cap = [cap_bps]
        self.flows = []


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


class _Req:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _make_req(i: int, *, pin: int = 1000, prefill_rem: float = 1000.0, deadline: float = 40.0, urgent: bool = False):
    return _Req(
        i=i,
        t=0.0,
        origin=0,
        long=True,
        pin=pin,
        gen=0,
        kv=pin,
        remaining=100.0,
        prefill_rem=prefill_rem,
        state="prefill" if prefill_rem > 0 else "decode",
        deadline=deadline,
        urgent=urgent,
        migrating=False,
        n_mig=0,
        anchor_node=-1,
        reaccess_remaining=0,
        reaccess_tile_bytes=0.0,
    )


def test_v10_policy_factory_and_id():
    pol = make_policy({"policy": "discrete_rh_tier_v10"})
    assert isinstance(pol, DiscreteThermalMigrator)
    assert pol.id == "discrete_rh_tier_v10"
    assert pol.selective_prefill_rescue is True
    assert pol.prefill_traj_aware is True
    assert pol.safe_urg_mix_order is False


def test_v10_prefill_traj_aware_penalizes_contention():
    cfg = _make_cfg()
    pm = DiscreteTierModel(cfg, n=2)
    r1 = _make_req(1, pin=1000, prefill_rem=1000.0, deadline=25.0)
    r2 = _make_req(2, pin=1000, prefill_rem=1000.0, deadline=25.0)
    w = SimpleNamespace(
        cfg=cfg,
        env=_DummyEnv(0.0),
        tr=_DummyTr(),
        net=_DummyNet(),
        gpu=cfg["gpu"],
        power_model=pm,
        thermal=None,
        thermal_f={0: 1.0, 1: 1.0},
        nodes={0: [r1, r2], 1: []},
        reserved={0: 0.0, 1: 0.0},
        bpt=1024,
        kv_cap=100_000_000,
        hop_lat=0.005,
        switch_s=0.1,
        mig_timeout=30.0,
        f_dark=0.3,
        kv_used=lambda j: 0.0,
        fits=lambda j, r: True,
        prefill_f_of=lambda j, t=None: 1.0,
        iter_of=lambda j, r=None: 0.02,
        upload_s=lambda j, r: 0.0,
    )
    v9 = make_policy({"policy": "discrete_rh_tier_v9"})
    v10 = make_policy({"policy": "discrete_rh_tier_v10"})
    _, fin_v9 = v9._eval_node_for_mig(w, 0, [r1, r2], 0.0, {})
    _, fin_v10 = v10._eval_node_for_mig(w, 0, [r1, r2], 0.0, {})
    # With 2 concurrent 1000-token prefills at 0.008 s/tok, prefill adds 16.0s
    assert fin_v10[r1] > fin_v9[r1] + 10.0


def test_v10_selective_prefill_rescue_blocks_healthy_single_but_allows_contended_or_urgent():
    cfg = _make_cfg()
    pm = DiscreteTierModel(cfg, n=2)
    pm.set_explicit_tier(0, 2)  # high tier (300W)
    pm._tier_idx[0] = 2

    def _build_world(reqs_0, reqs_1=None):
        return SimpleNamespace(
            cfg=cfg,
            env=_DummyEnv(1.0),
            tr=_DummyTr(),
            net=_DummyNet(),
            gpu=cfg["gpu"],
            power_model=pm,
            thermal=None,
            thermal_f={0: 1.0, 1: 1.0},
            nodes={0: reqs_0, 1: list(reqs_1 or [])},
            reserved={0: 0.0, 1: 0.0},
            bpt=1024,
            kv_cap=100_000_000,
            hop_lat=0.005,
            switch_s=0.1,
            mig_timeout=30.0,
            f_dark=0.3,
            kv_used=lambda j: 0.0,
            fits=lambda j, r: True,
            prefill_f_of=lambda j, t=None: 1.0,
            iter_of=lambda j, r=None: 0.02,
        )

    v10 = make_policy({"policy": "discrete_rh_tier_v10"})

    # 1. Healthy single non-urgent prefill on high-tier sunlit node -> no candidates
    r_single = _make_req(1, pin=1000, prefill_rem=800.0, deadline=100.0, urgent=False)
    w1 = _build_world([r_single])
    assert v10._collect_candidates(w1) == []

    # 2. Contended prefill (n_pref == 2) -> allowed to migrate via recompute to active non-prefilling node 1
    pm.set_explicit_tier(1, 2)
    pm._tier_idx[1] = 2
    r_a = _make_req(1, pin=1000, prefill_rem=900.0, deadline=15.0, urgent=False)
    r_b = _make_req(2, pin=1000, prefill_rem=900.0, deadline=15.0, urgent=False)
    r_dec = _make_req(3, pin=100, prefill_rem=0.0, deadline=100.0, urgent=False)
    w2 = _build_world([r_a, r_b], [r_dec])
    cands = v10._collect_candidates(w2)
    assert any(c.meta.get("mode") == "recompute" and c.dst == 1 for c in cands)

    # 3. Already recomputed once (n_recompute >= 1) -> blocked on sunlit cool node to prevent ping-pong
    r_a.n_recompute = 1
    r_b.n_recompute = 1
    assert not any(c.src == 0 for c in v10._collect_candidates(w2))
