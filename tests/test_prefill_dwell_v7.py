"""Unit tests for H1 (prefill_clock_f wiring), H2 (min_dwell_s), and discrete_rh_tier_v7."""
from __future__ import annotations

from pathlib import Path

import simpy

from satsim.cosim.policies import make_policy
from satsim.cosim.power_model import DiscreteTierModel
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, CRequest


def _tiny_trace() -> Trace:
    return Trace.load(Path(__file__).resolve().parents[1] / "input/traces/coop-w53.trace.json")


def _base_cfg(*, min_dwell_s: float = 0.0, t_prefill_tok_s: float = 0.0) -> dict:
    return {
        "delivery": "final",
        "control_period_s": 2.0,
        "frame_s": 10.0,
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": 40.0,
            "t_fixed_s": 0.003,
            "t_weights_s": 0.01,
            "t_kv_tok_s": 1.0e-7,
            "t_seq_s": 5.0e-5,
            "mem_freq_exp": 0.6,
            "t_prefill_tok_s": t_prefill_tok_s,
        },
        "power": {
            "mode": "discrete_tiers",
            "P_dark": 80.0,
            "prefer_high_clock": True,
            "min_dwell_s": min_dwell_s,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
                {"name": "low", "power_w": 75.0, "max_batch": 4, "clock_f": 0.78, "prefill_clock_f": 0.30},
                {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
            ],
        },
        "thermal": {
            "tau_s": 60.0,
            "R_th": 0.25,
            "T_amb_C": 20.0,
            "T0_C": 42.0,
            "T_max_C": 76.0,
            "thermal_horizon_s": 30.0,
        },
        "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 10.0}, "hop_latency_s": 0.005},
        "migration": {"timeout_s": 120.0, "switch_s": 0.02},
        "slo": {"deadline_factor_long": 2.0, "deadline_factor_short": 4.0, "min_deadline_s": 120, "max_stall_frac": 0.1},
        "policy": {"stall_weight": 1.0},
    }


def test_min_dwell_blocks_voluntary_switch_but_allows_eclipse_override() -> None:
    cfg = _base_cfg(min_dwell_s=30.0)
    pm = DiscreteTierModel(cfg, 1)
    # Enter high tier (idx=2) at t=0.0
    pm.sync_node(0, 0.0, 1, is_dark=False)
    assert pm._tier_idx[0] == 2

    # Voluntary explicit switch to low tier at t=10.0 (< 30s) must be held at tier 2
    pm.set_explicit_tier(0, 1)
    pm.sync_node(0, 10.0, 1, is_dark=False, prev_batch=1)
    assert pm._tier_idx[0] == 2

    # Once dwell >= 30s (t=30.0), voluntary switch to tier 1 succeeds
    pm.sync_node(0, 30.0, 1, is_dark=False, prev_batch=1)
    assert pm._tier_idx[0] == 1

    # Eclipse hardware cap overrides dwell even at t=32.0 if on high tier
    pm.set_explicit_tier(0, None)
    pm._tier_idx[0] = 2
    pm._power_w[0] = 300.0
    pm._tier_entered_t[0] = 31.0
    pm.sync_node(0, 32.0, 1, is_dark=True, prev_batch=1)
    assert pm._tier_idx[0] == 1


def test_prefill_clock_f_is_used_in_coworld_life() -> None:
    # Run identical 1000-token prefill + 10-token decode on explicit high vs low tier
    tr = _tiny_trace()
    rs_origin = tr.rs[0]
    compute_node = tr.compute[0]

    def _run_with_explicit(tier_idx: int) -> float:
        cfg = _base_cfg(t_prefill_tok_s=0.01)  # 1000 * 0.01 = 10s at high (1.0), 33.33s at low (0.30)
        env = simpy.Environment()
        pol = make_policy({"policy": "least_loaded"})
        w = CoWorld(env, tr, cfg, pol)
        for j in tr.compute:
            w.power_model.set_explicit_tier(j, tier_idx)
        req = CRequest(i=0, origin=rs_origin, t=0.0, long=True, pin=1000, pout=10, upload_bytes=0.0, deadline=500.0)
        w.submit(req)
        env.run(until=200.0)
        assert req.t_decoded is not None
        return float(req.t_decoded)

    t_high = _run_with_explicit(2)
    t_low = _run_with_explicit(1)
    # High prefill ~10s, low prefill ~33.33s -> difference > 20s
    assert t_low - t_high > 20.0


def test_v7_prefill_burst_holds_high_during_prefill_then_drops_to_low() -> None:
    pol = make_policy({"policy": "discrete_rh_tier_v7"})
    assert pol.id == "discrete_rh_tier_v7"
    assert pol.prefill_burst is True

    cfg = _base_cfg(t_prefill_tok_s=0.01)
    tr = _tiny_trace()
    c0 = tr.compute[0]
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, pol)
    req = CRequest(i=0, origin=tr.rs[0], t=0.0, long=True, pin=1000, pout=1000, upload_bytes=0.0, deadline=500.0)
    w.nodes[c0].append(req)
    req.node = c0
    req.state = "prefill"
    req.prefill_rem = 500.0

    # During prefill, v7 must return None (high tier 300W)
    assert pol._select_explicit_tier_for_node(w, c0) is None

    # After prefill finishes and request enters decode with ample slack, v7 selects low tier (1)
    req.prefill_rem = 0.0
    req.state = "decode"
    assert pol._select_explicit_tier_for_node(w, c0) == 1


def test_v9_prefill_dwell_place_avoids_low_locked_node_and_protects_prefill_mig() -> None:
    pol = make_policy({"policy": "discrete_rh_tier_v9"})
    assert pol.id == "discrete_rh_tier_v9"
    assert pol.prefill_dwell_place is True
    assert pol.dwell_aware_tier is True
    assert pol.prefill_aware_mig is True

    cfg = _base_cfg(min_dwell_s=20.0, t_prefill_tok_s=0.005)
    tr = _tiny_trace()
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, pol)

    # Lock compute[0] in low tier (tier_idx=1) at t=0, then submit at t=2 (within 20s dwell)
    from satsim.cosim.policies import upload_options

    req = CRequest(i=0, origin=tr.rs[0], t=2.0, long=True, pin=4000, pout=1000, upload_bytes=0.0, deadline=200.0)
    opts = list(upload_options(w, req))
    assert len(opts) >= 2
    locked_node = opts[0][0]
    w.power_model._tier_idx[locked_node] = 1
    w.power_model._tier_entered_t[locked_node] = 0.0
    env.run(until=2.0)

    chosen = pol.place(w, req)
    assert chosen is not None
    assert chosen != locked_node
