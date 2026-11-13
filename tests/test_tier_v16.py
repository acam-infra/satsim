"""Unit tests for discrete_rh_tier_v16 (slack_link_place + v13/v14/v15 synergy)."""
from __future__ import annotations

from pathlib import Path

import simpy

from satsim.cosim.policies import make_policy, upload_options
from satsim.cosim.run import load_config
from satsim.cosim.trace import Trace
from satsim.cosim.world import CRequest, CoWorld


def _tiny_cfg(isl_gbps: float = 1.5, t_prefill_tok_s: float = 0.005) -> dict:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/cosim_discrete_power_w53_v1.yaml")
    cfg["gpu"]["t_prefill_tok_s"] = t_prefill_tok_s
    cfg["net"]["link_gbps"]["intra"] = isl_gbps
    cfg["net"]["link_gbps"]["inter"] = isl_gbps
    cfg["power"]["min_dwell_s"] = 20.0
    cfg["power"]["tiers"] = [
        {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
        {"name": "low", "power_w": 75.0, "max_batch": 4, "clock_f": 0.78, "prefill_clock_f": 0.30},
        {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
    ]
    return cfg


def test_v16_policy_contract() -> None:
    p = make_policy({"policy": "discrete_rh_tier_v16"})
    assert p.id == "discrete_rh_tier_v16"
    assert p.slack_link_place is True
    assert p.link_aware_isolate is True
    assert p.wakeup_aware_mig is True
    assert p.hopeless_mig_guard is True
    assert p.wakeup_aware_place is True
    assert p.warm_high_reserve == 2


def test_v16_tight_urgent_prefers_earliest_finish_over_urg_mix() -> None:
    """When an urgent request has a tight deadline, v16 should not detour far over narrowband ISL just to avoid urg_mix."""
    cfg = _tiny_cfg(isl_gbps=1.5, t_prefill_tok_s=0.008)
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    p = make_policy({"policy": "discrete_rh_tier_v16"})
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, p)
    origin = tr.rs[0]
    c0 = tr.compute[0]
    r_norm = CRequest(i=1, t=0.0, deadline=60.0, long=True, origin=origin, pin=256, pout=256, upload_bytes=1e6, urgent=False)
    r_norm.node = c0
    r_norm.state = "decode"
    w.nodes[c0].append(r_norm)
    r_urg = CRequest(i=2, t=0.0, deadline=5.0, long=True, origin=origin, pin=256, pout=256, upload_bytes=1e6, urgent=True)
    dst = p.place(w, r_urg)
    assert dst is not None
    opts = dict(upload_options(w, r_urg))
    assert dst in opts
    assert opts[dst] <= min(opts.values()) + 1.0
