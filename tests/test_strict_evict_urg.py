"""Unit tests for strict_evict_urg and decode_only_low_tier."""
from __future__ import annotations

from pathlib import Path

import simpy

from satsim.cosim.policies import make_policy
from satsim.cosim.run import load_config
from satsim.cosim.trace import Trace
from satsim.cosim.world import CRequest, CoWorld


def _tiny_cfg() -> dict:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/cosim_discrete_power_w53_v1.yaml")
    cfg["gpu"]["t_prefill_tok_s"] = 0.005
    cfg["net"]["link_gbps"]["intra"] = 10.0
    cfg["net"]["link_gbps"]["inter"] = 10.0
    cfg["power"]["min_dwell_s"] = 20.0
    cfg["power"]["tiers"] = [
        {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
        {"name": "low", "power_w": 75.0, "max_batch": 4, "clock_f": 0.78, "prefill_clock_f": 0.30},
        {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
    ]
    return cfg


def test_strict_evict_urg_places_explicit_urg_isolated_even_under_tight_slack() -> None:
    cfg = _tiny_cfg()
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    p = make_policy({"policy": "discrete_rh_tier_v11", "strict_evict_urg": True})
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, p)
    sunlit = [j for j in tr.compute if not tr.is_dark(j, 1.0)]
    c0 = sunlit[0]

    # Put a tight-slack NORMAL request on c0 (would look urgent to _is_urgent, but urgent=False)
    r_tight_norm = CRequest(
        i=0, t=0.0, origin=tr.rs[0], pin=1000, pout=2000, long=True, deadline=30.0, urgent=False, upload_bytes=1e6
    )
    r_tight_norm.node = c0
    r_tight_norm.state = "decode"
    w.nodes[c0].append(r_tight_norm)
    env.run(until=1.0)

    # An explicit urgent request should avoid c0 (since c0 has a normal request)
    r_urg = CRequest(
        i=1, t=1.0, origin=tr.rs[0], pin=1000, pout=2000, long=True, deadline=600.0, urgent=True, upload_bytes=1e6
    )
    dst = p.place(w, r_urg)
    assert dst is not None and dst != c0


def test_decode_only_low_tier_allows_slack_rich_decode_but_blocks_prefill() -> None:
    cfg = _tiny_cfg()
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    p = make_policy({"policy": "discrete_rh_tier_v12"})
    assert p.id == "discrete_rh_tier_v12"
    assert p.strict_evict_urg is True
    assert p.decode_only_low_tier is True
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, p)
    sunlit = next(j for j in tr.compute if not tr.is_dark(j, 0.5))

    r_dec = CRequest(i=0, t=0.5, origin=tr.rs[0], pin=1000, pout=2000, long=True, deadline=2000.0, upload_bytes=1e6)
    w.nodes[sunlit].append(r_dec)
    r_dec.node = sunlit
    r_dec.state = "decode"
    env.run(until=0.5)
    assert p._select_explicit_tier_for_node(w, sunlit) == 1

    r_pref = CRequest(i=1, t=0.5, origin=tr.rs[0], pin=1000, pout=2000, long=True, deadline=2000.0, upload_bytes=1e6)
    w.nodes[sunlit].append(r_pref)
    r_pref.node = sunlit
    r_pref.state = "prefill"
    r_pref.prefill_rem = 500.0
    assert p._select_explicit_tier_for_node(w, sunlit) is None
