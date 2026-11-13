"""Unit tests for discrete_rh_tier_v11 (dwell_thermal_gate + isolate_lo_evict)."""
from __future__ import annotations

from pathlib import Path

import simpy

from satsim.cosim.policies import make_policy
from satsim.cosim.run import load_config, summarize
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


def test_v11_dwell_thermal_gate_keeps_cool_sunlit_node_at_high_tier() -> None:
    cfg = _tiny_cfg()
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    p = make_policy({"policy": "discrete_rh_tier_v11"})
    assert p.id == "discrete_rh_tier_v11"
    assert p.dwell_thermal_gate is True
    assert p.isolate_lo_evict is True
    assert p.prefill_traj_aware is True

    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, p)
    sunlit = next(j for j in tr.compute if not tr.is_dark(j, 0.5))
    r = CRequest(i=0, t=0.5, origin=tr.rs[0], pin=1000, pout=2000, long=True, deadline=2000.0, upload_bytes=1e6)
    w.nodes[sunlit].append(r)
    r.node = sunlit
    r.state = "decode"
    env.run(until=0.5)
    tier = p._select_explicit_tier_for_node(w, sunlit)
    assert tier is None


def test_v11_runs_end_to_end_narrow_isl() -> None:
    cfg = _tiny_cfg(isl_gbps=1.5, t_prefill_tok_s=0.005)
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    p = make_policy({"policy": "discrete_rh_tier_v11"})
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, p)
    for k in range(6):
        w.submit(
            CRequest(
                i=k,
                t=0.5 + k * 0.5,
                origin=tr.rs[0],
                pin=4000,
                pout=2000,
                long=True,
                deadline=600.0,
                urgent=(k % 3 == 0),
                upload_bytes=1e6,
            )
        )
    env.run(until=180.0)
    s = summarize(w)
    assert s["energy_kj_total"] > 0.0
    assert s["good_long_frac"] > 0.0


def test_v11_isolate_lo_evict_candidates() -> None:
    cfg = _tiny_cfg(isl_gbps=10.0, t_prefill_tok_s=0.005)
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    p = make_policy({"policy": "discrete_rh_tier_v11"})
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, p)
    sunlit = [j for j in tr.compute if not tr.is_dark(j, 1.0)]
    c0, c1 = sunlit[0], sunlit[1]
    r_urg = CRequest(i=0, t=0.0, origin=tr.rs[0], pin=1000, pout=2000, long=True, deadline=400.0, urgent=True, upload_bytes=1e6)
    r_norm = CRequest(i=1, t=0.0, origin=tr.rs[0], pin=1000, pout=2000, long=True, deadline=1000.0, urgent=False, upload_bytes=1e6)
    r_c1 = CRequest(i=2, t=0.0, origin=tr.rs[0], pin=1000, pout=2000, long=True, deadline=1000.0, urgent=False, upload_bytes=1e6)
    for r, node in ((r_urg, c0), (r_norm, c0), (r_c1, c1)):
        r.node = node
        r.state = "decode"
        w.nodes[node].append(r)
    env.run(until=1.0)
    cands = p._collect_candidates(w)
    assert any(c.req is r_norm and c.src == c0 for c in cands)
