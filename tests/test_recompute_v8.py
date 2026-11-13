"""Unit tests for discrete_rh_greedy placement fix and H3 token-recompute migration (v8)."""
from __future__ import annotations

from pathlib import Path

import simpy

from satsim.cosim.policies import make_policy
from satsim.cosim.run import load_config, summarize
from satsim.cosim.trace import Trace
from satsim.cosim.world import CRequest, CoWorld


def _tiny_cfg(isl_gbps: float = 1.0, t_prefill_tok_s: float = 0.005) -> dict:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/cosim_discrete_power_w53_v1.yaml")
    cfg["gpu"]["t_prefill_tok_s"] = t_prefill_tok_s
    cfg["net"]["isl_gbps"] = isl_gbps
    cfg["power"]["min_dwell_s"] = 20.0
    cfg["power"]["tiers"] = [
        {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
        {"name": "low", "power_w": 75.0, "max_batch": 4, "clock_f": 0.78, "prefill_clock_f": 0.30},
        {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
    ]
    return cfg


def test_discrete_rh_greedy_placement_not_none() -> None:
    cfg = _tiny_cfg()
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    env = simpy.Environment()
    p = make_policy({"policy": "discrete_rh_greedy"})
    w = CoWorld(env, tr, cfg, p)
    r = CRequest(i=0, t=0.5, origin=tr.rs[0], pin=4000, pout=2000, long=True, deadline=1000.0, upload_bytes=1e6)
    env.run(until=r.t)
    assert p.place(w, r) is not None


def test_world_recompute_migration_from_decode_and_prefill() -> None:
    cfg = _tiny_cfg(isl_gbps=1.0, t_prefill_tok_s=0.002)
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    env = simpy.Environment()
    p = make_policy({"place": "orbit_thermal", "migrate": "none"})
    w = CoWorld(env, tr, cfg, p)
    c0, c1 = tr.compute[0], tr.compute[1]
    r = CRequest(i=0, t=0.1, origin=tr.rs[0], pin=2000, pout=5000, long=True, deadline=1000.0, upload_bytes=1e6)
    w.submit(r)
    # Run until request is in decode with some generated tokens
    env.run(until=20.0)
    assert r.state == "decode"
    assert r.gen > 0
    dst = c1 if r.node == c0 else c0
    ok = w.migrate(r, dst, mode="recompute")
    assert ok
    env.run(until=500.0)
    assert r.n_recompute == 1
    assert w.acc["recompute_mig"] == 1
    assert w.acc["recompute_bytes"] < 100_000.0  # KB-scale, not GB-scale
    assert r.state == "done"
    assert r.remaining < 1e-9


def test_discrete_rh_tier_v8_instantiates_and_runs() -> None:
    cfg = _tiny_cfg(isl_gbps=1.0, t_prefill_tok_s=0.005)
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    p = make_policy({"policy": "discrete_rh_tier_v8"})
    assert p.id == "discrete_rh_tier_v8"
    assert p.recompute_aware is True
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, p)
    for k in range(4):
        w.submit(CRequest(i=k, t=0.5 + k * 0.2, origin=tr.rs[0], pin=4000, pout=1000, long=True, deadline=500.0, upload_bytes=1e6))
    env.run(until=120.0)
    s = summarize(w)
    assert s["energy_kj_total"] > 0.0
