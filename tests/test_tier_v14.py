"""Unit tests for discrete_rh_tier_v12 (link_aware_isolate + strict prefill rescue)."""
from __future__ import annotations

from pathlib import Path

import simpy

from satsim.cosim.policies import make_policy
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


def test_v12_policy_contract() -> None:
    p = make_policy({"policy": "discrete_rh_tier_v14"})
    assert p.id == "discrete_rh_tier_v14"
    assert p.link_aware_isolate is True
    assert p.isolate_lo_evict is True
    assert p.dwell_thermal_gate is True


def test_v12_does_not_rescue_solo_normal_prefill_on_tight_deadline() -> None:
    """With dl_long=2.0 (<=2.2), solo non-explicit-urgent prefill on cool sunlit node should not be migrated."""
    cfg = _tiny_cfg(isl_gbps=1.5)
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    p = make_policy({"policy": "discrete_rh_tier_v14"})
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, p)
    sunlit = [j for j in tr.compute if not tr.is_dark(j, 1.0)]
    c0 = sunlit[0]
    # dl_long = 2.0 -> tight slack, but NOT explicitly urgent
    r = CRequest(i=0, t=0.0, origin=tr.rs[0], pin=4000, pout=2000, long=True, deadline=60.0, urgent=False, upload_bytes=1e6)
    r.node = c0
    r.state = "prefill"
    r.prefill_rem = 2000.0
    w.nodes[c0].append(r)
    env.run(until=1.0)
    cands = p._collect_candidates(w)
    assert not any(c.req is r for c in cands)


def test_v12_blocks_isolate_lo_evict_on_narrow_isl_when_link_slow() -> None:
    """On 1.5 Gbps narrowband ISL, isolate_lo should not force KV eviction when relief_s <= 0.01 and kv_link_s > relief_t."""
    cfg = _tiny_cfg(isl_gbps=1.5)
    tr = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    p = make_policy({"policy": "discrete_rh_tier_v14"})
    env = simpy.Environment()
    w = CoWorld(env, tr, cfg, p)
    sunlit = [j for j in tr.compute if not tr.is_dark(j, 1.0)]
    c0, c1 = sunlit[0], sunlit[1]
    # Urgent request with plenty of slack (relief_s == 0) + normal decode request with huge KV
    r_urg = CRequest(i=0, t=0.0, origin=tr.rs[0], pin=1000, pout=2000, long=True, deadline=2000.0, urgent=True, upload_bytes=1e6)
    r_norm = CRequest(i=1, t=0.0, origin=tr.rs[0], pin=8000, pout=2000, long=True, deadline=2000.0, urgent=False, upload_bytes=1e6)
    r_c1 = CRequest(i=2, t=0.0, origin=tr.rs[0], pin=1000, pout=2000, long=True, deadline=2000.0, urgent=False, upload_bytes=1e6)
    for r, node in ((r_urg, c0), (r_norm, c0), (r_c1, c1)):
        r.node = node
        r.state = "decode"
        w.nodes[node].append(r)
    env.run(until=1.0)
    cands = p._collect_candidates(w)
    assert not any(c.req is r_norm and c.src == c0 for c in cands)
