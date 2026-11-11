"""T7: cool_aware migrator moves decoding requests to cool lit compute nodes."""
import copy
import pytest
import simpy

from satsim.cosim.policies import CoolAware, make_policy
from satsim.cosim.run import apply_rebalance_invariants
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, CRequest


def make_test_trace() -> Trace:
    # node 0: RS
    # node 1: Compute (source, in eclipse)
    # node 2: Compute (destination 1, sunlit, cool)
    # node 3: Compute (destination 2, sunlit, hot)
    return Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "t7-toy",
        "horizon_s": 1000.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0, 1000]]},
            {"u": 0, "v": 2, "kind": "rs", "up": [[0, 1000]]},
            {"u": 0, "v": 3, "kind": "rs", "up": [[0, 1000]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 1000]]},
            {"u": 1, "v": 3, "kind": "intra", "up": [[0, 1000]]},
            {"u": 2, "v": 3, "kind": "intra", "up": [[0, 1000]]},
        ],
        "dark": [
            [],              # node 0 (rs)
            [[0, 1000]],     # node 1: in eclipse whole time
            [],              # node 2: sunlit
            [],              # node 3: sunlit
        ],
    })


CFG = {
    "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
    "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7,
            "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
    "power": {"P_idle": 60, "P_max": 300, "gamma": 2.4, "f_min": 0.35, "P_dark": 80},
    "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
    "migration": {"timeout_s": 120.0, "switch_s": 0.02},
    "workload": {"arrival_window_s": 1000},
    "slo": {"deadline_factor_long": 3.0, "deadline_factor_short": 4.0, "min_deadline_s": 60,
            "max_stall_frac": 0.1},
    "policy": {"stall_weight": 1.0},
    "delivery": "final",
    "control_period_s": 2.0,
    "thermal": {
        "tau_s": 60.0, "R_th": 0.30, "T_amb_C": 20.0, "T0_C": 38.0, "T_throttle_C": 84.0,
        "T_release_C": 76.0, "f_throttle": 0.55, "P_busy": 300.0, "busy_util_floor": 0.40,
        "batch_scale": 4.0,
    },
}


def _create_world(policy_spec: dict, cfg_overrides: dict | None = None) -> CoWorld:
    cfg = copy.deepcopy(CFG)
    if cfg_overrides:
        cfg.update(cfg_overrides)
    env = simpy.Environment(initial_time=10.0)
    policy = make_policy(policy_spec)
    w = CoWorld(env, make_test_trace(), cfg, policy)
    return w


def test_cool_aware_defaults_and_invariants():
    pol = make_policy({"place": "nearest", "migrate": "cool_aware"})
    m = pol.migrator
    assert isinstance(m, CoolAware)
    assert m.lead_s == 60.0
    assert m.cool_margin_C == 10.0
    assert m.min_gain_s == 30.0
    assert m.max_inflight == 8
    assert m.max_mig_per_req == 2
    assert m.interval_s is None

    invariants = {"interval_s": 5.0, "max_inflight": 24, "max_mig_per_req": 4}
    spec = {"place": "nearest", "migrate": "cool_aware"}
    injected = apply_rebalance_invariants({"rebalance_invariants": invariants}, spec)
    assert injected["interval_s"] == 5.0
    assert injected["max_inflight"] == 24
    assert injected["max_mig_per_req"] == 4

    pol2 = make_policy(injected)
    assert pol2.migrator.interval_s == 5.0
    assert pol2.migrator.max_inflight == 24
    assert pol2.migrator.max_mig_per_req == 4


def test_migrates_from_eclipse_to_cool_destination():
    # Source node 1 in eclipse.
    # Destination node 2 sunlit and cool.
    # Destination node 3 sunlit and hot.
    w = _create_world({"place": "nearest", "migrate": "cool_aware"})

    # Setup temperatures / states:
    # Node 2 is cool: headroom = 84 - 35 = 49 > 10
    w.thermal_f[2] = 1.0
    w.thermal._T[2] = 35.0

    # Node 3 is hot: throttled
    w.thermal_f[3] = 0.55
    w.thermal._T[3] = 90.0

    # Put a decoding request on node 1
    req = CRequest(
        i=1, origin=0, t=0.0, long=True, pin=1000, pout=4000, upload_bytes=1e6,
        deadline=600.0, node=1, state="decode", gen=100.0
    )
    w.nodes[1].append(req)

    assert w.is_eclipse(1)
    assert w.is_cool(2, 10.0)
    assert not w.is_cool(3, 10.0)

    w.policy.rebalance(w)
    w.env.run(until=w.env.now + 0.001)

    # The request should have initiated migration to node 2 (the cool node)
    assert req.migrating is True
    # Verify migration process was scheduled targeting node 2
    assert w.reserved[2] > 0
    assert w.reserved[3] == 0


def test_does_not_migrate_when_bandwidth_is_too_low():
    # When link bandwidth is extremely low, transfer time dominates and gain < min_gain_s.
    w = _create_world(
        {"place": "nearest", "migrate": "cool_aware"},
        cfg_overrides={"net": {"link_gbps": {"intra": 0.00001, "inter": 0.00001, "rs": 5.0}, "hop_latency_s": 0.005}}
    )

    # Node 2 is cool
    w.thermal_f[2] = 1.0
    w.thermal._T[2] = 35.0

    req = CRequest(
        i=1, origin=0, t=0.0, long=True, pin=1000, pout=4000, upload_bytes=1e6,
        deadline=600.0, node=1, state="decode", gen=100.0
    )
    w.nodes[1].append(req)

    w.policy.rebalance(w)

    # Should NOT migrate because transfer latency is prohibitive
    assert req.migrating is False
    assert w.reserved[2] == 0


def test_destination_limit_per_cycle():
    # Each destination accepts at most 1 migration per cycle
    w = _create_world({"place": "nearest", "migrate": "cool_aware"})

    w.thermal_f[2] = 1.0
    w.thermal._T[2] = 35.0
    w.thermal_f[3] = 0.55
    w.thermal._T[3] = 90.0

    r1 = CRequest(i=1, origin=0, t=0.0, long=True, pin=1000, pout=4000, upload_bytes=1e6,
                  deadline=500.0, node=1, state="decode", gen=100.0)
    r2 = CRequest(i=2, origin=0, t=0.0, long=True, pin=1000, pout=4000, upload_bytes=1e6,
                  deadline=600.0, node=1, state="decode", gen=100.0)
    w.nodes[1].extend([r1, r2])

    w.policy.rebalance(w)
    w.env.run(until=w.env.now + 0.001)

    # Only r1 should migrate to node 2, r2 should stay because node 2 already took 1 in this cycle
    assert r1.migrating is True
    assert r2.migrating is False


def test_orbit_thermal_and_orbit_load_placer_composition():
    pol_th = make_policy({"place": "orbit_thermal", "migrate": "cool_aware"})
    assert pol_th.id == "orbit_thermal+cool_aware"
    assert pol_th.placer.thermal_aware is True

    pol_ld = make_policy({"place": "orbit_load", "place_kwargs": {"thermal_aware": False}, "migrate": "cool_aware"})
    assert pol_ld.id == "orbit_load+cool_aware"
    assert pol_ld.placer.thermal_aware is False

    pol_rr = make_policy({"place": "round_robin_all", "migrate": "cool_aware"})
    assert pol_rr.id == "round_robin_all+cool_aware"


def test_skips_short_lived_path_to_avoid_abort():
    trace = Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "t7-short-link",
        "horizon_s": 1000.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "compute"},
            {"id": 2, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0, 1000]]},
            {"u": 0, "v": 2, "kind": "rs", "up": [[0, 1000]]},
            # Link 1->2 goes down at t=10.5s, before a ~1.5s KV transfer at 1 Gbps can complete
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 10.5]]},
        ],
        "dark": [[], [[0, 1000]], []],
    })
    cfg = copy.deepcopy(CFG)
    cfg["net"] = {"link_gbps": {"intra": 1.0, "inter": 1.0, "rs": 5.0}, "hop_latency_s": 0.005}
    env = simpy.Environment(initial_time=10.0)
    w = CoWorld(env, trace, cfg, make_policy({"place": "nearest", "migrate": "cool_aware"}))
    w.thermal_f[2] = 1.0
    w.thermal._T[2] = 35.0

    req = CRequest(i=1, origin=0, t=0.0, long=True, pin=1000, pout=4000, upload_bytes=1e6,
                   deadline=600.0, node=1, state="decode", gen=500.0)
    w.nodes[1].append(req)
    w.policy.rebalance(w)
    assert req.migrating is False


def test_skips_destination_with_inflight_migration_across_ticks():
    w = _create_world(
        {"place": "nearest", "migrate": "cool_aware"},
        cfg_overrides={"net": {"link_gbps": {"intra": 0.2, "inter": 0.2, "rs": 5.0}, "hop_latency_s": 0.005}},
    )
    w.thermal_f[2] = 1.0
    w.thermal._T[2] = 35.0
    w.thermal_f[3] = 0.55
    w.thermal._T[3] = 90.0

    r1 = CRequest(i=1, origin=0, t=0.0, long=True, pin=1000, pout=8000, upload_bytes=1e6,
                  deadline=500.0, node=1, state="decode", gen=100.0)
    r2 = CRequest(i=2, origin=0, t=0.0, long=True, pin=1000, pout=8000, upload_bytes=1e6,
                  deadline=550.0, node=1, state="decode", gen=100.0)
    w.nodes[1].extend([r1, r2])

    w.policy.rebalance(w)
    w.env.run(until=w.env.now + 2.0)
    assert r1.migrating is True

    # Next tick while r1's KV flow to node 2 is still in flight: r2 must not pile onto node 2
    w.policy.rebalance(w)
    w.env.run(until=w.env.now + 0.01)
    assert r2.migrating is False


def test_skips_expired_requests_and_prioritizes_slo_rescue():
    w = _create_world({"place": "nearest", "migrate": "cool_aware"})
    w.thermal_f[2] = 1.0
    w.thermal._T[2] = 35.0
    w.thermal_f[3] = 0.55
    w.thermal._T[3] = 90.0

    # r_expired already missed its deadline (deadline=5.0 < now=10.0)
    r_expired = CRequest(i=1, origin=0, t=0.0, long=True, pin=1000, pout=4000, upload_bytes=1e6,
                         deadline=5.0, node=1, state="decode", gen=100.0)
    # r_rescue fails on eclipsed node 1 (~120s > slack=90s) but finishes in time on cool sunlit node 2 (~50s <= 90s)
    r_rescue = CRequest(i=2, origin=0, t=0.0, long=True, pin=1000, pout=4000, upload_bytes=1e6,
                        deadline=100.0, node=1, state="decode", gen=100.0)
    w.nodes[1].extend([r_expired, r_rescue])

    w.policy.rebalance(w)
    w.env.run(until=w.env.now + 0.001)
    assert r_expired.migrating is False
    assert r_rescue.migrating is True
