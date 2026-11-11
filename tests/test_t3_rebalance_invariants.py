"""T3: rebalance invariants are shared by every migration-capable strategy."""
import copy

import pytest
import simpy

from satsim.cosim.policies import MAX_INFLIGHT, MAX_MIG_PER_REQ, make_policy
from satsim.cosim.run import apply_rebalance_invariants, run_one
from satsim.cosim.trace import Trace


def toy_trace() -> Trace:
    return Trace({
        "schema": "orbitsim.contact_trace/v1",
        "name": "t3-toy",
        "horizon_s": 1.0,
        "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
        "edges": [
            {"u": 0, "v": 1, "kind": "rs", "up": [[0, 1]]},
            {"u": 1, "v": 2, "kind": "intra", "up": [[0, 1]]},
        ],
        "dark": [[], [], []],
    })


CFG = {
    "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
    "gpu": {"kv_gib": 24.0, "t_fixed_s": 0.003, "t_weights_s": 0.01, "t_kv_tok_s": 1e-7,
            "t_seq_s": 5e-5, "mem_freq_exp": 0.6},
    "power": {"P_idle": 60, "P_max": 300, "gamma": 2.4, "f_min": 0.35, "P_dark": 80},
    "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 5.0}, "hop_latency_s": 0.005},
    "migration": {"timeout_s": 120.0, "switch_s": 0.02},
    "workload": {"arrival_window_s": 0.1, "rate_per_rs_per_min": 0},
    "slo": {"deadline_factor_long": 3.0, "deadline_factor_short": 4.0, "min_deadline_s": 60,
            "max_stall_frac": 0.1},
    "policy": {"stall_weight": 1.0},
    "delivery": "final",
    "control_period_s": 2.0,
}

INVARIANTS = {"interval_s": 5.0, "max_inflight": 24, "max_mig_per_req": 4}


def test_default_quotas_are_unchanged():
    thermal = make_policy({"policy": "thermal_rebalance", "placement": {"policy": "greedy"}})
    assert (thermal.interval_s, thermal.max_inflight, thermal.max_mig_per_req) == (30.0, MAX_INFLIGHT, 1)

    torch = make_policy({"policy": "torch_rebalance", "placement": {"policy": "greedy"}})
    assert (torch.interval_s, torch.max_inflight, torch.max_mig_per_req) == (5.0, 24, 4)

    reactive = make_policy({"place": "nearest", "migrate": "reactive"})
    assert (reactive.migrator.interval_s, reactive.migrator.max_inflight,
            reactive.migrator.max_mig_per_req) == (None, MAX_INFLIGHT, MAX_MIG_PER_REQ)

    contact = make_policy({"place": "nearest", "migrate": "contact_aware"})
    assert (contact.migrator.interval_s, contact.migrator.max_inflight,
            contact.migrator.max_mig_per_req) == (None, MAX_INFLIGHT, MAX_MIG_PER_REQ)


def test_invariants_are_injected_into_every_migration_form():
    specs = [
        {"policy": "thermal_rebalance", "placement": {"policy": "greedy"}},
        {"policy": "torch_rebalance", "placement": {"policy": "greedy"}},
        {"place": "nearest", "migrate": "reactive"},
        {"place": "nearest", "migrate": "contact_aware"},
    ]
    for spec in specs:
        out = apply_rebalance_invariants({"rebalance_invariants": INVARIANTS}, spec)
        assert (out["interval_s"], out["max_inflight"], out["max_mig_per_req"]) == (5.0, 24, 4)


def test_run_one_injects_the_invariants():
    cfg = copy.deepcopy(CFG)
    cfg["rebalance_invariants"] = INVARIANTS
    w, _ = run_one(cfg, toy_trace(), {"policy": "thermal_rebalance", "placement": {"policy": "greedy"}}, 1)
    assert (w.policy.interval_s, w.policy.max_inflight, w.policy.max_mig_per_req) == (5.0, 24, 4)


def test_explicit_conflict_is_an_error_not_a_silent_override():
    spec = {"policy": "torch_rebalance", "placement": {"policy": "greedy"}, "max_inflight": 12}
    with pytest.raises(ValueError, match="conflicts"):
        apply_rebalance_invariants({"rebalance_invariants": INVARIANTS}, spec)


def test_non_invariant_policy_is_left_alone():
    spec = {"policy": "greedy"}
    assert apply_rebalance_invariants({"rebalance_invariants": INVARIANTS}, spec) == spec
