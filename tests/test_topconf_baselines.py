"""Tests for LAB-1919 top-conference-inspired discrete power-thermal baselines:
- llumnix_style_mig_only (OSDI'24 Llumnix: priority-isolation stateful migration, no explicit_tier_idx lock)
- tapas_reactive_thermal (ASPLOS'25 TAPAS + HotCarbon'26 TLB: temperature routing + reactive guardband step-down, slack-blind)
- mcs_slack_dvfs_no_mig (RTAS'20 MCS+DVFS: local tightest-slack explicit_tier_idx mapping, zero cross-satellite migration)
"""
from __future__ import annotations

import pytest

from satsim.cosim.interfaces import Migrator, Placer
from satsim.cosim.policies import MIGRATORS, PLACERS, POLICIES, make_policy
from satsim.cosim.run import apply_rebalance_invariants

INVARIANTS = {"interval_s": 5.0, "max_inflight": 24, "max_mig_per_req": 4}
BASELINE_IDS = (
    "llumnix_style_mig_only",
    "tapas_reactive_thermal",
    "mcs_slack_dvfs_no_mig",
)


def test_topconf_baselines_registered_and_honor_rebalance_invariants() -> None:
    for name in BASELINE_IDS:
        assert name in POLICIES
        assert name in PLACERS
        assert name in MIGRATORS

        standalone = make_policy({"policy": name})
        assert standalone.id == name
        assert isinstance(standalone, Placer)
        assert isinstance(standalone, Migrator)

        injected = apply_rebalance_invariants({"rebalance_invariants": INVARIANTS}, {"policy": name})
        assert (injected["interval_s"], injected["max_inflight"], injected["max_mig_per_req"]) == (5.0, 24, 4)
        pol = make_policy(injected)
        assert (pol.interval_s, pol.max_inflight, pol.max_mig_per_req) == (5.0, 24, 4)

        injected_comp = apply_rebalance_invariants(
            {"rebalance_invariants": INVARIANTS},
            {"place": "orbit_thermal", "migrate": name},
        )
        assert (
            injected_comp["interval_s"],
            injected_comp["max_inflight"],
            injected_comp["max_mig_per_req"],
        ) == (5.0, 24, 4)
        comp = make_policy(injected_comp)
        assert (
            comp.migrator.interval_s,
            comp.migrator.max_inflight,
            comp.migrator.max_mig_per_req,
        ) == (5.0, 24, 4)

        with pytest.raises(ValueError, match="conflicts"):
            apply_rebalance_invariants(
                {"rebalance_invariants": INVARIANTS},
                {"policy": name, "max_inflight": 12},
            )


def _mc_discrete_cfg(*, t0_c: float = 42.0, t_max_c: float = 80.0) -> dict:
    """Calibrated 3-tier discrete config with non-heating low tier:
    T_eq(75W) = 20.0 + 0.25 * 75.0 = 38.75 C <= T0_C = 42.0 C, high = 300W (T_eq = 95 C > T_max = 80 C).
    """
    return {
        "model": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        "gpu": {
            "kv_gib": 40.0,
            "t_fixed_s": 0.003,
            "t_weights_s": 0.01,
            "t_kv_tok_s": 1e-7,
            "t_seq_s": 5e-5,
            "mem_freq_exp": 0.6,
        },
        "power": {
            "mode": "discrete_tiers",
            "P_dark": 80.0,
            "prefer_high_clock": True,
            "tiers": [
                {"name": "idle_0w", "power_w": 0.0, "max_batch": 0, "clock_f": 0.0},
                {"name": "low", "power_w": 75.0, "max_batch": 6, "clock_f": 0.45, "prefill_clock_f": 0.35},
                {"name": "high", "power_w": 300.0, "max_batch": 12, "clock_f": 1.0, "prefill_clock_f": 1.0},
            ],
        },
        "thermal": {
            "tau_s": 60.0,
            "R_th": 0.25,
            "T_amb_C": 20.0,
            "T0_C": t0_c,
            "T_max_C": t_max_c,
            "thermal_horizon_s": 20.0,
        },
        "net": {"link_gbps": {"intra": 10.0, "inter": 10.0, "rs": 10.0}, "hop_latency_s": 0.005},
        "migration": {"timeout_s": 120.0, "switch_s": 0.02},
        "workload": {"arrival_window_s": 400.0, "long_out": [4000, 4000], "short_out": [300, 300]},
        "slo": {
            "deadline_factor_long": 3.5,
            "deadline_factor_short": 4.0,
            "min_deadline_s": 30.0,
            "max_stall_frac": 0.1,
        },
        "policy": {"stall_weight": 1.0},
        "rebalance_invariants": dict(INVARIANTS),
        "delivery": "final",
        "control_period_s": 2.0,
    }


def test_mcs_slack_dvfs_no_mig_maps_tightest_slack_and_never_migrates() -> None:
    import simpy
    from satsim.cosim.record import summarize
    from satsim.cosim.trace import Trace
    from satsim.cosim.world import CRequest, CoWorld

    # 1. Loose-slack request alone -> locks low tier immediately (0s on high tier), saves energy vs place_once
    tr = Trace(
        {
            "schema": "orbitsim.contact_trace/v1",
            "name": "mcs-loose",
            "horizon_s": 300.0,
            "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
            "edges": [
                {"u": 0, "v": 1, "kind": "rs", "up": [[0, 300]]},
                {"u": 1, "v": 2, "kind": "intra", "up": [[0, 300]]},
            ],
            "dark": [[], [[120.0, 260.0]], []],  # node 1 enters eclipse while node 2 stays sunlit & idle
        }
    )
    cfg = _mc_discrete_cfg()

    env_base = simpy.Environment()
    w_base = CoWorld(env_base, tr, cfg, make_policy({"place": "nearest", "migrate": "none"}))
    # 4000 tokens at 1.0x takes ~54s; loose 3.5x deadline = 195s
    w_base.submit(CRequest(0, 0, 2.0, True, 2000, 4000, 1e6, 195.0))
    env_base.run(until=tr.horizon)
    s_base = summarize(w_base)

    env_mcs = simpy.Environment()
    w_mcs = CoWorld(
        env_mcs,
        tr,
        cfg,
        make_policy({"place": "nearest", "migrate": "mcs_slack_dvfs_no_mig", **INVARIANTS}),
    )
    w_mcs.submit(CRequest(0, 0, 2.0, True, 2000, 4000, 1e6, 195.0))
    env_mcs.run(until=tr.horizon)
    s_mcs = summarize(w_mcs)

    assert s_mcs["good"] == 1
    assert s_mcs["migrations"] == 0
    assert s_mcs["power_tier_fracs"]["high"] == pytest.approx(0.0, abs=1e-9)
    assert s_mcs["power_tier_fracs"]["low"] > 0.0
    assert s_mcs["energy_kj_total"] < s_base["energy_kj_total"] * 0.60

    # 2. Mixed queue (urgent 1.5x + loose 4.0x on node 1, while node 1 later enters eclipse and node 2 is sunlit):
    #    - Tightest slack (urgent) forces high tier while urgent is resident;
    #    - Once urgent departs, loose drops to low tier;
    #    - Even when node 1 enters eclipse with node 2 sunlit, migrations MUST remain 0.
    env_mix = simpy.Environment()
    w_mix = CoWorld(
        env_mix,
        tr,
        cfg,
        make_policy({"policy": "mcs_slack_dvfs_no_mig", "placement": {"policy": "nearest"}, **INVARIANTS}),
    )
    r_loose = CRequest(0, 0, 2.0, True, 2000, 4000, 1e5, 260.0)   # loose ~4.8x (4000 tok matches long_out)
    r_urgent = CRequest(1, 0, 2.0, False, 500, 300, 1e5, 8.5)     # tight ~1.6x (300 tok matches short_out)
    w_mix.submit(r_loose)
    w_mix.submit(r_urgent)

    env_mix.run(until=4.0)
    # While urgent is active on node 1, explicit_tier_idx selects high tier (index 2, 300W)
    assert w_mix.power_model._tier_idx[1] == 2

    env_mix.run(until=12.0)
    # Urgent finished (~6.2s); remaining loose request immediately stepped down to low tier (index 1, 75W)
    assert r_urgent.t_delivered is not None and r_urgent.t_delivered <= r_urgent.deadline
    assert w_mix.power_model._tier_idx[1] == 1

    env_mix.run(until=tr.horizon)
    s_mix = summarize(w_mix)
    assert s_mix["good"] == 2
    assert s_mix["migrations"] == 0
    assert s_mix["power_tier_fracs"]["high"] > 0.0
    assert s_mix["power_tier_fracs"]["low"] > s_mix["power_tier_fracs"]["high"]


def test_llumnix_style_mig_only_isolates_urgent_without_explicit_low_tier() -> None:
    import simpy
    from satsim.cosim.record import summarize
    from satsim.cosim.trace import Trace
    from satsim.cosim.world import CRequest, CoWorld

    tr = Trace(
        {
            "schema": "orbitsim.contact_trace/v1",
            "name": "llumnix-iso",
            "horizon_s": 300.0,
            "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
            "edges": [
                {"u": 0, "v": 1, "kind": "rs", "up": [[0, 300]]},
                {"u": 0, "v": 2, "kind": "rs", "up": [[0, 300]]},
                {"u": 1, "v": 2, "kind": "intra", "up": [[0, 300]]},
            ],
            "dark": [[], [], []],
        }
    )
    cfg = _mc_discrete_cfg(t0_c=35.0, t_max_c=85.0)

    # Force initial admission onto node 1 (nearest), then let llumnix_style_mig_only rebalance:
    # node 1 has both an urgent HI request (1.5x slack) and a loose LO request (4.0x slack);
    # node 2 is sunlit, cool, and empty -> LO request must migrate 1 -> 2 to isolate HI on node 1,
    # while explicit_tier_idx stays None on all nodes (both active nodes sprint at 300W high tier).
    env = simpy.Environment()
    pol = make_policy({"place": "nearest", "place_kwargs": {"tie": "id"}, "migrate": "llumnix_style_mig_only", **INVARIANTS})
    w = CoWorld(env, tr, cfg, pol)

    r_urgent = CRequest(0, 0, 1.0, True, 2000, 4000, 1e5, 84.0)   # tight ~1.5x slack (HI)
    r_loose = CRequest(1, 0, 1.0, True, 2000, 4000, 1e5, 240.0)   # loose ~4.4x slack (LO)
    w.submit(r_urgent)
    w.submit(r_loose)

    env.run(until=12.0)
    assert all(x is None for x in w.power_model._explicit_tier_idx)
    assert w.power_model._tier_idx[1] == 2  # 300W high tier (no proactive low-tier lock)

    env.run(until=tr.horizon)
    s = summarize(w)
    assert all(x is None for x in w.power_model._explicit_tier_idx)
    assert s["migrations"] >= 1
    assert r_urgent.trail == [1]
    assert r_loose.trail == [1, 2]
    assert s["good"] == 2
    assert s["power_tier_fracs"]["high"] > 0.0
    assert s["power_tier_fracs"]["low"] == pytest.approx(0.0, abs=1e-9)


def test_tapas_reactive_thermal_routes_by_temp_reactively_throttles_and_is_slack_blind() -> None:
    import simpy
    from satsim.cosim.record import summarize
    from satsim.cosim.trace import Trace
    from satsim.cosim.world import CRequest, CoWorld

    tr = Trace(
        {
            "schema": "orbitsim.contact_trace/v1",
            "name": "tapas-thermal",
            "horizon_s": 250.0,
            "nodes": [{"id": 0, "role": "rs"}, {"id": 1, "role": "compute"}, {"id": 2, "role": "compute"}],
            "edges": [
                {"u": 0, "v": 1, "kind": "rs", "up": [[0, 250]]},
                {"u": 0, "v": 2, "kind": "rs", "up": [[0, 250]]},
                {"u": 1, "v": 2, "kind": "intra", "up": [[0, 250]]},
            ],
            "dark": [[], [], []],
        }
    )
    cfg = _mc_discrete_cfg(t0_c=40.0, t_max_c=80.0)
    env = simpy.Environment()
    pol = make_policy({"policy": "tapas_reactive_thermal", **INVARIANTS})
    w = CoWorld(env, tr, cfg, pol)

    # 1. Temperature routing + Slack-blindness:
    r_urgent_probe = CRequest(10, 0, 0.0, True, 2000, 4000, 1e6, 70.0)   # tight slack
    r_loose_probe = CRequest(11, 0, 0.0, True, 2000, 4000, 1e6, 500.0)   # huge slack
    w.power_model.set_temperature(1, 72.0)
    w.power_model.set_temperature(2, 38.0)
    assert pol.place(w, r_urgent_probe) == 2
    assert pol.place(w, r_loose_probe) == 2

    w.power_model.set_temperature(1, 38.0)
    w.power_model.set_temperature(2, 72.0)
    assert pol.place(w, r_urgent_probe) == 1
    assert pol.place(w, r_loose_probe) == 1

    # 2. Cool vs reactive guardband tier step-down + TLB evacuation:
    # Start node 1 hot in guardband (76.0 C >= 80.0 - 6.0 = 74.0 C) with an active decoding request,
    # while node 2 is cool (38.0 C) with a loose request arriving.
    w.power_model.set_temperature(1, 76.0)
    w.power_model.set_temperature(2, 38.0)
    r_hot = CRequest(0, 0, 0.0, True, 2000, 4000, 1e5, 220.0, node=1, state="decode")
    w.nodes[1].append(r_hot)
    pol.on_arrival(w, 1, r_hot)
    w._sync_discrete_node(1, 0)
    # Hot node 1 in guardband reactively steps down to low tier (index 1, 75W) before hitting T_max=80C
    assert w.power_model._explicit_tier_idx[1] == 1
    assert w.power_model._tier_idx[1] == 1

    # Loose request submitted to cool node 2: TAPAS is slack-blind, so cool node 2 stays at None (300W high tier)
    r_cool_loose = CRequest(1, 0, 1.0, True, 2000, 4000, 1e5, 240.0)
    w.submit(r_cool_loose)
    env.run(until=3.0)
    assert r_cool_loose.node == 2
    assert w.power_model._explicit_tier_idx[2] is None
    assert w.power_model._tier_idx[2] == 2

    # At rebalance cadence (t >= 5.0s), TLB evacuates r_hot from throttled hot node 1 to cool sunlit node 2
    env.run(until=tr.horizon)
    s = summarize(w)
    assert s["migrations"] >= 1
    assert r_hot.trail == [2] or r_hot.node == 2


def test_end_to_end_cosim_run_smoke_all_three_baselines_discrete_tiers(tmp_path) -> None:
    import json
    from satsim.cosim.run import run as cosim_run

    trace_doc = {
        "schema": "orbitsim.contact_trace/v1",
        "name": "mc-topconf-smoke",
        "horizon_s": 420.0,
        "nodes": [
            {"id": 0, "role": "rs"},
            {"id": 1, "role": "rs"},
            {"id": 2, "role": "compute"},
            {"id": 3, "role": "compute"},
            {"id": 4, "role": "compute"},
            {"id": 5, "role": "compute"},
        ],
        "edges": [
            {"u": 0, "v": 2, "kind": "rs", "up": [[0, 420]]},
            {"u": 0, "v": 4, "kind": "rs", "up": [[0, 420]]},
            {"u": 1, "v": 3, "kind": "rs", "up": [[0, 420]]},
            {"u": 1, "v": 5, "kind": "rs", "up": [[0, 420]]},
            {"u": 2, "v": 3, "kind": "intra", "up": [[0, 420]]},
            {"u": 2, "v": 4, "kind": "inter", "up": [[0, 420]]},
            {"u": 3, "v": 5, "kind": "inter", "up": [[0, 420]]},
            {"u": 4, "v": 5, "kind": "intra", "up": [[0, 420]]},
        ],
        "dark": [
            [],
            [],
            [[90.0, 260.0]],
            [[140.0, 310.0]],
            [],
            [],
        ],
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace_doc), encoding="utf-8")

    cfg = _mc_discrete_cfg(t0_c=42.0, t_max_c=76.0)
    cfg.update(
        {
            "name": "lab1919_topconf_smoke",
            "trace": str(trace_path),
            "scenario": "mc-topconf-smoke",
            "_dir": str(tmp_path),
            "workload": {
                "arrival_window_s": 220.0,
                "rate_per_rs_per_min": 3.0,
                "p_long": 0.55,
                "long_pin": 1800,
                "long_out": [2400, 2800],
                "long_upload_MB": 2.0,
                "short_pin": 600,
                "short_out": [400, 500],
                "short_upload_MB": 0.5,
            },
            "slo": {
                "deadline_factor_long": 3.8,
                "deadline_factor_short": 1.6,
                "min_deadline_s": 8.0,
                "max_stall_frac": 0.1,
            },
            "policies": {
                "place_once": {"place": "nearest", "migrate": "none"},
                "llumnix_style_mig_only": {"policy": "llumnix_style_mig_only"},
                "tapas_reactive_thermal": {"policy": "tapas_reactive_thermal"},
                "mcs_slack_dvfs_no_mig": {"policy": "mcs_slack_dvfs_no_mig"},
            },
            "seeds": [11],
        }
    )

    payload = cosim_run(cfg, quiet=True)
    mean = payload["mean"]

    for name in ("place_once", *BASELINE_IDS):
        row = mean[name]
        assert row["n_req"] > 0
        assert 0.0 <= row["zero_power_frac"] <= 1.0
        assert sum(row["power_tier_fracs"].values()) == pytest.approx(1.0, abs=1e-6)
        assert row["energy_kj_total"] > 0.0
        assert row["tier_unsat_s"] >= 0.0
        assert row["peak_temp_C"] is not None
        assert row["min_thermal_margin_C"] is not None

    # mcs_slack_dvfs_no_mig: strictly 0 migrations + lower energy than place_once via low-tier selection
    assert mean["mcs_slack_dvfs_no_mig"]["migrations"] == 0.0
    assert mean["mcs_slack_dvfs_no_mig"]["power_tier_fracs"]["low"] > 0.0
    assert mean["mcs_slack_dvfs_no_mig"]["energy_kj_total"] < mean["place_once"]["energy_kj_total"]

    # llumnix_style_mig_only: migrates across ISLs and maintains >= place_once SLO attainment
    assert mean["llumnix_style_mig_only"]["migrations"] >= 1.0
    assert mean["llumnix_style_mig_only"]["good_frac"] >= mean["place_once"]["good_frac"]

    # tapas_reactive_thermal: keeps peak temperature strictly below or equal to place_once
    assert mean["tapas_reactive_thermal"]["peak_temp_C"] <= mean["place_once"]["peak_temp_C"] + 1e-6




