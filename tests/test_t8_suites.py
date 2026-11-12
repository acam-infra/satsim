"""T8: validate suite YAML configs (E1, E3, E4) and policy instantiation."""
from __future__ import annotations

from pathlib import Path

import yaml

from satsim.cosim.policies import make_policy
from satsim.cosim.run import apply_rebalance_invariants, load_config
from satsim.cosim.suite import apply_set_block


ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    "e1_stage1_hotspot.yaml",
    "e2_hotspot_stress_paper.yaml",
    "e3_bw_kv_sweep.yaml",
    "e4_fair_quota.yaml",
    "diag_ephemeris_vs_orbit_thermal.yaml",
)


def test_t8_suite_configs_valid_and_instantiable():
    for fname in SUITES:
        suite_path = ROOT / "configs" / "suites" / fname
        assert suite_path.exists(), f"missing {fname}"
        data = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        base_cfg = load_config((suite_path.parent / data["base"]).resolve())
        if data.get("set"):
            apply_set_block(base_cfg, data["set"])
        assert data["report"]["baseline"] in data["policies"]
        for pol_name, spec in data["policies"].items():
            injected = apply_rebalance_invariants(base_cfg, spec)
            pol = make_policy(injected)
            assert pol is not None, f"failed to instantiate {pol_name} in {fname}"
