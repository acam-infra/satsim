"""YAML 加载、resolve、job 展开。"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from satsim.input.schema import (
    MAX_SEEDS,
    SpecError,
    _expect_keys,
    getp,
    setp,
    validate_base,
    validate_policy,
    validate_t_swap,
    validate_trace,
)


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SpecError(f"{path} 不是映射")
    return data


def apply_t_swap(cfg: dict, t_swap_s: float, p_cap: float) -> None:
    """定时限功时间表：切换前限制 node 2，切换后限制 node 0。"""
    n = cfg["nodes"]
    if n < 3:
        raise SpecError("限功切换写的是 node 2 → node 0，nodes 至少为 3")
    cfg["thermal"]["explicit_events"] = [
        [2, 0.0, float(t_swap_s), float(p_cap)],
        [0, float(t_swap_s), 1.0e9, float(p_cap)],
    ]


def resolve_spec(spec_path: Path) -> tuple[dict, dict, list[tuple[str, dict, dict, int]]]:
    spec_path = spec_path.resolve()
    spec = load_yaml(spec_path)
    if spec.get("schema") == "satsim.scene/v1":
        raise SpecError("scene 只是场景背景说明，不能当复现入口。请跑对应的 scenarios spec。")
    _expect_keys(
        spec,
        {
            "name",
            "base",
            "trace",
            "t_swap",
            "overrides",
            "policies",
            "seeds",
            "slo",
            "evidence",
            "metadata",
        },
        "spec",
    )
    base_path = (spec_path.parent / spec["base"]).resolve()
    cfg = load_yaml(base_path)
    validate_base(cfg)
    overrides = spec.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise SpecError("overrides 必须是映射")
    for k, v in overrides.items():
        setp(cfg, k, v)
        if k == "background.strict_priority" and v is not True:
            raise SpecError("background.strict_priority 只允许 true")
    validate_trace(spec["trace"])
    validate_t_swap(spec["t_swap"], trace_kind=str(spec["trace"].get("kind")))
    slo = spec["slo"]
    _expect_keys(slo, {"tpot_slo"}, "slo")
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or not seeds or len(seeds) > MAX_SEEDS:
        raise SpecError(f"seeds 需要 1..{MAX_SEEDS} 个")
    if "grid" in spec:
        raise SpecError("grid 会把一次实验展开成多组，当前最小实验不允许")
    policies = spec["policies"]
    if not isinstance(policies, dict) or not policies:
        raise SpecError("policies 为空")
    jobs = []
    for name, pol in policies.items():
        pol = copy.deepcopy(pol)
        validate_policy(name, pol)
        for seed in seeds:
            c2 = copy.deepcopy(cfg)
            for k, v in pol.get("knobs", {}).items():
                setp(c2, k, v)
                if k == "background.strict_priority" and v is not True:
                    raise SpecError("background.strict_priority 只允许 true")
            jobs.append((name, pol, c2, int(seed)))
    _check_scene(spec_path, spec, cfg)
    return spec, cfg, jobs


def _check_scene(spec_path: Path, spec: dict, cfg: dict) -> None:
    meta = spec.get("metadata") or {}
    _expect_keys(meta, {"scene"}, "metadata")
    scene_path = (spec_path.parent / meta["scene"]).resolve()
    scene = load_yaml(scene_path)
    if scene.get("schema") != "satsim.scene/v1":
        raise SpecError("metadata.scene 不是 satsim.scene/v1")
    _expect_keys(scene, {"schema", "name", "story", "not_modeled", "mirrors"}, "scene")
    mirrors = scene["mirrors"]
    _expect_keys(mirrors, {"cfg", "spec"}, "scene.mirrors")
    for path, want in mirrors["cfg"].items():
        got = getp(cfg, path)
        if got != want:
            raise SpecError(f"scene 与解析后的 cfg 不一致：{path} scene={want} cfg={got}")
    for path, want in mirrors["spec"].items():
        got = getp(spec, path)
        if got != want:
            raise SpecError(f"scene 与 spec 不一致：{path} scene={want} spec={got}")
