"""Declarative scenario loading.

A run spec points at a base YAML, applies a closed set of dotted overrides,
then crosses the admitted policies with the seeds. Unknown keys are refused.
A scene file may only mirror knobs that already drive the run.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Dotted paths a spec override or a policy cfg is allowed to write.
# Anything else is a new knob and has to be admitted here, with a reason.
OVERRIDE_REASONS = {
    "arrival_window_s": "goodput 的分母，spec 必须能对准到达窗。",
    "background.gbps_per_node": "连续背景流量，C1 靠它把 ISL 吃满。",
    "background.strict_priority": "背景是否先占链路；本复刻只准入 true。",
    "net.link_gbps": "抽象 ISL 口径。",
    "gpu.kv_gib": "单节点 KV 容量，放不下就不往该节点放。",
    "cost.mode": "free 瞬时搬走，full/net 走剩余带宽。",
    "cost.timeout_s": "剩余带宽上搬不完就 abort 的时限。",
    "thermal.explicit_events": "换位写成的功率帽时间线，由 t_swap 填，不手写第二套。",
}

# The six v3 policies. A new name is a new claim arm and is not admitted.
POLICY_REASONS = {
    "first": "首次分配：到来时放一次，换位后不动。",
    "rebalance": "运行时重分配：换位后再放，KV 不上链路。",
    "rebalance_link": "同一次运行时重分配，整段 KV 上链路。",
    "rebalance_link_net": "同样上链路，用来核对代价是不是来自链路。",
    "rebalance_nobg": "关掉背景后再做运行时重分配。",
    "rebalance_link_nobg": "关掉背景后仍把整段 KV 送上链路。",
}

_COST_MODE = {
    "rebalance": "free",
    "rebalance_link": "full",
    "rebalance_link_net": "net",
    "rebalance_nobg": "free",
    "rebalance_link_nobg": "full",
}

MAX_SEEDS = 8


class SpecError(ValueError):
    pass


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SpecError(f"{path} 不是映射")
    return data


def getp(d: dict, path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise SpecError(f"缺少 {path}")
        cur = cur[k]
    return cur


def setp(d: dict, path: str, value: Any) -> None:
    if path not in OVERRIDE_REASONS:
        raise SpecError(f"未准入的覆盖项 {path}")
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            raise SpecError(f"无法写入 {path}")
        cur = nxt
    cur[keys[-1]] = value


def _expect_keys(obj: dict, allowed: set[str], where: str) -> None:
    extra = set(obj) - allowed
    if extra:
        raise SpecError(f"{where} 有未准入字段 {sorted(extra)}")


def validate_base(cfg: dict) -> None:
    _expect_keys(
        cfg,
        {
            "nodes",
            "topology",
            "arrival_window_s",
            "drain_s",
            "control_period_s",
            "model",
            "gpu",
            "thermal",
            "net",
            "background",
            "cost",
            "workload",
        },
        "base",
    )
    if float(cfg["drain_s"]) <= 0 or float(cfg["control_period_s"]) <= 0:
        raise SpecError("drain_s 和 control_period_s 必须为正")
    if cfg["topology"] != "ring":
        raise SpecError("topology 只准入 ring（紧凑舱段的邻接，不是全星座）")
    if not isinstance(cfg["nodes"], int) or not 2 <= cfg["nodes"] <= 8:
        raise SpecError("nodes 只准入 2..8 的环")
    _expect_keys(cfg["model"], {"layers", "kv_heads", "head_dim", "bytes_per_elem"}, "model")
    _expect_keys(
        cfg["gpu"],
        {
            "kv_gib",
            "t_fixed_s",
            "t_weights_s",
            "t_kv_tok_s",
            "t_seq_s",
            "mem_freq_exp",
        },
        "gpu",
    )
    _expect_keys(cfg["thermal"], {"P_idle", "P_max", "gamma", "f_min"}, "thermal")
    _expect_keys(cfg["net"], {"link_gbps", "hop_latency_s"}, "net")
    _expect_keys(
        cfg["background"],
        {"gbps_per_node", "strict_priority", "pairs"},
        "background",
    )
    if cfg["background"]["strict_priority"] is not True:
        raise SpecError("background.strict_priority 只准入 true")
    _expect_keys(cfg["cost"], {"mode", "timeout_s", "switch_s"}, "cost")
    if cfg["cost"]["mode"] not in ("free", "full", "net"):
        raise SpecError("cost.mode 只准入 free/full/net")
    _expect_keys(cfg["workload"], {"p_long"}, "workload")
    n = cfg["nodes"]
    for pair in cfg["background"]["pairs"]:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(x, int) and 0 <= x < n for x in pair)
            or pair[0] == pair[1]
        ):
            raise SpecError(f"非法背景 pair {pair}")


def validate_trace(trace: dict) -> None:
    _expect_keys(
        trace,
        {"kind", "compress_horizon_s", "pack_window_s", "target_n", "jitter_s", "token_pad"},
        "trace",
    )
    if trace["kind"] != "synthetic_pass":
        raise SpecError("trace.kind 只准入 synthetic_pass（没有 STAC L1 文件）")
    pad = trace["token_pad"]
    _expect_keys(pad, {"mode", "long_out", "long_pin", "short_out", "short_pin"}, "token_pad")
    if pad["mode"] != "floor":
        raise SpecError("token_pad.mode 只准入 floor")
    h, w = float(trace["compress_horizon_s"]), float(trace["pack_window_s"])
    if not 0 < h <= w:
        raise SpecError("需要 0 < compress_horizon_s <= pack_window_s")
    if int(trace["target_n"]) < 1:
        raise SpecError("target_n 至少为 1")


def validate_policy(name: str, pol: dict) -> None:
    if name not in POLICY_REASONS:
        known = ", ".join(POLICY_REASONS)
        raise SpecError(f"未准入的策略 {name}。已准入：{known}")
    _expect_keys(pol, {"route", "migrate", "hysteresis", "cfg"}, f"policy {name}")
    if pol.get("route") != "late":
        raise SpecError(f"{name}: route 只准入 late")
    mig = pol.get("migrate", "none")
    cfg = pol.get("cfg", {})
    if name == "first":
        if mig not in ("none", None) or cfg:
            raise SpecError("first 是首次分配，换位后不再动")
        return
    if mig != "full":
        raise SpecError(f"{name}: migrate 只准入 full")
    h = pol.get("hysteresis")
    if not isinstance(h, (int, float)) or not 0 < float(h) <= 1:
        raise SpecError(f"{name}: hysteresis 需要在 (0, 1]")
    if not isinstance(cfg, dict):
        raise SpecError(f"{name}: cfg 必须是映射")
    allowed_cfg = {"cost.mode"}
    if name.endswith("_nobg"):
        allowed_cfg.add("background.gbps_per_node")
    extra = set(cfg) - allowed_cfg
    if extra:
        raise SpecError(f"{name}: cfg 有未准入项 {sorted(extra)}")
    if cfg.get("cost.mode") != _COST_MODE[name]:
        raise SpecError(f"{name}: cost.mode 必须是 {_COST_MODE[name]}")
    if name.endswith("_nobg") and cfg.get("background.gbps_per_node") != 0:
        raise SpecError(f"{name}: 对照必须把 background.gbps_per_node 写成 0")


def validate_t_swap(t_swap: dict) -> None:
    _expect_keys(t_swap, {"anchor", "offset_s", "P_cap"}, "t_swap")
    if t_swap["anchor"] != "max_cluster":
        raise SpecError("t_swap.anchor 只准入 max_cluster")


def apply_t_swap(cfg: dict, t_swap_s: float, p_cap: float) -> None:
    """Same two-ended cap as lab1616 apply_t_swap: node 2 until the swap, node 0 after."""
    n = cfg["nodes"]
    if n < 3:
        raise SpecError("换位写的是 node 2 → node 0，nodes 至少为 3")
    cfg["thermal"]["explicit_events"] = [
        [2, 0.0, float(t_swap_s), float(p_cap)],
        [0, float(t_swap_s), 1.0e9, float(p_cap)],
    ]


def resolve_spec(spec_path: Path) -> tuple[dict, dict, list[tuple[str, dict, dict, int]]]:
    spec_path = spec_path.resolve()
    spec = load_yaml(spec_path)
    if spec.get("schema") == "satsim.scene/v1":
        raise SpecError("scene 只是叙事镜像，不能当复现入口。请跑对应的 scenarios spec。")
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
            raise SpecError("background.strict_priority 只准入 true")
    validate_trace(spec["trace"])
    validate_t_swap(spec["t_swap"])
    slo = spec["slo"]
    _expect_keys(slo, {"tpot_slo"}, "slo")
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or not seeds or len(seeds) > MAX_SEEDS:
        raise SpecError(f"seeds 需要 1..{MAX_SEEDS} 个")
    if "grid" in spec:
        raise SpecError("grid 会把一次实验乘开，最小复刻不准入")
    policies = spec["policies"]
    if not isinstance(policies, dict) or not policies:
        raise SpecError("policies 为空")
    jobs = []
    for name, pol in policies.items():
        pol = copy.deepcopy(pol)
        validate_policy(name, pol)
        for seed in seeds:
            c2 = copy.deepcopy(cfg)
            for k, v in pol.get("cfg", {}).items():
                setp(c2, k, v)
                if k == "background.strict_priority" and v is not True:
                    raise SpecError("background.strict_priority 只准入 true")
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
