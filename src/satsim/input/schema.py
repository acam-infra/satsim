"""准入表与校验：旋钮、显式策略、base/trace/t_swap。"""
from __future__ import annotations

from typing import Any

# 覆盖/ knobs 可写路径（含由 link_cost 推导写入的 cost.mode）
OVERRIDE_REASONS = {
    "arrival_window_s": "goodput 的分母，spec 必须能对准到达窗。",
    "background.gbps_per_node": "连续背景流量；对照实验可写成 0。",
    "background.strict_priority": "背景是否先占链路；本复刻只准入 true。",
    "net.link_gbps": "抽象 ISL 口径。",
    "gpu.kv_gib": "单节点 KV 容量，放不下就不往该节点放。",
    "cost.mode": "由 link_cost 推导：off→free，full→full，net→net。",
    "cost.timeout_s": "剩余带宽上搬不完就 abort 的时限。",
    "thermal.explicit_events": "换位写成的功率帽时间线，由 t_swap 填，不手写第二套。",
}

LINK_COST_TO_MODE = {"off": "free", "full": "full", "net": "net"}

SCHEDULER_IDS = {"place_once", "rebalance_free", "rebalance_random", "rebalance_ours"}
LINK_COST_IDS = {"off", "full", "net"}
KNOB_KEYS = {"background.gbps_per_node"}

# 论文三档：初次 / 无消耗再分配 / 有消耗再分配（random|ours）
SCHEDULER_LINK_COST = {
    "place_once": {"off"},
    "rebalance_free": {"off"},
    "rebalance_random": {"full", "net"},
    "rebalance_ours": {"full", "net"},
}
SCHEDULER_NEEDS_HYSTERESIS = {"rebalance_free"}

MAX_SEEDS = 8

POISSON_DEFAULTS = {
    "window_s": 60.0,
    "rate_per_s": 0.1,
    "jitter_s": 0.0,
}

GAMMA_DEFAULTS = {
    "window_s": 60.0,
    "rate_per_s": 0.1,
    "cv": 1.0,
    "jitter_s": 0.0,
}

SHAREGPT_DEFAULTS = {
    "window_s": 180.0,
    "rate_per_s": 0.5,
    "arrival": "deterministic",
    "cv": 1.0,
    "jitter_s": 0.0,
    "long_threshold_tokens": 761,
    "max_seqlen": 4096,
}


class SpecError(ValueError):
    pass


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


def _validate_orbit_block(orbit: dict, n: int) -> None:
    """orbit_planes 的声明式壳层；生成边在 World 启动时做。"""
    if not isinstance(orbit, dict):
        raise SpecError("orbit 必须是映射")
    _expect_keys(
        orbit,
        {
            "earth_radius_km",
            "distance_threshold_km",
            "same_plane_neighbors",
            "plane_ring",
            "planes",
        },
        "orbit",
    )
    if "planes" not in orbit or not isinstance(orbit["planes"], list) or not orbit["planes"]:
        raise SpecError("orbit.planes 不能为空")
    if "distance_threshold_km" not in orbit:
        raise SpecError("orbit 需要 distance_threshold_km")
    if float(orbit["distance_threshold_km"]) <= 0:
        raise SpecError("distance_threshold_km 必须为正")
    if "earth_radius_km" in orbit and float(orbit["earth_radius_km"]) <= 0:
        raise SpecError("earth_radius_km 必须为正")
    for key in ("same_plane_neighbors", "plane_ring"):
        if key in orbit and not isinstance(orbit[key], bool):
            raise SpecError(f"orbit.{key} 必须是 bool")
    seen: set[int] = set()
    for i, plane in enumerate(orbit["planes"]):
        if not isinstance(plane, dict):
            raise SpecError(f"orbit.planes[{i}] 必须是映射")
        _expect_keys(
            plane,
            {
                "id",
                "role",
                "altitude_km",
                "inclination_deg",
                "raan_deg",
                "nodes",
            },
            f"orbit.planes[{i}]",
        )
        if plane.get("role") not in ("sensing", "compute"):
            raise SpecError(f"orbit.planes[{i}].role 只准入 sensing|compute")
        if float(plane["altitude_km"]) <= 0:
            raise SpecError(f"orbit.planes[{i}].altitude_km 必须为正")
        nodes = plane.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise SpecError(f"orbit.planes[{i}].nodes 不能为空")
        for j, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise SpecError(f"orbit.planes[{i}].nodes[{j}] 必须是映射")
            _expect_keys(node, {"id", "true_anomaly_deg"}, f"orbit.planes[{i}].nodes[{j}]")
            nid = node["id"]
            if not isinstance(nid, int) or not 0 <= nid < n:
                raise SpecError(f"节点 id 必须在 [0, {n})：得 {nid}")
            if nid in seen:
                raise SpecError(f"节点 id {nid} 重复")
            seen.add(nid)
    if seen != set(range(n)):
        raise SpecError(f"orbit 必须覆盖全部节点 0..{n - 1}，现有 {sorted(seen)}")
    # 连通性：与 World 同一生成器，拒绝一启动就拆开的图
    from satsim.physics.orbit_links import (
        adjacency_from_edges,
        build_orbit_edges,
        graph_is_connected,
    )

    edges = build_orbit_edges(orbit, n)
    adj = adjacency_from_edges(n, edges)
    if not graph_is_connected(n, adj):
        raise SpecError("orbit 生成的邻接图不连通；加大 distance_threshold_km 或打开 same_plane_neighbors")


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
            "orbit",  # 仅 topology=orbit_planes 时使用
        },
        "base",
    )
    if float(cfg["drain_s"]) <= 0 or float(cfg["control_period_s"]) <= 0:
        raise SpecError("drain_s 和 control_period_s 必须为正")
    topo = cfg["topology"]
    if topo not in ("ring", "orbit_planes"):
        raise SpecError("topology 只准入 ring | orbit_planes")
    if not isinstance(cfg["nodes"], int):
        raise SpecError("nodes 必须是整数")
    n = cfg["nodes"]
    if topo == "ring":
        if not 2 <= n <= 8:
            raise SpecError("ring 的 nodes 只准入 2..8")
        if "orbit" in cfg:
            raise SpecError("topology=ring 时不准写 orbit")
    else:
        if not 2 <= n <= 16:
            raise SpecError("orbit_planes 的 nodes 只准入 2..16（见 ADR 0003）")
        if "orbit" not in cfg:
            raise SpecError("topology=orbit_planes 需要 orbit 块")
        _validate_orbit_block(cfg["orbit"], n)
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


def _validate_token_pad(pad: dict) -> None:
    _expect_keys(pad, {"mode", "long_out", "long_pin", "short_out", "short_pin"}, "token_pad")
    if pad["mode"] != "floor":
        raise SpecError("token_pad.mode 只准入 floor")


def validate_trace(trace: dict) -> None:
    kind = trace.get("kind")
    if kind == "synthetic_pass":
        _expect_keys(
            trace,
            {"kind", "compress_horizon_s", "pack_window_s", "target_n", "jitter_s", "token_pad"},
            "trace",
        )
        _validate_token_pad(trace["token_pad"])
        h, w = float(trace["compress_horizon_s"]), float(trace["pack_window_s"])
        if not 0 < h <= w:
            raise SpecError("需要 0 < compress_horizon_s <= pack_window_s")
        if int(trace["target_n"]) < 1:
            raise SpecError("target_n 至少为 1")
        return
    if kind == "poisson":
        allowed = {
            "kind",
            "window_s",
            "compress_horizon_s",
            "rate_per_s",
            "jitter_s",
            "target_n",
            "token_pad",
            "pack_window_s",
        }
        _expect_keys(trace, allowed, "trace")
        if "token_pad" not in trace:
            raise SpecError("poisson 需要 token_pad")
        _validate_token_pad(trace["token_pad"])
        if "window_s" not in trace and "compress_horizon_s" not in trace:
            trace["window_s"] = POISSON_DEFAULTS["window_s"]
        if "rate_per_s" not in trace:
            trace["rate_per_s"] = POISSON_DEFAULTS["rate_per_s"]
        if "jitter_s" not in trace:
            trace["jitter_s"] = POISSON_DEFAULTS["jitter_s"]
        window = float(trace.get("window_s") or trace.get("compress_horizon_s"))
        rate = float(trace["rate_per_s"])
        if window <= 0 or rate <= 0:
            raise SpecError("poisson 需要 window_s > 0 且 rate_per_s > 0")
        if "target_n" in trace and int(trace["target_n"]) < 1:
            raise SpecError("target_n 至少为 1")
        return
    if kind == "gamma":
        allowed = {
            "kind",
            "window_s",
            "compress_horizon_s",
            "rate_per_s",
            "cv",
            "jitter_s",
            "target_n",
            "token_pad",
            "pack_window_s",
        }
        _expect_keys(trace, allowed, "trace")
        if "token_pad" not in trace:
            raise SpecError("gamma 需要 token_pad")
        _validate_token_pad(trace["token_pad"])
        if "window_s" not in trace and "compress_horizon_s" not in trace:
            trace["window_s"] = GAMMA_DEFAULTS["window_s"]
        if "rate_per_s" not in trace:
            trace["rate_per_s"] = GAMMA_DEFAULTS["rate_per_s"]
        if "cv" not in trace:
            trace["cv"] = GAMMA_DEFAULTS["cv"]
        if "jitter_s" not in trace:
            trace["jitter_s"] = GAMMA_DEFAULTS["jitter_s"]
        window = float(trace.get("window_s") or trace.get("compress_horizon_s"))
        rate = float(trace["rate_per_s"])
        cv = float(trace["cv"])
        if window <= 0 or rate <= 0:
            raise SpecError("gamma 需要 window_s > 0 且 rate_per_s > 0")
        if cv <= 0:
            raise SpecError("gamma 需要 cv > 0")
        if "target_n" in trace and int(trace["target_n"]) < 1:
            raise SpecError("target_n 至少为 1")
        return
    if kind == "sharegpt":
        allowed = {
            "kind",
            "path",
            "window_s",
            "compress_horizon_s",
            "rate_per_s",
            "arrival",
            "cv",
            "jitter_s",
            "num_prompts",
            "n",
            "target_n",
            "long_threshold_tokens",
            "max_seqlen",
            "len_scale",
            "pack_window_s",
        }
        _expect_keys(trace, allowed, "trace")
        if "path" not in trace or not str(trace["path"]).strip():
            raise SpecError("sharegpt 需要 path")
        if "window_s" not in trace and "compress_horizon_s" not in trace:
            trace["window_s"] = SHAREGPT_DEFAULTS["window_s"]
        if "rate_per_s" not in trace:
            trace["rate_per_s"] = SHAREGPT_DEFAULTS["rate_per_s"]
        if "arrival" not in trace:
            trace["arrival"] = SHAREGPT_DEFAULTS["arrival"]
        if "cv" not in trace:
            trace["cv"] = SHAREGPT_DEFAULTS["cv"]
        if "jitter_s" not in trace:
            trace["jitter_s"] = SHAREGPT_DEFAULTS["jitter_s"]
        if "long_threshold_tokens" not in trace:
            trace["long_threshold_tokens"] = SHAREGPT_DEFAULTS["long_threshold_tokens"]
        if "max_seqlen" not in trace:
            trace["max_seqlen"] = SHAREGPT_DEFAULTS["max_seqlen"]
        window = float(trace.get("window_s") or trace.get("compress_horizon_s"))
        rate = float(trace["rate_per_s"])
        if window <= 0 or rate <= 0:
            raise SpecError("sharegpt 需要 window_s > 0 且 rate_per_s > 0")
        arrival = str(trace["arrival"])
        if arrival not in {"deterministic", "poisson", "gamma"}:
            raise SpecError("sharegpt.arrival 只准入 deterministic / poisson / gamma")
        if arrival == "gamma" and float(trace["cv"]) <= 0:
            raise SpecError("sharegpt gamma 需要 cv > 0")
        for key in ("num_prompts", "n", "target_n"):
            if key in trace and int(trace[key]) < 1:
                raise SpecError(f"{key} 至少为 1")
        if int(trace["long_threshold_tokens"]) < 1:
            raise SpecError("long_threshold_tokens 至少为 1")
        if int(trace["max_seqlen"]) < 8:
            raise SpecError("max_seqlen 过小")
        return
    raise SpecError("trace.kind 只准入 synthetic_pass / poisson / gamma / sharegpt")


def _coerce_id(value: Any) -> str:
    """YAML 1.1 会把 off 读成 False。"""
    if isinstance(value, str):
        return value
    if value is False:
        return "off"
    if value is True:
        return "on"
    return str(value)


def validate_policy(name: str, pol: dict) -> None:
    """只认 scheduler + link_cost + knobs；就地规范化。"""
    _expect_keys(
        pol,
        {"scheduler", "link_cost", "hysteresis", "knobs"},
        f"policy {name}",
    )
    if "scheduler" not in pol or "link_cost" not in pol:
        raise SpecError(f"{name}: 需要 scheduler 与 link_cost")
    pol["scheduler"] = _coerce_id(pol["scheduler"])
    pol["link_cost"] = _coerce_id(pol["link_cost"])

    if pol["scheduler"] not in SCHEDULER_IDS:
        raise SpecError(f"{name}: scheduler 只准入 {sorted(SCHEDULER_IDS)}")
    if pol["link_cost"] not in LINK_COST_IDS:
        raise SpecError(f"{name}: link_cost 只准入 {sorted(LINK_COST_IDS)}")
    allowed_lc = SCHEDULER_LINK_COST[pol["scheduler"]]
    if pol["link_cost"] not in allowed_lc:
        raise SpecError(
            f"{name}: scheduler={pol['scheduler']} 时 link_cost 只准入 {sorted(allowed_lc)}"
        )

    if pol["scheduler"] in SCHEDULER_NEEDS_HYSTERESIS:
        h = pol.get("hysteresis")
        if not isinstance(h, (int, float)) or not 0 < float(h) <= 1:
            raise SpecError(f"{name}: hysteresis 需要在 (0, 1]")
    elif "hysteresis" in pol:
        raise SpecError(f"{name}: {pol['scheduler']} 不需要 hysteresis")

    knobs = pol.get("knobs") or {}
    if not isinstance(knobs, dict):
        raise SpecError(f"{name}: knobs 必须是映射")
    extra = set(knobs) - KNOB_KEYS
    if extra:
        raise SpecError(f"{name}: knobs 只准入 {sorted(KNOB_KEYS)}，多了 {sorted(extra)}")

    applied = dict(knobs)
    # link_cost → cfg cost.mode（off→free）
    applied["cost.mode"] = LINK_COST_TO_MODE[pol["link_cost"]]
    pol["knobs"] = applied


def validate_t_swap(t_swap: dict, *, trace_kind: str | None = None) -> None:
    anchor = t_swap.get("anchor")
    if anchor == "max_cluster":
        _expect_keys(t_swap, {"anchor", "offset_s", "P_cap"}, "t_swap")
    elif anchor == "absolute":
        _expect_keys(t_swap, {"anchor", "t_s", "P_cap"}, "t_swap")
        if float(t_swap["t_s"]) < 0:
            raise SpecError("t_swap.t_s 不能为负")
    elif anchor == "first_seed":
        _expect_keys(t_swap, {"anchor", "offset_s", "P_cap"}, "t_swap")
    else:
        raise SpecError("t_swap.anchor 只准入 max_cluster / absolute / first_seed")
