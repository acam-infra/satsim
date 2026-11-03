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
    raise SpecError("trace.kind 只准入 synthetic_pass / poisson / gamma")


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
