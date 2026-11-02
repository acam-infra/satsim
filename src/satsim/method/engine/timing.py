"""功率-频率与单步迭代时间。"""
from __future__ import annotations


def kv_bytes_per_tok(model: dict) -> int:
    return 2 * model["layers"] * model["kv_heads"] * model["head_dim"] * model["bytes_per_elem"]


def clock_factor(p_cap: float, thermal: dict) -> float:
    dyn = max(p_cap - thermal["P_idle"], 1e-9) / (thermal["P_max"] - thermal["P_idle"])
    f = dyn ** (1.0 / thermal["gamma"])
    return min(max(f, thermal["f_min"]), 1.0)


def iter_time(nd: int, ctx: float, f: float, gpu: dict) -> float:
    beta = gpu["mem_freq_exp"]
    mem = gpu["t_weights_s"] + gpu["t_kv_tok_s"] * ctx
    return gpu["t_fixed_s"] + mem / (f**beta) + (gpu["t_seq_s"] * nd) / f
