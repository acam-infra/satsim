"""ShareGPT 长度分布 + 合成到达（对齐 EcoServe：无时间戳，按 rate 造到达）。

YAML::
    trace:
      kind: sharegpt
      path: input/datasets/sharegpt_lengths_v1.csv   # prompt_len,output_len
      window_s: 180
      rate_per_s: 0.5
      arrival: deterministic   # | poisson | gamma
      cv: 1.0                  # gamma 用
      num_prompts: 120         # 可选；亦接受 n / target_n
      long_threshold_tokens: 761  # pin+pout >= 则 long；默认 761（char/4 CSV 约 p80）
      max_seqlen: 4096         # 防御再滤；与 len_scale 一起裁剪
      len_scale: 1.0           # 可选，乘在 CSV 长度上（冒烟可 >1 加压）
      jitter_s: 0
      # 不需要 token_pad
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from satsim.domain import Request
from satsim.input.generate.synthetic_pass import cluster_anchor
from satsim.input.schema import SpecError

_DEFAULT_RATE = 0.5
_DEFAULT_WINDOW = 180.0
_DEFAULT_ARRIVAL = "deterministic"
_DEFAULT_CV = 1.0
# 空白分词 sharegpt_lengths_v1.csv 上约 p80，贴近 workload.p_long=0.2
_DEFAULT_LONG_THR = 761
_DEFAULT_MAX_SEQLEN = 4096


def _resolve_path(path_s: str) -> Path:
    p = Path(path_s)
    if p.is_file():
        return p
    # 相对仓库根（…/satsim）
    root = Path(__file__).resolve().parents[4]
    cand = root / path_s
    if cand.is_file():
        return cand
    raise SpecError(f"sharegpt 找不到 path: {path_s}")


def _load_lengths(path: Path, max_seqlen: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "prompt_len" not in reader.fieldnames or "output_len" not in reader.fieldnames:
            raise SpecError("sharegpt CSV 需要表头 prompt_len,output_len")
        for row in reader:
            try:
                pin = int(row["prompt_len"])
                pout = int(row["output_len"])
            except (KeyError, TypeError, ValueError) as e:
                raise SpecError(f"sharegpt CSV 行非法: {row}") from e
            if pin < 4 or pout < 4:
                continue
            if pin > max_seqlen or pin + pout > max_seqlen:
                continue
            rows.append((pin, pout))
    if not rows:
        raise SpecError("sharegpt CSV 过滤后为空")
    return rows


def _arrival_times(
    *,
    window: float,
    rate: float,
    arrival: str,
    cv: float,
    jitter: float,
    num_prompts: int | None,
    rng: random.Random,
) -> list[float]:
    times: list[float] = []
    if arrival == "deterministic":
        # EcoServe 主 harness：interval = 1/rate；首请求 t=0
        t = 0.0
        while t < window:
            times.append(t)
            if num_prompts is not None and len(times) >= num_prompts:
                break
            t += 1.0 / rate
    elif arrival == "poisson":
        t = 0.0
        while True:
            t += rng.expovariate(rate)
            if t >= window:
                break
            times.append(t)
            if num_prompts is not None and len(times) >= num_prompts:
                break
    elif arrival == "gamma":
        shape = 1.0 / (cv * cv)
        scale = (cv * cv) / rate
        t = 0.0
        while True:
            t += rng.gammavariate(shape, scale)
            if t >= window:
                break
            times.append(t)
            if num_prompts is not None and len(times) >= num_prompts:
                break
    else:
        raise SpecError(f"sharegpt.arrival 只准入 deterministic/poisson/gamma，收到 {arrival}")

    if jitter:
        times = [min(window, max(0.0, ti + rng.uniform(-jitter, jitter))) for ti in times]
        times.sort()
    if not times:
        raise SpecError("sharegpt 到达为空（增大 rate_per_s / window_s / num_prompts）")
    return times


class ShareGPTArrival:
    """长度来自 CSV；到达由 rate + arrival 合成。p_long 在设了 long_threshold 时不用。"""

    def generate(
        self,
        trace: dict,
        *,
        p_long: float,
        seed: int,
    ) -> tuple[list[Request], float, int]:
        path_s = trace.get("path")
        if not path_s:
            raise SpecError("sharegpt 需要 path")
        window = float(trace.get("window_s") or trace.get("compress_horizon_s") or _DEFAULT_WINDOW)
        rate = float(trace.get("rate_per_s", _DEFAULT_RATE))
        arrival = str(trace.get("arrival") or _DEFAULT_ARRIVAL)
        cv = float(trace.get("cv", _DEFAULT_CV))
        jitter = float(trace.get("jitter_s") or 0.0)
        max_seqlen = int(trace.get("max_seqlen") or _DEFAULT_MAX_SEQLEN)
        if window <= 0 or rate <= 0:
            raise SpecError("sharegpt 需要 window_s > 0 且 rate_per_s > 0")
        if arrival == "gamma" and cv <= 0:
            raise SpecError("sharegpt gamma 需要 cv > 0")

        num_raw = trace.get("num_prompts", trace.get("n", trace.get("target_n")))
        num_prompts = int(num_raw) if num_raw is not None else None
        if num_prompts is not None and num_prompts < 1:
            raise SpecError("num_prompts 至少为 1")

        # long：优先阈值；否则回退 p_long 随机（一般不走到）
        thr = trace.get("long_threshold_tokens")
        if thr is None:
            thr = _DEFAULT_LONG_THR
        long_thr = int(thr) if thr is not None else None

        lengths = _load_lengths(_resolve_path(str(path_s)), max_seqlen)
        rng = random.Random(seed)
        times = _arrival_times(
            window=window,
            rate=rate,
            arrival=arrival,
            cv=cv,
            jitter=jitter,
            num_prompts=num_prompts,
            rng=rng,
        )

        n = len(times)
        if len(lengths) >= n:
            chosen = rng.sample(lengths, n)
        else:
            chosen = [lengths[rng.randrange(len(lengths))] for _ in range(n)]

        scale = float(trace.get("len_scale") or 1.0)
        if scale <= 0:
            raise SpecError("len_scale 必须 > 0")

        reqs: list[Request] = []
        for i, (ti, (pin, pout)) in enumerate(zip(times, chosen)):
            if scale != 1.0:
                pin = min(max_seqlen, max(4, int(round(pin * scale))))
                pout = min(max(4, int(round(pout * scale))), max(4, max_seqlen - pin))
            if long_thr is not None:
                long = (pin + pout) >= long_thr
            else:
                long = rng.random() < p_long
            reqs.append(
                Request(
                    i=i,
                    t=float(ti),
                    long=bool(long),
                    prompt_tokens=int(pin),
                    output_tokens=int(pout),
                )
            )
        anchor, count = cluster_anchor([r.t for r in reqs])
        return reqs, anchor, count


__all__ = ["ShareGPTArrival"]
