"""泊松到达：离线生成完整序列；已支持 trace.kind=poisson。"""
from __future__ import annotations

import random

from satsim.domain import Request
from satsim.input.generate.synthetic_pass import cluster_anchor
from satsim.input.schema import SpecError

# schema 填默认后仍可读；此处作防御默认
_DEFAULT_RATE = 0.1
_DEFAULT_WINDOW = 60.0


class PoissonArrival:
    """齐次泊松到达。

    YAML::
        trace:
          kind: poisson
          window_s: 60          # 默认 60；亦接受 compress_horizon_s
          rate_per_s: 0.1       # 默认 0.1
          jitter_s: 0           # 可选，对时刻再抖
          token_pad: ...
    """

    def generate(
        self,
        trace: dict,
        *,
        p_long: float,
        seed: int,
    ) -> tuple[list[Request], float, int]:
        window = float(trace.get("window_s") or trace.get("compress_horizon_s") or _DEFAULT_WINDOW)
        rate = float(trace.get("rate_per_s", _DEFAULT_RATE))
        jitter = float(trace.get("jitter_s") or 0.0)
        if window <= 0 or rate <= 0:
            raise SpecError("poisson 需要 window_s > 0 且 rate_per_s > 0")
        pad = trace.get("token_pad")
        if not isinstance(pad, dict):
            raise SpecError("poisson 需要 token_pad")
        rng = random.Random(seed)
        times: list[float] = []
        t = 0.0
        # 目标规模可选：有 target_n 时截断/提示，默认按过程采满窗
        target_n = trace.get("target_n")
        while True:
            t += rng.expovariate(rate)
            if t >= window:
                break
            ti = t
            if jitter:
                ti = min(window, max(0.0, ti + rng.uniform(-jitter, jitter)))
            times.append(ti)
            if target_n is not None and len(times) >= int(target_n):
                break
        if not times:
            raise SpecError("泊松到达为空（增大 rate_per_s 或 window_s）")
        times.sort()
        reqs = []
        for i, ti in enumerate(times):
            long = rng.random() < p_long
            reqs.append(
                Request(
                    i=i,
                    t=ti,
                    long=long,
                    prompt_tokens=int(pad["long_pin"] if long else pad["short_pin"]),
                    output_tokens=int(pad["long_out"] if long else pad["short_out"]),
                )
            )
        anchor, count = cluster_anchor([r.t for r in reqs])
        return reqs, anchor, count
