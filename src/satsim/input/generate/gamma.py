"""Gamma 间隔到达：用 CV 控制突发（Llumnix OSDI'24 风格）。

间隔 X ~ Gamma(k, θ)，均值 μ=1/rate_per_s，变异系数 CV=σ/μ。
关系：k = 1/CV²，θ = μ·CV² = CV²/rate。
CV=1 ⇒ k=1 ⇒ 指数间隔 ⇒ 齐次泊松。
"""
from __future__ import annotations

import random

from satsim.domain import Request
from satsim.input.generate.synthetic_pass import cluster_anchor
from satsim.input.schema import SpecError

_DEFAULT_RATE = 0.1
_DEFAULT_WINDOW = 60.0
_DEFAULT_CV = 1.0


class GammaArrival:
    """YAML::
        trace:
          kind: gamma
          window_s: 180
          rate_per_s: 0.4
          cv: 4.0
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
        cv = float(trace.get("cv", _DEFAULT_CV))
        jitter = float(trace.get("jitter_s") or 0.0)
        if window <= 0 or rate <= 0:
            raise SpecError("gamma 需要 window_s > 0 且 rate_per_s > 0")
        if cv <= 0:
            raise SpecError("gamma 需要 cv > 0")
        pad = trace.get("token_pad")
        if not isinstance(pad, dict):
            raise SpecError("gamma 需要 token_pad")

        # k = shape, θ = scale; mean = kθ = 1/rate, CV = 1/√k
        shape = 1.0 / (cv * cv)
        scale = (cv * cv) / rate
        rng = random.Random(seed)
        times: list[float] = []
        t = 0.0
        target_n = trace.get("target_n")
        while True:
            t += rng.gammavariate(shape, scale)
            if t >= window:
                break
            ti = t
            if jitter:
                ti = min(window, max(0.0, ti + rng.uniform(-jitter, jitter)))
            times.append(ti)
            if target_n is not None and len(times) >= int(target_n):
                break
        if not times:
            raise SpecError("gamma 到达为空（增大 rate_per_s / window_s 或减小 cv）")
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
