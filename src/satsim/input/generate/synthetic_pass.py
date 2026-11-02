"""压缩过境（synthetic_pass）：现网默认线下生成器。"""
from __future__ import annotations

import random
from collections import Counter

from satsim.domain import Request
from satsim.input.schema import SpecError

ROUND_S = 0.1


def cluster_anchor(times: list[float]) -> tuple[float, int]:
    """Densest 0.1s slot. A flat pass ties at 1; anchor the last slot so the whole pass is resident."""
    if not times:
        raise SpecError("空到达")
    slots = [round(t / ROUND_S) * ROUND_S for t in times]
    counts = Counter(slots)
    best = max(counts.values())
    winners = sorted(t for t, c in counts.items() if c == best)
    t_star = winners[-1] if best == 1 and len(winners) > 1 else winners[0]
    return float(t_star), int(best)


def make_pass(trace: dict, p_long: float, seed: int) -> tuple[list[Request], float, int]:
    n = int(trace["target_n"])
    horizon = float(trace["compress_horizon_s"])
    jitter = float(trace["jitter_s"])
    pad = trace["token_pad"]
    rng = random.Random(seed)
    times = []
    for i in range(n):
        t = 0.0 if n == 1 else i * horizon / (n - 1)
        if jitter:
            t += rng.uniform(-jitter, jitter)
        times.append(min(horizon, max(0.0, t)))
    anchor, count = cluster_anchor(times)
    reqs = []
    for i, t in enumerate(times):
        long = rng.random() < p_long
        reqs.append(
            Request(
                i=i,
                t=t,
                long=long,
                prompt_tokens=int(pad["long_pin"] if long else pad["short_pin"]),
                output_tokens=int(pad["long_out"] if long else pad["short_out"]),
            )
        )
    return reqs, anchor, count


class SyntheticPassArrival:
    """ArrivalProcess：线下生成完整 synthetic_pass 序列。"""

    def generate(
        self,
        trace: dict,
        *,
        p_long: float,
        seed: int,
    ) -> tuple[list[Request], float, int]:
        return make_pass(trace, p_long, seed)
