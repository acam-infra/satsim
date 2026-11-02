"""按 trace.kind 分发线下到达生成。"""
from __future__ import annotations

from satsim.domain import Request
from satsim.input.generate.poisson import PoissonArrival
from satsim.input.generate.synthetic_pass import SyntheticPassArrival
from satsim.input.schema import SpecError

_SYNTH = SyntheticPassArrival()
_POISSON = PoissonArrival()


def generate_arrivals(trace: dict, p_long: float, seed: int) -> tuple[list[Request], float, int]:
    kind = trace.get("kind")
    if kind == "synthetic_pass":
        return _SYNTH.generate(trace, p_long=p_long, seed=seed)
    if kind == "poisson":
        return _POISSON.generate(trace, p_long=p_long, seed=seed)
    raise SpecError(f"未知 trace.kind: {kind}")


# 兼容旧名：仅 synthetic_pass
__all__ = ["generate_arrivals"]
