"""按 trace.kind 分发线下到达生成。"""
from __future__ import annotations

from satsim.domain import Request
from satsim.input.generate.gamma import GammaArrival
from satsim.input.generate.poisson import PoissonArrival
from satsim.input.generate.sharegpt import ShareGPTArrival
from satsim.input.generate.synthetic_pass import SyntheticPassArrival
from satsim.input.schema import SpecError

_SYNTH = SyntheticPassArrival()
_POISSON = PoissonArrival()
_GAMMA = GammaArrival()
_SHAREGPT = ShareGPTArrival()


def generate_arrivals(trace: dict, p_long: float, seed: int) -> tuple[list[Request], float, int]:
    kind = trace.get("kind")
    if kind == "synthetic_pass":
        return _SYNTH.generate(trace, p_long=p_long, seed=seed)
    if kind == "poisson":
        return _POISSON.generate(trace, p_long=p_long, seed=seed)
    if kind == "gamma":
        return _GAMMA.generate(trace, p_long=p_long, seed=seed)
    if kind == "sharegpt":
        return _SHAREGPT.generate(trace, p_long=p_long, seed=seed)
    raise SpecError(f"未知 trace.kind: {kind}")


__all__ = ["generate_arrivals"]
