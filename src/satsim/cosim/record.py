"""Playback recording and per-run metrics.

The recorder samples the world every ``frame_s`` and keeps every request event, compact enough to
embed in one JSON per experiment that the 3-D viewer (orbitsim viz/play.html) replays.

Frame  ``{t, load: [resident requests per node], kv: [KV fill 0..1 per node], e: [[edge, Gbps]]}``
Event  ``[t, kind, {fields}]``  kinds: arrive, upload, admit, mig, mig_done, mig_abort, cut,
       deliver, lost.
"""
from __future__ import annotations

import statistics as st

from satsim.cosim.power_tiers import (
    ACC_ENERGY_J,
    ACC_TIER_TIME_S,
    ACC_TIER_UNSAT_S,
    ACC_ZERO_POWER_S,
    is_discrete_power,
    parse_discrete_power_spec,
)
from satsim.cosim.world import CoWorld


class Recorder:
    def __init__(self, frame_s: float):
        self.frame_s = float(frame_s)
        self.frames: list[dict] = []
        self.events: list[list] = []

    def event(self, t: float, kind: str, **kw) -> None:
        self.events.append([round(t, 2), kind, kw])

    def attach(self, w: CoWorld) -> None:
        w.env.process(self._sample(w))

    def _sample(self, w: CoWorld):
        n = w.tr.n
        while True:
            load = [0] * n
            kv = [0.0] * n
            for j in w.tr.compute:
                load[j] = len(w.nodes[j])
                kv[j] = round(min(1.0, w.kv_used(j) / w.kv_cap), 3)
            w.net.advance(w.env.now)
            rates = sorted(w.net.edge_rates().items())
            self.frames.append({
                "t": round(w.env.now, 1),
                "load": load,
                "kv": kv,
                "e": [[e, round(r * 8 / 1e9, 3)] for e, r in rates if r > 0],
            })
            yield w.env.timeout(self.frame_s)


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(w: CoWorld) -> dict:
    rs = w.reqs
    longs = [r for r in rs if r.long]
    delivered = [r for r in rs if r.t_delivered is not None]
    good = [r for r in rs if w.good(r)]
    lat_long = [r.t_delivered - r.t for r in longs if r.t_delivered is not None]
    wait = [r.t_admit - r.t for r in rs if r.t_admit is not None]
    gap = [r.t_delivered - r.t_decoded for r in delivered]
    tok = sum(r.gen for r in rs)
    window = float(w.cfg["workload"]["arrival_window_s"])
    bad = [r for r in rs if not w.good(r)]
    badput_gen = sum(r.gen for r in bad)
    goodput_gen = sum(r.gen for r in good)
    horizon = w.tr.horizon
    lat_all = [r.t_delivered - r.t if r.t_delivered is not None else horizon - r.t for r in rs]
    lat_long_all = [r.t_delivered - r.t if r.t_delivered is not None else horizon - r.t for r in longs]
    out = {
        "n_req": len(rs),
        "n_long": len(longs),
        "delivered": len(delivered),
        "good": len(good),
        "good_long": sum(1 for r in good if r.long),
        "good_frac": len(good) / max(1, len(rs)),
        "good_long_frac": sum(1 for r in good if r.long) / max(1, len(longs)),
        "goodput_tok_s": sum(r.pout for r in good) / window,
        "lat_long_p50_s": _pct(lat_long, 50),
        "lat_long_p90_s": _pct(lat_long, 90),
        "admit_wait_p90_s": _pct(wait, 90),
        "deliver_gap_p90_s": _pct(gap, 90),
        "dark_tok_share": sum(r.dark_tok for r in rs) / max(1.0, tok),
        "stall_s_mean": st.fmean([r.stall_s for r in rs]) if rs else 0.0,
        "migrations": w.acc["mig"],
        "mig_abort": w.acc["mig_abort"],
        "mig_GB": w.acc["mig_bytes"] / 1e9,
        "delta_kv_GB": w.acc.get("delta_kv_bytes", 0.0) / 1e9,
        "upload_fail": w.acc["upload_fail"],
        "link_GB_hops": w.net.bytes_hops / 1e9,
        "badput_tok_s": badput_gen / window,
        "waste_ratio": badput_gen / (goodput_gen + badput_gen) if goodput_gen + badput_gen > 0 else 0.0,
        "undelivered_frac": sum(1 for r in rs if r.t_delivered is None) / max(1, len(rs)),
        "e2e_p95_all_s": _pct(lat_all, 95),
        "long_p95_all_s": _pct(lat_long_all, 95),
        "throttle_s": float(w.acc.get("throttle_s", 0.0)),
    }
    if is_discrete_power(w.cfg):
        spec = parse_discrete_power_spec(w.cfg)
        assert spec is not None
        total_node_s = max(1e-9, len(w.tr.compute) * horizon)
        tier_time_s = w.acc.get(ACC_TIER_TIME_S) or {}
        zero_s = float(w.acc.get(ACC_ZERO_POWER_S, tier_time_s.get(0, 0.0)))
        tier_sum = sum(float(v) for v in tier_time_s.values())
        denom = tier_sum if tier_sum > 0.0 else total_node_s

        peak_temp_c: float | None = None
        diag = w.power_model.diagnostics() if hasattr(w, "power_model") else None
        if diag is None and w.thermal is not None:
            diag = w.thermal.diagnostics()
        if diag and diag.get("peak_T_C"):
            peaks = [float(diag["peak_T_C"][j]) for j in w.tr.compute]
            if peaks:
                peak_temp_c = max(peaks)

        out["zero_power_frac"] = zero_s / total_node_s
        out["power_tier_fracs"] = {
            t.name: float(tier_time_s.get(t.index, 0.0)) / denom for t in spec.tiers
        }
        out["energy_kj_total"] = float(w.acc.get(ACC_ENERGY_J, 0.0)) / 1000.0
        out["tier_unsat_s"] = float(w.acc.get(ACC_TIER_UNSAT_S, 0.0))
        out["peak_temp_C"] = peak_temp_c
        out["min_thermal_margin_C"] = (
            spec.t_max_c - peak_temp_c if peak_temp_c is not None else None
        )
    return out


def cumulative_good(w: CoWorld, step: float) -> list[list[float]]:
    """[t, good requests delivered by t] on a fixed grid, for the viewer's comparison chart."""
    ts = sorted(r.t_delivered for r in w.reqs if w.good(r))
    out, k, t = [], 0, 0.0
    while t <= w.tr.horizon + 1e-9:
        while k < len(ts) and ts[k] <= t:
            k += 1
        out.append([round(t, 1), k])
        t += step
    return out


def request_table(w: CoWorld) -> list[list]:
    """Per request: [i, origin, long, t, t_admit, t_decoded, t_delivered, good, trail]."""
    r2 = lambda x: None if x is None else round(x, 1)
    return [[r.i, r.origin, int(r.long), r2(r.t), r2(r.t_admit), r2(r.t_decoded), r2(r.t_delivered),
             int(w.good(r)), r.trail] for r in w.reqs]
