"""Compute load & bottleneck probe for co-simulation research experiments.

Measures why naive/standard task dispatchers (Nearest, Round-Robin, Random, Least-Loaded)
suffer from severe compute load imbalance, dark-satellite queue buildup, and iteration
time inflation on heterogeneous Multi-SSO constellations.
"""
from __future__ import annotations

import math
import statistics as st
from collections import defaultdict

from satsim.cosim.record import _pct, summarize
from satsim.cosim.world import CoWorld
from satsim.method.engine.timing import iter_time


class LoadProbe:
    """Samples per-node queue depth, dark vs. lit concurrency, and iteration slowdown."""

    def __init__(self, sample_s: float = 10.0):
        self.sample_s = float(sample_s)
        self.node_q_sum: dict[int, float] = defaultdict(float)
        self.node_q_max: dict[int, int] = defaultdict(int)
        self.n_samples: int = 0
        self.dark_q_obs: list[int] = []
        self.lit_q_obs: list[int] = []
        self.active_q_obs: list[int] = []
        self.req_slowdown_obs: list[float] = []

    def attach(self, w: CoWorld) -> None:
        w.env.process(self._loop(w))

    def _loop(self, w: CoWorld):
        window = float(w.cfg["workload"]["arrival_window_s"])
        base_it = iter_time(1, 4096, 1.0, w.gpu)
        while w.env.now < window:
            self.n_samples += 1
            for j in w.tr.compute:
                q = len(w.nodes[j])
                self.node_q_sum[j] += q
                if q > self.node_q_max[j]:
                    self.node_q_max[j] = q
                if w.tr.is_dark(j, w.env.now):
                    self.dark_q_obs.append(q)
                else:
                    self.lit_q_obs.append(q)
                if q > 0:
                    self.active_q_obs.append(q)
                    slowdown = w.iter_of(j) / base_it
                    self.req_slowdown_obs.extend([slowdown] * q)
            yield w.env.timeout(self.sample_s)


def jains_index(xs: list[float]) -> float:
    if not xs or sum(xs) <= 1e-12:
        return 1.0
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    return (s1 * s1) / (len(xs) * s2)


def analyze_run(w: CoWorld, probe: LoadProbe) -> dict:
    """Combine standard SLO summary with detailed compute-load & plane distribution diagnostics."""
    base = summarize(w)
    tr = w.tr

    # Identify which compute nodes have any umbra intervals vs 100% sunlit nodes
    eclipsed_nodes = {j for j in tr.compute if len(tr.dark[j]) > 0}
    sunlit_nodes = {j for j in tr.compute if len(tr.dark[j]) == 0}

    admitted = [r for r in w.reqs if r.t_admit is not None and r.trail]
    admit_dark = sum(1 for r in admitted if tr.is_dark(r.trail[0], r.t_admit))
    admit_eclipsed_plane = sum(1 for r in admitted if r.trail[0] in eclipsed_nodes)
    admit_sunlit_plane = sum(1 for r in admitted if r.trail[0] in sunlit_nodes)

    # Per-plane dispatch share (by cluster tag)
    plane_counts: dict[str, int] = defaultdict(int)
    for r in admitted:
        cluster = str(tr.nodes[r.trail[0]].get("cluster", "unknown"))
        plane_counts[cluster] += 1
    n_adm = max(1, len(admitted))
    plane_shares = {k: v / n_adm for k, v in sorted(plane_counts.items())}

    # Node queue averages over arrival window
    ns = max(1, probe.n_samples)
    node_q_means = [probe.node_q_sum[j] / ns for j in tr.compute]
    eclipsed_q_mean = (
        st.fmean([probe.node_q_sum[j] / ns for j in eclipsed_nodes]) if eclipsed_nodes else 0.0
    )
    sunlit_q_mean = (
        st.fmean([probe.node_q_sum[j] / ns for j in sunlit_nodes]) if sunlit_nodes else 0.0
    )

    q_dark_mean = st.fmean(probe.dark_q_obs) if probe.dark_q_obs else 0.0
    q_lit_mean = st.fmean(probe.lit_q_obs) if probe.lit_q_obs else 0.0

    # Decode time inflation ratio for completed long requests
    decode_stretches = []
    for r in w.reqs:
        if r.long and r.t_admit is not None and r.t_decoded is not None:
            ideal_s = r.pout * w.ideal_iter(r)
            if ideal_s > 0:
                decode_stretches.append((r.t_decoded - r.t_admit) / ideal_s)

    return {
        **base,
        # Dispatch distribution
        "admit_dark_frac": admit_dark / n_adm,
        "eclipsed_plane_disp_frac": admit_eclipsed_plane / n_adm,
        "sunlit_plane_disp_frac": admit_sunlit_plane / n_adm,
        "plane_shares": plane_shares,
        # Queue buildup & spatial load imbalance
        "q_mean_all": st.fmean(node_q_means) if node_q_means else 0.0,
        "q_eclipsed_plane_mean": eclipsed_q_mean,
        "q_sunlit_plane_mean": sunlit_q_mean,
        "q_dark_instant_mean": q_dark_mean,
        "q_lit_instant_mean": q_lit_mean,
        "q_dark_over_lit": q_dark_mean / max(1e-6, q_lit_mean) if q_lit_mean > 0 else 0.0,
        "q_active_p90": _pct([float(x) for x in probe.active_q_obs], 90) or 0.0,
        "q_node_max": float(max(probe.node_q_max.values(), default=0)),
        "load_jains_index": jains_index(node_q_means),
        # Compute slowdown
        "iter_slowdown_mean": st.fmean(probe.req_slowdown_obs) if probe.req_slowdown_obs else 1.0,
        "decode_stretch_p50": _pct(decode_stretches, 50) or 1.0,
        "decode_stretch_p90": _pct(decode_stretches, 90) or 1.0,
    }
