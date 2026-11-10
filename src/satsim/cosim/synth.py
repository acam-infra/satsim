"""Parametric on-the-fly orbit synthesis for co-simulation research experiments.

Eliminates the need to hand-edit ``orbitsim/export_cosim.py`` and dump static ``.trace.json``
files before running scheduling experiments. Directly synthesizes ``Trace`` objects in memory
for Multi-SSO (heterogeneous sunlit dawn-dusk + eclipsed noon/morning planes) and Walker shells.
"""
from __future__ import annotations

import sys
from pathlib import Path

from satsim.cosim.trace import Trace


def _ensure_orbitsim():
    try:
        from orbitsim import contact_trace, fleet_synth
        return contact_trace, fleet_synth
    except ImportError:
        # Auto-discover sibling orbitsim submodule in satcosim umbrella repo
        for parent in Path(__file__).resolve().parents:
            cand = parent / "orbitsim" / "src"
            if (cand / "orbitsim").is_dir():
                sys.path.insert(0, str(cand))
                from orbitsim import contact_trace, fleet_synth
                return contact_trace, fleet_synth
        raise


def synth_multi_sso(
    *,
    compute_ltans: tuple[float, ...] = (10.5, 13.5, 6.0),
    sats_per_plane: int = 10,
    compute_alt_km: float = 550.0,
    phase_f: int = 1,
    rs_specs: list[dict] | None = None,
    inter_mode: str = "grid",  # "grid" (+Grid cross-plane links) | "mesh" (all cross-plane in range)
    inter_range_km: float = 5000.0,
    rs_range_km: float = 4000.0,
    rs_policy: str = "sticky",
    doy: float = 81.0,
    horizon_s: float = 7200.0,
    dt_s: float = 5.0,
    name: str = "multi_sso",
) -> Trace:
    """Synthesize a Multi-SSO constellation with heterogeneous LTAN planes (eclipsed + sunlit).

    Default configuration creates:
    - 3 Remote-Sensing (RS) satellites on imaging SSO planes (RS-1030a, RS-1030b, RS-1330).
    - 3 Compute SSO planes × 10 satellites/plane (30 compute satellites total; 36° spacing keeps
      intra-plane ring line-of-sight above RE + 80 km):
      - Plane 0 (LTAN 10:30, morning): ~36% time in umbra (throttled to f_dark=0.35), co-planar with RS-1030
      - Plane 1 (LTAN 13:30, afternoon): ~36% time in umbra (throttled to f_dark=0.35), co-planar with RS-1330
      - Plane 2 (LTAN 06:00, dawn-dusk): 0% time in umbra (100% sunlit f=1.0)
    """
    ct, fs = _ensure_orbitsim()

    if rs_specs is None:
        rs_specs = [
            fs.make_sat("RS-1030a", 500.0, ltan_h=10.5, u0_deg=15.0, role="rs", group="rs", cluster="RS-1030a"),
            fs.make_sat("RS-1030b", 500.0, ltan_h=10.5, u0_deg=195.0, role="rs", group="rs", cluster="RS-1030b"),
            fs.make_sat("RS-1330", 500.0, ltan_h=13.5, u0_deg=105.0, role="rs", group="rs", cluster="RS-1330"),
        ]
    else:
        rs_specs = [dict(s, role="rs", group="rs", cluster=s.get("cluster", s["name"])) for s in rs_specs]

    n_planes = len(compute_ltans)
    cs: list[dict] = []
    for p_idx, ltan in enumerate(compute_ltans):
        tag = f"LTAN{ltan:04.1f}h"
        for k in range(sats_per_plane):
            u0 = (360.0 * k / sats_per_plane + 360.0 * phase_f * p_idx / max(1, n_planes)) % 360.0
            cs.append(
                fs.make_sat(
                    f"C/p{p_idx}s{k}",
                    compute_alt_km,
                    ltan_h=float(ltan),
                    u0_deg=u0,
                    role="compute",
                    group="compute",
                    plane=p_idx,
                    cluster=tag,
                )
            )

    sats = rs_specs + cs
    pairs: list[tuple[str, str, str, float]] = []
    intra_range = ct.chord_m(compute_alt_km * 1e3, 360.0 / sats_per_plane) * 1.05
    for p_idx in range(n_planes):
        names = [s["name"] for s in cs if s["plane"] == p_idx]
        pairs += [(a, b, "intra", intra_range) for a, b in ct.ring_pairs(names, closed=True)]

    inter_m = inter_range_km * 1e3
    if inter_mode == "grid":
        pairs += [(a, b, "inter", inter_m) for a, b in ct.nearest_plane_pairs(cs, doy)]
    elif inter_mode == "mesh":
        for i, a in enumerate(cs):
            for b in cs[i + 1 :]:
                if a["plane"] != b["plane"]:
                    pairs.append((a["name"], b["name"], "inter", inter_m))
    else:
        raise ValueError(f"unknown inter_mode {inter_mode!r}; expected 'mesh' or 'grid'")

    attach = [ct.AttachRule("rs", "compute", "rs", rs_range_km * 1e3, rs_policy)]
    raw = ct.build_trace(
        sats,
        doy=doy,
        horizon_s=horizon_s,
        dt_s=dt_s,
        pairs=pairs,
        attach=attach,
        name=name,
    )
    # Enrich node dicts with plane index and LTAN so probes can group by orbital plane
    for i, s in enumerate(sats):
        raw["nodes"][i]["plane"] = s.get("plane", -1)
        raw["nodes"][i]["ltan_h"] = s.get("ltan_h")
    tr = Trace(raw)
    tr.raw_stats = raw["stats"]
    return tr
