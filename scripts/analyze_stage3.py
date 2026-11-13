"""Analyze Stage 3 runs.csv: paired per-seed tables, frontier crossover, and S5 Go/No-Go verdict."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path


def _pct(val: float, base: float) -> str:
    if base == 0:
        return "N/A"
    d = (val - base) / abs(base) * 100.0
    return f"{d:+.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/stage3_discrete_power")
    args = ap.parse_args()

    out_dir = Path(args.dir).resolve()
    rows = list(csv.DictReader((out_dir / "runs.csv").open("r", encoding="utf-8")))

    pol_order = ["place_once", "least_loaded", "cool_aware", "deadline_rc", "discrete_rh_greedy"]

    for sc in ["coop_sso2", "coop_w53"]:
        print(f"\n=== MAIN 5-SEED: {sc} ===")
        sc_rows = [r for r in rows if r["scenario"] == sc and r["sweep"] == "main"]
        base_rows = {int(r["seed"]): r for r in sc_rows if r["policy"] == "place_once"}
        means: dict[str, dict[str, float]] = {}
        for pol in pol_order:
            pr = [r for r in sc_rows if r["policy"] == pol]
            if not pr:
                continue
            means[pol] = {
                m: st.fmean(float(r[m]) for r in pr if r.get(m) not in (None, "", "None"))
                for m in [
                    "good_long_frac",
                    "goodput_tok_s",
                    "waste_ratio",
                    "long_p95_all_s",
                    "link_GB_hops",
                    "zero_power_frac",
                    "energy_kj_total",
                    "min_thermal_margin_C",
                    "migrations",
                ]
            }
        b = means["place_once"]
        for pol in pol_order:
            if pol not in means:
                continue
            m = means[pol]
            print(
                f"{pol:20s} | SLO={m['good_long_frac']*100:5.1f}% ({_pct(m['good_long_frac'], b['good_long_frac'])}) "
                f"| goodput={m['goodput_tok_s']:7.1f} ({_pct(m['goodput_tok_s'], b['goodput_tok_s'])}) "
                f"| waste={m['waste_ratio']*100:5.1f}% "
                f"| p95={m['long_p95_all_s']:6.1f}s "
                f"| linkGB={m['link_GB_hops']:6.1f} "
                f"| 0W={m['zero_power_frac']*100:5.1f}% "
                f"| E={m['energy_kj_total']:7.1f}kJ ({_pct(m['energy_kj_total'], b['energy_kj_total'])}) "
                f"| mig={m['migrations']:5.1f}"
            )

        # Paired per-seed delta: discrete_rh_greedy vs deadline_rc
        dr_rows = {int(r["seed"]): r for r in sc_rows if r["policy"] == "discrete_rh_greedy"}
        dl_rows = {int(r["seed"]): r for r in sc_rows if r["policy"] == "deadline_rc"}
        common_seeds = sorted(set(dr_rows) & set(dl_rows))
        if common_seeds:
            d_slo = [
                (float(dr_rows[s]["good_long_frac"]) - float(dl_rows[s]["good_long_frac"])) * 100.0
                for s in common_seeds
            ]
            d_ene = [
                (float(dr_rows[s]["energy_kj_total"]) - float(dl_rows[s]["energy_kj_total"]))
                / float(dl_rows[s]["energy_kj_total"])
                * 100.0
                for s in common_seeds
            ]
            print(
                f"  Paired (discrete_rh_greedy - deadline_rc, seeds={common_seeds}): "
                f"ΔSLO_pp={st.fmean(d_slo):+.2f}pp (min={min(d_slo):+.2f}, max={max(d_slo):+.2f}), "
                f"ΔEnergy%={st.fmean(d_ene):+.2f}% (min={min(d_ene):+.2f}%, max={max(d_ene):+.2f}%)"
            )

    for sw in ["bw_gbps", "t_max_c"]:
        print(f"\n=== FRONTIER SWEEP (coop_w53): {sw} ===")
        sw_rows = [r for r in rows if r["sweep"] == sw]
        params = sorted({float(r["param"]) for r in sw_rows})
        for p in params:
            line = f"param={p:5.1f}"
            for pol in ["place_once", "deadline_rc", "discrete_rh_greedy"]:
                pr = [r for r in sw_rows if float(r["param"]) == p and r["policy"] == pol]
                if not pr:
                    continue
                slo = st.fmean(float(r["good_long_frac"]) for r in pr) * 100.0
                ene = st.fmean(float(r["energy_kj_total"]) for r in pr)
                line += f" | {pol}: SLO={slo:5.1f}% E={ene:7.1f}kJ"
            print(line)


if __name__ == "__main__":
    main()
