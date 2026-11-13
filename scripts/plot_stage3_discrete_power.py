"""Plot Stage 3 discrete power results directly from runs.csv."""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]
    if not vals:
        return 0.0, 0.0
    return st.fmean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/stage3_discrete_power")
    args = ap.parse_args()

    out_dir = Path(args.dir).resolve()
    rows = _load_rows(out_dir / "runs.csv")

    # 1. Main 5-seed comparison (coop_sso2 vs coop_w53)
    pol_order = ["place_once", "least_loaded", "cool_aware", "deadline_rc", "discrete_rh_greedy"]
    pol_labels = ["place_once", "least_loaded", "cool_aware", "deadline_rc", "discrete_rh\n_greedy"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), dpi=160)
    scenarios = [("coop_sso2", "coop-sso2 (16 sats, 13 req/min)"), ("coop_w53", "coop-w53 (40 sats, 30 req/min)")]

    for col, (sc, sc_title) in enumerate(scenarios):
        sc_rows = [r for r in rows if r["scenario"] == sc and r["sweep"] == "main"]
        slo_m, slo_s, ene_m, ene_s = [], [], [], []
        for pol in pol_order:
            pr = [r for r in sc_rows if r["policy"] == pol]
            m1, s1 = _mean_std(pr, "good_long_frac")
            m2, s2 = _mean_std(pr, "energy_kj_total")
            slo_m.append(m1 * 100.0)
            slo_s.append(s1 * 100.0)
            ene_m.append(m2)
            ene_s.append(s2)

        colors = ["#6b7280", "#9ca3af", "#60a5fa", "#f59e0b", "#10b981"]
        ax0 = axes[0, col]
        ax0.bar(range(len(pol_order)), slo_m, yerr=slo_s, capsize=3, color=colors)
        ax0.set_xticks(range(len(pol_order)))
        ax0.set_xticklabels(pol_labels, fontsize=8)
        ax0.set_ylabel("Long SLO Attainment (%) ↑")
        ax0.set_title(sc_title, fontsize=10, fontweight="bold")
        ax0.grid(axis="y", alpha=0.3)

        ax1 = axes[1, col]
        ax1.bar(range(len(pol_order)), ene_m, yerr=ene_s, capsize=3, color=colors)
        ax1.set_xticks(range(len(pol_order)))
        ax1.set_xticklabels(pol_labels, fontsize=8)
        ax1.set_ylabel("Total Constellation Energy (kJ) ↓")
        ax1.grid(axis="y", alpha=0.3)

    fig.suptitle("Stage 3 (5-Seed Mean ± Std, Fair Quota 5s / 24 / 4)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_stage3_main.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Frontier sweeps on coop_w53
    fig2, axes2 = plt.subplots(2, 2, figsize=(11.5, 7.2), dpi=160)
    sweeps = [
        ("bw_gbps", "Link Bandwidth (Gbps)", [1.0, 2.5, 5.0, 10.0, 25.0]),
        ("t_max_c", "Thermal Cap T_max_C (°C)", [72.0, 76.0, 80.0, 84.0]),
    ]
    sw_pols = [("place_once", "#6b7280", "o--"), ("deadline_rc", "#f59e0b", "s-"), ("discrete_rh_greedy", "#10b981", "^-")]

    for col, (sw_name, xlabel, xs) in enumerate(sweeps):
        for pol, color, fmt in sw_pols:
            ys_slo, ys_ene = [], []
            for x in xs:
                pr = [r for r in rows if r["sweep"] == sw_name and float(r["param"]) == x and r["policy"] == pol]
                m1, _ = _mean_std(pr, "good_long_frac")
                m2, _ = _mean_std(pr, "energy_kj_total")
                ys_slo.append(m1 * 100.0)
                ys_ene.append(m2)
            axes2[0, col].plot(xs, ys_slo, fmt, color=color, label=pol, linewidth=2, markersize=6)
            axes2[1, col].plot(xs, ys_ene, fmt, color=color, label=pol, linewidth=2, markersize=6)
        axes2[0, col].set_ylabel("Long SLO Attainment (%) ↑")
        axes2[1, col].set_ylabel("Total Energy (kJ) ↓")
        axes2[1, col].set_xlabel(xlabel)
        axes2[0, col].grid(alpha=0.3)
        axes2[1, col].grid(alpha=0.3)
        axes2[0, col].legend(fontsize=8)

    fig2.suptitle("coop-w53 Compute-Comm-Energy Frontier (3-Seed Mean)", fontsize=12, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_stage3_frontier.png", bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved figures in {out_dir}")


if __name__ == "__main__":
    main()
