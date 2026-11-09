"""Plot the paired placement study outputs (requires matplotlib)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "round_robin": ("Round robin", "#767676", "o"),
    "nearest": ("Min. upload ETA", "#B64342", "s"),
    "greedy": ("Fastest now", "#42949E", "^"),
    "orbit_thermal": ("Orbit + heat placement", "#3775BA", "D"),
    "orbit_thermal_speed": ("Speed only", "#B64342", "s"),
    "orbit_thermal_self": ("Request SLO", "#A76C23", "v"),
    "orbit_thermal_joint": ("Fixed heat cap", "#9A4D8E", "o"),
    "orbit_thermal_no_forecast": ("No heat forecast", "#42949E", "^"),
    "orbit_thermal_rc": ("RC recovery", "#0F4D92", "D"),
}


def load(root, run):
    with (root / run / "runs.csv").open() as source:
        return list(csv.DictReader(source))


def select(rows, policy, *, rate=None, arrival="burst", heat=0.30):
    return [r for r in rows if r["policy"] == policy and r["arrival"] == arrival
            and float(r["resistance"]) == heat and r["length_mode"] == "mixed"
            and (rate is None or int(r["rate_per_rs_min"]) == rate)]


def avg(rows, field):
    return mean(float(r[field]) for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="parent directory of the run directories")
    ap.add_argument("out", type=Path, help="figure path without extension")
    args = ap.parse_args()
    placement = load(args.root, "placement")
    incident = load(args.root, "incident")
    runtime = [r for group in ("rescue", "speed", "no_forecast", "rc") for r in load(args.root, group)]
    incident_seed1 = {r["policy"]: r for r in
                      json.loads((args.root / "incident_rescue" / "case_seed1.json").read_text())}

    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 10,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": 1.1, "pdf.fonttype": 42})
    fig, ax = plt.subplots(2, 2, figsize=(11.1, 7.7), layout="constrained")
    ax = ax.ravel()
    policies = ("round_robin", "nearest", "greedy", "orbit_thermal")
    for policy in policies:
        label, color, marker = LABELS[policy]
        points = [(rate, select(placement, policy, rate=rate)) for rate in (8, 12, 16, 24)]
        points = [(rate, 100 * avg(rows, "good_long_frac")) for rate, rows in points if len(rows) == 5]
        ax[0].plot(*zip(*points), color=color, marker=marker, lw=1.8, label=label)
    ax[0].set(xlabel="Offered requests per sensing satellite per min", ylabel="Long jobs on time (%)",
              title="a   Placement alone under burst load", xticks=sorted({x for line in ax[0].lines for x in line.get_xdata()}))
    ax[0].legend(loc="lower left", frameon=False, fontsize=8)

    for i, policy in enumerate(policies):
        _, color, _ = LABELS[policy]
        cells = [select(incident, policy, rate=12, arrival=arrival) for arrival in ("smooth", "burst")]
        values = [100 * avg(rows, "top_quartile_placements") for rows in cells]
        ax[1].plot([i - 0.15, i + 0.15], values, color=color, lw=1.5)
        ax[1].scatter(i - 0.15, values[0], marker="o", facecolors="white", edgecolors=color, s=53, zorder=3)
        ax[1].scatter(i + 0.15, values[1], marker="o", color=color, s=53, zorder=3)
    ax[1].axhline(25, color="#767676", ls=":", lw=1)
    ax[1].set(xticks=range(4), xticklabels=["Round\nrobin", "Min. upload\nETA", "Fastest\nnow", "Orbit + heat"],
              ylabel="First placements on busiest 10 of 40 (%)", title="b   Single incident source concentrates placements",
              ylim=(20, 76))
    ax[1].text(0.02, 0.97, "Open: smooth release    Filled: buffered release",
               transform=ax[1].transAxes, va="top", fontsize=8)

    controls = ("orbit_thermal", "orbit_thermal_speed", "orbit_thermal_self", "orbit_thermal_joint",
                "orbit_thermal_no_forecast", "orbit_thermal_rc")
    for policy in controls:
        label, color, marker = LABELS[policy]
        rows = select(runtime, policy, rate=12)
        ax[2].scatter(avg(rows, "mig_GB"), avg(rows, "good_long_req_min"), color=color, marker=marker,
                      label=label, s=48)
    ax[2].set(xlabel="KV transferred (GB per run)", ylabel="Long-job goodput (jobs/min)",
              title="c   Same placement with runtime migration")
    ax[2].legend(loc="lower right", frameon=False, fontsize=8, ncol=2, columnspacing=0.8)

    for policy, label, color in (("orbit_thermal", "Placement alone", "#767676"),
                                 ("orbit_thermal_joint", "Runtime migration", "#0F4D92")):
        case = incident_seed1[policy]
        longs = [x[1] for x in case["reqs"] if x[0]]
        complete = sorted(x for x in longs if x is not None)
        ax[3].step(complete, [(i + 1) / len(longs) * 100 for i in range(len(complete))],
                   where="post", color=color, lw=1.8, label=label)
    ax[3].axvline(1, color="#767676", ls=":", lw=1)
    ax[3].set(xlim=(0.25, 1.6), ylim=(0, 102), xlabel="Delivery latency / SLO",
              ylabel="Fraction of submitted long jobs (%)", title="d   Incident delivery distribution (seed 1)")
    ax[3].legend(loc="lower right", frameon=False, fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.out.with_suffix(suffix), dpi=300)


if __name__ == "__main__":
    main()
