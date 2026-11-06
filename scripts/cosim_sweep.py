"""Sweep one knob of a co-simulation config and print the policy comparison per value.

  .venv/Scripts/python scripts/cosim_sweep.py configs/cosim_coop_w53_v1.yaml workload.rate_per_rs_per_min 10 20 40
  ... --set delivery=stream --seeds 1,2 --csv runs/sweep.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import yaml

from satsim.cosim.run import load_config, run

KEYS = ["good_frac", "good_long_frac", "goodput_tok_s", "lat_long_p90_s", "dark_tok_share",
        "migrations", "mig_abort", "upload_fail"]


def set_path(cfg: dict, key: str, val) -> None:
    node = cfg
    *head, last = key.split(".")
    for k in head:
        node = node[k]
    node[last] = val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("knob")
    ap.add_argument("values", nargs="+")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--seeds", default="")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    base = load_config(Path(args.config), args.set)
    if args.seeds:
        base["seeds"] = [int(s) for s in args.seeds.split(",")]
    out_rows = []
    for raw in args.values:
        cfg = copy.deepcopy(base)
        val = yaml.safe_load(raw)
        set_path(cfg, args.knob, val)
        res = run(cfg, None, quiet=True)
        print(f"{args.knob}={raw}")
        for pol, row in res["mean"].items():
            cells = " ".join(f"{k}={row[k]:.3g}" if isinstance(row[k], float) else f"{k}={row[k]}" for k in KEYS)
            print(f"  {pol:14} {cells}")
            out_rows.append({"knob": args.knob, "value": raw, "policy": pol, **{k: row[k] for k in KEYS}})
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            wr.writeheader()
            wr.writerows(out_rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
