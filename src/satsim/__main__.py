"""Run one admitted spec. Scene files are refused."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from satsim.claims import claim_summary
from satsim.sim import make_pass, run_one
from satsim.spec import SpecError, apply_t_swap, resolve_spec


def run_spec(spec_path: Path, outdir: Path | None = None) -> dict:
    spec, cfg, jobs = resolve_spec(spec_path)
    # T_SWAP is a property of the pass shape; seeds only jitter inside a bin
    # that does not move the flat-pass anchor. Compute it from seed 0's pass
    # and check every seed lands on the same anchor.
    trace = spec["trace"]
    anchors = []
    for seed in spec["seeds"]:
        _reqs, anchor, count = make_pass(trace, float(cfg["workload"]["p_long"]), int(seed))
        anchors.append((anchor, count))
    anchor = anchors[0][0]
    if any(abs(a - anchor) > 1e-6 for a, _ in anchors):
        raise SpecError("各种子的 max_cluster 锚点不一致，拒绝把换位时间做成隐蔽旋钮")
    t_swap_s = anchor + float(spec["t_swap"]["offset_s"])
    apply_t_swap(cfg, t_swap_s, float(spec["t_swap"]["P_cap"]))
    # jobs were copied before events existed; stamp the same timeline onto each.
    rows = []
    for name, pol, c2, seed in jobs:
        apply_t_swap(c2, t_swap_s, float(spec["t_swap"]["P_cap"]))
        out = run_one(c2, pol, seed, trace, t_swap_s, float(spec["slo"]["tpot_slo"]))
        rows.append({"policy": name, "seed": seed, **out})
    evidence = spec["evidence"]
    claims = claim_summary(rows, evidence)
    payload = {
        "name": spec["name"],
        "evidence": evidence,
        "scene": spec["metadata"]["scene"],
        "T_SWAP": t_swap_s,
        "max_cluster_n": anchors[0][1],
        "rows": rows,
        "claims": claims,
    }
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="跑一份已准入的 satsim spec")
    ap.add_argument("spec")
    ap.add_argument("--out", default="", help="默认写入 ./runs/<name>")
    args = ap.parse_args(argv)
    spec_path = Path(args.spec)
    try:
        # resolve name for the default directory without a second admission pass
        out = Path(args.out) if args.out else None
        payload = run_spec(spec_path, outdir=None)
        dest = out if out is not None else Path("runs") / payload["name"]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    except SpecError as e:
        raise SystemExit(f"spec: {e}") from e
    claims = payload["claims"]
    print(f"{payload['name']}  T_SWAP={payload['T_SWAP']:.4g}  evidence={payload['evidence']}")
    for pol, row in claims["policies"].items():
        print(
            f"  {pol:16} good_long={_fmt(row['good_long_mean'])}"
            f"  goodput={_fmt(row['goodput_tok_s_mean'])}"
            f"  byteHop_GB={_fmt(row['mig_byteHop_GB_mean'])}"
            f"  abort={_fmt(row['mig_abort_mean'])}"
        )
    c1, c2 = claims["C1"], claims["C2"]
    print(
        f"C1 good_long {c1['static_good_long']:.4g}→{c1['free_good_long']:.4g}"
        f"  gain={c1['goodput_gain_vs_static']:.4g}"
    )
    print(
        f"C2 casual byteHop_GB={c2['casual_byteHop_GB']:.4g}"
        f"  free byteHop_GB={c2['free_byteHop_GB']:.4g}"
        f"  net_tracks_full={c2['net_tracks_full']}"
    )
    print(f"wrote {dest / 'summary.json'}")


if __name__ == "__main__":
    main()
