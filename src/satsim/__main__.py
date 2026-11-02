"""Run one admitted spec. Scene files are refused."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from satsim.input.generate.dispatch import generate_arrivals
from satsim.input.load import apply_t_swap, resolve_spec
from satsim.input.schema import SpecError
from satsim.method.engine.loop import run_one
from satsim.output.claims import claim_summary


def resolve_t_swap_s(spec: dict, cfg: dict) -> tuple[float, float, int]:
    """返回 (t_swap_s, anchor_t, cluster_n)。"""
    trace = spec["trace"]
    t_swap = spec["t_swap"]
    anchor_kind = t_swap["anchor"]
    seeds = [int(s) for s in spec["seeds"]]
    p_long = float(cfg["workload"]["p_long"])

    if anchor_kind == "absolute":
        t_swap_s = float(t_swap["t_s"])
        _reqs, anchor, count = generate_arrivals(trace, p_long, seeds[0])
        return t_swap_s, float(anchor), int(count)

    anchors = []
    for seed in seeds:
        _reqs, anchor, count = generate_arrivals(trace, p_long, seed)
        anchors.append((anchor, count))
    anchor = anchors[0][0]
    if anchor_kind == "max_cluster":
        if any(abs(a - anchor) > 1e-6 for a, _ in anchors):
            raise SpecError("各种子的 max_cluster 锚点不一致，拒绝把换位时间做成隐蔽旋钮")
    t_swap_s = anchor + float(t_swap["offset_s"])
    return t_swap_s, float(anchor), int(anchors[0][1])


def run_spec(spec_path: Path, outdir: Path | None = None) -> dict:
    spec, cfg, jobs = resolve_spec(spec_path)
    t_swap_s, _anchor, cluster_n = resolve_t_swap_s(spec, cfg)
    apply_t_swap(cfg, t_swap_s, float(spec["t_swap"]["P_cap"]))
    rows = []
    for name, pol, c2, seed in jobs:
        apply_t_swap(c2, t_swap_s, float(spec["t_swap"]["P_cap"]))
        out = run_one(c2, pol, seed, spec["trace"], t_swap_s, float(spec["slo"]["tpot_slo"]))
        rows.append(
            {
                "policy": name,
                "seed": seed,
                "scheduler": pol["scheduler"],
                "link_cost": pol["link_cost"],
                "background_gbps_per_node": float(c2["background"]["gbps_per_node"]),
                **out,
            }
        )
    evidence = spec["evidence"]
    claims = claim_summary(rows, evidence)
    payload = {
        "name": spec["name"],
        "evidence": evidence,
        "scene": spec["metadata"]["scene"],
        "T_SWAP": t_swap_s,
        "max_cluster_n": cluster_n,
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
            f"  {pol:28} {row.get('scheduler')}/{row.get('link_cost')}"
            f"  good_long={_fmt(row['good_long_mean'])}"
            f"  goodput={_fmt(row['goodput_tok_s_mean'])}"
            f"  link_traffic_GB={_fmt(row['link_traffic_GB_mean'])}"
            f"  abort={_fmt(row['migration_abort_mean'])}"
        )
    if "C1" in claims:
        c1 = claims["C1"]
        print(
            f"C1 首次 {c1['first_good_long']:.4g} → 运行时重分配 {c1['rebalance_good_long']:.4g}"
            f"  gain={c1['goodput_gain']:.4g}"
        )
    if "C2" in claims:
        c2 = claims["C2"]
        nobg = c2.get("nobg") or {}
        nobg_drop = nobg.get("goodput_drop", float("nan"))
        print(
            f"C2 cap={c2['link_cap_gbps']:.4g} Gbps"
            f"  flows/link={c2['max_flows_on_link']:.4g}"
            f"  上链路 goodput {c2['rebalance_link_goodput_tok_s']:.4g}"
            f"  不上链路 {c2['rebalance_goodput_tok_s']:.4g}"
            f"  drop={c2['goodput_drop']:.4g}"
            f"  nobg_drop={nobg_drop:.4g}"
        )
    print(f"wrote {dest / 'summary.json'}")


if __name__ == "__main__":
    main()
