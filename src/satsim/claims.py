"""C1: migration has a benefit. C2: casual migration puts a large cost on the ISL."""
from __future__ import annotations

import collections


def _mean(rows: list[dict], key: str) -> float:
    xs = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def claim_summary(rows: list[dict], evidence: str) -> dict:
    by_pol = collections.defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        by_pol[r["policy"]].append(r)

    summary: dict = {"evidence": evidence, "claims_frame": "benefit_and_link_cost", "policies": {}}
    for pol, rs in sorted(by_pol.items()):
        summary["policies"][pol] = dict(
            n=len(rs),
            good_long_mean=_mean(rs, "good_long"),
            n_long_mean=_mean(rs, "n_long"),
            goodput_tok_s_mean=_mean(rs, "goodput_tok_s"),
            mig_mean=_mean(rs, "mig"),
            mig_abort_mean=_mean(rs, "mig_abort"),
            mig_GB_mean=_mean(rs, "mig_GB"),
            mig_byteHop_GB_mean=_mean(rs, "mig_byteHop_GB"),
            T_SWAP_mean=_mean(rs, "T_SWAP"),
        )

    st = summary["policies"].get("static_late", {})
    fr = summary["policies"].get("free_mig", {})
    cm = summary["policies"].get("cost_mig", {})
    cn = summary["policies"].get("cost_net_only", {})

    def by_seed(pol: str) -> dict[int, dict]:
        return {int(r["seed"]): r for r in by_pol.get(pol, [])}

    if st and fr:
        g_st, g_fr = st["goodput_tok_s_mean"], fr["goodput_tok_s_mean"]
        per = []
        for seed in sorted(set(by_seed("static_late")) & set(by_seed("free_mig"))):
            a, b = by_seed("static_late")[seed], by_seed("free_mig")[seed]
            per.append(
                dict(
                    seed=seed,
                    static_good_long=int(a["good_long"]),
                    free_good_long=int(b["good_long"]),
                    static_goodput_tok_s=float(a["goodput_tok_s"]),
                    free_goodput_tok_s=float(b["goodput_tok_s"]),
                )
            )
        summary["C1"] = dict(
            claim="migration_has_benefit",
            narrative="换位之后把驻留改放到仍有功率的节点，换位后的 TPOT 好于不动",
            static_good_long=st["good_long_mean"],
            free_good_long=fr["good_long_mean"],
            good_long_gain=fr["good_long_mean"] - st["good_long_mean"],
            static_goodput_tok_s=g_st,
            free_goodput_tok_s=g_fr,
            goodput_gain_vs_static=(g_fr - g_st) / g_st if g_st else float("nan"),
            per_seed=per,
        )

    if cm and fr:
        per = []
        for seed in sorted(set(by_seed("cost_mig")) & set(by_seed("free_mig"))):
            c, f = by_seed("cost_mig")[seed], by_seed("free_mig")[seed]
            per.append(
                dict(
                    seed=seed,
                    casual_byteHop_GB=float(c["mig_byteHop_GB"]),
                    casual_mig_GB=float(c["mig_GB"]),
                    casual_abort=c["mig_abort"],
                    free_byteHop_GB=float(f["mig_byteHop_GB"]),
                )
            )
        c2 = dict(
            claim="casual_migration_link_cost",
            narrative="不设预算地把整段 KV 送上 ISL，byte·hop 远大于免费改放置（免费不占链路）",
            casual_byteHop_GB=cm["mig_byteHop_GB_mean"],
            casual_mig_GB=cm["mig_GB_mean"],
            casual_abort_mean=cm["mig_abort_mean"],
            free_byteHop_GB=fr["mig_byteHop_GB_mean"],
            per_seed=per,
        )
        if cn:
            c2["net_only_byteHop_GB"] = cn["mig_byteHop_GB_mean"]
            base = cm["mig_byteHop_GB_mean"] or 1.0
            c2["net_tracks_full"] = abs(cn["mig_byteHop_GB_mean"] - cm["mig_byteHop_GB_mean"]) / base < 0.05
        summary["C2"] = c2
    return summary
