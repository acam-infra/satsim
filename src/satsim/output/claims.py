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
            link_cap_gbps_mean=_mean(rs, "link_cap_gbps"),
            max_flows_on_link_mean=_mean(rs, "max_flows_on_link"),
            T_SWAP_mean=_mean(rs, "T_SWAP"),
        )

    st = summary["policies"].get("first", {})
    fr = summary["policies"].get("rebalance", {})
    cm = summary["policies"].get("rebalance_link", {})
    cn = summary["policies"].get("rebalance_link_net", {})
    fr0 = summary["policies"].get("rebalance_nobg", {})
    cm0 = summary["policies"].get("rebalance_link_nobg", {})

    def by_seed(pol: str) -> dict[int, dict]:
        return {int(r["seed"]): r for r in by_pol.get(pol, [])}

    if st and fr:
        g_st, g_fr = st["goodput_tok_s_mean"], fr["goodput_tok_s_mean"]
        per = []
        for seed in sorted(set(by_seed("first")) & set(by_seed("rebalance"))):
            a, b = by_seed("first")[seed], by_seed("rebalance")[seed]
            per.append(
                dict(
                    seed=seed,
                    first_good_long=int(a["good_long"]),
                    rebalance_good_long=int(b["good_long"]),
                    first_goodput_tok_s=float(a["goodput_tok_s"]),
                    rebalance_goodput_tok_s=float(b["goodput_tok_s"]),
                )
            )
        summary["C1"] = dict(
            claim="runtime_rebalance_needed",
            narrative="首次分配解决不了换位。运行时重分配把请求再放到仍有功率的节点",
            first_good_long=st["good_long_mean"],
            rebalance_good_long=fr["good_long_mean"],
            good_long_gain=fr["good_long_mean"] - st["good_long_mean"],
            first_goodput_tok_s=g_st,
            rebalance_goodput_tok_s=g_fr,
            goodput_gain=(g_fr - g_st) / g_st if g_st else float("nan"),
            per_seed=per,
        )

    if cm and fr:
        g_fr, g_cm = fr["goodput_tok_s_mean"], cm["goodput_tok_s_mean"]
        per = []
        for seed in sorted(set(by_seed("rebalance_link")) & set(by_seed("rebalance"))):
            c, f = by_seed("rebalance_link")[seed], by_seed("rebalance")[seed]
            per.append(
                dict(
                    seed=seed,
                    rebalance_goodput_tok_s=float(f["goodput_tok_s"]),
                    rebalance_link_goodput_tok_s=float(c["goodput_tok_s"]),
                    rebalance_link_abort=c["mig_abort"],
                    max_flows_on_link=c.get("max_flows_on_link"),
                )
            )
        c2 = dict(
            claim="link_cap_contention",
            narrative="每条链路有带宽上限。多条流抢同一条上限时，上链路的重分配完不成，收益掉回去",
            link_cap_gbps=cm.get("link_cap_gbps_mean"),
            max_flows_on_link=cm.get("max_flows_on_link_mean"),
            rebalance_goodput_tok_s=g_fr,
            rebalance_link_goodput_tok_s=g_cm,
            goodput_drop=(g_fr - g_cm) / g_fr if g_fr else float("nan"),
            rebalance_link_abort_mean=cm["mig_abort_mean"],
            per_seed=per,
        )
        if fr0 and cm0:
            g0f, g0c = fr0["goodput_tok_s_mean"], cm0["goodput_tok_s_mean"]
            c2["nobg"] = dict(
                rebalance_goodput_tok_s=g0f,
                rebalance_link_goodput_tok_s=g0c,
                goodput_drop=(g0f - g0c) / g0f if g0f else float("nan"),
                rebalance_link_abort_mean=cm0["mig_abort_mean"],
            )
        if cn:
            c2["net_goodput_tok_s"] = cn["goodput_tok_s_mean"]
            c2["net_tracks_link"] = abs(cn["goodput_tok_s_mean"] - g_cm) < 1.0
        summary["C2"] = c2
    return summary
