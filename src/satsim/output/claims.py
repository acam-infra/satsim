"""C1/C2：按论文三档 scheduler × link_cost × nobg 选臂。"""
from __future__ import annotations

import collections


def _mean(rows: list[dict], key: str) -> float:
    xs = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _is_nobg(row: dict) -> bool:
    bg = row.get("background_gbps_per_node")
    if bg is None:
        return False
    return float(bg) == 0.0


def _select(
    rows: list[dict],
    *,
    scheduler: str,
    link_cost: str,
    nobg: bool,
) -> list[dict]:
    out = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("scheduler") != scheduler:
            continue
        if r.get("link_cost") != link_cost:
            continue
        if _is_nobg(r) != nobg:
            continue
        out.append(r)
    return out


def _policy_block(rs: list[dict]) -> dict:
    traffic = _mean(rs, "link_traffic_GB")
    return dict(
        n=len(rs),
        scheduler=rs[0]["scheduler"] if rs else None,
        link_cost=rs[0]["link_cost"] if rs else None,
        background_gbps_per_node=_mean(rs, "background_gbps_per_node"),
        good_long_mean=_mean(rs, "good_long"),
        n_long_mean=_mean(rs, "n_long"),
        goodput_tok_s_mean=_mean(rs, "goodput_tok_s"),
        migrations_mean=_mean(rs, "migrations"),
        migration_abort_mean=_mean(rs, "migration_abort"),
        migrated_GB_mean=_mean(rs, "migrated_GB"),
        link_traffic_GB_mean=traffic,
        link_cap_gbps_mean=_mean(rs, "link_cap_gbps"),
        max_flows_on_link_mean=_mean(rs, "max_flows_on_link"),
        T_SWAP_mean=_mean(rs, "T_SWAP"),
    )


def claim_summary(rows: list[dict], evidence: str) -> dict:
    by_pol = collections.defaultdict(list)
    ok_rows = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        by_pol[r["policy"]].append(r)
        ok_rows.append(r)

    summary: dict = {
        "evidence": evidence,
        "claims_frame": "place_once_free_costly",
        "tiers": {
            "1_place_once": "初次分配 only",
            "2_rebalance_free": "实时再分配 · 无消耗",
            "3_rebalance_costly": "有消耗再分配（random 基线；ours 待解决）",
        },
        "policies": {},
    }
    for pol, rs in sorted(by_pol.items()):
        summary["policies"][pol] = _policy_block(rs)

    place = _select(ok_rows, scheduler="place_once", link_cost="off", nobg=False)
    free = _select(ok_rows, scheduler="rebalance_free", link_cost="off", nobg=False)
    costly = _select(ok_rows, scheduler="rebalance_random", link_cost="full", nobg=False)
    free0 = _select(ok_rows, scheduler="rebalance_free", link_cost="off", nobg=True)
    costly0 = _select(ok_rows, scheduler="rebalance_random", link_cost="full", nobg=True)
    ours = _select(ok_rows, scheduler="rebalance_ours", link_cost="full", nobg=False)

    def by_seed(rs: list[dict]) -> dict[int, dict]:
        return {int(r["seed"]): r for r in rs}

    if place and free:
        st, fr = _policy_block(place), _policy_block(free)
        g_st, g_fr = st["goodput_tok_s_mean"], fr["goodput_tok_s_mean"]
        per = []
        for seed in sorted(set(by_seed(place)) & set(by_seed(free))):
            a, b = by_seed(place)[seed], by_seed(free)[seed]
            per.append(
                dict(
                    seed=seed,
                    place_once_good_long=int(a["good_long"]),
                    rebalance_free_good_long=int(b["good_long"]),
                    place_once_goodput_tok_s=float(a["goodput_tok_s"]),
                    rebalance_free_goodput_tok_s=float(b["goodput_tok_s"]),
                )
            )
        summary["C1"] = dict(
            claim="runtime_rebalance_needed",
            narrative="初次分配解决不了换位；无消耗的实时再分配把请求再放到仍有功率的节点",
            arms=dict(place_once="place_once/off", rebalance_free="rebalance_free/off"),
            first_good_long=st["good_long_mean"],
            rebalance_good_long=fr["good_long_mean"],
            good_long_gain=fr["good_long_mean"] - st["good_long_mean"],
            first_goodput_tok_s=g_st,
            rebalance_goodput_tok_s=g_fr,
            goodput_gain=(g_fr - g_st) / g_st if g_st else float("nan"),
            per_seed=per,
        )

    if costly and free:
        fr, cm = _policy_block(free), _policy_block(costly)
        g_fr, g_cm = fr["goodput_tok_s_mean"], cm["goodput_tok_s_mean"]
        per = []
        for seed in sorted(set(by_seed(costly)) & set(by_seed(free))):
            c, f = by_seed(costly)[seed], by_seed(free)[seed]
            per.append(
                dict(
                    seed=seed,
                    rebalance_free_goodput_tok_s=float(f["goodput_tok_s"]),
                    rebalance_random_goodput_tok_s=float(c["goodput_tok_s"]),
                    rebalance_random_abort=c["migration_abort"],
                    max_flows_on_link=c.get("max_flows_on_link"),
                )
            )
        c2 = dict(
            claim="link_cap_contention",
            narrative="有消耗再分配走链路；与背景流争用带宽时，随机基线搬不完、收益掉回去",
            arms=dict(
                rebalance_free="rebalance_free/off",
                rebalance_random="rebalance_random/full",
            ),
            link_cap_gbps=cm.get("link_cap_gbps_mean"),
            max_flows_on_link=cm.get("max_flows_on_link_mean"),
            rebalance_goodput_tok_s=g_fr,
            rebalance_link_goodput_tok_s=g_cm,
            goodput_drop=(g_fr - g_cm) / g_fr if g_fr else float("nan"),
            rebalance_link_abort_mean=cm["migration_abort_mean"],
            per_seed=per,
        )
        if free0 and costly0:
            fr0, cm0 = _policy_block(free0), _policy_block(costly0)
            g0f, g0c = fr0["goodput_tok_s_mean"], cm0["goodput_tok_s_mean"]
            c2["nobg"] = dict(
                rebalance_goodput_tok_s=g0f,
                rebalance_link_goodput_tok_s=g0c,
                goodput_drop=(g0f - g0c) / g0f if g0f else float("nan"),
                rebalance_link_abort_mean=cm0["migration_abort_mean"],
            )
        summary["C2"] = c2

    if ours:
        summary["ours"] = dict(
            status="todo",
            note="rebalance_ours：有消耗再分配待解决，当前 maybe_rebalance 为空操作",
            n_rows=len(ours),
            goodput_tok_s_mean=_policy_block(ours)["goodput_tok_s_mean"],
            migrations_mean=_policy_block(ours)["migrations_mean"],
        )
    return summary
