"""C1/C2：按 placement/rebalance/link_cost + 是否 nobg 选臂，不依赖旧拼盘策略名。"""
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
    placement: str,
    rebalance: str,
    link_cost: str,
    nobg: bool,
) -> list[dict]:
    out = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("placement") != placement:
            continue
        if r.get("rebalance") != rebalance:
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
        placement=rs[0]["placement"] if rs else None,
        rebalance=rs[0]["rebalance"] if rs else None,
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

    summary: dict = {"evidence": evidence, "claims_frame": "benefit_and_link_cost", "policies": {}}
    for pol, rs in sorted(by_pol.items()):
        summary["policies"][pol] = _policy_block(rs)

    place = _select(ok_rows, placement="first", rebalance="none", link_cost="off", nobg=False)
    local = _select(ok_rows, placement="first", rebalance="local", link_cost="off", nobg=False)
    over_link = _select(ok_rows, placement="first", rebalance="with_link", link_cost="full", nobg=False)
    over_net = _select(ok_rows, placement="first", rebalance="with_link", link_cost="net", nobg=False)
    local0 = _select(ok_rows, placement="first", rebalance="local", link_cost="off", nobg=True)
    link0 = _select(ok_rows, placement="first", rebalance="with_link", link_cost="full", nobg=True)

    def by_seed(rs: list[dict]) -> dict[int, dict]:
        return {int(r["seed"]): r for r in rs}

    if place and local:
        st, fr = _policy_block(place), _policy_block(local)
        g_st, g_fr = st["goodput_tok_s_mean"], fr["goodput_tok_s_mean"]
        per = []
        for seed in sorted(set(by_seed(place)) & set(by_seed(local))):
            a, b = by_seed(place)[seed], by_seed(local)[seed]
            per.append(
                dict(
                    seed=seed,
                    place_once_good_long=int(a["good_long"]),
                    rebalance_local_good_long=int(b["good_long"]),
                    place_once_goodput_tok_s=float(a["goodput_tok_s"]),
                    rebalance_local_goodput_tok_s=float(b["goodput_tok_s"]),
                )
            )
        summary["C1"] = dict(
            claim="runtime_rebalance_needed",
            narrative="首次分配解决不了换位。运行时重分配把请求再放到仍有功率的节点",
            arms=dict(place_once="first/none/off", rebalance_local="first/local/off"),
            first_good_long=st["good_long_mean"],
            rebalance_good_long=fr["good_long_mean"],
            good_long_gain=fr["good_long_mean"] - st["good_long_mean"],
            first_goodput_tok_s=g_st,
            rebalance_goodput_tok_s=g_fr,
            goodput_gain=(g_fr - g_st) / g_st if g_st else float("nan"),
            per_seed=per,
        )

    if over_link and local:
        fr, cm = _policy_block(local), _policy_block(over_link)
        g_fr, g_cm = fr["goodput_tok_s_mean"], cm["goodput_tok_s_mean"]
        per = []
        for seed in sorted(set(by_seed(over_link)) & set(by_seed(local))):
            c, f = by_seed(over_link)[seed], by_seed(local)[seed]
            per.append(
                dict(
                    seed=seed,
                    rebalance_local_goodput_tok_s=float(f["goodput_tok_s"]),
                    rebalance_over_link_goodput_tok_s=float(c["goodput_tok_s"]),
                    rebalance_over_link_abort=c["migration_abort"],
                    max_flows_on_link=c.get("max_flows_on_link"),
                )
            )
        c2 = dict(
            claim="link_cap_contention",
            narrative="每条链路有带宽上限。多条流抢同一条上限时，上链路的重分配完不成，收益掉回去",
            arms=dict(rebalance_local="first/local/off", rebalance_over_link="first/with_link/full"),
            link_cap_gbps=cm.get("link_cap_gbps_mean"),
            max_flows_on_link=cm.get("max_flows_on_link_mean"),
            rebalance_goodput_tok_s=g_fr,
            rebalance_link_goodput_tok_s=g_cm,
            goodput_drop=(g_fr - g_cm) / g_fr if g_fr else float("nan"),
            rebalance_link_abort_mean=cm["migration_abort_mean"],
            per_seed=per,
        )
        if local0 and link0:
            fr0, cm0 = _policy_block(local0), _policy_block(link0)
            g0f, g0c = fr0["goodput_tok_s_mean"], cm0["goodput_tok_s_mean"]
            c2["nobg"] = dict(
                rebalance_goodput_tok_s=g0f,
                rebalance_link_goodput_tok_s=g0c,
                goodput_drop=(g0f - g0c) / g0f if g0f else float("nan"),
                rebalance_link_abort_mean=cm0["migration_abort_mean"],
            )
        if over_net:
            cn = _policy_block(over_net)
            c2["net_goodput_tok_s"] = cn["goodput_tok_s_mean"]
            c2["net_tracks_link"] = abs(cn["goodput_tok_s_mean"] - g_cm) < 1.0
        summary["C2"] = c2
    return summary
