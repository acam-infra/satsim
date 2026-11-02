"""单次 run 行指标：规范化字段，不解释 C1/C2。"""
from __future__ import annotations


def row_from_run(
    *,
    good_long: int,
    n_long: int,
    goodput_tok_s: float,
    migrations: int,
    migration_abort: int,
    migrated_GB: float,
    link_traffic_GB: float,
    link_cap_gbps: float,
    max_flows_on_link: int,
    T_SWAP: float,
    max_cluster_t: float,
    max_cluster_n: int,
    n_req: int,
) -> dict:
    return dict(
        status="ok",
        good_long=good_long,
        n_long=n_long,
        goodput_tok_s=goodput_tok_s,
        migrations=migrations,
        migration_abort=migration_abort,
        migrated_GB=migrated_GB,
        link_traffic_GB=link_traffic_GB,
        link_cap_gbps=link_cap_gbps,
        max_flows_on_link=max_flows_on_link,
        T_SWAP=T_SWAP,
        max_cluster_t=max_cluster_t,
        max_cluster_n=max_cluster_n,
        n_req=n_req,
    )
