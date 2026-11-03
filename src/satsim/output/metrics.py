"""单次 run 行指标：规范化字段，不解释 C1/C2。"""
from __future__ import annotations


def row_from_run(
    *,
    good_long: int,
    n_long: int,
    n_completed: int,
    completed_long: int,
    request_throughput_rps: float,
    total_token_throughput_tok_s: float,
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
    horizon_s: float,
) -> dict:
    """字段说明见 docs/SPEC 或 README「吞吐定义」。

    - request_throughput_rps: 完成请求数 / horizon_s
    - total_token_throughput_tok_s: 完成请求 (prompt+output) token 之和 / horizon_s
    - goodput_tok_s: 旧对照——换位后 TPOT 过 SLO 的 *输出* token / arrival_window_s
    """
    return dict(
        status="ok",
        good_long=good_long,
        n_long=n_long,
        n_completed=n_completed,
        completed_long=completed_long,
        request_throughput_rps=request_throughput_rps,
        total_token_throughput_tok_s=total_token_throughput_tok_s,
        goodput_tok_s=goodput_tok_s,
        goodput_slo_tok_s=goodput_tok_s,
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
        horizon_s=horizon_s,
    )
