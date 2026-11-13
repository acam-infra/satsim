import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from satsim.cosim.run import load_config, run_one, summarize
from satsim.cosim.trace import Trace
from scripts.exp_h1_h2_h3_regime import POLICIES, _make_cfg

VARIANTS = {
    "v13": {"policy": "discrete_rh_tier_v13"},
    "v13_wake_place": {"policy": "discrete_rh_tier_v13", "wakeup_aware_place": True},
    "v13_warm_res2": {"policy": "discrete_rh_tier_v13", "warm_high_reserve": 2},
    "v13_urg_fast": {"policy": "discrete_rh_tier_v13", "urg_fast_place": True},
    "v14": {"policy": "discrete_rh_tier_v14"},
}


def _worker(task):
    sc, name, cfg, trace_path = task
    tr = Trace.load(Path(trace_path))
    t0 = time.perf_counter()
    w, _ = run_one(copy.deepcopy(cfg), tr, VARIANTS[name], 11, record=False)
    s = summarize(w)
    return {
        "sc": sc,
        "pol": name,
        "wall_s": round(time.perf_counter() - t0, 1),
        "slo": round(s["good_long_frac"] * 100, 2),
        "urg": round(s["good_urgent_frac"] * 100, 2),
        "bad": round(s["waste_ratio"] * 100, 2),
        "kj": round(s["energy_kj_total"], 1),
        "wake": s["wakeup_count"],
        "mig": s["migrations"],
        "rec": int(w.acc.get("recompute_mig", 0)),
        "gbh": round(s["link_GB_hops"], 1),
    }


def main():
    base_cfg = load_config(Path("configs/cosim_discrete_power_w53_v1.yaml"))
    trace_path = str((Path(base_cfg["_dir"]) / base_cfg["trace"]).resolve())
    scenarios = {
        "h1_h2_dwell": _make_cfg(base_cfg, t_prefill=0.005, min_dwell=20.0, isl_gbps=10.0, dl_long=2.40, dl_urg=1.55),
        "tight_h1_h2": _make_cfg(base_cfg, t_prefill=0.008, min_dwell=30.0, isl_gbps=10.0, dl_long=1.75, dl_urg=1.35),
    }
    tasks = [(sc, name, cfg, trace_path) for sc, cfg in scenarios.items() for name in VARIANTS]
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
