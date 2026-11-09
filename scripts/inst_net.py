"""Instrument FluidNet._allocate to measure how often max-min fair sharing actually
throttles flows (vs. degenerate exclusive-link case). Run as:

  uv run python scripts/inst_net.py configs/cosim_typhoon_agentic_v1.yaml \
        --policy contact_aware --seed 1
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import statistics as st
import time

import simpy
import yaml

from satsim.cosim.net import FluidNet
from satsim.cosim.policies import make_policy
from satsim.cosim.trace import Trace
from satsim.cosim.world import CoWorld, make_requests


class AllocStats:
    def __init__(self):
        self.calls = 0                 # how many times _allocate ran
        self.throttled = 0             # calls where >=1 link was shared (share < cap)
        self.throttled_mig = 0         # throttled calls that involve >=1 migration flow
        self.throttled_up = 0          # throttled calls that involve >=1 upload flow
        self.user_hist = []            # max users on a link within a throttled call
        self.rate_obs = 0
        self.rate_shared = 0
        self.max_share_frac = []       # min over links of (share*users)/cap  -> how far below cap
        self.flows_at_call = []        # number of live flows at each call
        self.rates_seen = []           # final per-flow rate / link-cap ratio


_STATS = AllocStats()


def _patched_allocate(self: FluidNet):
    for f in self.flows:
        f.rate = 0.0
    rem = {}
    for f in self.flows:
        for a, b, e in f.hops:
            rem[(a, b)] = self.cap[e]
    if not self.flows:
        return
    _STATS.calls += 1
    _STATS.flows_at_call.append(len(self.flows))
    live = list(self.flows)
    max_users = 0
    has_mig = any(f.tag == "mig" for f in self.flows)
    has_up = any(f.tag == "up" for f in self.flows)
    while live:
        users = {}
        for f in live:
            for a, b, _ in f.hops:
                users[(a, b)] = users.get((a, b), 0) + 1
        max_users = max(max_users, max(users.values()))
        share = min(rem[k] / c for k, c in users.items())
        for f in live:
            f.rate += share
            for a, b, _ in f.hops:
                rem[(a, b)] -= share
        full = {k for k, v in rem.items() if v <= 1e-6 * max(1.0, share)}
        live = [f for f in live if not any((a, b) in full for a, b, _ in f.hops)]
    if max_users > 1:
        _STATS.throttled += 1
        if has_mig:
            _STATS.throttled_mig += 1
        if has_up:
            _STATS.throttled_up += 1
        _STATS.user_hist.append(max_users)
    _STATS.max_share_frac.append(max_users)
    for f in self.flows:
        # rate / bottleneck edge cap for this flow
        rates = [f.rate / self.cap[e] for _, _, e in f.hops]
        r = min(rates)
        _STATS.rates_seen.append(r)
        _STATS.rate_obs += 1
        if r < 0.99:
            _STATS.rate_shared += 1


def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        *head, last = key.split(".")
        for k in head:
            node = node[k]
        node[last] = yaml.safe_load(raw)
    cfg["_dir"] = str(Path(path).parent.resolve())
    return cfg


from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--policy", default="contact_aware")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    FluidNet._allocate = _patched_allocate

    cfg = load_config(args.config, args.overrides)
    trace = Trace.load(Path(cfg["_dir"]) / cfg["trace"])
    pol_spec = cfg["policies"][args.policy]
    env = simpy.Environment()
    w = CoWorld(env, trace, cfg, make_policy(pol_spec), seed=args.seed)
    for r in make_requests(cfg, trace, args.seed):
        w.submit(r)
    t0 = time.perf_counter()
    env.run(until=trace.horizon)
    wall = time.perf_counter() - t0

    s = _STATS
    print(f"policy={args.policy} seed={args.seed} wall={wall:.1f}s")
    print(f"  _allocate calls         : {s.calls}")
    print(f"  calls with throttling   : {s.throttled}  ({100*s.throttled/max(s.calls,1):.1f}%)")
    if s.flows_at_call:
        print(f"  live flows/call  median : {st.median(s.flows_at_call):.0f}  max={max(s.flows_at_call)}")
    if s.max_share_frac:
        print(f"  max flows on a link/call: median={st.median(s.max_share_frac):.0f} "
              f"max={max(s.max_share_frac)}  (>1 == real contention)")
    if s.user_hist:
        from collections import Counter
        hist = Counter(s.user_hist)
        top = ", ".join(f"{k}x:{hist[k]}" for k in sorted(hist)[:8])
        print(f"  sharing-depth hist (throttled calls): {top}")
    if s.rate_obs:
        print(f"  flow-obs below 0.99 cap : {100*s.rate_shared/s.rate_obs:.1f}% ({s.rate_shared}/{s.rate_obs})")
    if s.rates_seen:
        print(f"  per-flow rate/cap       : median={st.median(s.rates_seen):.3f} "
              f"min={min(s.rates_seen):.3f} (1.0 == exclusive)")
    print(f"  throttled w/ mig flow    : {s.throttled_mig}")
    print(f"  throttled w/ up  flow    : {s.throttled_up}")
    # migrations / aborts from world accumulator
    print(f"  migrations={w.acc['mig']} aborts={w.acc['mig_abort']} "
          f"upload_fails={w.acc['upload_fail']}")


if __name__ == "__main__":
    main()
