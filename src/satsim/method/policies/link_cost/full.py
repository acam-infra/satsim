"""full：整段 KV 走剩余带宽；超时 abort。"""
from __future__ import annotations


class FullLinkCost:
    mode = "full"

    def transfer(self, world, req, src, dst, nbytes: float):
        path = world.pick_path(src, dst)
        if path:
            sharing = max(world.net.flows_on(e) for e in path) + 1
            world.acc["max_flows_on_link"] = max(world.acc["max_flows_on_link"], sharing)
        flow = world.net.add(path, nbytes, priority=1, demand=float("inf"))
        timeout = float(world.cfg["cost"]["timeout_s"])
        yield flow.done | world.env.timeout(timeout)
        if not flow.done.triggered:
            world.net.cancel(flow)
            world.acc["abort"] += 1
            req.migrating = False
            return
        tail = len(path) * float(world.cfg["net"]["hop_latency_s"]) + float(world.cfg["cost"]["switch_s"])
        if tail > 0:
            yield world.env.timeout(tail)
        world._commit(req, src, dst, nbytes)
