"""rebalance_random：目标须比源更亮（更高 f_of）；无更亮目标则不迁。"""
from __future__ import annotations

import random

from satsim.method.policies.schedulers.rebalance_random import RebalanceRandomScheduler


class _FakeReq:
    def __init__(self, i: int, node: int, kv: float = 10.0):
        self.i = i
        self.node = node
        self.kv = kv
        self.migrating = False
        self.nmig = 0
        self.t = 0.0


class _FakeEnv:
    def __init__(self):
        self.now = 0.0
        self.started = []

    def process(self, gen):
        try:
            next(gen)
        except StopIteration:
            pass
        self.started.append(gen)
        return gen


class _FakeWorld:
    def __init__(self, fs: list[float], *, seed: int = 0):
        self.n = len(fs)
        self._fs = list(fs)
        self.nodes = [[] for _ in range(self.n)]
        self.cfg = {"gpu": {"kv_gib": 100.0}}
        self.bpt = 1.0
        self.env = _FakeEnv()
        self.rng = random.Random(seed)
        self.migrate_calls: list[tuple[int, int, int]] = []

    def f_of(self, node: int, t: float) -> float:
        return self._fs[node]

    def migrate_proc(self, req, src, dst):
        self.migrate_calls.append((req.i, src, dst))
        yield


def test_random_only_picks_brighter_dst():
    # node0 最暗，1/2 更亮；即使 3 空载也不能选（更暗）
    world = _FakeWorld([0.4, 0.9, 1.0, 0.2], seed=1)
    req = _FakeReq(0, node=0)
    world.nodes[0].append(req)
    sched = RebalanceRandomScheduler()
    for _ in range(20):
        world.migrate_calls.clear()
        world.env.started.clear()
        sched.maybe_rebalance(world)
        assert world.migrate_calls, "应能迁向更亮节点"
        _, src, dst = world.migrate_calls[0]
        assert src == 0
        assert dst in (1, 2)
        assert world.f_of(dst, 0.0) > world.f_of(src, 0.0)


def test_random_skips_when_no_brighter_dst():
    # 源已最亮 → 不迁（即便其它节点可能更空）
    world = _FakeWorld([1.0, 0.5, 0.2], seed=0)
    req = _FakeReq(1, node=0)
    world.nodes[0].append(req)
    RebalanceRandomScheduler().maybe_rebalance(world)
    assert world.migrate_calls == []
    assert world.env.started == []
