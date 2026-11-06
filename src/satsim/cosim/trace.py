"""orbitsim contact trace -> time-indexed topology, power and reachability lookups.

All queries are pure functions of time, so every process in the event loop sees the same
topology at the same instant regardless of SimPy's ordering within a timestamp.

Epoch k is ``[epochs[k], epochs[k+1])``: the link set is constant inside it. Reachability between
an RS satellite and any node is precomputed as merged runs of epochs, so the policies' ephemeris
queries (when does this pair connect, how long does it stay connected) are a bisect each.
"""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

SCHEMA = "orbitsim.contact_trace/v1"
INF = math.inf


def _components(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        a, b = find(u), find(v)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(n)]


def _find_run(runs: list[tuple[float, float]], starts: list[float], t: float) -> int:
    """Index of the run containing t, or -1."""
    k = bisect.bisect_right(starts, t) - 1
    if k >= 0 and runs[k][0] <= t < runs[k][1]:
        return k
    return -1


class Trace:
    def __init__(self, data: dict):
        if data.get("schema") != SCHEMA:
            raise ValueError(f"not an {SCHEMA} file")
        self.name = str(data["name"])
        self.horizon = float(data["horizon_s"])
        self.nodes = data["nodes"]
        self.n = len(self.nodes)
        self.role = [str(x["role"]) for x in self.nodes]
        self.rs = [i for i, r in enumerate(self.role) if r == "rs"]
        self.compute = [i for i, r in enumerate(self.role) if r != "rs"]
        self.edges = [(int(e["u"]), int(e["v"])) for e in data["edges"]]
        self.kind = [str(e["kind"]) for e in data["edges"]]
        self.up = [[(float(a), float(b)) for a, b in e["up"]] for e in data["edges"]]
        self._up_starts = [[a for a, _ in iv] for iv in self.up]
        self.dark = [[(float(a), float(b)) for a, b in d] for d in data["dark"]]
        self._dark_starts = [[a for a, _ in d] for d in self.dark]
        if len(self.dark) != self.n:
            raise ValueError("dark intervals must cover every node")

        times = {0.0}
        for iv in self.up:
            for a, b in iv:
                times.update((a, b))
        self.epochs = sorted(t for t in times if 0.0 <= t < self.horizon)
        self.epoch_edges: list[list[int]] = []
        self.epoch_comp: list[list[int]] = []
        for t0 in self.epochs:
            ids = [i for i in range(len(self.edges)) if self.edge_up(i, t0)]
            self.epoch_edges.append(ids)
            self.epoch_comp.append(_components(self.n, [self.edges[i] for i in ids]))
        self._adj_cache: dict[int, list[list[tuple[int, int]]]] = {}

        self._conn: dict[tuple[int, int], list[tuple[float, float]]] = {}
        self._conn_starts: dict[tuple[int, int], list[float]] = {}
        for o in self.rs:
            for j in range(self.n):
                runs: list[tuple[float, float]] = []
                for k, t0 in enumerate(self.epochs):
                    if self.epoch_comp[k][o] != self.epoch_comp[k][j]:
                        continue
                    t1 = self.epochs[k + 1] if k + 1 < len(self.epochs) else self.horizon
                    if runs and runs[-1][1] == t0:
                        runs[-1] = (runs[-1][0], t1)
                    else:
                        runs.append((t0, t1))
                self._conn[(o, j)] = runs
                self._conn_starts[(o, j)] = [a for a, _ in runs]

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- time grid ---------------------------------------------------------------------------
    def change_times(self) -> list[float]:
        """Every instant in (0, horizon) at which a link or a satellite's power state changes."""
        ts = set(self.epochs)
        for d in self.dark:
            for a, b in d:
                ts.update((a, b))
        return sorted(t for t in ts if 0.0 < t < self.horizon)

    def epoch_index(self, t: float) -> int:
        return max(0, bisect.bisect_right(self.epochs, t) - 1)

    # -- links -------------------------------------------------------------------------------
    def edge_up(self, e: int, t: float) -> bool:
        return _find_run(self.up[e], self._up_starts[e], t) >= 0

    def edge_up_until(self, e: int, t: float) -> float:
        """End of the contact containing t (t itself when the link is down)."""
        k = _find_run(self.up[e], self._up_starts[e], t)
        if k < 0:
            return t
        end = self.up[e][k][1]
        return INF if end >= self.horizon else end

    def adjacency(self, t: float) -> list[list[tuple[int, int]]]:
        """Per node, sorted ``(neighbour, edge id)`` of the links up at t."""
        k = self.epoch_index(t)
        adj = self._adj_cache.get(k)
        if adj is None:
            adj = [[] for _ in range(self.n)]
            for e in self.epoch_edges[k]:
                u, v = self.edges[e]
                adj[u].append((v, e))
                adj[v].append((u, e))
            for row in adj:
                row.sort()
            self._adj_cache[k] = adj
        return adj

    def path(self, s: int, d: int, t: float, *, via_rs: bool = False) -> tuple[list[int], list[int]] | None:
        """Fewest-hop path at t as (nodes, edge ids); RS satellites are endpoints, not relays."""
        if s == d:
            return [s], []
        adj = self.adjacency(t)
        prev: dict[int, tuple[int, int] | None] = {s: None}
        q = [s]
        i = 0
        while i < len(q) and d not in prev:
            u = q[i]
            i += 1
            if u != s and not via_rs and self.role[u] == "rs":
                continue
            for v, e in adj[u]:
                if v not in prev:
                    prev[v] = (u, e)
                    q.append(v)
        if d not in prev:
            return None
        nodes, eids = [d], []
        x = d
        while prev[x] is not None:
            u, e = prev[x]
            nodes.append(u)
            eids.append(e)
            x = u
        return nodes[::-1], eids[::-1]

    # -- power -------------------------------------------------------------------------------
    def is_dark(self, j: int, t: float) -> bool:
        return _find_run(self.dark[j], self._dark_starts[j], t) >= 0

    def power_segments(self, j: int, t0: float):
        """Yield ``(start, end, dark)`` from t0 on; the last segment is open-ended (end = inf)."""
        t = t0
        d = self.dark[j]
        k = bisect.bisect_right(self._dark_starts[j], t) - 1
        while True:
            if k >= 0 and d[k][0] <= t < d[k][1]:
                end = d[k][1]
                if end >= self.horizon:
                    yield t, INF, True
                    return
                yield t, end, True
                t = end
            nxt = d[k + 1][0] if k + 1 < len(d) else INF
            if nxt >= self.horizon:
                yield t, INF, False
                return
            yield t, nxt, False
            t = nxt
            k += 1

    # -- reachability RS <-> node ------------------------------------------------------------
    def connected(self, o: int, j: int, t: float) -> bool:
        return _find_run(self._conn[(o, j)], self._conn_starts[(o, j)], t) >= 0

    def next_conn(self, o: int, j: int, t: float) -> float:
        """Earliest time >= t at which o and j share a component (inf if not before the horizon)."""
        runs, starts = self._conn[(o, j)], self._conn_starts[(o, j)]
        if _find_run(runs, starts, t) >= 0:
            return t
        k = bisect.bisect_right(starts, t)
        return runs[k][0] if k < len(runs) else INF

    def conn_until(self, o: int, j: int, t: float) -> float:
        runs, starts = self._conn[(o, j)], self._conn_starts[(o, j)]
        k = _find_run(runs, starts, t)
        if k < 0:
            return t
        return INF if runs[k][1] >= self.horizon else runs[k][1]

    def disconnected_time(self, o: int, j: int, t0: float, t1: float) -> float:
        """Seconds of [t0, t1) during which o cannot reach j (time past the horizon counts as up)."""
        t1c = min(t1, self.horizon)
        if t1c <= t0:
            return 0.0
        runs, starts = self._conn[(o, j)], self._conn_starts[(o, j)]
        k = max(0, bisect.bisect_right(starts, t0) - 1)
        up = 0.0
        while k < len(runs) and runs[k][0] < t1c:
            a, b = max(runs[k][0], t0), min(runs[k][1], t1c)
            if b > a:
                up += b - a
            k += 1
        return (t1c - t0) - up
