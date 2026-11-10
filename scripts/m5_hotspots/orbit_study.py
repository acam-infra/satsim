"""Orbit-driven initial placement and hotspot feasibility characterization."""
import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np

R = 6378.137
MU = 398600.4418
J2 = 1.08262668e-3
SIDEREAL = 86164.0905
DT = 10
DAY = 86400
N = 16
HORIZON = DAY + 3600
EO_RANGE = 2500
BRIDGE_RANGE = 2300
HOT = 120
COLD = 30
PHASES = (0, 90, 180, 270)


def orbital_points(t, altitude, inclination, raans, anomalies):
    a = R + altitude
    n = np.sqrt(MU / a**3)
    i = np.deg2rad(inclination)
    drift = -1.5 * J2 * n * (R / a)**2 * np.cos(i)
    node = np.deg2rad(raans)[None, :] + drift * t[:, None]
    u = np.deg2rad(anomalies)[None, :] + n * t[:, None]
    return a * np.stack([
        np.cos(node) * np.cos(u) - np.sin(node) * np.sin(u) * np.cos(i),
        np.sin(node) * np.cos(u) + np.cos(node) * np.sin(u) * np.cos(i),
        np.sin(u) * np.sin(i),
    ], axis=-1)


def visible(p, q, limit):
    delta = p[:, :, None, :] - q[:, None, :, :]
    d = np.linalg.norm(delta, axis=-1)
    pair_radii = np.linalg.norm(p, axis=-1)[:, :, None], np.linalg.norm(q, axis=-1)[:, None, :]
    horizon = sum(np.sqrt(np.maximum(0, r**2 - R**2)) for r in pair_radii)
    return np.where((d <= limit) & (d <= horizon), d, np.inf)


def geometry(phase):
    t = np.arange(0, HORIZON, DT)
    mean_motion = np.sqrt(MU / (R + 600)**3)
    desired_drift = np.deg2rad(360 / 365.2422) / DAY
    sso_inc = np.rad2deg(np.arccos(-desired_drift / (1.5 * J2 * mean_motion * (R / (R + 600))**2)))
    eo = orbital_points(t, 600, sso_inc, np.repeat([24, 59], 4), np.tile([0, 90, 180, 270], 2) + np.repeat([0, 37], 4))
    compute = orbital_points(t, 550, 53, np.repeat([0, 12, 24, 36], 4),
                             phase + np.repeat([0, 7, 14, 21], 4) + np.tile([-3, -1, 1, 3], 4))
    contacts = visible(eo, compute, EO_RANGE)
    cross = visible(compute, compute, BRIDGE_RANGE)
    direct = np.argsort(contacts, axis=-1, kind="stable")[:, :, :2]
    direct = np.where(np.take_along_axis(contacts, direct, axis=-1) < np.inf, direct, -1)
    hops = np.full((len(t), N, N), 99, dtype=np.int16)
    for c in range(4):
        hops[:, c*4:c*4+4, c*4:c*4+4] = 1
    diagonal = np.arange(N)
    hops[:, diagonal, diagonal] = 0
    # Two shortest visible bridges on each adjacent-cluster boundary, with distinct endpoints.
    for c in range(3):
        for k, distances in enumerate(cross[:, c*4:(c+1)*4, (c+1)*4:(c+2)*4]):
            left, right = set(), set()
            for idx in np.argsort(distances, axis=None):
                a, b = divmod(int(idx), 4)
                if not np.isfinite(distances[a, b]):
                    break
                if a in left or b in right:
                    continue
                i, j = 4*c+a, 4*(c+1)+b
                hops[k, i, j] = hops[k, j, i] = 1
                left.add(a)
                right.add(b)
                if len(left) == 2:
                    break
    for j in range(N):
        np.minimum(hops, hops[:, :, j:j+1] + hops[:, j:j+1, :], out=hops)
    # Fixed ground targets independent of compute-orbit phase. Synthetic imaging swath.
    rng = np.random.default_rng(112)
    lat = np.deg2rad(np.repeat([-55, -30, -10, 10, 30, 55], 8) + rng.uniform(-4, 4, 48))
    lon = np.deg2rad(np.tile(np.arange(8)*45, 6) + rng.uniform(-12, 12, 48))
    angle = lon[None, :] + 2*np.pi*t[:, None]/SIDEREAL
    ground = np.stack([np.cos(lat)*np.cos(angle), np.cos(lat)*np.sin(angle),
                       np.broadcast_to(np.sin(lat), angle.shape)], axis=-1)
    seen = np.einsum("tei,tri->ter", eo/(R+600), ground) >= np.cos(380/R)
    entries = seen & ~np.concatenate([np.zeros_like(seen[:1]), seen[:-1]], axis=0)
    passes = np.argwhere(entries[:DAY//DT])
    return t, direct, hops, passes, sso_inc


def traffic(t, direct, passes, jobs_per_pass):
    rng = np.random.default_rng(47)
    created = [[] for _ in t]
    for k, eo, roi in passes:
        for _ in range(jobs_per_pass):
            kind = int(rng.choice(3, p=[.55, .30, .15]))
            work = [5, 19, 52][kind] * rng.lognormal(-.5*.35**2, .35)
            created[k].append((int(eo), float(work), int(roi)))
    buffers = [deque() for _ in range(8)]
    admissions = [[] for _ in t]
    for k, jobs in enumerate(created):
        for eo, work, roi in jobs:
            buffers[eo].append((k, work, roi))
        for eo in range(8):
            if direct[k, eo, 0] < 0:
                continue
            for _ in range(min(16, len(buffers[eo]))):
                generated, work, roi = buffers[eo].popleft()
                admissions[k].append((eo, generated, work, roi))
    return admissions, sum(map(len, created)), sum(map(len, buffers))


def run(t, direct, hops, admissions, policy, hot_threshold=HOT):
    pending = [deque() for _ in range(N)]
    active = [[] for _ in range(N)]
    cursor = np.zeros(8, dtype=int)
    work_by_node = np.zeros(N)
    count_by_node = np.zeros(N, dtype=int)
    backlog = np.zeros((len(t), N))
    served = []
    compute_delays = []
    placement = []
    admission_hops = []
    hot_n = reachable = cross_cluster_only = spare_unreachable = no_cold = active_candidate = 0
    hop_samples = []
    for k, now in enumerate(t):
        for eo, generated, work, roi in admissions[k]:
            options = direct[k, eo]
            if policy == "nearest":
                node = int(options[0])
            elif policy == "rr":
                # Per-source rotating cursor over the same feasible antenna peers.
                candidates = options[options >= 0]
                node = int(min(candidates, key=lambda j: (j - cursor[eo]) % N))
                cursor[eo] = (node + 1) % N
            else:
                ingress = int(options[0])
                candidates = np.flatnonzero(hops[k, ingress] < 99)
                node = int(min(candidates, key=lambda j: (j - cursor[eo]) % N))
                cursor[eo] = (node + 1) % N
            peers = options[options >= 0]
            admission_hops.append(int(hops[k, options[0], node]) if policy == "routed_rr" else int(hops[k, peers, node].min()))
            pending[node].append([work, generated, now])
            work_by_node[node] += work
            count_by_node[node] += 1
            placement.append((int(now), eo, roi, generated*DT, node, work))
        for j in range(N):
            backlog[k, j] = sum(x[0] for x in active[j]) + sum(x[0] for x in pending[j])
        if now < DAY:
            cold = np.flatnonzero(backlog[k] < COLD)
            for j in np.flatnonzero(backlog[k] >= hot_threshold):
                hot_n += 1
                candidates = cold[hops[k, j, cold] < 99]
                if not len(candidates):
                    if len(cold):
                        spare_unreachable += 1
                    else:
                        no_cold += 1
                    continue
                reachable += 1
                cross_cluster_only += int(all(i//4 != j//4 for i in candidates))
                hop_samples.append(int(hops[k, j, candidates].min()))
                active_candidate += int(bool(active[j]))
        for j in range(N):
            dt = float(DT)
            while dt > 1e-9:
                while len(active[j]) < 8 and pending[j]:
                    active[j].append(pending[j].popleft())
                if not active[j]:
                    break
                step = min(dt, min(x[0] for x in active[j])*len(active[j]))
                for task in active[j]:
                    task[0] -= step/len(active[j])
                dt -= step
                for remaining, generated, admitted_at in active[j]:
                    if remaining <= 1e-9:
                        served.append(now+DT-dt-generated*DT)
                        compute_delays.append(now+DT-dt-admitted_at)
                active[j] = [x for x in active[j] if x[0] > 1e-9]
    day = backlog[:DAY//DT]
    contact_delay = [x[0]-x[3] for x in placement]
    assert hot_n == reachable + spare_unreachable + no_cold
    assert count_by_node.sum() == len(placement) == len(served) + sum(map(len, pending)) + sum(map(len, active))
    summary = dict(policy=policy, admitted=len(placement), completed=len(served),
                   offered_work_over_day_capacity=float(work_by_node.sum()/(N*DAY)),
                   node_count=count_by_node.tolist(), node_work_s=work_by_node.tolist(),
                   arrival_work_cv=float(work_by_node.std()/work_by_node.mean()),
                   top4_work_share=float(np.sort(work_by_node)[-4:].sum()/work_by_node.sum()),
                   backlog_peak_s=float(day.max()),
                   backlog_p95_node_time_s=float(np.percentile(day, 95)),
                   e2e_p99_finished_s=float(np.percentile(served,99)),
                   e2e_mean_finished_s=float(np.mean(served)),
                   ingress_to_done_p99_finished_s=float(np.percentile(compute_delays,99)),
                   input_compute_hops_mean=float(np.mean(admission_hops)),
                   contact_wait_p95_s=float(np.percentile(contact_delay,95)),
                   hot_node_minutes=hot_n*DT/60,
                   hot_samples=hot_n, reachable_cold_samples=reachable,
                   cold_only_cross_cluster_samples=cross_cluster_only,
                   cold_exists_but_unreachable=spare_unreachable,
                   no_cold_samples=no_cold,
                   hot_reachable_with_active_samples=active_candidate,
                   nearest_cold_hops=dict(zip(*[x.tolist() for x in np.unique(hop_samples, return_counts=True)])))
    return summary, backlog, placement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for phase in PHASES:
        t, direct, hops, passes, inc = geometry(phase)
        contact = dict(phase_deg=phase, sso_inclination_deg=float(inc), passes=len(passes),
                       eo_contact_share_by_source=(direct[:DAY//DT,:,0]>=0).mean(axis=0).tolist(),
                       no_contact_share=float((direct[:DAY//DT,:,0]<0).mean()),
                       two_contact_share=float((direct[:DAY//DT,:,1]>=0).mean()),
                       all_compute_connected_share=float((hops[:DAY//DT]<99).all(axis=(1,2)).mean()))
        for jobs_per_pass in (12, 72, 144):
            admissions, n_generated, undelivered = traffic(t, direct, passes, jobs_per_pass)
            for policy in ("nearest","rr","routed_rr"):
                row, backlog, placement = run(t, direct, hops, admissions, policy)
                row.update(contact, jobs_per_pass=jobs_per_pass, generated=n_generated,
                           undelivered=undelivered, admitted_with_two_peers=sum(
                               int(direct[now//DT, eo,1]>=0) for now,eo,*_ in placement))
                rows.append(row)
                if phase == 0:
                    name = f"{policy}_{jobs_per_pass}"
                    np.savez_compressed(args.out/f"{name}.npz", t=t, backlog=backlog)
                    with (args.out/f"{name}_placement.csv").open("w") as file:
                        writer = csv.writer(file)
                        writer.writerow(["admit_s","eo","region","capture_s","compute","work_s"])
                        writer.writerows(placement)
                print(phase, jobs_per_pass, policy, row['admitted'], round(row['offered_work_over_day_capacity'],3),
                      round(row['hot_node_minutes'],1), round(row['arrival_work_cv'],3), flush=True)
    (args.out/"summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
