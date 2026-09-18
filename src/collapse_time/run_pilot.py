"""Run the extinction-time pilot (r = 0.5, 1, 8 at the smaller sizes): full trajectories from the existing initial state
to the small off rectangle (recording threshold entry, large-rectangle entry,
extinction and attempts on the way), in parallel chunks.

Usage:  python run_pilot.py [--workers 10] [--only r:N,r:N,...]
Each (r, N) point is written to data/collapse_time/raw/r{r}_N{N}.npz with a JSON sidecar and is
skipped if already present (checkpointing).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common import (BIN, C_COST, ON_BOX, RAW, RECT_LARGE, RECT_SMALL, fixed_points,
                    initial_counts, saddle_actions)

# Pilot grid: the smaller population sizes of the existing threshold campaign.
PLAN = {
    0.5: [40, 50, 60, 70, 80, 90],
    1.0: [40, 47, 55, 62, 70, 77, 85, 92, 100],
    8.0: [60, 80, 100, 120, 140],
}
N_REP = 1500
N_CHUNK = 10
# Horizon H = K_H * exp(N dV_num(r)). Calibration runs (300 replicas at the
# three smallest sizes per rate) gave E[tau_off] / exp(N dV_num) = 2.8-4.2, so
# K_H = 30 is about 7-10 mean off-target times: expected censoring < 0.1%.
K_H = 30.0
SEED_TAG = 2026091600          # distinct from the existing campaign's 20260812


def run_chunk(r, N, chunk, nrep, H):
    x_star, x_on = fixed_points()
    X0, W0 = initial_counts(N)
    with tempfile.TemporaryDirectory() as td:
        ft, fa = f"{td}/t.bin", f"{td}/a.bin"
        cmd = [str(BIN), "off", str(N), repr(r), repr(C_COST), str(X0), str(W0),
               repr(x_star), repr(H), str(nrep), str(SEED_TAG + chunk),
               str(int(round(100 * r))), str(N),
               *map(repr, (*RECT_LARGE, *RECT_SMALL, x_on, 2 * x_on, *ON_BOX)), ft, fa]
        t0 = time.time()
        subprocess.run(cmd, check=True)
        secs = time.time() - t0
        traj = np.fromfile(ft).reshape(-1, 12)
        att = np.fromfile(fa).reshape(-1, 5)
    return r, N, chunk, traj, att, secs, cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    dV = saddle_actions()
    points = [(r, N) for r, Ns in PLAN.items() for N in Ns]
    if args.only:
        want = {(float(a), int(b)) for a, b in (s.split(":") for s in args.only.split(","))}
        points = [p for p in points if p in want]
    points = [p for p in points if not (RAW / f"r{p[0]:g}_N{p[1]}.npz").exists()]
    # largest expected cost first so that the big points get the pool early
    points.sort(key=lambda p: -(p[1] * (1 + p[0]) * np.exp(p[1] * dV[p[0]])))
    tasks = []
    per = N_REP // N_CHUNK
    for r, N in points:
        H = K_H * float(np.exp(N * dV[r]))
        for ch in range(N_CHUNK):
            tasks.append((r, N, ch, per, H))
    print(f"{len(points)} points, {len(tasks)} chunks", flush=True)
    store = {}
    t_start = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_chunk, *t) for t in tasks]
        for f in as_completed(futs):
            r, N, ch, traj, att, secs, cmd = f.result()
            d = store.setdefault((r, N), {})
            d[ch] = (traj, att, secs, cmd)
            if len(d) == N_CHUNK:
                trajs, atts, secs_l = [], [], []
                offset = 0
                for c in range(N_CHUNK):
                    tr, at, s, _ = d[c]
                    at = at.copy()
                    at[:, 0] += offset
                    offset += len(tr)
                    trajs.append(tr); atts.append(at); secs_l.append(s)
                traj = np.vstack(trajs); att = np.vstack(atts)
                H = K_H * float(np.exp(N * dV[r]))
                np.savez_compressed(
                    RAW / f"r{r:g}_N{N}.npz",
                    tau_thr=traj[:, 0], X_thr=traj[:, 1].astype(np.int32),
                    W_thr=traj[:, 2].astype(np.int32),
                    tau_off=traj[:, 3], X_off=traj[:, 4].astype(np.int32),
                    W_off=traj[:, 5].astype(np.int32),
                    tau_small=traj[:, 6], tau_ext=traj[:, 7],
                    n_attempts=traj[:, 8].astype(np.int32), n_events=traj[:, 9],
                    t_last_attempt=traj[:, 10], t_end=traj[:, 11],
                    att_traj=att[:, 0].astype(np.int32), att_t=att[:, 1],
                    att_X=att[:, 2].astype(np.int32), att_W=att[:, 3].astype(np.int32),
                    att_outcome=att[:, 4].astype(np.int8))
                meta = dict(r=r, N=N, c=C_COST, n_rep=len(traj), horizon=H, K_H=K_H,
                            dV_num=dV[r], X0=int(initial_counts(N)[0]),
                            W0=int(initial_counts(N)[1]), rect_large=RECT_LARGE,
                            rect_small=RECT_SMALL, on_box_halfwidths=ON_BOX,
                            seeds=[[SEED_TAG + c, int(round(100 * r)), N] for c in range(N_CHUNK)],
                            chunk_secs=secs_l, core_secs=float(sum(secs_l)),
                            total_events=float(traj[:, 9].sum()),
                            example_cmd=d[0][3])
                (RAW / f"r{r:g}_N{N}.json").write_text(json.dumps(meta, indent=1) + "\n")
                print(f"done r={r:g} N={N}: core-s {sum(secs_l):.0f}, "
                      f"mean tau_thr {np.mean(traj[:,0]):.1f}, mean tau_off {np.mean(traj[:,3]):.1f}, "
                      f"censored off {np.mean(~np.isfinite(traj[:,3])):.4f}, "
                      f"wall {time.time()-t_start:.0f}s", flush=True)
                del store[(r, N)]


if __name__ == "__main__":
    main()
