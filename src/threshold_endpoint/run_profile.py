"""Threshold barrier, stages A/B: fixed-endpoint profile w -> V^Q((x*, w)) and
free-terminal-signal optimisations, all r in R_GRID, several seeds.

Screening discretisation: T = 40, dt = 0.04 (K = 1000).
Output: results_profile.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from exp1_core import (R_GRID, U_ON, U_STAR, W_ON, W_STAR, X_STAR, polyline,
                       seed_paths, solve_point)

T0, DT0 = 40.0, 0.04
K0 = int(round(T0 / DT0))
OFFSETS_BELOW = (-0.3, -0.1)
OFFSETS_ABOVE = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "threshold_endpoint")
os.makedirs(HERE, exist_ok=True)


def slim(res):
    return dict(S=res["S"], S_explicit=res["S_explicit"], success=res["success"],
                nit=res["nit"], message=res["message"], diag=res["diag"])


def saddle_task(r):
    res = solve_point(r, W_STAR, polyline([U_ON, U_STAR], K0), T0, DT0,
                      maxiter=60000)
    return r, res


def point_task(args):
    r, w, seedname, seed, constrained = args
    t = time.time()
    res = solve_point(r, w, seed, T0, DT0, constrained=constrained)
    out = slim(res)
    out.update(r=r, w=w, seed=seedname, constrained=constrained,
               secs=time.time() - t)
    return out


def continuation_task(args):
    """Sequential sweep in w (upward from just above w*, and downward from the
    top), each point warm-started from the previous optimum."""
    r, ws, direction = args
    order = ws if direction == "up" else ws[::-1]
    rows, prev = [], None
    for w in order:
        seed = (polyline([U_ON, [X_STAR, w]], K0) if prev is None
                else np.vstack([prev[:-1], [[X_STAR, w]]]))
        t = time.time()
        res = solve_point(r, w, seed, T0, DT0)
        prev = res["path"]
        out = slim(res)
        out.update(r=r, w=w, seed=f"continuation_{direction}",
                   constrained=True, secs=time.time() - t)
        rows.append(out)
    return rows


def free_task(args):
    """Free terminal signal, x(T) = x* pinned.  bounds: 'above' -> w_T >= w*,
    'all' -> w_T > 0 (FLOOR)."""
    r, seedname, seed, bound, w0 = args
    t = time.time()
    wb = (W_STAR, None) if bound == "above" else (None, None)
    res = solve_point(r, w0, seed, T0, DT0, constrained=True, free_end=True,
                      wT_bounds=wb)
    out = slim(res)
    out.update(r=r, seed=seedname, bound=bound, w_init=w0,
               w_final=float(res["path"][-1, 1]), secs=time.time() - t)
    return out


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t_start = time.time()
    with Pool(nproc) as pool:
        saddles = dict(pool.map(saddle_task, R_GRID))
        print("saddles:", {r: saddles[r]["S"] for r in R_GRID}, flush=True)

        tasks = []
        for r in R_GRID:
            sp = saddles[r]["path"]
            for off in OFFSETS_BELOW + OFFSETS_ABOVE:
                w = W_STAR + off
                seeds = seed_paths(w, K0, sp)
                if off < 0:   # solver sanity checks only
                    seeds = {k: seeds[k] for k in ("line", "saddle", "deplete")}
                for name, sd in seeds.items():
                    tasks.append((r, w, name, sd, True))
                # unconstrained (no first-hit constraint), direct seed
                tasks.append((r, w, "line", seeds["line"], False))
        rows = pool.map(point_task, tasks, chunksize=1)
        print(f"fixed-endpoint runs: {len(rows)}  ({time.time()-t_start:.0f}s)",
              flush=True)

        ws_above = [W_STAR + o for o in OFFSETS_ABOVE]
        cont = pool.map(continuation_task,
                        [(r, ws_above, dname) for r in R_GRID
                         for dname in ("up", "down")])
        for c in cont:
            rows.extend(c)
        print(f"continuation done ({time.time()-t_start:.0f}s)", flush=True)

        ftasks = []
        for r in R_GRID:
            sp = saddles[r]["path"]
            for w0 in (W_STAR + 0.3, W_STAR + 1.0, W_ON, W_STAR + 3.0):
                sd = seed_paths(w0, K0, sp)
                for name in ("line", "retain", "deplete", "saddle"):
                    for bound in ("above", "all"):
                        ftasks.append((r, name, sd[name], bound, w0))
        frows = pool.map(free_task, ftasks, chunksize=1)
        print(f"free-end runs: {len(frows)} ({time.time()-t_start:.0f}s)",
              flush=True)

    out = dict(
        settings=dict(c=0.36, T=T0, dt=DT0, K=K0, x_star=X_STAR, w_star=W_STAR,
                      x_on=float(U_ON[0]), w_on=W_ON, r_grid=list(R_GRID),
                      offsets_below=OFFSETS_BELOW, offsets_above=OFFSETS_ABOVE,
                      maxiter=20000, saddle_maxiter=60000),
        saddle={str(r): slim(saddles[r]) for r in R_GRID},
        fixed=rows, free=frows, wallclock_s=time.time() - t_start)
    with open(os.path.join(HERE, "results_profile.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results_profile.json", flush=True)


if __name__ == "__main__":
    main()
