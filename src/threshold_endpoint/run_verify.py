"""Threshold barrier, stage E: verification of the r = 0.5 candidate and a fine
grid just above w* for r >= 1.  Reads results_refine.json, writes
results_verify.json.

r = 0.5:
  * candidate w_c = refined w_best; four seeds NOT derived from the saddle
    path (line, late_line, retain, overshoot) at (T, dt) in
    {(40,0.02), (80,0.02), (40,0.01), (80,0.01)}; x >= x* enforced.
  * saddle endpoint recomputed at the same settings from the straight line
    AND from a seed that follows the optimized candidate path and then goes
    down the threshold line to U_* (tests whether the saddle MAM value is a
    spurious local minimum above a cheaper route).
  * polygon (continuum) action with 8- and 16-point Gauss rules.
  * open-quadrant check and forward deterministic integration of the
    candidate endpoint (basin membership).
r in {1,2,4,8}: w = w* + {0.001, 0.002, 0.005, 0.01, 0.02}, seeds line and
  retain, T=40, dt=0.02, plus the saddle at the same settings.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp

from exp1_core import (C, P, U_ON, U_STAR, W_STAR, X_STAR, b, d, polyline,
                       seed_paths, solve_point)
from run_refine import polygon_action

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "threshold_endpoint")
os.makedirs(HERE, exist_ok=True)
SETTINGS = ((40.0, 0.02), (80.0, 0.02), (40.0, 0.01), (80.0, 0.01))
FINE_OFFS = (0.001, 0.002, 0.005, 0.01, 0.02)


def slim(res, **kw):
    out = dict(S=res["S"], S_explicit=res["S_explicit"], success=res["success"],
               nit=res["nit"], message=res["message"], diag=res["diag"],
               min_x_all=float(res["path"][:, 0].min()),
               min_w_all=float(res["path"][:, 1].min()))
    out.update(kw)
    return out


def cand_task(args):
    r, w, name, seed, T, dt = args
    res = solve_point(r, w, seed, T, dt, maxiter=60000)
    out = slim(res, r=r, w=w, seed=name, T=T, dt=dt)
    out["polygon8"] = polygon_action(res["path"], T, r, 8)
    out["polygon16"] = polygon_action(res["path"], T, r, 16)
    return out


def saddle_task(args):
    r, name, seed, T, dt = args
    res = solve_point(r, W_STAR, seed, T, dt, maxiter=60000)
    out = slim(res, r=r, w=W_STAR, seed=name, T=T, dt=dt)
    out["polygon16"] = polygon_action(res["path"], T, r, 16)
    return out


def fine_task(args):
    r, w, name, seed = args
    res = solve_point(r, w, seed, 40.0, 0.02, maxiter=60000)
    return slim(res, r=r, w=w, seed=name, T=40.0, dt=0.02)


def flow(t, u, r):
    x, w = u
    return [x * (b(w) - d(x)), r * (P["aC"] * x - P["kappa"] * w)]


def main():
    ref = {q["r"]: q for q in json.load(open(os.path.join(HERE,
                                                          "results_refine.json")))["rows"]}
    r = 0.5
    wc = ref[r]["w_best"]
    cand_path = np.array(ref[r]["paths"]["threshold"])   # T=40, dt=0.01, every 10th node
    tasks, stasks = [], []
    for T, dt in SETTINGS:
        K = int(round(T / dt))
        sd = seed_paths(wc, K)                           # no saddle-derived seeds
        for name in ("line", "late_line", "retain", "overshoot"):
            tasks.append((r, wc, name, sd[name], T, dt))
        stasks.append((r, "line", polyline([U_ON, U_STAR], K), T, dt))
        via = np.vstack([cand_path,
                         np.column_stack([np.full(50, X_STAR),
                                          np.linspace(wc, W_STAR, 51)[1:]])])
        stasks.append((r, "via_candidate", via, T, dt))
    ftasks = []
    for rr in (1.0, 2.0, 4.0, 8.0):
        for off in FINE_OFFS:
            w = W_STAR + off
            sd = seed_paths(w, 2000)
            for name in ("line", "retain"):
                ftasks.append((rr, w, name, sd[name]))
        stasks.append((rr, "line", polyline([U_ON, U_STAR], 2000), 40.0, 0.02))

    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t0 = time.time()
    with Pool(nproc) as pool:
        a = pool.map_async(cand_task, tasks, chunksize=1)
        s = pool.map_async(saddle_task, stasks, chunksize=1)
        f = pool.map_async(fine_task, ftasks, chunksize=1)
        cand, sad, fine = a.get(), s.get(), f.get()

    sol = solve_ivp(flow, (0, 400), [X_STAR, wc], args=(r,), rtol=1e-10,
                    atol=1e-12)
    basin = dict(w_start=wc, final_state=sol.y[:, -1].tolist(),
                 dist_to_U_on=float(np.linalg.norm(sol.y[:, -1] - U_ON)),
                 min_x_along=float(sol.y[0].min()))
    out = dict(candidate_r05=dict(w_c=wc, runs=cand), saddle=sad,
               fine_grid=fine, basin_check_r05=basin,
               settings=dict(SETTINGS=SETTINGS, FINE_OFFS=FINE_OFFS,
                             maxiter=60000, c=C),
               wallclock_s=time.time() - t0)
    json.dump(out, open(os.path.join(HERE, "results_verify.json"), "w"),
              indent=1)
    print("wrote results_verify.json", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
