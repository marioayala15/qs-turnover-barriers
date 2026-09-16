"""Threshold barrier, stages C/D: refine the best threshold candidate for each r
and compare with the saddle under identical horizon/mesh refinements.

Reads results_profile.json, writes results_refine.json.

For each r:
  1. w_best: bounded 1-D minimisation of the constrained fixed-endpoint action
     (T=40, dt=0.04) on a bracket around the best screened grid point.
  2. Horizon refinement at dt=0.02: T in {40, 80, 160};
     mesh refinement at T=40: dt in {0.04, 0.02, 0.01};
     identical settings for the saddle endpoint.
  3. Free terminal signal with w_T >= w* at T=80, dt=0.02, seeded from the
     refined candidate path.
  4. Continuum check: the exact action of the piecewise-linear interpolant of
     the finest computed path (8-point Gauss-Legendre per segment, closed-form
     Lagrangian).  That polygon is an admissible interior AC path, so this
     number is an upper bound on V^Q of its endpoint up to quadrature error.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize_scalar

from exp1_core import (P, R_GRID, U_ON, U_STAR, W_STAR, X_STAR, _ell, b, d,
                       polyline, regrid, solve_point)

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "threshold_endpoint")
os.makedirs(HERE, exist_ok=True)
HORIZONS = (40.0, 80.0, 160.0)   # at dt = 0.02
MESHES = (0.04, 0.02, 0.01)      # at T = 40


def polygon_action(path, T, r, ngl=8):
    path = np.asarray(path)
    K = path.shape[0] - 1
    h = T / K
    xg, wg = np.polynomial.legendre.leggauss(ngl)
    s = 0.5 * (xg + 1)
    A, B = path[:-1], path[1:]
    V = (B - A) / h
    total = 0.0
    for sk, wk in zip(s, wg):
        U = A + sk * (B - A)
        x, w = U[:, 0], U[:, 1]
        L = (_ell(V[:, 0], x * b(w), x * d(x))
             + _ell(V[:, 1], r * P["aC"] * x, r * P["kappa"] * w))
        total += 0.5 * wk * np.sum(L) * h
    return float(total)


def slim(res):
    return dict(S=res["S"], S_explicit=res["S_explicit"], success=res["success"],
                nit=res["nit"], message=res["message"], diag=res["diag"])


def ladder(r, w, seed, maxiter):
    """Horizon and mesh refinement from one seed path (warm-started)."""
    out = {"horizon_dt0.02": [], "mesh_T40": []}
    path = seed
    for T in HORIZONS:
        res = solve_point(r, w, path, T, 0.02, maxiter=maxiter)
        path = res["path"]
        row = slim(res)
        row.update(T=T, dt=0.02)
        out["horizon_dt0.02"].append(row)
        last_h = res
    path = seed
    for dt in MESHES:
        res = solve_point(r, w, path, 40.0, dt, maxiter=maxiter)
        path = res["path"]
        row = slim(res)
        row.update(T=40.0, dt=dt)
        out["mesh_T40"].append(row)
        last_m = res
    return out, last_h, last_m


def refine_task(args):
    r, w_grid_best, bracket, best_seed = args
    t0 = time.time()
    rec = dict(r=r, w_grid_best=w_grid_best)

    cache = {}

    def f(w):
        sd = np.vstack([best_seed[:-1], [[X_STAR, w]]])
        res = solve_point(r, w, sd, 40.0, 0.04)
        cache[w] = res
        return res["S"]

    opt = minimize_scalar(f, bounds=bracket, method="bounded",
                          options=dict(xatol=2e-3))
    w_best = float(opt.x)
    rec.update(w_best=w_best, S_best_T40_dt004=float(opt.fun),
               bracket=list(bracket), nfev=int(opt.nfev),
               scan=[dict(w=k, S=v["S"], success=v["success"])
                     for k, v in sorted(cache.items())])
    cand_seed = cache[min(cache, key=lambda k: abs(k - w_best))]["path"]

    rec["threshold"], last_h, last_m = ladder(r, w_best, cand_seed, 40000)
    sad_seed = polyline([U_ON, U_STAR], 1000)
    rec["saddle"], sad_h, sad_m = ladder(r, W_STAR, sad_seed, 60000)

    # free terminal signal, w_T >= w*, T=80, dt=0.02
    res80 = solve_point(r, w_best, last_m["path"], 80.0, 0.02,
                        free_end=True, wT_bounds=(W_STAR, None), maxiter=40000)
    fr = slim(res80)
    fr.update(T=80.0, dt=0.02, w_final=float(res80["path"][-1, 1]))
    rec["free_end_T80_dt002"] = fr

    rec["polygon_action"] = dict(
        threshold_finest_mesh=polygon_action(last_m["path"], 40.0, r),
        threshold_longest_T=polygon_action(last_h["path"], 160.0, r),
        saddle_finest_mesh=polygon_action(sad_m["path"], 40.0, r),
        saddle_longest_T=polygon_action(sad_h["path"], 160.0, r),
    )
    rec["paths"] = dict(threshold=last_m["path"][::10].tolist(),
                        saddle=sad_m["path"][::10].tolist())
    rec["secs"] = time.time() - t0
    return rec


def main():
    with open(os.path.join(HERE, "results_profile.json")) as fh:
        prof = json.load(fh)
    tasks = []
    for r in R_GRID:
        rows = [q for q in prof["fixed"] if q["r"] == r and q["constrained"]
                and q["success"] and q["w"] > W_STAR]
        best = min(rows, key=lambda q: q["S"])
        ws = sorted({q["w"] for q in rows})
        i = ws.index(best["w"])
        lo = ws[max(i - 1, 0)]
        hi = ws[min(i + 1, len(ws) - 1)]
        # seed: a straight line to the best grid point (the refinement does not
        # rely on stored paths; seed dependence is covered in stage A)
        seed = polyline([U_ON, [X_STAR, best["w"]]], 1000)
        seed = solve_point(r, best["w"], seed, 40.0, 0.04)["path"]
        tasks.append((r, best["w"], (lo, hi), seed))
        print(f"r={r}: grid best w={best['w']:.4f} S={best['S']:.8f} "
              f"bracket=({lo:.3f},{hi:.3f})", flush=True)
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    with Pool(nproc) as pool:
        recs = pool.map(refine_task, tasks)
    out = dict(settings=dict(horizons_dt002=HORIZONS, meshes_T40=MESHES,
                             maxiter_threshold=40000, maxiter_saddle=60000),
               rows=recs, wallclock_s=time.time() - t0)
    with open(os.path.join(HERE, "results_refine.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results_refine.json", flush=True)


if __name__ == "__main__":
    main()
