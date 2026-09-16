"""Probability of reaching R_off before the on-box, started near the saddle.

Grid: c = 0.36, r in {0.5, 1, 2, 4, 8}, N in {50, 100, 200, 400, 800, 1600, 3200},
start U_* + s N^{-1/2} v_u rounded to the lattice, s in {-1, 0, +1}, where v_u is
the unit unstable eigenvector (both components positive; s < 0 points to the off side).
NREP trajectories per case, in chunks, run in parallel.

R_off = [0,0.3]x[0,0.7] (prop:off-target) and the on-box [x_on +- 0.15]x[w_on +- 0.30],
which lies in {x > x*, w > w*} and hence in the on basin (lem:threshold-line).

Usage: python run_committor.py [--workers 10] [--nrep 4000]
Writes data/committor/raw/r{r}_N{N}_s{s}.npz (not tracked) and data/committor/results.json.
"""
import argparse
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parents[2] / "data" / "committor"
BIN = Path(__file__).resolve().parent / "ssa_commit"
RAW = HERE / "raw"
b0, b1, Kh, h, d0, th, aC, ka = 0.5, 2.0, 1.0, 4, 0.8, 1.0, 2.0, 1.0
C = 0.36
RECT = (0.3, 0.7)
ONBOX = (0.15, 0.30)
H = 2000.0
RS = [0.5, 1.0, 2.0, 4.0, 8.0]
NS = [50, 100, 200, 400, 800, 1600, 3200]
SS = [-1, 0, 1]
SEED_TAG = 2026091700


def b(w): return b0 + b1 * w**h / (Kh**h + w**h)
def db(w): return b1 * h * Kh**h * w**(h - 1) / (Kh**h + w**h) ** 2
def d(x): return d0 + C + th * x


G = lambda x: b(aC * x / ka) - d(x)
XS = brentq(G, 0.3, 0.8); XON = brentq(G, 1.0, 2.0)
WS = aC * XS / ka; WON = aC * XON / ka


def eig(r):
    J = np.array([[-th * XS, XS * db(WS)], [r * aC, -r * ka]])
    lam, V = np.linalg.eig(J)
    iu, is_ = int(np.argmax(lam)), int(np.argmin(lam))
    vu, vs = V[:, iu], V[:, is_]
    vu = vu / np.linalg.norm(vu) * np.sign(vu[0])
    vs = vs / np.linalg.norm(vs) * np.sign(vs[0])
    return lam[iu], lam[is_], vu, vs


def start(r, N, s):
    lu, ls, vu, vs = eig(r)
    target = np.array([XS, WS]) + s * vu / np.sqrt(N)
    X0, W0 = int(round(N * target[0])), int(round(N * target[1]))
    delta = np.array([X0 / N - XS, W0 / N - WS])
    coef = np.linalg.solve(np.column_stack([vs, vu]), delta)
    return X0, W0, float(coef[1] * np.sqrt(N)), float(coef[0] * np.sqrt(N))


def run_chunk(r, N, s, ch, n):
    X0, W0, _, _ = start(r, N, s)
    with tempfile.TemporaryDirectory() as td:
        f = f"{td}/o.bin"
        cmd = [str(BIN), str(N), repr(r), repr(C), str(X0), str(W0), repr(RECT[0]), repr(RECT[1]),
               repr(XON), repr(WON), repr(ONBOX[0]), repr(ONBOX[1]), repr(H), str(n),
               str(SEED_TAG + ch), str(int(round(100 * r))), str(N * 10 + (s + 1))]
        t0 = time.time()
        subprocess.run(cmd + [f], check=True)
        return np.fromfile(f).reshape(-1, 3), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--nrep", type=int, default=4000)
    ap.add_argument("--nchunk", type=int, default=10)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    cases = [(r, N, s) for r in RS for N in NS for s in SS]
    per = args.nrep // args.nchunk
    t_all = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {}
        for r, N, s in cases:
            if (RAW / f"r{r:g}_N{N}_s{s}.npz").exists():
                continue
            futs[(r, N, s)] = [ex.submit(run_chunk, r, N, s, ch, per) for ch in range(args.nchunk)]
        for (r, N, s), fl in futs.items():
            parts = [f.result() for f in fl]
            out = np.vstack([p[0] for p in parts])
            X0, W0, xi_u, xi_s = start(r, N, s)
            np.savez_compressed(RAW / f"r{r:g}_N{N}_s{s}.npz", outcome=out[:, 0].astype(np.int8),
                                t=out[:, 1], nev=out[:, 2], X0=X0, W0=W0, xi_u=xi_u, xi_s=xi_s,
                                core_secs=sum(p[1] for p in parts))
            print(f"r={r:g} N={N} s={s:+d}: q={np.mean(out[:,0]==1):.3f} unresolved={np.mean(out[:,0]==2):.4f} "
                  f"xi_u={xi_u:+.3f} core-s={sum(p[1] for p in parts):.1f} wall={time.time()-t_all:.0f}s",
                  flush=True)
    # summary
    res = dict(settings=dict(c=C, x_star=XS, w_star=WS, x_on=XON, w_on=WON, rect=RECT, onbox=ONBOX,
                             H=H, nrep=args.nrep, seed_tag=SEED_TAG), cases=[])
    for r in RS:
        lu, ls, vu, vs = eig(r)
        for N in NS:
            for s in SS:
                z = np.load(RAW / f"r{r:g}_N{N}_s{s}.npz")
                o = z["outcome"]; n1 = int((o == 1).sum()); n0 = int((o == 0).sum()); n2 = int((o == 2).sum())
                m = n0 + n1
                q = n1 / m if m else float("nan")
                # Wilson 95% interval on resolved trials
                zc = 1.959964
                den = 1 + zc**2 / m
                cen = (q + zc**2 / (2 * m)) / den
                half = zc * np.sqrt(q * (1 - q) / m + zc**2 / (4 * m * m)) / den
                res["cases"].append(dict(r=r, N=N, s=s, X0=int(z["X0"]), W0=int(z["W0"]),
                                         xi_u=float(z["xi_u"]), xi_s=float(z["xi_s"]),
                                         n_off=n1, n_on=n0, n_unresolved=n2, q=q,
                                         q_ci95=[cen - half, cen + half],
                                         mean_t=float(np.mean(z["t"])), core_secs=float(z["core_secs"]),
                                         mu_u=float(lu), mu_s=float(ls), v_u=vu.tolist(), v_s=vs.tolist()))
    (HERE / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print("wrote results.json")


if __name__ == "__main__":
    main()
