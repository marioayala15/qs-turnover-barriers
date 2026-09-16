"""Simulator validation for the collapse-time campaign.

(1) A numpy port of the existing vectorised SSA (ldp_crossover_exit.first_passage,
    re-typed here, using net_wellmixed_explicit from src/ldp_action.py) with the
    existing seed must reproduce stored threshold data points EXACTLY.
(2) The C simulator's propensities are compared with net_wellmixed_explicit.
(3) The C simulator in threshold-only mode, run with independent seeds at the
    stored horizons, must agree statistically with the stored estimates.

Writes data/collapse_time/validation.json.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time

import numpy as np

from common import (BASE_SEED, BIN, C_COST, EXP, ON_BOX, RECT_LARGE, RECT_SMALL,
                    censored_mle, fixed_points, initial_counts, net_wellmixed_explicit,
                    old_threshold_points)


def first_passage_port(net, N, counts0, thr_density, n_rep, horizon, rng):
    D = np.rint(net.D).astype(np.int64)
    counts = np.tile(np.asarray(counts0, np.int64), (n_rep, 1))
    T = np.zeros(n_rep)
    tau = np.full(n_rep, np.inf)
    alive = np.arange(n_rep)
    thr = thr_density * N
    while alive.size:
        a = N * net.rates(counts[alive] / N)
        tot = a.sum(axis=-1)
        newT = T[alive] + rng.exponential(1.0 / np.maximum(tot, 1e-300))
        over = newT > horizon
        if over.any():
            alive, a, tot, newT = alive[~over], a[~over], tot[~over], newT[~over]
            if alive.size == 0:
                break
        pick = (rng.random(alive.size) * tot)[:, None] < np.cumsum(a, axis=-1)
        counts[alive] += D[np.argmax(pick, axis=-1)]
        T[alive] = newT
        hit = counts[alive, 0] <= thr
        if hit.any():
            tau[alive[hit]] = T[alive[hit]]
            alive = alive[~hit]
    return tau


def c_rates(X, W, N, r, c):
    q = W / N
    b = 0.5 + 2.0 * q**4 / (1 + q**4)
    return np.array([X * b, X * (0.8 + c + X / N), r * 2.0 * X, r * 1.0 * W])


def run_c_thr(N, r, H, nrep, seeds):
    x_star, x_on = fixed_points()
    X0, W0 = initial_counts(N)
    with tempfile.TemporaryDirectory() as td:
        ft, fa = f"{td}/t.bin", f"{td}/a.bin"
        cmd = [str(BIN), "thr", str(N), repr(r), repr(C_COST), str(X0), str(W0),
               repr(x_star), repr(H), str(nrep), *map(str, seeds),
               *map(repr, (*RECT_LARGE, *RECT_SMALL, x_on, 2 * x_on, *ON_BOX)), ft, fa]
        subprocess.run(cmd, check=True)
        d = np.fromfile(ft).reshape(-1, 12)
    return d[:, 0]


def main():
    out = {}
    old = old_threshold_points()
    x_star, x_on = fixed_points()

    # (1) exact reproduction
    exact = []
    for r, N in [(0.5, 40), (0.5, 50), (1.0, 40)]:
        row = next(q for q in old[f"{r:g}" if r != 1.0 else "1.0"] if q["N"] == N)
        net = net_wellmixed_explicit(C_COST, rN=r)
        rng = np.random.default_rng(row["seed"])
        t0 = time.time()
        tau = first_passage_port(net, N, initial_counts(N), x_star, row["n_rep"],
                                 row["horizon"], rng)
        E, n = censored_mle(tau, np.isfinite(tau), row["horizon"])
        exact.append(dict(r=r, N=N, stored_Etau=row["Etau"], port_Etau=E,
                          stored_n_exit=row["n_exit"], port_n_exit=n,
                          identical=bool(E == row["Etau"] and n == row["n_exit"]),
                          rel_diff=float(E / row["Etau"] - 1), secs=time.time() - t0))
        print(exact[-1], flush=True)
    out["exact_port_reproduction"] = exact

    # (2) propensities
    rng = np.random.default_rng(1)
    maxdiff = 0.0
    for _ in range(200):
        N = int(rng.integers(20, 200)); r = float(rng.choice([0.5, 1, 8]))
        X = int(rng.integers(0, 3 * N)); W = int(rng.integers(0, 6 * N))
        net = net_wellmixed_explicit(C_COST, rN=r)
        ref = N * net.rates(np.array([X / N, W / N]))
        maxdiff = max(maxdiff, float(np.max(np.abs(ref - c_rates(X, W, N, r, C_COST))
                                            / np.maximum(1e-12, np.abs(ref)))))
    out["propensity_max_rel_diff"] = maxdiff
    print("propensity max rel diff", maxdiff)

    # (3) statistical agreement of the C code, threshold-only mode
    stat = []
    for r, N, key in [(0.5, 40, "0.5"), (0.5, 50, "0.5"), (1.0, 40, "1.0"),
                      (1.0, 55, "1.0"), (8.0, 60, "8.0")]:
        row = next(q for q in old[key] if q["N"] == N)
        nrep = 6000
        t0 = time.time()
        tau = run_c_thr(N, r, row["horizon"], nrep, (BASE_SEED + 777, int(round(100 * r)), N))
        E, n = censored_mle(tau, np.isfinite(tau), row["horizon"])
        se = np.sqrt(1 / n + 1 / row["n_exit"])
        stat.append(dict(r=r, N=N, horizon=row["horizon"], stored_Etau=row["Etau"],
                         stored_n_exit=row["n_exit"], c_Etau=E, c_n_exit=n, c_nrep=nrep,
                         log_ratio=float(np.log(E / row["Etau"])), se_log_ratio=float(se),
                         z=float(np.log(E / row["Etau"]) / se), secs=time.time() - t0))
        print(stat[-1], flush=True)
    out["c_threshold_statistical"] = stat
    zs = np.array([q["z"] for q in stat])
    from scipy import stats as st
    out["c_threshold_chi2"] = dict(sum_z2=float(np.sum(zs**2)), dof=len(zs),
                                   p_value=float(st.chi2.sf(np.sum(zs**2), len(zs))))

    # (4) C simulator vs the numpy port, both with fresh seeds, many replicas
    head = []
    for r, N, H in [(0.5, 40, 201.26553545125734), (8.0, 60, 77.96227907454048)]:
        nrep = 20000
        net = net_wellmixed_explicit(C_COST, rN=r)
        t0 = time.time()
        tp = first_passage_port(net, N, initial_counts(N), x_star, nrep, H,
                                np.random.default_rng([BASE_SEED + 555, int(100 * r), N]))
        tc = run_c_thr(N, r, H, nrep, (BASE_SEED + 999, int(round(100 * r)), N))
        Ep, npe = censored_mle(tp, np.isfinite(tp), H)
        Ec, nce = censored_mle(tc, np.isfinite(tc), H)
        ks = st.ks_2samp(np.minimum(tp, H), np.minimum(tc, H))
        se = np.sqrt(1 / npe + 1 / nce)
        head.append(dict(r=r, N=N, horizon=H, nrep=nrep, port_Etau=Ep, port_n_exit=npe,
                         c_Etau=Ec, c_n_exit=nce, log_ratio=float(np.log(Ec / Ep)),
                         se_log_ratio=float(se), z=float(np.log(Ec / Ep) / se),
                         ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                         secs=time.time() - t0))
        print(head[-1], flush=True)
    out["c_vs_port_fresh_seeds"] = head
    (EXP / "validation.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
