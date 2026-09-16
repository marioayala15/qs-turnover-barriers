"""Threshold barrier: shared machinery.

Interior action from U_on to threshold points (x*, w) of the four-channel
cell-signal network, c = 0.36.  The solver is the published minimum-action
code `src/ldp_action.py` (only
`Network`, `lagrangian`, `mam_action`, `hill`, `dhill`, `PAR`,
`net_wellmixed_explicit` are used).  `mam_action_bounded` below is a copy of
`ldp_action.mam_action` whose only change is that L-BFGS-B receives box
bounds (used for the first-hit constraint x_k >= x* at interior nodes and
for bounding a free terminal signal); with no active bounds it is the same
discretisation (midpoint rule, forward differences, uniform mesh).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize

COMPANION_SRC = Path(__file__).resolve().parents[1]  # src/, for ldp_action
sys.path.insert(0, str(COMPANION_SRC))
import ldp_action as la  # noqa: E402

C = 0.36
P = la.PAR
R_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
FLOOR = 1e-9  # same positivity floor as ldp_action.mam_action


def b(w):
    return P["b0"] + P["b1"] * la.hill(w, P["Kh"], P["h"])


def d(x):
    return P["d0"] + C + P["theta"] * x


def wbar(x):
    return P["aC"] * x / P["kappa"]


def equilibria():
    g = lambda x: b(wbar(x)) - d(x)
    xs = np.linspace(1e-3, 3.0, 3000)
    gv = g(xs)
    roots = [brentq(g, xs[i], xs[i + 1], xtol=1e-15)
             for i in range(len(xs) - 1) if gv[i] * gv[i + 1] < 0]
    assert len(roots) == 2, roots
    return roots[0], roots[1]


X_STAR, X_ON = equilibria()
W_STAR, W_ON = wbar(X_STAR), wbar(X_ON)
U_ON = np.array([X_ON, W_ON])
U_STAR = np.array([X_STAR, W_STAR])


def net(r):
    return la.net_wellmixed_explicit(C, rN=r)


# ------------------------------------------------ independent action check
def _ell(v, lp, lm):
    """Closed-form Legendre transform for a pair of opposite unit jumps
    (eq:explicit-legendre), written to avoid cancellation for v < 0."""
    s = np.sqrt(v * v + 4 * lp * lm)
    p = np.where(v >= 0, np.log((v + s) / (2 * lp)),
                 np.log(2 * lm / np.maximum(s - v, 1e-300)))
    return v * p - s + lp + lm


def explicit_action(path, T, r):
    """Discrete action (eq:discrete-action) with the closed form eq:explicit-L."""
    path = np.asarray(path)
    K = path.shape[0] - 1
    h = T / K
    U = 0.5 * (path[:-1] + path[1:])
    V = (path[1:] - path[:-1]) / h
    x, w = U[:, 0], U[:, 1]
    L = (_ell(V[:, 0], x * b(w), x * d(x))
         + _ell(V[:, 1], r * P["aC"] * x, r * P["kappa"] * w))
    return float(np.sum(L) * h)


# ---------------------------------------------------- bounded MAM (copy)
def mam_action_bounded(nt, U_A, U_B, T, K, path0, xlow=None, end_free=None,
                       wT_bounds=(None, None), maxiter=20000):
    """ldp_action.mam_action with L-BFGS-B box bounds.

    xlow      : lower bound on x at interior nodes 1..K-1 (first-hit
                constraint x >= x*); None -> FLOOR as in the original.
    end_free  : mask of terminal components left free (as in the original).
    wT_bounds : bounds on a free terminal w.
    """
    U_A = np.asarray(U_A, float)
    U_B = np.asarray(U_B, float)
    n = nt.n
    h = T / K
    path = np.array(path0, float)
    free = np.zeros(n, bool) if end_free is None else np.asarray(end_free, bool)
    nf = int(free.sum())

    def unpack(z):
        Pm = np.empty((K + 1, n))
        Pm[0] = U_A
        Pm[1:-1] = z[:(K - 1) * n].reshape(K - 1, n)
        Pm[-1] = U_B
        if nf:
            Pm[-1, free] = z[(K - 1) * n:]
        return np.maximum(Pm, FLOOR)

    cache = {"p": None}

    def fun_and_grad(z):
        Pm = unpack(z)
        Ubar = 0.5 * (Pm[:-1] + Pm[1:])
        V = (Pm[1:] - Pm[:-1]) / h
        L, pstar = la.lagrangian(nt, Ubar, V, p0=cache["p"])
        cache["p"] = pstar
        S = float(np.sum(L) * h)
        dU = -nt.gradU_H(Ubar, pstar)
        G = np.zeros((K + 1, n))
        G[:-1] += 0.5 * h * dU
        G[1:] += 0.5 * h * dU
        G[1:] += pstar
        G[:-1] -= pstar
        g = (np.concatenate([G[1:-1].ravel(), G[-1, free]]) if nf
             else G[1:-1].ravel())
        return S, g

    lo = FLOOR if xlow is None else xlow
    bnds = [(lo, None), (FLOOR, None)] * (K - 1)
    if nf:
        for j in np.nonzero(free)[0]:
            bnds.append(wT_bounds if j == 1 else (FLOOR, None))
    z0 = path[1:-1].copy()
    z0[:, 0] = np.maximum(z0[:, 0], lo)
    z0 = z0.ravel()
    if nf:
        tail = path[-1, free].copy()
        if nf and free[1]:
            lo_w, hi_w = wT_bounds
            if lo_w is not None:
                tail[-1] = max(tail[-1], lo_w)
            if hi_w is not None:
                tail[-1] = min(tail[-1], hi_w)
        z0 = np.concatenate([z0, tail])
    res = minimize(fun_and_grad, z0, jac=True, method="L-BFGS-B", bounds=bnds,
                   options=dict(maxiter=maxiter, maxfun=maxiter * 2,
                                ftol=1e-15, gtol=1e-10))
    return dict(S=float(res.fun), path=unpack(res.x), T=T,
                success=bool(res.success), nit=int(res.nit),
                message=str(res.message))


# ------------------------------------------------------------------ seeds
def regrid(path, K):
    s_old = np.linspace(0, 1, path.shape[0])
    s_new = np.linspace(0, 1, K + 1)
    return np.stack([np.interp(s_new, s_old, path[:, j]) for j in range(2)], 1)


def polyline(points, K, weights=None):
    """Piecewise-linear path through `points`, uniform in arclength-parameter
    per segment (segment i gets fraction weights[i] of the nodes)."""
    pts = np.asarray(points, float)
    m = len(pts) - 1
    wts = np.ones(m) / m if weights is None else np.asarray(weights) / np.sum(weights)
    knots = np.concatenate([[0], np.cumsum(wts)])
    s = np.linspace(0, 1, K + 1)
    return np.stack([np.interp(s, knots, pts[:, j]) for j in range(2)], 1)


def seed_paths(w_end, K, saddle_path=None):
    """Named initial paths U_on -> (x*, w_end).  All keep x >= x* at nodes."""
    z = np.array([X_STAR, w_end])
    xm = 0.5 * (X_ON + X_STAR)
    seeds = {
        # direct interpolation (the default guess of mam_action)
        "line": polyline([U_ON, z], K),
        # wait at U_on for half the horizon, then straight line
        "late_line": polyline([U_ON, U_ON, z], K, [0.5, 0.5]),
        # early signal depletion: signal dropped well below w* first
        "deplete": polyline([U_ON, [X_ON, 0.4 * W_STAR],
                             [X_STAR + 0.02, 0.4 * W_STAR], z], K,
                            [0.4, 0.3, 0.3]),
        # signal retention: cells fall while signal stays at/above w_on
        "retain": polyline([U_ON, [xm, max(W_ON, w_end)],
                            [X_STAR + 0.02, max(W_ON, w_end)], z], K,
                           [0.4, 0.3, 0.3]),
        # overshoot: signal first rises above w_on, then cells fall
        "overshoot": polyline([U_ON, [X_ON, 1.5 * max(W_ON, w_end)], z], K,
                              [0.5, 0.5]),
    }
    if saddle_path is not None:
        sp = regrid(saddle_path, K)
        # saddle path with the terminal offset blended in over the last 30 %
        s = np.linspace(0, 1, K + 1)
        ramp = np.clip((s - 0.7) / 0.3, 0, 1) ** 2
        sp2 = sp + ramp[:, None] * (z - U_STAR)[None, :]
        sp2[:, 0] = np.maximum(sp2[:, 0], X_STAR)
        seeds["saddle"] = sp2
        # saddle path followed by a straight vertical leg along x = x*+eps
        sp3 = regrid(np.vstack([saddle_path[:: max(1, saddle_path.shape[0] // K)],
                                [[X_STAR, w_end]]]), K)
        seeds["saddle_then_leg"] = sp3
    return seeds


def diagnostics(path, xstar=X_STAR, tol=1e-7):
    """First-hit diagnostics on the discrete path (piecewise-linear
    interpolation: nodes >= x* implies the whole polygon is >= x*)."""
    x = path[:-1, 0]
    below = np.nonzero(x < xstar - tol)[0]
    touch = np.nonzero(np.abs(x - xstar) <= 1e-6)[0]
    touch = touch[touch > 0]
    return dict(
        min_x_before_end=float(x[1:].min()) if len(x) > 1 else float(x[0]),
        first_below_index=int(below[0]) if len(below) else None,
        w_at_first_below=float(path[below[0], 1]) if len(below) else None,
        n_nodes_touching=int(len(touch)),
        w_at_first_touch=float(path[touch[0], 1]) if len(touch) else None,
        w_min=float(path[:, 1].min()), w_max=float(path[:, 1].max()),
        w_end=float(path[-1, 1]),
    )


def solve_point(r, w_end, seed, T, dt, constrained=True, free_end=False,
                wT_bounds=(None, None), maxiter=20000):
    """One fixed-(or free-)endpoint optimisation from a given seed path."""
    K = int(round(T / dt))
    nt = net(r)
    path0 = regrid(np.asarray(seed, float), K)
    res = mam_action_bounded(
        nt, U_ON, [X_STAR, w_end], T, K, path0,
        xlow=X_STAR if constrained else None,
        end_free=[False, True] if free_end else None,
        wT_bounds=wT_bounds, maxiter=maxiter)
    res["S_explicit"] = explicit_action(res["path"], T, r)
    res["diag"] = diagnostics(res["path"])
    return res
