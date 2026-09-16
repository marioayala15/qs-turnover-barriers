"""Independent check of the r=0.5 threshold barrier (own code, no shared solver).

Discrete action as in eq:discrete-action: midpoint state, forward velocity,
local rate from eq:explicit-L with analytic envelope gradients.
"""
import sys
import numpy as np
from scipy.optimize import minimize, brentq

b0, b1, K, h, d0, th, al, ka = 0.5, 2.0, 1.0, 4, 0.8, 1.0, 2.0, 1.0
c = 0.36


def b(w): return b0 + b1 * w**h / (K**h + w**h)
def db(w): return b1 * h * K**h * w**(h - 1) / (K**h + w**h) ** 2
def d(x): return d0 + c + th * x


def ell(v, lp, lm):
    s = np.sqrt(v * v + 4 * lp * lm)
    p = np.log((v + s) / (2 * lp))
    val = v * p - s + lp + lm
    # envelope derivatives
    return val, p, -(np.exp(p) - 1), -(np.exp(-p) - 1)


def action(z, r, A, B, M, T):
    dt = T / M
    U = np.vstack([A, z.reshape(M - 1, 2), B])
    m = 0.5 * (U[1:] + U[:-1])
    v = (U[1:] - U[:-1]) / dt
    x, w = m[:, 0], m[:, 1]
    lx, px, gxp, gxm = ell(v[:, 0], x * b(w), x * d(x))
    lw, pw, gwp, gwm = ell(v[:, 1], r * al * x, r * ka * w)
    S = dt * np.sum(lx + lw)
    gm = np.empty_like(m)
    gm[:, 0] = gxp * b(w) + gxm * (d(x) + th * x) + gwp * r * al
    gm[:, 1] = gxp * x * db(w) + gwm * r * ka
    gv = np.stack([px, pw], axis=1)
    gU = np.zeros_like(U)
    gU[:-1] += dt * 0.5 * gm - gv
    gU[1:] += dt * 0.5 * gm + gv
    return S, gU[1:-1].ravel()


G = lambda x: b(al * x / ka) - d(x)
xs = brentq(G, 0.3, 0.8); xon = brentq(G, 1.0, 2.0)
Uon = np.array([xon, al * xon / ka]); ws = al * xs / ka


def solve(r, B, M, T, seed):
    t = np.linspace(0, 1, M + 1)[1:-1, None]
    bump = np.array(seed) * np.sin(np.pi * t)
    z0 = (Uon + t * (B - Uon) + bump).ravel()
    bnds = [(1e-6, None)] * z0.size
    res = minimize(action, z0, args=(r, Uon, B, M, T), jac=True,
                   method="L-BFGS-B", bounds=bnds,
                   options=dict(maxiter=200000, maxfun=400000, ftol=1e-15, gtol=1e-10))
    Z = res.x.reshape(M - 1, 2)
    return res.fun, Z[:, 0].min(), res.message


r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
wc = float(sys.argv[2]) if len(sys.argv) > 2 else 1.8083669
for T, M in [(40, 800), (80, 1600)]:
    for name, B in [("saddle", np.array([xs, ws])), ("candidate", np.array([xs, wc]))]:
        for seed in [(0.0, 0.0), (0.0, 0.5), (0.1, -0.3)]:
            S, mx, msg = solve(r, B, M, T, seed)
            print(f"r={r} T={T} M={M} {name:9s} seed={seed} S={S:.7f} min_x_interior={mx:.4f} {msg}")
