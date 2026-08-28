"""
Large-deviation machinery for the quorum-sensing reaction network.

Everything here works with the EXACT reaction-network Hamiltonian

    H(U,p) = sum_rho  beta_rho(U) * ( exp(<p, Delta_rho>) - 1 ),

i.e. the Hamiltonian of equation (10) of the paper, whose large-deviation
principle is Theorem 6.  Nothing is slaved unless a network is explicitly built
that way.  This matters, because eliminating the signal and taking the large
deviations do not commute: the barrier of the eliminated-signal model is the
r -> infinity limit of the barrier computed here, not its value at finite r,
and measuring that gap is what the paper does.

Contents
--------
Network            reaction network: jump vectors + density-dependent rates
lagrangian         L(U,v) = sup_p [<p,v> - H(U,p)]  by Newton, with gradients
mam_action         minimum action over paths U_A -> U_B at fixed horizon T
quasipotential_1d  closed form  V = int log(beta_minus/beta_plus)  (validation)

Run `python ldp_action.py` for the self-test against the 1-D closed form.
"""
import numpy as np
from scipy.optimize import minimize

# ``np.trapz`` was renamed ``np.trapezoid`` in NumPy 2.0 and then removed.
# The two are the same rule, so either gives identical quadrature.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# ----------------------------------------------------------------- network


class Network:
    """A density-dependent reaction network.

    jumps : (R, n) integer array of jump vectors Delta_rho
    rates : callable  U -> beta,  accepting (..., n) and returning (..., R)
    jac   : callable  U -> dbeta/dU, accepting (..., n), returning (..., R, n)
    """

    def __init__(self, jumps, rates, jac, name="", sparse=None):
        self.D = np.asarray(jumps, float)          # (R, n)
        self.rates = rates
        self.jac = jac
        self.n = self.D.shape[1]
        self.R = self.D.shape[0]
        self.name = name
        # Jump vectors have <=2 non-zeros, so Delta Delta^T is very sparse.
        # Precompute it flattened, (R, n*n), to make the Newton Hessian cheap.
        if sparse is None:
            sparse = self.n >= 8
        self._Msp = None
        if sparse:
            import scipy.sparse as sp
            rows, cols, vals = [], [], []
            for r in range(self.R):
                nz = np.nonzero(self.D[r])[0]
                for i in nz:
                    for j in nz:
                        rows.append(r)
                        cols.append(i * self.n + j)
                        vals.append(self.D[r, i] * self.D[r, j])
            self._Msp = sp.csr_matrix((vals, (rows, cols)),
                                      shape=(self.R, self.n * self.n))

    # mean-field drift  F(U) = sum_rho Delta_rho beta_rho(U)
    def F(self, U):
        return np.einsum("...r,rn->...n", self.rates(U), self.D)

    def H(self, U, p):
        e = np.exp(np.einsum("...n,rn->...r", p, self.D))
        return np.einsum("...r,...r->...", self.rates(U), e - 1.0)

    def gradp_H(self, U, p):
        e = np.exp(np.einsum("...n,rn->...r", p, self.D))
        return np.einsum("...r,...r,rn->...n", self.rates(U), e, self.D)

    def hessp_H(self, U, p):
        e = np.exp(np.einsum("...n,rn->...r", p, self.D))
        w = self.rates(U) * e                                    # (..., R)
        if self._Msp is not None:                # sparse route: O(K * nnz)
            flat = np.atleast_2d(w.reshape(-1, self.R))
            out = (self._Msp.T @ flat.T).T       # (K, n*n)
            return out.reshape(w.shape[:-1] + (self.n, self.n))
        return np.einsum("...r,rn,rm->...nm", w, self.D, self.D)

    def gradU_H(self, U, p):
        e = np.exp(np.einsum("...n,rn->...r", p, self.D))
        return np.einsum("...rn,...r->...n", self.jac(U), e - 1.0)


# --------------------------------------------------------------- Lagrangian


def lagrangian(net, U, v, p0=None, tol=1e-11, itmax=200, ridge=1e-10):
    """L(U,v) = sup_p [<p,v> - H(U,p)], solved by damped Newton on
    grad_p H(U,p) = v.  Vectorised over leading axes.

    Returns (L, p_star).  By the envelope theorem
        dL/dv = p_star,     dL/dU = -grad_U H(U, p_star).
    """
    U = np.atleast_2d(U)
    v = np.atleast_2d(v)
    p = np.zeros_like(v) if p0 is None else p0.copy()
    Id = np.eye(net.n)
    for _ in range(itmax):
        g = net.gradp_H(U, p) - v                                # residual
        if np.max(np.abs(g)) < tol:
            break
        Hs = net.hessp_H(U, p) + ridge * Id
        step = np.linalg.solve(Hs, g[..., None])[..., 0]
        # damped: never let <p,Delta> explode
        for _ in range(60):
            pn = p - step
            if np.all(np.abs(np.einsum("...n,rn->...r", pn, net.D)) < 60.0):
                break
            step = 0.5 * step
        p = p - step
    L = np.einsum("...n,...n->...", p, v) - net.H(U, p)
    return L, p


# ---------------------------------------------- minimum action over paths


def mam_min_over_T(net, U_A, U_B, Ts, K=200, end_free=None, verbose=False):
    """inf over BOTH path and horizon.

    When U_B is a fixed point the infimum is attained as T -> infinity and the
    sequence plateaus; when it is not (e.g. a coordinate pinned at a level the
    flow does not sustain) the infimum sits at a finite T and the sequence is
    U-shaped.  Taking the min covers both.
    """
    best, rows, path = np.inf, [], None
    for T in Ts:
        if path is not None:
            s_old = np.linspace(0, 1, path.shape[0])
            s_new = np.linspace(0, 1, K + 1)
            path = np.stack([np.interp(s_new, s_old, path[:, j])
                             for j in range(net.n)], axis=1)
        r = mam_action(net, U_A, U_B, T=T, K=K, path0=path,
                       end_free=end_free, verbose=False)
        path = r["path"]
        rows.append((T, r["S"]))
        if verbose:
            print(f"      T={T:7.2f}  S={r['S']:.6f}")
        if r["S"] < best:
            best, bestpath = r["S"], r["path"]
    return best, rows, bestpath


def mam_action(net, U_A, U_B, T=40.0, K=400, path0=None, maxiter=600,
               end_free=None, verbose=False):
    """Minimise the Freidlin-Wentzell action

        S_T[U] = int_0^T L(U, dU/dt) dt,   U(0)=U_A, U(T)=U_B

    over the interior nodes of a K-interval discretisation (midpoint rule).
    Gradients are analytic via the envelope theorem, so this scales to networks
    of many species, where finite differences would not.

    Returns dict with 'S', 'path', 'T', 'success'.
    """
    U_A = np.asarray(U_A, float)
    U_B = np.asarray(U_B, float)
    n = net.n
    h = T / K

    if path0 is None:                       # straight line as the initial guess
        s = np.linspace(0.0, 1.0, K + 1)[:, None]
        path = U_A[None, :] * (1 - s) + U_B[None, :] * s
    else:
        path = path0.copy()

    floor = 1e-9                            # keep strictly inside the positive orthant
    # end_free: boolean mask of components of U_B that are NOT pinned, so the
    # terminal state may relax inside the target set (e.g. pin the cell density
    # and let the signal find its cheapest configuration).
    free = (np.zeros(n, bool) if end_free is None else np.asarray(end_free, bool))
    nf = int(free.sum())

    def unpack(z):
        P = np.empty((K + 1, n))
        P[0] = U_A
        P[1:-1] = z[:(K - 1) * n].reshape(K - 1, n)
        P[-1] = U_B
        if nf:
            P[-1, free] = z[(K - 1) * n:]
        return np.maximum(P, floor)

    cache = {"p": None}

    def fun_and_grad(z):
        P = unpack(z)
        Ubar = 0.5 * (P[:-1] + P[1:])                    # (K, n)
        V = (P[1:] - P[:-1]) / h                         # (K, n)
        L, pstar = lagrangian(net, Ubar, V, p0=cache["p"])
        cache["p"] = pstar
        S = float(np.sum(L) * h)

        dU = -net.gradU_H(Ubar, pstar)                   # (K, n)
        G = np.zeros((K + 1, n))
        # d/dU_j from the U-slot of intervals j-1 and j (midpoint weight 1/2)
        G[:-1] += 0.5 * h * dU
        G[1:] += 0.5 * h * dU
        # d/dU_j from the v-slot: v_k = (U_{k+1}-U_k)/h, dL/dv = pstar
        G[1:] += pstar
        G[:-1] -= pstar
        g = np.concatenate([G[1:-1].ravel(), G[-1, free]]) if nf else G[1:-1].ravel()
        return S, g

    z0 = (np.concatenate([path[1:-1].ravel(), path[-1, free]]) if nf
          else path[1:-1].ravel())
    res = minimize(fun_and_grad, z0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=1e-14,
                                gtol=1e-10))
    if verbose:
        print(f"    L-BFGS: S={res.fun:.6f}  nit={res.nit}  {res.message}")
    return dict(S=float(res.fun), path=unpack(res.x), T=T, success=bool(res.success))


def mam_converged(net, U_A, U_B, Ts=(10, 20, 40, 80), K=400, verbose=False):
    """Run mam_action over increasing horizons, warm-starting, and report the
    plateau.  Escape actions are T -> infinity limits, so the sequence should
    settle from above."""
    out, path = [], None
    for T in Ts:
        if path is not None:                       # re-grid the previous optimum
            s_old = np.linspace(0, 1, path.shape[0])
            s_new = np.linspace(0, 1, K + 1)
            path = np.stack([np.interp(s_new, s_old, path[:, j])
                             for j in range(net.n)], axis=1)
        r = mam_action(net, U_A, U_B, T=T, K=K, path0=path, verbose=verbose)
        path = r["path"]
        out.append((T, r["S"]))
        if verbose:
            print(f"  T={T:6.1f}   S={r['S']:.6f}")
    return out, path


# ------------------------------------------------- 1-D closed form (check)


def quasipotential_1d(beta_plus, beta_minus, a, b, steps=200000):
    """V(b)-V(a) = int_a^b log(beta_minus/beta_plus) dx  for a 1-D
    birth-death network (the H=0 fluctuation branch)."""
    x = np.linspace(a, b, steps + 1)
    f = np.log(beta_minus(x) / beta_plus(x))
    return float(_trapezoid(f, x))


# ------------------------------------------------------------ QS networks

PAR = dict(b0=0.5, b1=2.0, d0=0.8, theta=1.0, aC=2.0, aH=0.5, kappa=1.0,
           Kh=1.0, h=4.0, beta=0.5, Denv=1.0)


def hill(r, Kh=1.0, hh=4.0):
    r = np.maximum(r, 0.0)
    rp = r ** hh
    return rp / (Kh ** hh + rp)


def dhill(r, Kh=1.0, hh=4.0):
    r = np.maximum(r, 1e-300)
    rp = r ** hh
    return hh * (Kh ** hh) * r ** (hh - 1) / (Kh ** hh + rp) ** 2


def net_1d_slaved(c, P=PAR):
    """Cooperator-only, well mixed, medium SLAVED (rho = aC x / kappa).
    n = 1.  This is the system whose barrier has a closed form."""
    jumps = np.array([[+1.0], [-1.0]])

    def rates(U):
        x = U[..., 0]
        rho = P["aC"] * x / P["kappa"]
        bp = x * (P["b0"] + P["b1"] * hill(rho, P["Kh"], P["h"]))
        bm = x * (P["d0"] + c + P["theta"] * x)
        return np.stack([bp, bm], axis=-1)

    def jac(U):
        x = U[..., 0]
        rho = P["aC"] * x / P["kappa"]
        b = P["b0"] + P["b1"] * hill(rho, P["Kh"], P["h"])
        db = P["b1"] * dhill(rho, P["Kh"], P["h"]) * P["aC"] / P["kappa"]
        d = P["d0"] + c + P["theta"] * x
        return np.stack([b + x * db, d + x * P["theta"]], axis=-1)[..., None]

    return Network(jumps, rates, jac, name=f"1d-slaved c={c}")


def net_wellmixed_explicit(c, P=PAR, rN=1.0):
    """Cooperator-only, well mixed, medium EXPLICIT.  n = 2, U = (x, w).
    Same mean-field fixed points as net_1d_slaved, but a different rate
    functional -- this is the pair that tests averaging vs. LDP."""
    jumps = np.array([[+1.0, 0.0],      # birth
                      [-1.0, 0.0],      # death
                      [0.0, +1.0],      # production
                      [0.0, -1.0]])     # degradation

    def rates(U):
        x, w = U[..., 0], U[..., 1]
        bp = x * (P["b0"] + P["b1"] * hill(w, P["Kh"], P["h"]))
        bm = x * (P["d0"] + c + P["theta"] * x)
        pr = rN * P["aC"] * x
        dg = rN * P["kappa"] * w
        return np.stack([bp, bm, pr, dg], axis=-1)

    def jac(U):
        x, w = U[..., 0], U[..., 1]
        sh = x.shape + (4, 2)
        J = np.zeros(sh)
        b = P["b0"] + P["b1"] * hill(w, P["Kh"], P["h"])
        J[..., 0, 0] = b
        J[..., 0, 1] = x * P["b1"] * dhill(w, P["Kh"], P["h"])
        J[..., 1, 0] = P["d0"] + c + 2 * P["theta"] * x
        J[..., 2, 0] = rN * P["aC"]
        J[..., 3, 1] = rN * P["kappa"]
        return J

    return Network(jumps, rates, jac, name=f"wellmixed-explicit c={c} rN={rN}")



if __name__ == "__main__":
    print("=" * 68)
    print("SELF-TEST: minimum action vs. the 1-D closed-form quasipotential")
    print("=" * 68)
    for c in (0.0, 0.36):
        net = net_1d_slaved(c)

        # fixed points of the per-capita growth
        xs = np.linspace(1e-6, 3.0, 400000)
        r = PAR["aC"] * xs / PAR["kappa"]
        g = (PAR["b0"] + PAR["b1"] * hill(r) - PAR["d0"] - c - PAR["theta"] * xs)
        idx = np.where(np.diff(np.sign(g)))[0]
        roots = []
        for i in idx:
            lo, hi = xs[i], xs[i + 1]
            for _ in range(100):
                m = 0.5 * (lo + hi)
                rm = PAR["aC"] * m / PAR["kappa"]
                gm = (PAR["b0"] + PAR["b1"] * hill(rm) - PAR["d0"] - c
                      - PAR["theta"] * m)
                gl = np.interp(lo, xs, g)
                if np.sign(gm) == np.sign(gl):
                    lo = m
                else:
                    hi = m
            roots.append(0.5 * (lo + hi))
        x_star, x_on = roots[0], roots[1]

        bp = lambda x: x * (PAR["b0"] + PAR["b1"] * hill(PAR["aC"] * x / PAR["kappa"]))
        bm = lambda x: x * (PAR["d0"] + c + PAR["theta"] * x)
        V_exact = quasipotential_1d(bp, bm, x_on, x_star)

        print(f"\nc = {c}:  x_on={x_on:.4f}  x_star={x_star:.4f}")
        print(f"  closed form  dV = {V_exact:.6f}")
        seq, _ = mam_converged(net, [x_on], [x_star], Ts=(10, 20, 40, 80),
                               K=500, verbose=True)
        print(f"  MAM (T=80)   dV = {seq[-1][1]:.6f}   "
              f"rel.err = {abs(seq[-1][1]-V_exact)/V_exact:.2%}")
