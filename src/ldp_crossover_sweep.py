"""
The finite-r extinction barriers behind the crossover figure.

`fig_proceedings_numerics.py` plots two things: the barrier DeltaV(r) at the
working cost across eight values of r, and the coefficient A(c) at five
costs.  This module is the driver behind both: it computes them from
`ldp_action.mam_action` and writes them to `data/crossover_barriers.json`, so
the numbers quoted in the paper have a script behind them.

WHAT IS COMPUTED.  For each (c, r) the barrier is the minimum of the discretised
Freidlin-Wentzell action over paths from the stable cooperative state to the
unstable threshold, with the medium an explicit coordinate:

    DeltaV_num(r; c) = min over paths of  S_{T,M}(U),   U(0) = U_on, U(T) = U_*,

both endpoints on the quasi-steady manifold w = wbar(x) = aC x / kappa.  The
medium is never slaved: `net_wellmixed_explicit` carries production and
degradation as their own reaction channels, which is the whole point -- slaving
it first is exactly the shortcut the paper measures the cost of.

TWO SEPARATE KNOBS: THE HORIZON AND THE STEP.  The target is a free-time
infimum and T is a fixed argument, so each value is formally an upper bound.
The bound is attained once T clears the slowest timescale in the problem, which
is the medium relaxation 1/(r kappa) -- so the binding case is the SLOW end of
the rate grid, not the fast end.  At r >= 1 the barrier is already converged at
T=20; at r=0.5 it is not (T=20 overstates it by 7e-5, enough to move the fourth
decimal), and converges by T=30.  We take T=40 for the whole grid.

Check the small-r end when changing this.  Sampling only r=4 and r=64 makes T
look irrelevant, because at those rates it is.

The other knob is dt = T/M, and it is the one that sets the accuracy.  Fixing M
instead of dt couples the two: raising T then silently coarsens the mesh, and
the resulting drift looks like a horizon effect when it is a resolution one.

MAXITER MATTERS TOO.  `mam_action` defaults to maxiter=600, which does NOT
converge at these mesh sizes -- L-BFGS stops on the iteration cap and returns a
value that is too high, increasingly so as M grows.  At M=2000 the default
overstates DeltaV(0.5) by 5 percent, which looks exactly like a discretisation
error and is not one.  We raise the budget and assert convergence.

EXTRACTING A, AND WHY THE MESH DECIDES WHERE.  Equation (2) of the paper gives
DeltaV(r) = DeltaV_inf + A/r + O(r^-2), so r (DeltaV_num(r) - DeltaV_inf) tends
to A.  The excess shrinks like 1/r while the discretisation error in DeltaV does
not, so this quantity multiplies a fixed absolute error by r -- on a coarse mesh
a 5e-6 error in the barrier becomes a 1 percent error in A at r=64, and the
sequence turns back upward at the top of the grid.  That is an artifact, and it
is the trap to avoid: it makes the largest rates look unusable and invites
averaging over a middle window instead, which bakes in the O(1/r) remainder the
expansion is trying to expose.

At dt=0.02 the amplified error is below the remainder and the sequence is clean:
monotone in r with increments that roughly halve as r doubles.  A monotone
sequence with geometrically shrinking increments is read at its far end, so we
take A_num at the largest rate and use the gap to the previous rate as the
O(1/r) residual.  This is a statement about the numerics, not about the answer;
`--check` is what licenses it.

    python src/ldp_crossover_sweep.py            # the sweep, ~30 s
    python src/ldp_crossover_sweep.py --write    # and write it
    python src/ldp_crossover_sweep.py --check    # T- and M-stability
"""
import json
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

from ldp_action import (PAR, hill, dhill, net_wellmixed_explicit, mam_action,
                        quasipotential_1d)
from paths import data_path

P = PAR
RESULTS = "crossover_barriers.json"

# The grids the paper displays.
R_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
C_GRID = (0.20, 0.30, 0.36, 0.45, 0.50)
C_WORK = 0.36                    # the cost at which the r-sweep is shown
# A is read at the largest rate, with the gap to the previous one as the
# residual.  On a coarser mesh this would be the worst choice, not the best;
# see the docstring.

# Discretisation.  T is set by the slow end of the rate grid (r=0.5 needs
# T >= 30; see `--check`), dt by the accuracy wanted in A.
T_HORIZON = 40.0
DT = 0.02
M_NODES = int(round(T_HORIZON / DT))
MAXITER = 60000


def pieces(c):
    """b, b', wbar, d, F and the two positive equilibria (x_star, x_on)."""
    b = lambda w: P["b0"] + P["b1"] * hill(w, P["Kh"], P["h"])
    db = lambda w: P["b1"] * dhill(w, P["Kh"], P["h"])
    wb = lambda x: P["aC"] * x / P["kappa"]
    d = lambda x: P["d0"] + c + P["theta"] * x
    F = lambda x: x * (b(wb(x)) - d(x))
    g = lambda x: b(wb(x)) - d(x)
    xs = np.linspace(1e-6, 3.0, 400000)
    idx = np.where(np.diff(np.sign(g(xs))))[0]
    rt = [brentq(g, xs[i], xs[i + 1]) for i in idx]
    return b, db, wb, d, F, rt[0], rt[1]


def Vinf(c):
    """Eliminated-medium barrier, closed form: int log(beta_minus/beta_plus)."""
    b, db, wb, d, F, x_st, x_on = pieces(c)
    return quasipotential_1d(lambda x: x * b(wb(x)),
                             lambda x: x * d(x), x_on, x_st)


def A_theory(c):
    """The analytical crossover coefficient, eq. (A) of the paper."""
    b, db, wb, d, F, x_st, x_on = pieces(c)
    f = lambda x: (db(wb(x)) * F(x) / b(wb(x))) * (1 - x * db(wb(x)) / b(wb(x)))
    return (P["aC"] / P["kappa"] ** 2) * quad(f, x_st, x_on, limit=400)[0]


def barrier(c, r, T=T_HORIZON, K=M_NODES, maxiter=MAXITER, strict=True):
    """DeltaV_num(r; c) by minimum action, medium explicit.

    Raises on a non-converged optimisation rather than returning the number,
    because the failure mode here is silent and upward: L-BFGS stopping on the
    iteration cap returns a plausible-looking value that is simply too big.
    """
    _, _, wb, _, _, x_st, x_on = pieces(c)
    net = net_wellmixed_explicit(c, rN=r)
    res = mam_action(net, [x_on, wb(x_on)], [x_st, wb(x_st)],
                     T=T, K=K, maxiter=maxiter)
    if strict and not res["success"]:
        raise RuntimeError(
            f"minimisation did not converge at c={c}, r={r} "
            f"(T={T}, M={K}, maxiter={maxiter}); raise maxiter")
    return res["S"]


def sweep(c, rs=R_GRID, verbose=True):
    """DeltaV_num over the rate grid at one cost, plus the A plateau."""
    V0 = Vinf(c)
    rows = []
    for r in rs:
        t0 = time.time()
        dV = barrier(c, r)
        rows.append(dict(r=r, dV=dV, excess=dV - V0, plateau=r * (dV - V0),
                         secs=time.time() - t0))
        if verbose:
            print(f"    r={r:5g}  DeltaV={dV:.6f}  r*(DeltaV-Vinf)={rows[-1]['plateau']:.6f}",
                  flush=True)
    tail = sorted(rows, key=lambda q: q["r"])[-2:]
    A_num = tail[-1]["plateau"]
    return rows, V0, float(A_num), float(abs(tail[-1]["plateau"] - tail[0]["plateau"]))


def check(c=C_WORK, rs=(0.5, 64.0)):
    """Separate the two things that could move the answer: T and dt."""
    V0 = Vinf(c)
    print(f"c={c}  Vinf={V0:.6f}\n")
    print("(1) horizon, at fixed dt -- M is scaled with T so the mesh is held.")
    print("    r=0.5 is the binding case: the medium relaxes at rate r*kappa,")
    print("    so the slowest rate needs the longest horizon.")
    print(f"    {'dt':>6}{'T':>6}{'M':>7}" +
          "".join(f"{'dV(r=%g)' % r:>12}" for r in rs))
    for dt in (0.04, 0.02):
        for T in (20.0, 30.0, 40.0, 60.0):
            M = int(round(T / dt))
            vs = [barrier(c, r, T=T, K=M) for r in rs]
            print(f"    {dt:6.3f}{T:6.0f}{M:7d}" +
                  "".join(f"{v:12.6f}" for v in vs))
    print("\n(2) step, at fixed T -- this is what actually moves it")
    print(f"    {'dt':>6}{'M':>7}" +
          "".join(f"{'dV(r=%g)' % r:>12}{'r*excess':>11}" for r in rs))
    for dt in (0.08, 0.04, 0.02, 0.01):
        M = int(round(T_HORIZON / dt))
        cells = ""
        for r in rs:
            v = barrier(c, r, K=M)
            cells += f"{v:12.6f}{r * (v - V0):11.5f}"
        print(f"    {dt:6.3f}{M:7d}" + cells)
    print("\n(3) the mam_action default budget, for contrast")
    for M in (400, 2000):
        v = barrier(c, rs[0], K=M, maxiter=600, strict=False)
        print(f"    maxiter=600, M={M}: dV={v:.6f}   <- not converged")


def main(write=False):
    out = {"settings": dict(T=T_HORIZON, M=M_NODES, maxiter=MAXITER,
                            dt=DT, r_grid=list(R_GRID), c_grid=list(C_GRID),
                            A_read_at="largest rate", par=dict(PAR)),
           "costs": {}}

    print(f"minimum-action barriers, medium explicit "
          f"(T={T_HORIZON:g}, M={M_NODES}, maxiter={MAXITER})\n")
    for c in C_GRID:
        print(f"  c = {c}")
        rows, V0, A_num, spread = sweep(c)
        out["costs"][f"{c}"] = dict(Vinf=V0, A_num=A_num,
                                    A_residual=spread,
                                    A_theory=A_theory(c), rows=rows)
        print(f"    Vinf={V0:.6f}  A_num={A_num:.6f} (residual {spread:.1e})"
              f"  A_theory={A_theory(c):.6f}\n")

    print(f"{'c':>6}{'Vinf':>11}{'A_theory':>11}{'A_num':>11}{'ratio':>8}")
    for c in C_GRID:
        q = out["costs"][f"{c}"]
        print(f"{c:6.2f}{q['Vinf']:11.6f}{q['A_theory']:11.6f}"
              f"{q['A_num']:11.6f}{q['A_num']/q['A_theory']:8.4f}")

    if write:
        with open(data_path(RESULTS), "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {data_path(RESULTS)}")
    return out


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main(write="--write" in sys.argv)
