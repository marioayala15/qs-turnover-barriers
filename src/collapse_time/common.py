"""Shared definitions for the collapse-time campaign (threshold and off-target times).

Parameters and rates follow the parameter table of the paper and
``net_wellmixed_explicit`` in ``src/ldp_action.py`` (used for cross-checks).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]                     # repository root
EXP = PAPER / "data" / "collapse_time"      # stored output of this experiment
COMPANION_SRC = PAPER / "src"               # src/, for ldp_action
os.environ.setdefault("MPLCONFIGDIR", str(PAPER / ".mplconfig"))
sys.path.insert(0, str(COMPANION_SRC))
from ldp_action import PAR, hill, net_wellmixed_explicit  # noqa: E402

C_COST = 0.36
BIN = HERE / "ssa_offtarget"
RAW = EXP / "raw"

# Off rectangles in density units (Proposition 1 of the revision plan).
RECT_LARGE = (0.30, 0.70)
RECT_SMALL = (0.15, 0.35)
# On-box used only to define a "return near U_on" (retry) after a threshold entry.
ON_BOX = (0.15, 0.30)   # half-widths in x and w

BASE_SEED = 20260812


def fixed_points(c=C_COST):
    """Same root search as ldp_crossover_exit.fixed_points (reimplemented)."""
    P = PAR
    g = lambda x: (P["b0"] + P["b1"] * hill(P["aC"] * x / P["kappa"], P["Kh"], P["h"])
                   - P["d0"] - c - P["theta"] * x)
    xs = np.linspace(1e-6, 3.0, 400000)
    idx = np.where(np.diff(np.sign(g(xs))))[0]
    rt = [brentq(g, xs[i], xs[i + 1]) for i in idx]
    return rt[0], rt[1]


def b_of(w):
    return PAR["b0"] + PAR["b1"] * hill(w, PAR["Kh"], PAR["h"])


def rect_conditions(a, b, c=C_COST):
    x_star, _ = fixed_points(c)
    return dict(a_off=a, b_off=b,
                b_of_b_off=float(b_of(b)), d0_plus_c=PAR["d0"] + c,
                cond1=bool(b_of(b) < PAR["d0"] + c),
                alphaC_a=PAR["aC"] * a, kappa_b=PAR["kappa"] * b,
                cond2=bool(PAR["aC"] * a < PAR["kappa"] * b),
                a_below_xstar=bool(a < x_star))


def initial_counts(N, c=C_COST):
    _, x_on = fixed_points(c)
    U0 = np.array([x_on, PAR["aC"] * x_on / PAR["kappa"]])
    return np.rint(U0 * N).astype(np.int64)   # as in ldp_crossover_exit


def saddle_actions(c=C_COST):
    d = json.loads((PAPER / "data" / "crossover_barriers.json").read_text())
    return {row["r"]: row["dV"] for row in d["costs"][f"{c:g}"]["rows"]}


def old_threshold_points():
    return json.loads((PAPER / "data" / "crossover_exit_times.json").read_text())


# ------------------------------------------------------------------ estimators

def censored_mle(t, event, horizon):
    """Right-censored exponential MLE: sum(exposure)/n_events.

    t: event times (inf if not observed by `horizon`); exposure min(t, horizon).
    """
    t = np.asarray(t, float)
    ev = np.asarray(event, bool) & (t <= horizon)
    n = int(ev.sum())
    if n == 0:
        return np.nan, 0
    # same summation order as ldp_crossover_exit.censored_mle
    total = float(t[ev].sum()) + horizon * (len(t) - n)
    return total / n, n


def fit_slope(Ns, Etau, nexit):
    """Weighted LS of log E on (N, log N, 1), weights n_exit, errors scaled by
    max(1, chi2_red)^(1/2): identical to ldp_crossover_exit.fit_barrier."""
    N = np.asarray(Ns, float)
    y = np.log(np.asarray(Etau, float))
    w = np.asarray(nexit, float)
    A = np.vstack([N, np.log(N), np.ones_like(N)]).T
    rt = np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq(rt * A, rt[:, 0] * y, rcond=None)
    cov = np.linalg.inv(A.T @ (w[:, None] * A))
    dof = len(N) - 3
    red = float(np.sum(w * (y - A @ sol) ** 2)) / max(dof, 1)
    return dict(slope=float(sol[0]), se_nominal=float(np.sqrt(cov[0, 0])),
                se=float(np.sqrt(cov[0, 0] * max(1.0, red))) if dof > 0 else float("nan"),
                b_logN=float(sol[1]), a=float(sol[2]), chi2_red=red if dof > 0 else float("nan"),
                dof=dof)


def fit_slope_linear(Ns, Etau, nexit):
    """Secondary model without the log N nuisance term: log E = a + N dV."""
    N = np.asarray(Ns, float)
    y = np.log(np.asarray(Etau, float))
    w = np.asarray(nexit, float)
    A = np.vstack([N, np.ones_like(N)]).T
    rt = np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq(rt * A, rt[:, 0] * y, rcond=None)
    cov = np.linalg.inv(A.T @ (w[:, None] * A))
    dof = len(N) - 2
    red = float(np.sum(w * (y - A @ sol) ** 2)) / max(dof, 1)
    return dict(slope=float(sol[0]), se=float(np.sqrt(cov[0, 0] * max(1.0, red))),
                a=float(sol[1]), chi2_red=red, dof=dof)
