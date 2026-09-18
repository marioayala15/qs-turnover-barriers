"""
The crossover barrier measured from particle-system exit times.

Everything else supporting DeltaV(r) shares one piece of machinery: the
action, its minimisers, and the boundary-layer ansatz.  This module shares
none of it.  The barrier also sets the mean extinction time of the jump
process itself,

    E[tau_N] = C(N) exp(N DeltaV(r)),

so DeltaV(r) is the large-N slope of log E[tau_N] and can be read off without
constructing any path.  The target here is the well-mixed crossover, where the
medium is an explicit species and r is a free parameter.

The measurement discriminates: at r = 0.5 the two candidate answers are
DeltaV(0.5) = 0.1225 and the eliminated DeltaV_inf = 0.0614.

Three choices worth stating.

CENSORING.  Replicas run to a fixed horizon rather than until every one
escapes.  The slowest of R replicas takes about log(R) times the mean, so
waiting for it costs a factor ~7 at R = 1500 and adds no information.  Escaped
replicas contribute their exit time, survivors contribute the horizon, and the
rate is the right-censored exponential MLE

    lambda_hat = n_exit / sum_i t_i,        E[tau] = 1 / lambda_hat,

which is unbiased for any horizon.  The horizon is sized from a predicted
DeltaV, but that only sets the time budget: it never enters the estimate.

VECTORISATION.  SSA is sequential per trajectory, but the replicas are
independent, so one reaction is taken for all live replicas per numpy step.
Cost is then governed by the horizon and barely by the replica count, which is
why 1500 replicas cost little more than 200.

READING THE SLOPE.  Do NOT fit a straight line in N: the prefactor C(N) is
absorbed into the slope and understates the barrier by 5-20% at reachable
population sizes.  Do not read the slope between the two largest N either --
it looks assumption-free but samples the residual N-dependence of C at one
point, and at r = 0.5 two adjacent windows differ by five sigma.  What is
quoted is a regression on N and log N over every size at that rate, with the
uncertainty scaled to unit reduced chi-square.

That scaling would be a fudge if the residual dependence were an artifact of
the estimator, so `--verify` checks that it is not: independent seeds
reproduce the nominal 1/sqrt(n_exit) spread, tau is exponential to about one
percent, and re-applying the censoring post hoc to one stored sample leaves
the estimate unchanged across escaped fractions from 0.5 to 1.  The
N-dependence therefore belongs to the process, and the scaling is an
allowance for it.
"""
import json
import time
import numpy as np
from scipy.optimize import brentq

from ldp_action import PAR, hill, net_wellmixed_explicit
from paths import data_path

P = PAR
RESULTS = "crossover_exit_times.json"
C_COST = 0.36                    # the cost at which the crossover is reported


def fixed_points(c):
    g = lambda x: (P["b0"] + P["b1"] * hill(P["aC"] * x / P["kappa"],
                                            P["Kh"], P["h"])
                   - P["d0"] - c - P["theta"] * x)
    xs = np.linspace(1e-6, 3.0, 400000)
    idx = np.where(np.diff(np.sign(g(xs))))[0]
    rt = [brentq(g, xs[i], xs[i + 1]) for i in idx]
    return rt[0], rt[1]                     # x_star (unstable), x_on (stable)


def first_passage(net, N, U0, coord, threshold, n_rep, horizon, rng):
    """Exact SSA on a Network until coordinate `coord` falls to `threshold`.

    Works for any Network, since it needs only `rates` and the jump vectors.
    Returns (tau, n_steps), with tau[i] = inf for replicas still alive at the
    horizon.  Keeping the individual times rather than only their total is
    what lets verify() re-apply the censoring post hoc to one fixed sample.
    """
    D = np.rint(net.D).astype(np.int64)                  # (R, n)
    counts = np.rint(np.asarray(U0, float) * N).astype(np.int64)
    counts = np.tile(counts, (n_rep, 1))
    T = np.zeros(n_rep)
    tau = np.full(n_rep, np.inf)
    alive = np.arange(n_rep)
    thr = threshold * N
    steps = 0

    while alive.size:
        steps += 1
        a = N * net.rates(counts[alive] / N)             # (n_alive, R)
        tot = a.sum(axis=-1)
        newT = T[alive] + rng.exponential(1.0 / np.maximum(tot, 1e-300))

        over = newT > horizon                            # right-censored
        if over.any():
            alive, a, tot, newT = (alive[~over], a[~over], tot[~over],
                                   newT[~over])
            if alive.size == 0:
                break

        pick = (rng.random(alive.size) * tot)[:, None] < np.cumsum(a, axis=-1)
        counts[alive] += D[np.argmax(pick, axis=-1)]
        T[alive] = newT

        hit = counts[alive, coord] <= thr
        if hit.any():
            tau[alive[hit]] = T[alive[hit]]
            alive = alive[~hit]

    return tau, steps


def censored_mle(tau, horizon):
    """Right-censored exponential MLE.  Unbiased for any horizon, which the
    third check in verify() confirms empirically rather than assuming."""
    hit = np.isfinite(tau) & (tau <= horizon)
    n = int(np.count_nonzero(hit))
    if n == 0:
        return np.nan, 0
    total = float(tau[hit].sum()) + horizon * (len(tau) - n)
    return total / n, n


def run_point(N, r, c, dV_guess, n_rep=1500, base_seed=20260812,
              horizon_mult=1.5):
    """One (r, N).  Its own random stream, derived reproducibly."""
    rng = np.random.default_rng([base_seed, int(round(r * 100)), N])
    x_st, x_on = fixed_points(c)
    net = net_wellmixed_explicit(c, rN=r)
    U0 = [x_on, P["aC"] * x_on / P["kappa"]]
    horizon = horizon_mult * np.exp(N * dV_guess)

    t0 = time.time()
    tau, steps = first_passage(net, N, U0, 0, x_st, n_rep, horizon, rng)
    Etau, n_exit = censored_mle(tau, horizon)
    if n_exit == 0:
        return None
    return dict(N=N, r=r, c=c, Etau=Etau, n_exit=n_exit,
                n_rep=n_rep, frac=n_exit / n_rep, horizon=horizon,
                steps=steps, secs=time.time() - t0,
                seed=[base_seed, int(round(r * 100)), N])


def fit_barrier(rows):
    """Weighted regression of log E[tau] on N and log N; returns (dV, err, a).

    Deliberately NOT the slope between the two largest N.  That estimator
    samples the residual N-dependence of C(N) at a single point instead of
    averaging over it, and at r = 0.5 two adjacent windows differ by five
    sigma.  Uncertainties are scaled to unit reduced chi-square -- never
    scaled down -- as an empirical allowance for that dependence, which
    verify() shows belongs to the process and not to the estimator.
    """
    N = np.array([q["N"] for q in rows], float)
    y = np.log([q["Etau"] for q in rows])
    w = np.array([q["n_exit"] for q in rows], float)   # var(log) ~ 1/n_exit
    A = np.vstack([N, np.log(N), np.ones_like(N)]).T
    rt = np.diag(np.sqrt(w))
    sol, *_ = np.linalg.lstsq(rt @ A, rt @ y, rcond=None)
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    red = float(np.sum(w * (y - A @ sol) ** 2)) / max(len(N) - 3, 1)
    return sol[0], float(np.sqrt(cov[0, 0] * max(1.0, red))), sol[1], red


def report(all_rows, dV_inst, dV_inf):
    print(f"\n{'r':>6}{'sizes':>7}{'measured':>11}{'+-':>9}{'a':>7}"
          f"{'chi2/dof':>10}{'DeltaV(r)':>11}{'sigma':>8}{'vs elim':>9}")
    for r in sorted(all_rows, key=float):
        rows = sorted(all_rows[r], key=lambda d: d["N"])
        if len(rows) < 4:
            print(f"{float(r):6.1f}   fewer than four sizes; fit not attempted")
            continue
        s, e, a, red = fit_barrier(rows)
        dV = dV_inst[float(r)]
        print(f"{float(r):6.1f}{len(rows):7d}{s:11.5f}{e:9.5f}{a:+7.2f}"
              f"{red:10.2f}{dV:11.5f}{(s-dV)/e:+8.1f}{(s-dV_inf)/e:+9.1f}")


def verify(r=0.5, N=80, c=C_COST, n_rep=3000, base_seed=4242):
    """The three checks that license the quoted uncertainties."""
    x_st, x_on = fixed_points(c)
    net = net_wellmixed_explicit(c, rN=r)
    U0 = [x_on, P["aC"] * x_on / P["kappa"]]
    dV = DV_INSTANTON[r]
    H = 1.5 * np.exp(N * dV)

    print(f"(1) independent seeds at r={r:g}, N={N}: is 1/sqrt(n_exit) real?")
    ly, nom = [], []
    for s in range(6):
        rng = np.random.default_rng([base_seed, s])
        tau, _ = first_passage(net, N, U0, 0, x_st, 1500, H, rng)
        Etau, n = censored_mle(tau, H)
        ly.append(np.log(Etau)); nom.append(1 / np.sqrt(n))
        print(f"      seed {s}: E[tau]={Etau:10.1f}  exits={n:5d}", flush=True)
    obs = float(np.std(ly, ddof=1))
    print(f"      observed sd {obs:.4f} vs nominal {np.mean(nom):.4f}"
          f"   ratio {obs/np.mean(nom):.2f}\n")

    rng = np.random.default_rng([base_seed, 99])
    tau, _ = first_passage(net, N, U0, 0, x_st, n_rep,
                           25.0 * np.exp(N * dV) * 0.19, rng)
    t = tau[np.isfinite(tau)]
    m = float(t.mean())
    print(f"(2) is tau exponential?  {len(t)}/{n_rep} escaped")
    print(f"      sd/mean {t.std(ddof=1)/m:.4f} (1.000)    "
          f"median/mean {np.median(t)/m:.4f} (0.693)\n")

    print("(3) estimate vs truncation, same stored sample (paired)")
    for mult in (0.15, 0.3, 0.6, 1.0, 1.5, 3.0):
        est, n = censored_mle(tau, mult * np.exp(N * dV))
        print(f"      mult {mult:4.2f}   exit frac {n/n_rep:.3f}   "
              f"E[tau] {est:9.1f}   ratio to full mean {est/m:.4f}")


# DeltaV(r) from the minimum-action runs, as plotted in fig_proceedings_numerics.py.
DV_INSTANTON = {0.5: 0.122479, 1.0: 0.095261, 2.0: 0.079060,
                4.0: 0.070316, 8.0: 0.065846}
DV_INF = 0.061379

# The runs behind the table in the paper.  Each r carries a list of segments
# rather than a bare list of N, so settings that differ within one r stay
# visible instead of being averaged into a single default.
#
# The r = 0.5 split is the interesting one.  Its measured prefactor is
# C ~ 0.19, so horizon = 1.5 exp(N dV) is roughly eight times E[tau]: every
# replica escapes long before the horizon and the loop ends up waiting on the
# slowest, paying a log(n_rep) straggler factor for nothing.  The N >= 90
# segment sizes the horizon from the measured C instead, which is what makes
# those points affordable at all, and spends the saving on more replicas --
# which is where the paper's r = 0.5 row comes from.
#
# The whole plan is about five hours.  Results are checkpointed after every
# point under --write, so an interrupted run is not lost.
PLAN = {
    0.5: [dict(Ns=[40, 50, 60, 70, 80], n_rep=1500, horizon_mult=1.5),
          dict(Ns=[90, 100, 110], n_rep=2500, horizon_mult=0.3)],
    1.0: [dict(Ns=[40, 47, 55, 62, 70, 77, 85, 92, 100],
               n_rep=1500, horizon_mult=1.5)],
    2.0: [dict(Ns=[50, 65, 80, 95, 110], n_rep=1500, horizon_mult=1.5)],
    4.0: [dict(Ns=[50, 60, 70, 80, 90, 100, 110],
               n_rep=1500, horizon_mult=1.5)],
    8.0: [dict(Ns=[60, 80, 100, 120, 140], n_rep=1500, horizon_mult=1.5)],
}

QUICK = {1.0: [dict(Ns=[20, 30, 40], n_rep=1500, horizon_mult=1.5)]}


if __name__ == "__main__":
    import sys

    if "--verify" in sys.argv:
        verify()
        sys.exit(0)

    c = 0.36
    write = "--write" in sys.argv
    plan = QUICK if "--quick" in sys.argv else PLAN

    out = {}
    for r, segments in plan.items():
        dV = DV_INSTANTON[r]
        print(f"\nr = {r:g}   DeltaV(r) = {dV:.6f}", flush=True)
        rows = []
        for seg in segments:
            for N in seg["Ns"]:
                q = run_point(N, r, c, dV, n_rep=seg["n_rep"],
                              horizon_mult=seg["horizon_mult"])
                if q is None:
                    print(f"  N={N}: no escapes, raise horizon_mult",
                          flush=True)
                    continue
                rows.append(q)
                print(f"  N={N:4d}  E[tau]={q['Etau']:11.2f}  "
                      f"exits={q['n_exit']:5d}  frac={q['frac']:.2f}  "
                      f"{q['secs']:8.1f}s", flush=True)
                out[str(r)] = sorted(rows, key=lambda d: d["N"])
                if write:                       # checkpoint every point
                    with open(data_path(RESULTS), "w") as f:
                        json.dump(out, f, indent=1)

    if write:
        print(f"\nwrote {data_path(RESULTS)}")
    report(out, DV_INSTANTON, DV_INF)
