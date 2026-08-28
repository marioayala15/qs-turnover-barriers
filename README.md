# Turnover-dependent collapse barriers in a quorum-sensing reaction network

Code and data for

> M. Ayala and J. Zimmer, *Computing Turnover-Dependent Rare-Event Barriers in a
> Quorum-Sensing Reaction Network*.

A four-channel cell-signal reaction network couples a cell density `x` to a signal
density `w` through cell division, cell death, signal production and signal removal.
Multiplying signal production and removal by a common turnover factor `r` leaves every
deterministic equilibrium and its stability type unchanged, hence any effect of `r` on
the rare-event barrier is invisible in the mean-field ODE. The barrier is nevertheless
`r`-dependent, and this repository computes it.

The quasipotential barrier `DeltaV(r)` against collapse from the cooperative state to
the unstable threshold is computed twice, by two routes that share no machinery:

1. minimization of the Freidlin-Wentzell action, with the signal kept as a fluctuating
   coordinate rather than slaved to `wbar(x) = aC x / kappa`;
2. exact stochastic simulation, with right-censored collapse times regressed in the
   population scale `N`, which uses no action solver at all.

At the working cost `c = 0.36` the two routes agree within two standard errors at every
common rate, and both exclude the eliminated-signal value.

## The numbers

Eliminating the signal first gives a barrier available in closed form as a
one-dimensional quadrature:

    DeltaV_inf = 0.0613795 .

Keeping the signal explicit gives, at the production settings below:

    r      0.5        1         2         4         8         16        32        64
    dV     0.122477   0.095259  0.079057  0.070313  0.065843  0.063605  0.062490  0.061935

So `DeltaV(1)` exceeds `DeltaV_inf` by 55 percent, `DeltaV(r)` falls by a factor near
1.98 across the rate grid, and `r (DeltaV(r) - DeltaV_inf)` settles at `0.03553`,
against the predicted coefficient `A = 0.03544`. Since the mean collapse time grows as
`exp(N DeltaV(r))`, these differences are amplified exponentially in the population
scale.

## Layout

    src/ldp_action.py            reaction network, Lagrangian by Newton, minimum action
    src/ldp_crossover_sweep.py   the barrier sweep over (c, r); writes crossover_barriers.json
    src/ldp_crossover_exit.py    SSA with censored exit times; writes crossover_exit_times.json
    src/paths.py                 where stored measurements go
    src/generate_optimal_paths.py   representative collapse paths for the landscape figure
    src/verify_hessian.py        the action solver against the Gaussian regime
    src/verify_exit_multiseed.py five-seed diagnostics for the exit-time estimator
    src/fig_proceedings_numerics.py  the four figures of the paper
    data/                        stored output of the runs that are too slow to repeat

Everything in `data/` is regenerable by the scripts above. It is tracked because the
stochastic sweeps take hours and the figures should not require them.

## Requirements

Python 3.9 or later, with `numpy`, `scipy` and `matplotlib`. The results quoted here
were produced with Python 3.9.6, numpy 1.26.3, scipy 1.12.0 and matplotlib 3.8.2. Run
the scripts from the repository root:

    python src/fig_proceedings_numerics.py

## Reproducing the paper

The figures read only the stored data and take a few seconds:

    python src/fig_proceedings_numerics.py

This writes `fig_nullclines.pdf`, `fig_landscape.pdf`, `fig_action_results.pdf` and
`fig_exit_scaling.pdf` to the repository root.

Three checks are cheap enough to run directly. The first calibrates the action solver
against the closed-form quadrature in the eliminated-signal model, where the answer is
known:

    python src/ldp_action.py                    # agrees to 0.07 percent at c = 0.36

The second checks the solver against the Gaussian regime near the stable state, where
the quasipotential Hessian must satisfy `H J + J' H + H D H = 0` and is therefore the
inverse of the stationary covariance of the central limit regime. This shares nothing
with the first check:

    python src/verify_hessian.py                # worst relative error 2.5e-04

The third checks the horizon and the mesh separately, i.e. horizon convergence at fixed
step and mesh convergence at fixed horizon:

    python src/ldp_crossover_sweep.py --check

The production sweep itself is `T = 40`, `M = 2000` and an outer budget of 60000
iterations, over `r` in `{0.5, 1, 2, 4, 8, 16, 32, 64}` and `c` in
`{0.20, 0.30, 0.36, 0.45, 0.50}`:

    python src/ldp_crossover_sweep.py --write   # 40 minimizations, about four minutes

Notice that at the default 600 iterations L-BFGS stops on the cap and returns
`0.128412` at `r = 0.5` instead of `0.122477`, i.e. the budget itself moves the answer
upward by five percent, which looks exactly like a discretisation error and is not one.

The exit-time route is the expensive one. Each pair `(r, N)` gets its own random stream,
seeded reproducibly from the triple `(20260812, 100r, N)`:

    python src/ldp_crossover_exit.py --quick    # a reduced grid, for a sanity check
    python src/ldp_crossover_exit.py --write    # the full sweep, hours
    python src/ldp_crossover_exit.py --verify   # estimator diagnostics at one point

    python src/verify_exit_multiseed.py         # five seeds at r = 0.5, N = 60

## What this does not settle

The barrier is computed for one network at one working cost, and the exit-time route
covers `r` in `{0.5, 1, 2, 4, 8}` only, since the horizon needed at larger `r` grows
with the barrier. The prefactor `C(N)` in `E[tau_N] = C(N) exp(N DeltaV(r))` is not
computed: it is absorbed into a regression on `N` and `log N`, with the uncertainty
scaled to unit reduced chi-square, and `src/ldp_crossover_exit.py --verify` is what
licenses that scaling. The minimizer is local, and no claim is made that it is globally
reliable in higher dimensions.
