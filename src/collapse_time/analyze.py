"""Analysis of the collapse-time campaign.

For each r: event-specific censored exponential MLE of E[tau] for
  thr   = first entry into {X <= x* N}
  off   = first entry into the large rectangle [0,0.3]x[0,0.7]
  small = first entry into the small rectangle [0,0.15]x[0,0.35]
(and cell extinction counts), the paper's weighted regression
log E = a + b log N + N dV, a paired trajectory-level bootstrap for slope
differences, retry statistics, and threshold-entry states.

Writes data/collapse_time/fit_results.json and entry_states.npz, and fig_offtarget_full.pdf
at the repository root.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats as st

from common import (C_COST, EXP, ON_BOX, RAW, RECT_LARGE, RECT_SMALL, censored_mle,
                    fit_slope, fit_slope_linear, fixed_points, old_threshold_points,
                    rect_conditions, saddle_actions)

EVENTS = ("thr", "off", "small")
KEYS = {"thr": "tau_thr", "off": "tau_off", "small": "tau_small"}
B_BOOT = 2000


def load(r):
    pts = []
    for f in sorted(RAW.glob(f"r{r:g}_N*.npz")):
        meta = json.loads(f.with_suffix(".json").read_text())
        pts.append((meta, dict(np.load(f))))
    pts.sort(key=lambda p: p[0]["N"])
    return pts


def estimates(d, H, idx=None):
    out = {}
    for e in EVENTS:
        t = d[KEYS[e]] if idx is None else d[KEYS[e]][idx]
        out[e] = censored_mle(t, np.isfinite(t), H)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(cen - half), float(cen + half)]


def main():
    x_star, x_on = fixed_points()
    w_star = 2 * x_star
    dV = saddle_actions()
    old = old_threshold_points()
    rng = np.random.default_rng(20260916)
    results = dict(
        settings=dict(c=C_COST, x_star=x_star, w_star=w_star, x_on=x_on, w_on=2 * x_on,
                      rect_large=rect_conditions(*RECT_LARGE),
                      rect_small=rect_conditions(*RECT_SMALL),
                      on_box=dict(center=[x_on, 2 * x_on], halfwidths=ON_BOX),
                      estimator="right-censored exponential MLE, event-specific indicators",
                      regression="WLS log E = a + b log N + N dV, weights n_exit, "
                                 "se inflated by max(1,chi2_red)^0.5",
                      bootstrap=f"{B_BOOT} paired resamples of trajectories within each N"),
        rates={})
    entry_rows = []
    for r in (0.5, 1.0, 2.0, 4.0, 8.0):
        pts = load(r)
        if len(pts) < 4:
            print(f"r={r}: only {len(pts)} points; skipping fit")
            continue
        Ns = np.array([m["N"] for m, _ in pts])
        rows = []
        for meta, d in pts:
            N, H = meta["N"], meta["horizon"]
            n = len(d["tau_thr"])
            est = estimates(d, H)
            row = dict(N=int(N), n_rep=n, horizon=H, core_secs=meta["core_secs"],
                       total_events=meta["total_events"])
            for e in EVENTS:
                E, k = est[e]
                row[e] = dict(Etau=E, n_exit=k, censored_frac=1 - k / n,
                              se_log=float(1 / np.sqrt(k)))
            for e in EVENTS:
                # censoring expected if tau were exactly exponential with the fitted mean
                row[e]["expected_censored_if_exponential"] = float(n * np.exp(-H / row[e]["Etau"]))
                row[e]["naive_mean_observed_only"] = float(np.mean(d[KEYS[e]][np.isfinite(d[KEYS[e]])]))
            row["Eoff_over_Ethr"] = row["off"]["Etau"] / row["thr"]["Etau"]
            row["Eoff_over_exp_NdV"] = row["off"]["Etau"] / np.exp(N * dV[r])
            row["Ethr_over_exp_NdV"] = row["thr"]["Etau"] / np.exp(N * dV[r])
            # extinction
            ext = np.isfinite(d["tau_ext"])
            row["extinction_before_small_rect"] = int(ext.sum())
            row["extinction_before_large_rect"] = int(np.sum(d["tau_ext"] < d["tau_off"]))
            # comparison with the stored threshold campaign under its own horizon
            oldrow = next((q for q in old[f"{r:.1f}"] if q["N"] == N), None)
            if oldrow is not None:
                Eo, ko = censored_mle(d["tau_thr"], np.isfinite(d["tau_thr"]), oldrow["horizon"])
                se = np.sqrt(1 / ko + 1 / oldrow["n_exit"])
                row["thr_vs_stored"] = dict(stored_Etau=oldrow["Etau"], stored_n_exit=oldrow["n_exit"],
                                            new_Etau_at_stored_horizon=Eo, new_n_exit=ko,
                                            log_ratio=float(np.log(Eo / oldrow["Etau"])),
                                            z=float(np.log(Eo / oldrow["Etau"]) / se))
                Eoo, koo = censored_mle(d["tau_off"], np.isfinite(d["tau_off"]), oldrow["horizon"])
                row["off_at_stored_horizon"] = dict(Etau=Eoo, n_exit=koo,
                                                    ratio_to_full=Eoo / row["off"]["Etau"])
            # attempts / retries
            outc = d["att_outcome"]
            natt = len(outc)
            fail, succ, cens = int(np.sum(outc == 0)), int(np.sum(outc == 1)), int(np.sum(outc == 2))
            first_mask = np.r_[True, d["att_traj"][1:] != d["att_traj"][:-1]]
            first_fail = int(np.sum(outc[first_mask] == 0))
            row["attempts"] = dict(
                total=natt, returned_to_on=fail, reached_off=succ, censored=cens,
                retry_fraction=fail / natt, retry_fraction_ci=wilson(fail, natt),
                success_prob=succ / max(succ + fail, 1),
                success_prob_ci=wilson(succ, succ + fail),
                first_entry_returned_fraction=first_fail / int(first_mask.sum()),
                first_entry_returned_ci=wilson(first_fail, int(first_mask.sum())),
                mean_attempts_per_traj=float(np.mean(d["n_attempts"])),
                reached_large_without_attempt=int(np.sum(np.isfinite(d["tau_off"]) & (d["n_attempts"] == 0))),
            )
            ok = np.isfinite(d["tau_off"])
            row["final_descent_time_mean"] = float(np.mean(d["tau_off"][ok] - d["t_last_attempt"][ok]))
            fs = np.isfinite(d["tau_small"])
            row["small_minus_large_mean"] = float(np.mean(d["tau_small"][fs] - d["tau_off"][fs]))
            # threshold-entry signal
            wthr = d["W_thr"] / N
            fin = np.isfinite(d["tau_thr"])
            dw = wthr[fin] - w_star
            wa = d["att_W"] / N - w_star
            row["first_entry_signal"] = dict(
                quantiles_w_minus_wstar=dict(zip(["q05", "q25", "q50", "q75", "q95"],
                                                 np.quantile(dw, [.05, .25, .5, .75, .95]).tolist())),
                mean_w_minus_wstar=float(dw.mean()),
                frac_above_wstar=float(np.mean(dw > 0)),
                frac_below_wstar_first_entry=float(np.mean(dw < 0)),
                frac_below_wstar_first_entry_ci=wilson(int(np.sum(dw < 0)), int(dw.size)),
                frac_below_wstar_all_attempts=float(np.mean(wa < 0)),
                X_at_entry_values=sorted(set(d["X_thr"][fin].tolist())),
                success_prob_attempts_above_wstar=float(np.mean(outc[(wa > 0) & (outc < 2)] == 1))
                if np.any((wa > 0) & (outc < 2)) else None,
                success_prob_attempts_at_or_below_wstar=float(np.mean(outc[(wa <= 0) & (outc < 2)] == 1))
                if np.any((wa <= 0) & (outc < 2)) else None,
                n_attempts_above=int(np.sum((wa > 0) & (outc < 2))),
                n_attempts_at_or_below=int(np.sum((wa <= 0) & (outc < 2))),
            )
            row["lattice"] = dict(
                threshold_count=int(np.floor(x_star * N)), threshold_density=float(np.floor(x_star * N) / N),
                large_rect_counts=[int(np.floor(RECT_LARGE[0] * N + 1e-9)), int(np.floor(RECT_LARGE[1] * N + 1e-9))],
                small_rect_counts=[int(np.floor(RECT_SMALL[0] * N + 1e-9)), int(np.floor(RECT_SMALL[1] * N + 1e-9))],
                n_lattice_points_small_rect=int((np.floor(RECT_SMALL[0] * N + 1e-9) + 1) * (np.floor(RECT_SMALL[1] * N + 1e-9) + 1)),
            )
            rows.append(row)
            for j in range(natt):
                entry_rows.append((r, N, d["att_traj"][j], d["att_t"][j], d["att_X"][j] / N,
                                   d["att_W"][j] / N, outc[j], first_mask[j]))

        # direct paired look at the ratio E_off/E_thr: fits of log ratio on (1, N) and on (1, log N)
        def ratio_fits(Eo, Et):
            y = np.log(np.asarray(Eo) / np.asarray(Et))
            A1 = np.vstack([Ns, np.ones_like(Ns)]).T
            A2 = np.vstack([np.log(Ns), np.ones_like(Ns)]).T
            s1, res1, *_ = np.linalg.lstsq(A1, y, rcond=None)
            s2, res2, *_ = np.linalg.lstsq(A2, y, rcond=None)
            return y, s1, s2, float(np.sum((y - A1 @ s1) ** 2)), float(np.sum((y - A2 @ s2) ** 2))
        y0, s1, s2, rss1, rss2 = ratio_fits([q["off"]["Etau"] for q in rows], [q["thr"]["Etau"] for q in rows])
        # fits
        fits = {}
        for e in EVENTS:
            E = [q[e]["Etau"] for q in rows]
            k = [q[e]["n_exit"] for q in rows]
            f = fit_slope(Ns, E, k)
            f["ci95_regression"] = [f["slope"] - 1.96 * f["se"], f["slope"] + 1.96 * f["se"]]
            f["linear_model"] = fit_slope_linear(Ns, E, k)
            f["drop_smallest_N"] = fit_slope(Ns[1:], E[1:], k[1:])
            fits[e] = f
        # paired bootstrap
        boot = {e: np.empty(B_BOOT) for e in EVENTS}
        bootlin = {e: np.empty(B_BOOT) for e in EVENTS}
        rb1 = np.empty(B_BOOT); rb2 = np.empty(B_BOOT); yb = []
        for b in range(B_BOOT):
            Eb = {e: [] for e in EVENTS}
            kb = {e: [] for e in EVENTS}
            for meta, d in pts:
                idx = rng.integers(0, len(d["tau_thr"]), len(d["tau_thr"]))
                es = estimates(d, meta["horizon"], idx)
                for e in EVENTS:
                    Eb[e].append(es[e][0]); kb[e].append(es[e][1])
            _, t1, t2, _, _ = ratio_fits(Eb["off"], Eb["thr"])
            rb1[b], rb2[b] = t1[0], t2[0]
            for e in EVENTS:
                boot[e][b] = fit_slope(Ns, Eb[e], kb[e])["slope"]
                bootlin[e][b] = fit_slope_linear(Ns, Eb[e], kb[e])["slope"]
        pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        for e in EVENTS:
            fits[e]["bootstrap_se"] = float(np.std(boot[e], ddof=1))
            fits[e]["ci95_bootstrap"] = pct(boot[e])
        diffs = {}
        for a, b_ in (("off", "thr"), ("small", "off"), ("small", "thr")):
            dd = boot[a] - boot[b_]
            dl = bootlin[a] - bootlin[b_]
            diffs[f"{a}_minus_{b_}"] = dict(
                estimate=fits[a]["slope"] - fits[b_]["slope"],
                bootstrap_se=float(np.std(dd, ddof=1)), ci95_bootstrap=pct(dd),
                corr_bootstrap=float(np.corrcoef(boot[a], boot[b_])[0, 1]),
                linear_model_estimate=fits[a]["linear_model"]["slope"] - fits[b_]["linear_model"]["slope"],
                linear_model_ci95_bootstrap=pct(dl))
        ratio = dict(log_ratio_by_N=dict(zip(map(int, Ns), y0.tolist())),
                     linear_in_N=dict(slope=float(s1[0]), ci95_bootstrap=pct(rb1), rss=rss1),
                     linear_in_logN=dict(coef=float(s2[0]), ci95_bootstrap=pct(rb2), rss=rss2),
                     note="unweighted OLS; bootstrap = sampling error only; the two forms are "
                          "not distinguishable over this N range unless rss differ clearly")
        vs_saddle = {e: dict(diff=fits[e]["slope"] - dV[r], z_regression=(fits[e]["slope"] - dV[r]) / fits[e]["se"],
                             saddle_in_bootstrap_ci=bool(fits[e]["ci95_bootstrap"][0] <= dV[r] <= fits[e]["ci95_bootstrap"][1]),
                             saddle_in_regression_ci=bool(fits[e]["ci95_regression"][0] <= dV[r] <= fits[e]["ci95_regression"][1]))
                     for e in EVENTS}
        results["rates"][f"{r:g}"] = dict(saddle_dV_num=dV[r], Ns=Ns.tolist(), points=rows,
                                          fits=fits, slope_differences=diffs, off_thr_ratio=ratio, vs_saddle=vs_saddle,
                                          core_secs=float(sum(q["core_secs"] for q in rows)))
        print(f"r={r:g}: saddle {dV[r]:.4f}; ratio {json.dumps(ratio)}")
        for e in EVENTS:
            f = fits[e]
            print(f"   {e:5s} {f['slope']:.4f} +- {f['se']:.4f} (boot {f['ci95_bootstrap'][0]:.4f},"
                  f"{f['ci95_bootstrap'][1]:.4f}) chi2r {f['chi2_red']:.2f} lin {f['linear_model']['slope']:.4f}"
                  f" dropN {f['drop_smallest_N']['slope']:.4f}")
        for k_, v in diffs.items():
            print(f"   {k_}: {v['estimate']:.4f} CI {v['ci95_bootstrap']}")

    ent = np.array(entry_rows, float)
    np.savez_compressed(EXP / "entry_states.npz", r=ent[:, 0], N=ent[:, 1].astype(int),
                        traj=ent[:, 2].astype(int), t=ent[:, 3], x=ent[:, 4], w=ent[:, 5],
                        outcome=ent[:, 6].astype(np.int8), is_first_entry=ent[:, 7].astype(bool),
                        w_star=w_star, x_star=x_star,
                        README="one row per attempt (threshold entry after being in the on-box); "
                               "outcome 0=returned to on-box before large rectangle, 1=reached large "
                               "rectangle first, 2=censored; x,w in density units at entry")
    (EXP / "fit_results.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    plot(results)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    col = {"thr": "#2a78d6", "off": "#eb6834", "small": "#1baf7a"}
    mk = {"thr": "o", "off": "s", "small": "^"}
    lab = {"thr": r"threshold $\tau^{\rm thr}$", "off": r"off, $[0,.3]\times[0,.7]$",
           "small": r"off, $[0,.15]\times[0,.35]$"}
    rs = list(results["rates"].keys())
    fig, ax = plt.subplots(2, len(rs), figsize=(3.3 * len(rs), 5.6), squeeze=False)
    for j, rk in enumerate(rs):
        R = results["rates"][rk]
        N = np.array(R["Ns"], float)
        dV = R["saddle_dV_num"]
        Ng = np.linspace(N.min(), N.max(), 100)
        for e in EVENTS:
            E = np.array([q[e]["Etau"] for q in R["points"]])
            f = R["fits"][e]
            ax[0, j].plot(N, np.log(E), mk[e], color=col[e], ms=5, label=lab[e])
            ax[0, j].plot(Ng, f["a"] + f["b_logN"] * np.log(Ng) + f["slope"] * Ng, "-", color=col[e], lw=1.2)
            ax[1, j].plot(N, np.log(E) - N * dV, mk[e], color=col[e], ms=5,
                          label=f"{e}: fit {f['slope']:.4f}±{f['se']:.4f}")
        # saddle slope reference, anchored at the off-target mean point (anchor arbitrary)
        Eoff = np.log([q["off"]["Etau"] for q in R["points"]])
        anchor = np.mean(Eoff - N * dV)
        ax[0, j].plot(Ng, anchor + dV * Ng, "--", color="0.35", lw=1,
                      label=rf"slope $\Delta V_*={dV:.4f}$ (anchored)")
        ax[1, j].axhline(anchor, ls="--", color="0.35", lw=1)
        ax[0, j].set_title(f"r = {rk}, c = 0.36")
        ax[1, j].set_xlabel("N")
        ax[0, j].set_ylabel(r"$\log \widehat{E}[\tau]$")
        ax[1, j].set_ylabel(r"$\log \widehat{E}[\tau] - N\,\Delta V_*$")
        ax[0, j].legend(fontsize=6.5, frameon=False, loc="upper left")
        ax[1, j].legend(fontsize=6.5, frameon=False, loc="best")
        for a in ax[:, j]:
            a.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Exp. 3 full grid: threshold vs off-target mean times (error bars smaller than symbols; "
                 r"fits: $a+b\log N+N\Delta V$)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(EXP.parents[1] / "fig_offtarget_full.pdf")
    

if __name__ == "__main__":
    main()
