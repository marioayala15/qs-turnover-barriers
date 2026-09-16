"""Basin geometry: saddle manifolds, Hypothesis H, R_off checks, threshold-line classification.

Mean field of the paper (parameters as in its parameter table):
    dx/dt = x [b(w) - d(x)],   dw/dt = r [alpha_C x - kappa w],
    b(w) = b0 + b1 w^h / (K^h + w^h),   d(x) = d0 + c + theta x.

Writes data/basins/results.json (all numbers) and the manifold traces
data/basins/*.npy (not tracked; about 20 s to regenerate).  Self-contained: it uses
its own implementation of the vector field (no import of ldp_action is needed here).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parents[2] / "data" / "basins"
HERE.mkdir(parents=True, exist_ok=True)

B0, B1, KH, HILL = 0.5, 2.0, 1.0, 4
D0, THETA, ALPHA, KAPPA = 0.8, 1.0, 2.0, 1.0
COSTS = [0.20, 0.30, 0.36, 0.45, 0.50]      # crossover_barriers.json c_grid
RATES = [0.5, 1.0, 2.0, 4.0, 8.0]           # rates of the exit-time campaign
A_OFF, B_OFF = 0.3, 0.7                     # R_off = [0,0.3] x [0,0.7]


def b(w):
    return B0 + B1 * w**HILL / (KH**HILL + w**HILL)


def db(w):
    return B1 * HILL * KH**HILL * w ** (HILL - 1) / (KH**HILL + w**HILL) ** 2


def d(x, c):
    return D0 + c + THETA * x


def G(x, c):                      # slaved growth rate b(wbar(x)) - d(x)
    return b(ALPHA * x / KAPPA) - d(x, c)


def field(t, u, r, c, sign=1.0):
    x, w = u
    return [sign * x * (b(w) - d(x, c)), sign * r * (ALPHA * x - KAPPA * w)]


def jac(x, w, r, c):
    return np.array([[b(w) - d(x, c) - THETA * x, x * db(w)],
                     [r * ALPHA, -r * KAPPA]])


def eta(x, c):
    """x-nullcline b(eta) = d(x); defined while b0 < d(x) < b0 + b1."""
    y = d(x, c)
    return ((y - B0) / (B0 + B1 - y)) ** (1.0 / HILL) * KH


# ------------------------------------------------------------ equilibria


def equilibria(c):
    """All equilibria in the closed quadrant.  dw/dt=0 forces w = alpha x/kappa, so
    equilibria are x=0 or zeros of G.  For x > b0+b1-d0-c, G(x) < 0 strictly, so the
    scan interval [0, X_MAX] is exhaustive."""
    x_max = B0 + B1 - D0 - c + 1e-9
    grid = np.linspace(0.0, x_max, 200001)
    g = G(grid, c)
    roots = []
    for i in np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]:
        roots.append(brentq(G, grid[i], grid[i + 1], args=(c,), xtol=1e-15, rtol=1e-15))
    # tangency guard: smallest |G| at local minima of |G| away from the found roots
    ag = np.abs(g)
    locmin = np.nonzero((ag[1:-1] < ag[:-2]) & (ag[1:-1] < ag[2:]))[0] + 1
    far = [i for i in locmin if all(abs(grid[i] - x0) > 1e-3 for x0 in roots)]
    min_abs_G_away = float(min(ag[far])) if far else None
    return roots, x_max, min_abs_G_away, float(g[0])


def classify_eq(x, w, r, c):
    ev = np.linalg.eigvals(jac(x, w, r, c))
    if np.any(np.abs(ev.imag) > 0):
        kind = "stable focus" if np.all(ev.real < 0) else "other (complex)"
        return kind, [complex(e).__repr__() for e in ev]
    ev = np.sort(ev.real)
    if np.all(ev < 0):
        kind = "stable node"
    elif ev[0] < 0 < ev[1]:
        kind = "saddle"
    else:
        kind = "other"
    return kind, ev.tolist()


def saddle_data(c, r):
    xs = [x for x in equilibria(c)[0]][0]
    ws = ALPHA * xs / KAPPA
    J = jac(xs, ws, r, c)
    lam, vec = np.linalg.eig(J)
    order = np.argsort(lam.real)
    lam_s, lam_u = lam.real[order]
    # eigenvectors in the closed form (x* b'(w*), lambda + theta x*)
    v_s = np.array([xs * db(ws), lam_s + THETA * xs]); v_s /= np.linalg.norm(v_s)
    v_u = np.array([xs * db(ws), lam_u + THETA * xs]); v_u /= np.linalg.norm(v_u)
    # cross-check against numpy eigenvectors
    for lam_i, v in ((lam_s, v_s), (lam_u, v_u)):
        assert np.linalg.norm(J @ v - lam_i * v) < 1e-10
    return dict(x_star=xs, w_star=ws, lam_s=lam_s, lam_u=lam_u, v_s=v_s, v_u=v_u,
                slope_s=v_s[1] / v_s[0], slope_u=v_u[1] / v_u[0],
                eta_prime=THETA / db(ws))


# -------------------------------------------------- forward classification

RHO_ON = 0.05   # "on" = enters the closed ball of radius RHO_ON about U_on


def make_classifier(c, r, rtol=1e-10, atol=1e-12, t_max=5000.0):
    roots = equilibria(c)[0]
    x_on = roots[1]
    w_on = ALPHA * x_on / KAPPA

    def ev_on(t, u, *a):
        return np.hypot(u[0] - x_on, u[1] - w_on) - RHO_ON
    ev_on.terminal = True

    def ev_off(t, u, *a):     # inside R_off  <=>  max(x - a, w - b) <= 0
        return max(u[0] - A_OFF, u[1] - B_OFF)
    ev_off.terminal = True

    def classify(u0):
        if ev_off(0, u0) <= 0:
            return "off", 0.0
        if ev_on(0, u0) <= 0:
            return "on", 0.0
        sol = solve_ivp(field, (0.0, t_max), u0, args=(r, c), method="DOP853",
                        rtol=rtol, atol=atol, events=(ev_on, ev_off))
        if sol.t_events[0].size:
            return "on", float(sol.t_events[0][0])
        if sol.t_events[1].size:
            return "off", float(sol.t_events[1][0])
        return "unresolved", float(sol.t[-1])
    return classify, (x_on, w_on)


def check_on_ball(c, r, n=64):
    """Confirm (numerically) that the ball used as the 'on' target lies in the on basin:
    points on its boundary circle converge to U_on to within 1e-8 by t=400 without
    leaving the ball of radius 2*RHO_ON."""
    x_on = equilibria(c)[0][1]
    w_on = ALPHA * x_on / KAPPA
    worst_final, worst_excursion = 0.0, 0.0
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        u0 = [x_on + RHO_ON * np.cos(th), w_on + RHO_ON * np.sin(th)]
        sol = solve_ivp(field, (0, 400), u0, args=(r, c), method="DOP853",
                        rtol=1e-10, atol=1e-12, dense_output=True)
        tt = np.linspace(0, 400, 4001)
        yy = sol.sol(tt)
        dist = np.hypot(yy[0] - x_on, yy[1] - w_on)
        worst_final = max(worst_final, dist[-1])
        worst_excursion = max(worst_excursion, dist.max())
    return dict(max_final_distance=float(worst_final), max_distance_along=float(worst_excursion))


# ------------------------------------------------------------ manifolds

BOX_X, BOX_W = 4.0, 30.0


def trace_stable(c, r, eps, rtol, atol, branch):
    """Backward integration from U* + branch*eps*v_s.  branch=+1: upper (w>w*)."""
    s = saddle_data(c, r)
    v = s["v_s"] * (1 if s["v_s"][1] > 0 else -1)          # v points to w increasing
    u0 = np.array([s["x_star"], s["w_star"]]) + branch * eps * v

    def hit_w0(t, u, *a): return u[1]
    hit_w0.terminal = True
    def hit_x0(t, u, *a): return u[0] - 1e-9
    hit_x0.terminal = True
    def box(t, u, *a): return min(BOX_X - u[0], BOX_W - u[1])
    box.terminal = True
    sol = solve_ivp(field, (0.0, 400.0), u0, args=(r, c, -1.0), method="DOP853",
                    rtol=rtol, atol=atol, events=(hit_w0, hit_x0, box),
                    dense_output=True, max_step=0.05)
    reason = ("w=0" if sol.t_events[0].size else "x->0" if sol.t_events[1].size
              else "box" if sol.t_events[2].size else "t_max")
    # sample the dense output finely: linear interpolation between raw solver steps
    # (a few dozen points) would otherwise dominate the comparison errors
    tt = np.linspace(0.0, sol.t[-1], 40001)
    return sol.sol(tt).T, reason


def trace_unstable(c, r, eps, branch, rtol=1e-11, atol=1e-13, t_max=400.0):
    s = saddle_data(c, r)
    v = s["v_u"]                                            # both components > 0
    u0 = np.array([s["x_star"], s["w_star"]]) + branch * eps * v
    classify, (x_on, w_on) = make_classifier(c, r)
    sol = solve_ivp(field, (0.0, t_max), u0, args=(r, c), method="DOP853",
                    rtol=rtol, atol=atol, dense_output=True, max_step=0.05)
    y = sol.y.T
    info = dict(final=y[-1].tolist(), label=classify(u0)[0])
    if branch < 0:
        xs = s["x_star"]
        # first entry into R_off; is the orbit inside F_off = {0<x<x*, 0<w<eta(x)}?
        inside = [(yy[0] < xs and 0 < yy[1] < eta(yy[0], c)) for yy in y[1:]]
        info["all_points_after_start_in_F_off"] = bool(all(inside))
        k = np.nonzero((y[:, 0] <= A_OFF) & (y[:, 1] <= B_OFF))[0]
        info["t_enter_R_off"] = float(sol.t[k[0]]) if k.size else None
    return y, info


def curve_discrepancy(ref, other, branch):
    """Max |difference| between two traced branches, as w(x) (lower) or x(w) (upper),
    on the overlap of their ranges (monotonicity checked)."""
    if branch < 0:
        a, bq = 0, 1
    else:
        a, bq = 1, 0
    def mono(y):
        dy = np.diff(y[:, a])
        return bool(np.all(dy > 0) or np.all(dy < 0))
    R = ref[np.argsort(ref[:, a])]
    O = other[np.argsort(other[:, a])]
    lo = max(R[0, a], O[0, a]); hi = min(R[-1, a], O[-1, a])
    grid = np.linspace(lo, hi, 2000)[1:-1]
    diff = np.interp(grid, R[:, a], R[:, bq]) - np.interp(grid, O[:, a], O[:, bq])
    return float(np.max(np.abs(diff))), mono(ref) and mono(other)


def bisect_line(classify, p_on, p_off, tol=1e-9):
    """Bisection between a point classified 'on' and one classified 'off'."""
    lo, hi = np.array(p_off, float), np.array(p_on, float)
    assert classify(lo)[0] == "off" and classify(hi)[0] == "on", (classify(lo), classify(hi))
    n_unres = 0
    while np.linalg.norm(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        lab = classify(mid)[0]
        if lab == "on":
            hi = mid
        elif lab == "off":
            lo = mid
        else:
            n_unres += 1
            break
    return 0.5 * (lo + hi), float(np.linalg.norm(hi - lo)), n_unres


def main():
    t0 = time.time()
    out = dict(parameters=dict(b0=B0, b1=B1, K_h=KH, h=HILL, d0=D0, theta=THETA,
                               alpha_C=ALPHA, kappa=KAPPA),
               costs=COSTS, rates=RATES, R_off=[A_OFF, B_OFF], rho_on=RHO_ON)

    # ---------------- Task 2: Hypothesis H at all five costs
    H = {}
    for c in COSTS:
        roots, x_max, min_abs, g0 = equilibria(c)
        rows = {"U_off": (0.0, 0.0)}
        names = ["U_star", "U_on"]
        ok_count = len(roots) == 2
        for nm, x0 in zip(names, roots):
            rows[nm] = (x0, ALPHA * x0 / KAPPA)
        types = {}
        for rr in [0.5, 1, 2, 4, 8, 16, 64]:
            types[str(rr)] = {nm: classify_eq(*rows[nm], rr, c) for nm in rows}
        H[str(c)] = dict(
            H1_b0_lt_d0_plus_c=bool(B0 < D0 + c), d0_plus_c=D0 + c,
            positive_zeros_of_G=roots, n_equilibria_closed_quadrant=1 + len(roots),
            scan_interval=[0.0, x_max], G_at_0=g0,
            min_abs_G_at_other_local_minima=min_abs,
            equilibria={k: list(v) for k, v in rows.items()},
            eigenvalues_by_r={rr: {nm: t[1] for nm, t in tv.items()} for rr, tv in types.items()},
            types_by_r={rr: {nm: t[0] for nm, t in tv.items()} for rr, tv in types.items()},
            H_ok=bool(ok_count and B0 < D0 + c and 0 < roots[0] < roots[1] and all(
                tv["U_off"][0] in ("stable node", "stable focus")
                and tv["U_on"][0] in ("stable node", "stable focus")
                and tv["U_star"][0] == "saddle" for tv in types.values())),
        )
    out["hypothesis_H"] = H

    # ---------------- Task 3: rectangle + Lemma B numerical remark
    rect = {}
    for c in COSTS:
        xs = H[str(c)]["positive_zeros_of_G"][0]
        rect[str(c)] = dict(b_of_b_off=b(B_OFF), d0_plus_c=D0 + c,
                            cond_b=bool(b(B_OFF) < D0 + c),
                            alpha_a=ALPHA * A_OFF, kappa_b=KAPPA * B_OFF,
                            cond_ab=bool(ALPHA * A_OFF < KAPPA * B_OFF),
                            x_star=xs, cond_a=bool(0 < A_OFF < xs),
                            margin_gamma=D0 + c - b(B_OFF))
    out["rectangle"] = rect
    lemB = {}
    for c in COSTS:
        for rr in RATES:
            s = saddle_data(c, rr)
            lemB[f"c={c},r={rr}"] = dict(slope_u=s["slope_u"], eta_prime=s["eta_prime"],
                                         slope_s=s["slope_s"], lam_u=s["lam_u"], lam_s=s["lam_s"],
                                         slope_u_gt_eta_prime=bool(s["slope_u"] > s["eta_prime"]))
    out["lemma_B_remark"] = lemB

    # ---------------- Task 1 and 4 at c = 0.36
    c = 0.36
    per_r = {}
    for r in RATES:
        s = saddle_data(c, r)
        rec = dict(x_star=s["x_star"], w_star=s["w_star"], lam_s=s["lam_s"], lam_u=s["lam_u"],
                   v_s=s["v_s"].tolist(), v_u=s["v_u"].tolist(),
                   slope_s=s["slope_s"], slope_u=s["slope_u"], eta_prime=s["eta_prime"])
        rec["on_ball_check"] = check_on_ball(c, r)
        classify, _ = make_classifier(c, r)

        # stable manifold: reference + variations
        ref_eps, ref_tol = 1e-6, 1e-12
        branches = {}
        for br, nm in ((+1, "upper"), (-1, "lower")):
            ref, reason = trace_stable(c, r, ref_eps, ref_tol, ref_tol * 1e-2, br)
            variants = {}
            for eps in (1e-4, 1e-5, 1e-6, 1e-7):
                for tol in (1e-8, 1e-10, 1e-12):
                    if eps == ref_eps and tol == ref_tol:
                        continue
                    y, rs = trace_stable(c, r, eps, tol, tol * 1e-2, br)
                    dmax, mono = curve_discrepancy(ref, y, br)
                    variants[f"eps={eps:g},rtol={tol:g}"] = dict(max_diff=dmax, monotone=mono, stop=rs)
            info = dict(stop_reason=reason, n_points=len(ref),
                        end_point=ref[-1].tolist(),
                        x_range=[float(ref[:, 0].min()), float(ref[:, 0].max())],
                        w_range=[float(ref[:, 1].min()), float(ref[:, 1].max())],
                        max_diff_over_variants=max(v["max_diff"] for v in variants.values()),
                        all_monotone=all(v["monotone"] for v in variants.values()),
                        variants=variants)
            if nm == "upper":
                # does the upper branch ever return to x >= x* ?
                info["max_x_after_start"] = float(ref[5:, 0].max())
                info["min_x"] = float(ref[:, 0].min())
                info["crosses_threshold_line_above_saddle"] = bool(np.any(ref[5:, 0] >= s["x_star"]))
            else:
                info["min_w"] = float(ref[:, 1].min())
                info["w0_intercept_x"] = float(ref[-1, 0]) if reason == "w=0" else None
            branches[nm] = info
            np.save(HERE / f"stable_{nm}_r{r:g}.npy", ref)
        rec["stable_manifold"] = branches

        # bisection cross-checks
        ups = np.load(HERE / f"stable_upper_r{r:g}.npy")
        los = np.load(HERE / f"stable_lower_r{r:g}.npy")
        bis = []
        # lower branch: vertical lines x = const, bisect in w (off below, on above)
        for xl in (0.6, 0.65, 0.7, 0.9, 1.1, 1.30, 1.6, 2.0):
            if not (los[:, 0].min() < xl < los[:, 0].max()):
                lab = classify([xl, 1e-6])[0]
                bis.append(dict(line=f"x={xl}", note=f"line not crossed by lower branch; (x,1e-6) classified {lab}"))
                continue
            wman = float(np.interp(xl, los[np.argsort(los[:, 0]), 0], los[np.argsort(los[:, 0]), 1]))
            p, width, nu = bisect_line(classify, [xl, wman + 0.2], [xl, max(wman - 0.2, 1e-9) if wman > 0.2 else 1e-9])
            bis.append(dict(line=f"x={xl}", w_bisect=float(p[1]), w_manifold=wman,
                            diff=float(p[1] - wman), bracket=width, unresolved_hits=nu))
        # upper branch: horizontal lines w = const, bisect in x (off left, on right)
        for wl in (1.5, 2.0, 3.0, 5.0, 10.0, 20.0):
            if not (ups[:, 1].min() < wl < ups[:, 1].max()):
                bis.append(dict(line=f"w={wl}", note="outside traced range"))
                continue
            o = np.argsort(ups[:, 1])
            xman = float(np.interp(wl, ups[o, 1], ups[o, 0]))
            lo = max(xman * 0.5, 1e-12)
            hi = min(xman * 1.5 + 1e-3, s["x_star"])
            p, width, nu = bisect_line(classify, [hi, wl], [lo, wl])
            bis.append(dict(line=f"w={wl}", x_bisect=float(p[0]), x_manifold=xman,
                            diff=float(p[0] - xman), rel_diff=float((p[0] - xman) / xman),
                            bracket=width, unresolved_hits=nu))
        rec["bisection"] = bis

        # unstable branches
        yu_on, info_on = trace_unstable(c, r, 1e-7, +1)
        yu_off, info_off = trace_unstable(c, r, 1e-7, -1)
        np.save(HERE / f"unstable_on_r{r:g}.npy", yu_on)
        np.save(HERE / f"unstable_off_r{r:g}.npy", yu_off)
        rec["unstable_manifold"] = dict(towards_on=info_on, towards_off=info_off)

        # Task 4: threshold line x = x*, w > w*
        ws = s["w_star"]
        offsets = np.concatenate([np.logspace(-10, -1, 46), np.linspace(0.1, 30.0, 300)])
        labels, times = [], []
        for delta in offsets:
            lab, tt = classify([s["x_star"], ws + delta])
            labels.append(lab); times.append(tt)
        # tolerance variation on a subset
        classify_loose, _ = make_classifier(c, r, rtol=1e-7, atol=1e-9)
        loose = [classify_loose([s["x_star"], ws + dlt])[0] for dlt in offsets[::5]]
        rec["threshold_line"] = dict(
            w_minus_wstar=offsets.tolist(), labels=labels, entry_times=times,
            counts={k: labels.count(k) for k in ("on", "off", "unresolved")},
            loose_tolerance_labels_subset=dict(counts={k: loose.count(k) for k in ("on", "off", "unresolved")},
                                               agree=bool(loose == labels[::5])),
            max_entry_time=max(times))
        # below the saddle, a few sanity points (Lemma B(2) says off)
        below = [classify([s["x_star"], ws - dl])[0] for dl in (1e-8, 1e-4, 0.01, 0.5, 1.0, ws - 1e-6)]
        rec["threshold_line_below_sanity"] = below
        # manuscript caption check: x = 1.30 thresholds
        per_r[str(r)] = rec
        print(f"r={r} done  t={time.time()-t0:.1f}s", flush=True)
    out["c036"] = per_r
    out["wall_seconds"] = time.time() - t0
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote results.json")


if __name__ == "__main__":
    main()
