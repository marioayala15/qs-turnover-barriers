"""Phase portrait of the paper (fig_nullclines.pdf): nullclines, separatrices at
r = 0.5 and 8, the threshold line, the extinction target R_off, minimum-action and
relaxation paths.

Needs the .npy manifolds written by src/basins/separatrix.py and
data/basins/action_path_r8.json.  Writes fig_nullclines.pdf (and a version without
relaxation paths, fig_nullclines_norelax.pdf) to the repository root.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "data" / "basins"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

import separatrix as S  # noqa: E402

BLUE, VERMILLION = "#0072B2", "#D55E00"
matplotlib.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8,
                            "ytick.labelsize": 8, "legend.fontsize": 6.8, "lines.linewidth": 1.25})
COST = 0.36
XMAX, WMAX = 1.45, 3.15


def main(with_relaxation=True, suffix=""):
    fig, ax = plt.subplots(figsize=(3.4, 3.35))
    xs = S.saddle_data(COST, 1.0)["x_star"]; ws = S.ALPHA * xs / S.KAPPA
    x_on = S.equilibria(COST)[0][1]; w_on = S.ALPHA * x_on / S.KAPPA

    # off rectangle
    ax.add_patch(Rectangle((0, 0), S.A_OFF, S.B_OFF, facecolor="0.88", edgecolor="0.45",
                           linewidth=0.8, zorder=0))
    ax.text(0.5 * S.A_OFF, 0.5 * S.B_OFF, r"$R_{\mathrm{off}}$", ha="center", va="center",
            fontsize=7.5, color="0.25", zorder=1)

    # nullclines (as in the existing figure)
    xg = np.linspace(0.0, XMAX, 400)
    ax.plot(xg, S.ALPHA * xg / S.KAPPA, "--", color="0.55", linewidth=1.0, zorder=1,
            label=r"$\dot w=0$")
    tgt = S.d(xg, COST)
    ok = (tgt > S.B0) & (tgt < S.B0 + S.B1)
    ax.plot(xg[ok], ((tgt[ok] - S.B0) / (S.B0 + S.B1 - tgt[ok])) ** 0.25, color="0.55",
            linewidth=1.0, zorder=1, label=r"$\dot x=0$")
    ax.axvline(0.0, color="0.55", linewidth=1.0, zorder=1)

    # threshold line
    ax.axvline(xs, color="0.3", linestyle=(0, (1, 1.5)), linewidth=0.9, zorder=1)
    ax.text(xs + 0.015, WMAX - 0.08, r"$x=x^\ast$", fontsize=7, va="top", color="0.25")

    for rate, tone in ((0.5, VERMILLION), (8.0, BLUE)):
        up = np.load(HERE / f"stable_upper_r{rate:g}.npy")
        lo = np.load(HERE / f"stable_lower_r{rate:g}.npy")
        sep = np.vstack([up[::-1], lo])
        ax.plot(sep[:, 0], sep[:, 1], color=tone, linewidth=1.6, zorder=3,
                label=fr"separatrix, $r={rate:g}$")

    # minimum-action paths U_on -> U*
    stored = json.loads((ROOT / "data" / "optimal_paths.json").read_text())
    p05 = np.asarray(next(r["path"] for r in stored["rows"] if r["r"] == 0.5))
    p8 = np.asarray(json.loads((HERE / "action_path_r8.json").read_text())["path"])
    for path, tone, rate in ((p05, VERMILLION, 0.5), (p8, BLUE, 8.0)):
        ax.plot(path[:, 0], path[:, 1], color=tone, linestyle=(0, (4, 1.6)), linewidth=1.0,
                zorder=2, label=fr"min.-action path, $r={rate:g}$")

    if with_relaxation:
        start = (0.72, 2.85)
        for rate, tone in ((0.5, VERMILLION), (8.0, BLUE)):
            sol = solve_ivp(S.field, (0.0, 260.0), start, args=(rate, COST),
                            rtol=1e-9, atol=1e-11, dense_output=True)
            path = sol.sol(np.linspace(0, 260, 20001))
            ax.plot(path[0], path[1], color=tone, linewidth=0.7, alpha=0.75, zorder=2)
            step = np.hypot(np.diff(path[0]), np.diff(path[1]))
            arc = np.concatenate([[0.0], np.cumsum(step)])
            j = int(np.clip(np.searchsorted(arc, 0.33 * arc[-1]), 1, len(arc) - 2))
            ax.annotate("", xy=(path[0][j + 1], path[1][j + 1]), xytext=(path[0][j], path[1][j]),
                        zorder=2, arrowprops={"arrowstyle": "-|>", "color": tone, "lw": 0.7,
                                              "mutation_scale": 7})
        ax.plot(*start, "s", color="0.25", markersize=3.0, zorder=4)

    for xe, we, stable in ((0.0, 0.0, True), (xs, ws, False), (x_on, w_on, True)):
        ax.plot(xe, we, "o", markersize=5.0, zorder=5, color="0.1",
                markerfacecolor="0.1" if stable else "white")
    ax.annotate(r"$U_\ast$", (xs, ws), textcoords="offset points", xytext=(10, -12), fontsize=7.5)
    ax.annotate(r"$U_{\mathrm{on}}$", (x_on, w_on), textcoords="offset points", xytext=(-4, 5),
                fontsize=7.5)
    ax.annotate(r"$U_{\mathrm{off}}$", (0, 0), textcoords="offset points", xytext=(4, -9),
                fontsize=7.0)

    ax.set_xlim(-0.06, XMAX)
    ax.set_ylim(-0.22, WMAX)
    ax.set_xlabel(r"cell density $x$")
    ax.set_ylabel(r"signal density $w$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False,
              handlelength=2.4, labelspacing=0.25, columnspacing=1.0)
    fig.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        if ext == "pdf":
            fig.savefig(ROOT / f"fig_nullclines{suffix}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", ROOT / f"fig_nullclines{suffix}.pdf")


if __name__ == "__main__":
    main(True, "")
    main(False, "_norelax")
