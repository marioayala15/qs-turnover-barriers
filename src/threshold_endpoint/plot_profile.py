"""Profile plot for the threshold barrier -> fig_threshold_profile.pdf.

(a) r = 0.5: w -> min over seeds of the constrained fixed-endpoint action
    (T=40, dt=0.04), saddle location and value marked.
(b) all r: the same profile minus the saddle action at identical settings.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "threshold_endpoint")
os.makedirs(HERE, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # r ascending
ACCENT = "#eb6834"


def profile(rows, r):
    by_w = {}
    for q in rows:
        if q["r"] == r and q["constrained"] and q["success"]:
            by_w[q["w"]] = min(by_w.get(q["w"], np.inf), q["S"])
    ws = np.array(sorted(by_w))
    return ws, np.array([by_w[w] for w in ws])


def main():
    prof = json.load(open(os.path.join(HERE, "results_profile.json")))
    st = prof["settings"]
    wst = st["w_star"]
    rows = prof["fixed"]
    refine_path = os.path.join(HERE, "results_refine.json")
    refine = (json.load(open(refine_path))["rows"]
              if os.path.exists(refine_path) else [])
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    r0 = 0.5
    S0 = prof["saddle"][str(r0)]["S"]
    ws, S = profile(rows, r0)
    a.grid(True, color=GRID, lw=0.6)
    a.plot(ws, S, color=RAMP[2], lw=2, marker="o", ms=4, label="threshold point $(x^*,w)$")
    a.axhline(S0, color=INK2, lw=1, ls="--")
    a.axvline(wst, color=INK2, lw=0.8, ls=":")
    a.plot([wst], [S0], marker="*", ms=12, color=ACCENT, mec="white", mew=1,
           zorder=5, label=r"saddle $U_*$")
    a.text(3.15, S0, r" $\Delta V_*$", va="bottom", ha="right", color=INK2)
    for q in refine:
        if q["r"] == r0:
            h = [t for t in q["threshold"]["mesh_T40"] if t["dt"] == 0.04][0]
            a.plot([q["w_best"]], [h["S"]], marker="D", ms=7, mfc="none",
                   mec=INK, mew=1.2, zorder=6, label="refined minimum")
    a.set_ylim(0.9 * min(S.min(), S0), 1.6 * S0)
    a.set_xlim(0.8, 3.2)
    a.set_xlabel("terminal signal $w$ (on $x=x^*$)")
    a.set_ylabel("action, $T=40$, $\\Delta t=0.04$")
    a.set_title(f"(a) $r={r0:g}$", loc="left", color=INK)
    a.legend(frameon=False, loc="upper left")

    b.grid(True, color=GRID, lw=0.6)
    b.axhline(0, color=INK2, lw=1, ls="--")
    b.axvline(wst, color=INK2, lw=0.8, ls=":")
    lo = 0.0
    for col, r in zip(RAMP, st["r_grid"]):
        ws, S = profile(rows, r)
        dS = S - prof["saddle"][str(r)]["S"]
        lo = min(lo, dS.min())
        b.plot(ws, dS, color=col, lw=2, marker="o", ms=3, label=f"$r={r:g}$")
        m = int(np.argmin(dS))
        b.plot([ws[m]], [dS[m]], marker="o", ms=8, mfc="none", mec=col, mew=1.5)
    b.plot([wst], [0], marker="*", ms=12, color=ACCENT, mec="white", mew=1,
           zorder=5)
    b.set_xlim(0.8, 3.2)
    b.set_ylim(1.3 * lo, -1.3 * lo if lo < 0 else 0.01)
    b.set_xlabel("terminal signal $w$ (on $x=x^*$)")
    b.set_ylabel("action minus saddle action")
    b.set_title("(b) all $r$ (open circles: grid minima)", loc="left", color=INK)
    b.legend(frameon=False, ncol=1, loc="upper right", fontsize=8)
    out = os.path.join(ROOT, "fig_threshold_profile.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
