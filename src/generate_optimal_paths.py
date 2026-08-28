"""Minimum-action collapse paths at four turnover rates.

Writes ``data/optimal_paths.json``, from which the landscape figure of
Section 5 is drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

from ldp_action import mam_action, net_wellmixed_explicit
from ldp_crossover_sweep import pieces


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cost = 0.36
    horizon = 40.0
    intervals = 2000
    _, _, wbar, _, _, x_star, x_on = pieces(cost)
    initial = [x_on, wbar(x_on)]
    target = [x_star, wbar(x_star)]

    rows = []
    for rate in (0.5, 1.0, 4.0, 16.0):
        result = mam_action(
            net_wellmixed_explicit(cost, rN=rate),
            initial,
            target,
            T=horizon,
            K=intervals,
            maxiter=60000,
        )
        if not result["success"]:
            raise RuntimeError(f"path optimization failed at r={rate}")
        rows.append(
            {
                "r": rate,
                "action": result["S"],
                "path": result["path"].tolist(),
            }
        )
        print(f"r={rate:g}: action={result['S']:.8f}", flush=True)

    output = {
        "settings": {
            "c": cost,
            "T": horizon,
            "M": intervals,
            "x_star": x_star,
            "x_on": x_on,
            "alpha_C_over_kappa": 2.0,
        },
        "rows": rows,
    }
    destination = ROOT / "data" / "optimal_paths.json"
    destination.write_text(json.dumps(output) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
