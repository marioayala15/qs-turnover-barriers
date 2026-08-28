"""Multi-seed diagnostics for the proceedings-paper exit-time estimator.

The expensive barrier sweep is read from ``data/crossover_exit_times.json``.
This script reruns only a moderate validation point, ``r=0.5, N=60``, where
the process is already metastable but several independent ensembles remain
affordable.  It reports ranges rather than a single noisy diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ldp_action import PAR, net_wellmixed_explicit
from ldp_crossover_exit import censored_mle, first_passage, fixed_points


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    r = 0.5
    N = 60
    c = 0.36
    dV = 0.12247715856976549
    x_star, x_on = fixed_points(c)
    network = net_wellmixed_explicit(c, rN=r)
    initial = [x_on, PAR["aC"] * x_on / PAR["kappa"]]

    # About seventeen estimated means: enough that all 1500 replicas escape
    # with overwhelming probability, while remaining much cheaper than N=80.
    full_horizon = 25.0 * 0.19 * np.exp(N * dV)
    horizon_multipliers = (0.15, 0.30, 0.60, 1.00, 1.50, 3.00)

    rows = []
    for offset in range(args.seeds):
        seed = 6100 + offset
        rng = np.random.default_rng([20260826, seed])
        tau, steps = first_passage(
            network, N, initial, 0, x_star, args.replicas, full_horizon, rng
        )
        finite = tau[np.isfinite(tau)]
        if len(finite) != args.replicas:
            raise RuntimeError(
                f"Only {len(finite)}/{args.replicas} trajectories escaped; "
                "increase full_horizon."
            )
        mean = float(finite.mean())
        row = {
            "seed": seed,
            "mean": mean,
            "log_mean": float(np.log(mean)),
            "cv": float(finite.std(ddof=1) / mean),
            "median_over_mean": float(np.median(finite) / mean),
            "steps": int(steps),
            "censoring": [],
        }
        for multiplier in horizon_multipliers:
            horizon = multiplier * np.exp(N * dV)
            estimate, exits = censored_mle(tau, horizon)
            row["censoring"].append(
                {
                    "multiplier": multiplier,
                    "exit_fraction": exits / args.replicas,
                    "estimate_over_full_mean": float(estimate / mean),
                }
            )
        rows.append(row)
        print(
            f"seed {seed}: mean={mean:.2f}, cv={row['cv']:.3f}, "
            f"median/mean={row['median_over_mean']:.3f}",
            flush=True,
        )

    def span(key: str) -> list[float]:
        values = [row[key] for row in rows]
        return [float(min(values)), float(max(values))]

    censoring_ranges = []
    for index, multiplier in enumerate(horizon_multipliers):
        fractions = [row["censoring"][index]["exit_fraction"] for row in rows]
        ratios = [row["censoring"][index]["estimate_over_full_mean"] for row in rows]
        censoring_ranges.append(
            {
                "multiplier": multiplier,
                "exit_fraction_range": [float(min(fractions)), float(max(fractions))],
                "estimate_ratio_range": [float(min(ratios)), float(max(ratios))],
            }
        )

    output = {
        "settings": {
            "r": r,
            "N": N,
            "c": c,
            "dV_for_horizon": dV,
            "replicas_per_seed": args.replicas,
            "number_of_seeds": args.seeds,
            "full_horizon": float(full_horizon),
        },
        "summary": {
            "between_seed_sd_log_mean": float(
                np.std([row["log_mean"] for row in rows], ddof=1)
            ),
            "nominal_sd_log_mean": float(1.0 / np.sqrt(args.replicas)),
            "cv_range": span("cv"),
            "median_over_mean_range": span("median_over_mean"),
            "censoring_ranges": censoring_ranges,
        },
        "rows": rows,
    }

    destination = ROOT / "data" / "exit_verification_multiseed.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
