"""Check the action solver against the Gaussian regime near the stable state.

Near an asymptotically stable equilibrium the quasipotential is quadratic,
V(U) = 1/2 (U-U_on)' H (U-U_on) + O(|U-U_on|^3), and its Hessian satisfies

    H J + J' H + H D H = 0 ,

which is solved by H = Sigma^{-1}, where Sigma is the stationary covariance of
the central limit regime, i.e. the solution of the Lyapunov equation

    J Sigma + Sigma J' + D = 0 .

So the minimum action over short excursions from U_on is predicted in closed
form by a 2x2 Lyapunov solve, with no path optimisation involved. This gives a
check on the minimiser that shares nothing with the eliminated-signal
calibration or with the exit-time experiment.

For each value of r we minimise the action from U_on to U_on + eps*z over a
set of unit directions z, at two values of eps, Richardson-extrapolate the
resulting quadratic forms to eps -> 0, and recover H by least squares. The
recovered matrix is compared entrywise with Sigma^{-1}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import fsolve

from ldp_action import mam_action, net_wellmixed_explicit
from ldp_crossover_sweep import pieces

ROOT = Path(__file__).resolve().parent.parent

COST = 0.36
RATES = (0.5, 1.0, 2.0, 4.0, 8.0)
ANGLES = np.deg2rad([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
EPSILONS = (0.02, 0.01)
HORIZON = 10.0
INTERVALS = 1000


def gaussian_hessian(net, equilibrium):
    """H = Sigma^{-1} from the Lyapunov equation, plus the pieces that built it."""
    jacobian = np.einsum("rn,rm->nm", net.D, net.jac(equilibrium))
    diffusion = net.hessp_H(equilibrium, np.zeros(net.n))  # sum_rho Delta Delta' beta
    covariance = solve_continuous_lyapunov(jacobian, -diffusion)
    hessian = np.linalg.inv(covariance)
    residual = np.abs(
        hessian @ jacobian + jacobian.T @ hessian + hessian @ diffusion @ hessian
    ).max()
    return jacobian, diffusion, covariance, hessian, float(residual)


def main() -> None:
    _, _, wbar, _, _, _, x_on = pieces(COST)
    directions = np.column_stack([np.cos(ANGLES), np.sin(ANGLES)])

    rows = []
    for rate in RATES:
        net = net_wellmixed_explicit(COST, rN=rate)
        equilibrium = np.asarray(
            fsolve(lambda U: net.F(np.asarray(U)), [x_on, wbar(x_on)]), float
        )
        jacobian, diffusion, covariance, hessian, residual = gaussian_hessian(
            net, equilibrium
        )

        # quadratic form along each direction, at two excursion sizes
        forms = np.zeros((len(directions), len(EPSILONS)))
        for i, direction in enumerate(directions):
            for j, eps in enumerate(EPSILONS):
                result = mam_action(
                    net,
                    equilibrium,
                    equilibrium + eps * direction,
                    T=HORIZON,
                    K=INTERVALS,
                    maxiter=60000,
                )
                if not result["success"]:
                    raise RuntimeError(f"action failed at r={rate}, dir={i}, eps={eps}")
                forms[i, j] = 2.0 * result["S"] / eps**2

        # error in the quadratic form is O(eps), so one Richardson step suffices
        extrapolated = 2.0 * forms[:, 1] - forms[:, 0]

        # least squares for (H11, H12, H22) from z' H z
        design = np.column_stack(
            [
                directions[:, 0] ** 2,
                2.0 * directions[:, 0] * directions[:, 1],
                directions[:, 1] ** 2,
            ]
        )
        (h11, h12, h22), *_ = np.linalg.lstsq(design, extrapolated, rcond=None)
        recovered = np.array([[h11, h12], [h12, h22]])

        relative = np.abs(recovered - hessian).max() / np.abs(hessian).max()
        rows.append(
            {
                "r": rate,
                "U_on": equilibrium.tolist(),
                "J": jacobian.tolist(),
                "D": diffusion.tolist(),
                "Sigma": covariance.tolist(),
                "H_lyapunov": hessian.tolist(),
                "H_recovered": recovered.tolist(),
                "hjb_residual": residual,
                "max_relative_error": float(relative),
            }
        )
        print(
            f"r={rate:g}: H_lyap=[{hessian[0,0]:.5f},{hessian[0,1]:.5f},"
            f"{hessian[1,1]:.5f}] recovered=[{recovered[0,0]:.5f},"
            f"{recovered[0,1]:.5f},{recovered[1,1]:.5f}] "
            f"rel={relative:.2e} hjb={residual:.1e}",
            flush=True,
        )

    output = {
        "settings": {
            "c": COST,
            "T": HORIZON,
            "K": INTERVALS,
            "angles_deg": np.rad2deg(ANGLES).tolist(),
            "epsilons": list(EPSILONS),
        },
        "rows": rows,
        "worst_relative_error": max(row["max_relative_error"] for row in rows),
    }
    destination = ROOT / "data" / "hessian_verification.json"
    destination.write_text(json.dumps(output, indent=1))
    print(f"wrote {destination}")
    print(f"worst relative error over all rates: {output['worst_relative_error']:.2e}")


if __name__ == "__main__":
    main()
