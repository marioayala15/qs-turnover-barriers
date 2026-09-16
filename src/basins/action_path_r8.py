"""Minimum-action path U_on -> U* at c=0.36, r=8 (data/optimal_paths.json stores only r in {0.5,1,4,16}).

Uses mam_action and net_wellmixed_explicit from src/ldp_action.py.  Warm start: the stored r=4
path.  Writes data/basins/action_path_r8.json; the action is compared with
data/crossover_barriers.json (r=8, c=0.36).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "data" / "basins"
sys.path.insert(0, str(ROOT / "src"))
from ldp_action import mam_action, net_wellmixed_explicit  # noqa: E402

stored = json.loads((ROOT / "data" / "optimal_paths.json").read_text())
ref = json.loads((ROOT / "data" / "crossover_barriers.json").read_text())
dV_ref = next(r["dV"] for r in ref["costs"]["0.36"]["rows"] if r["r"] == 8.0)
s = stored["settings"]
U_on = np.array([s["x_on"], 2.0 * s["x_on"]])
U_st = np.array([s["x_star"], 2.0 * s["x_star"]])
seed = np.asarray(next(r["path"] for r in stored["rows"] if r["r"] == 4.0))

net = net_wellmixed_explicit(0.36, rN=8.0)
out = {"dV_reference_crossover_barriers": dV_ref, "runs": []}
path = None
for K in (400, 800):
    s_old = np.linspace(0, 1, (seed if path is None else path).shape[0])
    s_new = np.linspace(0, 1, K + 1)
    base = seed if path is None else path
    p0 = np.stack([np.interp(s_new, s_old, base[:, j]) for j in range(2)], axis=1)
    t0 = time.time()
    res = mam_action(net, U_on, U_st, T=40.0, K=K, path0=p0, maxiter=20000)
    path = res["path"]
    row = dict(K=K, T=40.0, S=res["S"], success=res["success"], seconds=time.time() - t0)
    out["runs"].append(row)
    print(row, "ref", dV_ref, flush=True)
out["path"] = path.tolist()
(HERE / "action_path_r8.json").write_text(json.dumps(out))
