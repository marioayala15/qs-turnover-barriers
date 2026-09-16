"""Print the markdown tables used in README.md from results.json and action_path_r8.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "data" / "basins"
d = json.loads((HERE / "results.json").read_text())

print("### Hypothesis H (types checked at r in {0.5,1,2,4,8,16,64})\n")
print("| c | b0 < d0+c | #equilibria | x* | x_on | eigenvalues U* (r=0.5; r=8) | eigenvalues U_on (r=0.5; r=8) | types all r | H ok |")
print("|---|---|---|---|---|---|---|---|---|")
for c, h in d["hypothesis_H"].items():
    e = h["eigenvalues_by_r"]
    f = lambda v: ", ".join(f"{x:.4f}" for x in v)
    types = {tuple(t.values()) for t in h["types_by_r"].values()}
    print(f"| {c} | {h['H1_b0_lt_d0_plus_c']} (0.5 < {h['d0_plus_c']:.2f}) | {h['n_equilibria_closed_quadrant']} | "
          f"{h['positive_zeros_of_G'][0]:.6f} | {h['positive_zeros_of_G'][1]:.6f} | {f(e['0.5']['U_star'])}; {f(e['8']['U_star'])} | "
          f"{f(e['0.5']['U_on'])}; {f(e['8']['U_on'])} | {'/'.join(next(iter(types)))} ({len(types)} pattern) | {h['H_ok']} |")

print("\n### Rectangle R_off = [0,0.3] x [0,0.7]\n")
print("| c | b(0.7) | d0+c | gamma = d0+c-b(0.7) | alpha_C*0.3 | kappa*0.7 | x* | all three hold |")
print("|---|---|---|---|---|---|---|---|")
for c, h in d["rectangle"].items():
    print(f"| {c} | {h['b_of_b_off']:.6f} | {h['d0_plus_c']:.2f} | {h['margin_gamma']:.6f} | {h['alpha_a']:.1f} | {h['kappa_b']:.1f} | {h['x_star']:.6f} | {h['cond_b'] and h['cond_ab'] and h['cond_a']} |")

print("\n### Lemma B remark: unstable slope vs eta'(x*), and stable slope (all five r)\n")
print("| c | r | lambda_s | lambda_u | slope v_s | slope v_u | eta'(x*) | slope v_u > eta' |")
print("|---|---|---|---|---|---|---|---|")
for k, v in d["lemma_B_remark"].items():
    c, r = [s.split("=")[1] for s in k.split(",")]
    print(f"| {c} | {r} | {v['lam_s']:.4f} | {v['lam_u']:.4f} | {v['slope_s']:.4f} | {v['slope_u']:.4f} | {v['eta_prime']:.4f} | {v['slope_u_gt_eta_prime']} |")

print("\n### c = 0.36: stable manifold (reference eps=1e-6, rtol=1e-12)\n")
print("| r | upper branch: stop, end point | upper: x* - max x over samples 5.. (> 0 means never back on x = x*) | lower branch: crosses w=0 at x = | max curve diff over 11 (eps, rtol) variants: upper; lower | max abs(bisection - manifold) |")
print("|---|---|---|---|---|---|")
for r, rec in d["c036"].items():
    u, l = rec["stable_manifold"]["upper"], rec["stable_manifold"]["lower"]
    bd = max(abs(b["diff"]) for b in rec["bisection"] if "diff" in b)
    nb = sum("diff" in b for b in rec["bisection"])
    print(f"| {r} | {u['stop_reason']}, ({u['end_point'][0]:.3g}, {u['end_point'][1]:.3g}) | {rec['x_star'] - u['max_x_after_start']:.2e} | "
          f"{l['w0_intercept_x']:.4f} | {u['max_diff_over_variants']:.1e}; {l['max_diff_over_variants']:.1e} | {bd:.1e} ({nb} lines) |")

print("\n### c = 0.36: bisection lines (bracket < 1e-9)\n")
print("| r | line | bisection | manifold | diff |")
print("|---|---|---|---|---|")
for r, rec in d["c036"].items():
    for b in rec["bisection"]:
        if "diff" in b:
            val, man = (b["w_bisect"], b["w_manifold"]) if "w_bisect" in b else (b["x_bisect"], b["x_manifold"])
            print(f"| {r} | {b['line']} | {val:.8f} | {man:.8f} | {b['diff']:.1e} |")
        else:
            print(f"| {r} | {b['line']} | - | - | {b['note']} |")

print("\n### c = 0.36: unstable branches (eps=1e-7) and threshold line x = x*, w > w*\n")
print("| r | + branch ends at | - branch: stays in F_off / enters R_off at t | threshold line: on / off / unresolved (346 pts, w-w* in [1e-10, 30]) | loose-tol subset agrees | max entry time | below-saddle sanity (6 pts) |")
print("|---|---|---|---|---|---|---|")
for r, rec in d["c036"].items():
    um = rec["unstable_manifold"]; t = rec["threshold_line"]
    print(f"| {r} | ({um['towards_on']['final'][0]:.6f}, {um['towards_on']['final'][1]:.6f}) | "
          f"{um['towards_off']['all_points_after_start_in_F_off']} / {um['towards_off']['t_enter_R_off']:.2f} | "
          f"{t['counts']['on']} / {t['counts']['off']} / {t['counts']['unresolved']} | {t['loose_tolerance_labels_subset']['agree']} | "
          f"{t['max_entry_time']:.1f} | {'/'.join(rec['threshold_line_below_sanity'])} |")
print("\nOn-ball check (64 points on the circle of radius 0.05 about U_on, t=400):")
for r, rec in d["c036"].items():
    print(f"- r={r}: max final distance {rec['on_ball_check']['max_final_distance']:.1e}, max distance along orbit {rec['on_ball_check']['max_distance_along']:.4f}")
a = json.loads((HERE / "action_path_r8.json").read_text())
print("\nr=8 action path recomputation:", [(x["K"], round(x["S"], 7)) for x in a["runs"]], "reference", round(a["dV_reference_crossover_barriers"], 7))
