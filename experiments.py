"""Benchmark suite for Grover on constraint satisfaction problems.

    python experiments.py            # full run (~15-25 min on a laptop)
    python experiments.py --quick    # small subset, ~1-2 min

Writes CSVs to results/ and figures to figures/csp_*.png. The Streamlit page
(pages/1_Constraint_Satisfaction.py) reads the CSVs, so it can show the study
without re-running it.
"""

import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import csp
import csp_plots

RESULTS = "results"
FIGURES = "figures"
NOISE_LEVELS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def heatmap_suite(quick=False):
    path = lambda n: csp.colouring_problem(n, [(i, i + 1) for i in range(n - 1)], k=2,  # noqa: E731
                                           name=f"2-colouring P{n}")
    P = csp.presets()
    suite = [
        ("2-colouring (path graphs)", path(3)),
        ("2-colouring (path graphs)", path(4)),
        ("2-colouring (path graphs)", path(5)),
        ("3-SAT (unique solution)", P["SAT · 3 vars, 7 clauses (1 solution)"]),
        ("4x4 Sudoku (unique solution)", csp.sudoku_4x4_with_blanks([(0, 0), (1, 1)])),
    ]
    if not quick:
        suite += [
            ("2-colouring (path graphs)", path(6)),
            ("3-SAT (unique solution)", P["SAT · 4 vars, 8 clauses (1 solution)"]),
            ("4x4 Sudoku (unique solution)", csp.sudoku_4x4_with_blanks([(0, 0), (0, 1), (2, 2)])),
            ("4x4 Sudoku (unique solution)", csp.sudoku_4x4_with_blanks([(0, 0), (1, 2), (2, 1), (3, 3)])),
        ]
    return suite


def advantage_threshold(noise_levels, ratios):
    """Largest p2 at which the Grover/classical expected-query ratio is still < 1.

    Interpolates log(ratio) linearly in log(p2) between grid points (linearly in
    p2 for the first step up from p2 = 0). Returns (threshold, status), status
    'ok', 'above_grid' (still an advantage at the noisiest level, so the value is
    a lower bound) or 'never' (no advantage even without noise).
    """
    if ratios[0] >= 1:
        return float("nan"), "never"
    prev_p, prev_r = noise_levels[0], ratios[0]
    for p, r in zip(noise_levels[1:], ratios[1:]):
        if r >= 1:
            lr0, lr1 = math.log(prev_r), math.log(min(r, 1e6))
            t = -lr0 / (lr1 - lr0)
            if prev_p == 0:
                return t * p, "ok"
            return math.exp(math.log(prev_p) + t * (math.log(p) - math.log(prev_p))), "ok"
        prev_p, prev_r = p, r
    return noise_levels[-1], "above_grid"


def run_heatmap(shots, quick=False):
    levels = NOISE_LEVELS if not quick else [0.0, 1e-3, 1e-2]
    records, rows = [], []
    for family, prob in heatmap_suite(quick):
        M = len(prob.solutions())
        k = csp.optimal_iterations(prob.N, M)
        tqc = csp.transpile_grover(prob, k)
        stats = csp.circuit_stats(tqc)
        classical = csp.brute_force_expected(prob.N, M)
        succ, ratios = [], []
        for p2 in levels:
            t0 = time.time()
            r = csp.run_grover(prob, k, shots=shots, p2=p2, tqc=tqc)
            q = csp.grover_expected_queries(k, r["success"])
            succ.append(r["success"])
            ratios.append(q / classical if np.isfinite(q) else 1e6)
            records.append({"family": family, "problem": prob.name, "n_vars": prob.n_vars,
                            "qubits": stats["qubits"], "cx": stats["cx"], "depth": stats["depth"],
                            "N": prob.N, "M": M, "iterations": k, "p2": p2, "shots": shots,
                            "success": r["success"], "grover_queries": q,
                            "classical_queries": classical})
            print(f"  {prob.name:28s} p2={p2:<7g} success={r['success']:.3f}  ({time.time() - t0:.1f}s)")
        thr, status = advantage_threshold(levels, ratios)
        rows.append({"family": family, "problem": prob.name, "cx": stats["cx"],
                     "qubits": stats["qubits"], "threshold": thr, "status": status})
    return pd.DataFrame(records), pd.DataFrame(rows), levels


def heatmap_rows(df):
    """Reshape the long heatmap table into rows for csp_plots.plot_noise_heatmap."""
    levels = sorted(df["p2"].unique())
    rows = []
    order = df.drop_duplicates("problem").sort_values(["family", "cx"])
    for _, meta in order.iterrows():
        sub = df[df["problem"] == meta["problem"]].set_index("p2").loc[levels]
        rows.append({
            "label": f"{meta['problem']}  ·  {meta['qubits']}q, {meta['cx']} CX, M={meta['M']}",
            "success": list(sub["success"]),
            "advantage": list(sub["grover_queries"] < sub["classical_queries"]),
        })
    return rows, levels


def multi_solution_instances(n_vars, M, count, seed):
    """Up to ``count`` random 3-SAT instances with exactly M solutions, using the fewest clauses that work."""
    for m in range(3, 12):
        first = csp.random_ksat_with_solutions(n_vars, m, M, seed=seed, max_tries=2000)
        if first is None:
            continue  # too few or too many clauses for this solution count
        found = [first]
        s = seed + 1
        while len(found) < count and s < seed + 20:
            p = csp.random_ksat_with_solutions(n_vars, m, M, seed=s, max_tries=2000)
            if p is not None and p not in found:
                found.append(p)
            s += 1
        return found
    return []


def run_multi_solution(shots, noisy_p2=1e-3, quick=False, max_noisy_qubits=15):
    configs = [(4, M) for M in (1, 2, 4, 8)] + ([] if quick else [(5, M) for M in (1, 2, 4, 8)])
    rows = []
    for n, M in configs:
        insts = multi_solution_instances(n, M, count=1 if quick else 3, seed=100 * n + M)
        if not insts:
            print(f"  no instance for n={n} M={M}")
            continue
        N = 2 ** n
        cls = [csp.classical_summary(p, walksat_runs=20 if quick else 100) for p in insts]
        k = csp.optimal_iterations(N, M)
        p_ideal = csp.ideal_success_probability(N, M, k)
        main = insts[0]
        row = {
            "n_vars": n, "N": N, "M": M, "instances": len(insts),
            "clauses": len(main.constraints), "qubits": main.n_qubits, "iterations": k,
            "brute_force_expected": csp.brute_force_expected(N, M),
            "dpll_nodes": float(np.mean([c["dpll_nodes"] for c in cls])),
            "walksat_mean_flips": float(np.mean([c["walksat_mean_flips"] for c in cls])),
            "walksat_mean_evals": float(np.mean([c["walksat_mean_evals"] for c in cls])),
            "grover_ideal_success": p_ideal,
            "grover_ideal_queries": csp.grover_expected_queries(k, p_ideal),
            "bbht_ideal_queries": csp.bbht_expected_queries(N, M, trials=2000 if quick else 20000),
            "noisy_p2": noisy_p2,
            "clauses_json": json.dumps(main.meta["clauses"]),
        }
        if main.n_qubits <= max_noisy_qubits:
            t0 = time.time()
            r = csp.run_grover(main, k, shots=shots, p2=noisy_p2)
            row.update(grover_noisy_success=r["success"], cx=r["cx"],
                       grover_noisy_queries=csp.grover_expected_queries(k, r["success"]))
            print(f"  n={n} M={M} k={k} {main.n_qubits}q noisy success={r['success']:.3f} ({time.time() - t0:.1f}s)")
        else:
            row.update(grover_noisy_success=float("nan"), cx=float("nan"), grover_noisy_queries=float("nan"))
            print(f"  n={n} M={M} k={k} {main.n_qubits}q (noisy run skipped: too many qubits)")
        rows.append(row)
    return pd.DataFrame(rows)


def run_iteration_sweep(shots, noisy_p2=1e-3, n_vars=4, Ms=(1, 2, 4)):
    """Simulated success vs iteration count on real instances (for the 'more solutions' figure)."""
    rows = []
    for M in Ms:
        inst = multi_solution_instances(n_vars, M, 1, seed=100 * n_vars + M)[0]
        for k in range(0, 6):
            r = csp.run_grover(inst, k, shots=shots, p2=noisy_p2)
            rows.append({"n_vars": n_vars, "M": M, "iterations": k, "p2": noisy_p2,
                         "success": r["success"], "problem": inst.name})
        print(f"  sweep M={M} done")
    return pd.DataFrame(rows)


def run_classical_table():
    rows = []
    for name, p in csp.presets().items():
        c = csp.classical_summary(p, walksat_runs=100)
        M = c["M"]
        k = csp.optimal_iterations(p.N, M)
        p_ideal = csp.ideal_success_probability(p.N, M, k)
        rows.append({"preset": name, "n_vars": p.n_vars, "qubits": p.n_qubits, **c,
                     "grover_iterations": k,
                     "grover_ideal_queries": csp.grover_expected_queries(k, p_ideal) if M else float("inf")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small subset for a fast check")
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--only", choices=["heatmap", "multi", "sweep", "classical"], help="run one part")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    parts = [args.only] if args.only else ["classical", "multi", "sweep", "heatmap"]

    if "classical" in parts:
        print("Classical baselines on presets")
        df = run_classical_table()
        df.to_csv(f"{RESULTS}/classical_comparison.csv", index=False)
        print(df[["preset", "N", "M", "brute_force_sequential", "dpll_nodes",
                  "walksat_mean_evals", "grover_ideal_queries"]].to_string(index=False))

    if "multi" in parts:
        print("Multiple-solution study")
        df = run_multi_solution(args.shots, quick=args.quick)
        df.to_csv(f"{RESULTS}/multi_solution.csv", index=False)
        for n in sorted(df["n_vars"].unique()):
            fig = csp_plots.plot_queries_vs_solutions(df.to_dict("records"), n, noisy_label="1e-3")
            fig.savefig(f"{FIGURES}/csp_queries_vs_solutions_n{n}.png", dpi=150)

    if "sweep" in parts:
        print("Iteration sweep")
        df = run_iteration_sweep(args.shots)
        df.to_csv(f"{RESULTS}/iteration_sweep.csv", index=False)
        noisy = {M: (list(g["iterations"]), list(g["success"])) for M, g in df.groupby("M")}
        fig = csp_plots.plot_success_vs_iterations(16, [1, 2, 4], max_iter=5, noisy=noisy)
        fig.savefig(f"{FIGURES}/csp_success_vs_iterations.png", dpi=150)

    if "heatmap" in parts:
        print("Noise-threshold heatmap")
        df, thr, _ = run_heatmap(args.shots, quick=args.quick)
        df.to_csv(f"{RESULTS}/noise_heatmap.csv", index=False)
        thr.to_csv(f"{RESULTS}/noise_thresholds.csv", index=False)
        rows, levels = heatmap_rows(df)
        csp_plots.plot_noise_heatmap(rows, levels).savefig(f"{FIGURES}/csp_noise_heatmap.png", dpi=150)
        csp_plots.plot_threshold_vs_cx(thr.to_dict("records")).savefig(
            f"{FIGURES}/csp_threshold_vs_cx.png", dpi=150)
        print(thr.to_string(index=False))


if __name__ == "__main__":
    main()
