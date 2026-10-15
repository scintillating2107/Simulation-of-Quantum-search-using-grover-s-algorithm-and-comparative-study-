"""Paper-grade study: statistics, oracle synthesis, real-device noise, zero-noise extrapolation.

    python study.py threshold   # noise sweep over many random instances, 2 oracle designs
    python study.py device      # IBM fake-backend layouts + calibration noise
    python study.py zne         # zero-noise extrapolation by global folding
    python study.py analyze     # tables + figures from the CSVs above
    python study.py all

Every simulation is appended to a CSV as soon as it finishes, and finished
rows are skipped on a re-run, so an interrupted run can simply be restarted.
Add --quick for a small smoke test (writes to results/quick/).
"""

import argparse
import csv
import gc
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import csp
import csp_plots

NOISE = [1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 1e-2]
SYNTH = ["baseline", "aux+rp"]
DEVICES = ["FakeBrisbane", "FakeTorino", "FakeKingston"]
MAX_QUBITS = 13
# alpha of the decay model is fitted on these families only; Sudoku is the held-out test.
MODEL_FIT_FAMILIES = ("2-colouring", "3-SAT")
OUT = "results"
FIG = "figures"


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

def sat_unique(n, seed):
    for m in range(n + 2, 14):
        p = csp.random_ksat_with_solutions(n, m, 1, seed=seed, max_tries=3000)
        if p is not None:
            return p
    return None


def instance_suite(n_seeds=4, quick=False):
    """(family, size label, seed, Problem). Sizes are chosen to stay <= MAX_QUBITS."""
    specs = [("2-colouring (random trees)", f"n={n}", n) for n in (4, 5, 6)]
    specs += [("3-SAT (unique solution)", f"n={n}", n) for n in (3, 4)]
    specs += [("4x4 Sudoku (unique solution)", f"{b} blanks", b) for b in (2, 3, 4)]
    if quick:
        specs = [specs[0], specs[3], specs[5]]
        n_seeds = 1
    out = []
    for family, label, size in specs:
        seeds_found, s = 0, 0
        seen = set()
        while seeds_found < n_seeds and s < 200:
            if family.startswith("2-col"):
                p = csp.colouring_problem(size, csp.random_tree(size, seed=s), 2,
                                          name=f"tree n={size} #{s}")
            elif family.startswith("3-SAT"):
                p = sat_unique(size, seed=1000 * size + s)
                if p is not None:
                    p.name = f"3-SAT n={size} m={len(p.constraints)} #{s}"
            else:
                p = csp.random_sudoku_4x4_puzzle(size, seed=100 * size + s, max_qubits=MAX_QUBITS)
                if p is not None:
                    p.name = f"Sudoku {size} blanks #{s}"
            s += 1
            if p is None or p.n_qubits > MAX_QUBITS:
                continue
            key = tuple(p.constraints)
            if key in seen:
                continue
            seen.add(key)
            out.append((family, label, seeds_found, p))
            seeds_found += 1
    return out


# ---------------------------------------------------------------------------
# Resumable CSV
# ---------------------------------------------------------------------------

class Log:
    def __init__(self, path, key_cols):
        self.path, self.key_cols = path, key_cols
        self.done = set()
        if os.path.exists(path):
            for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
                self.done.add(tuple(row[c] for c in key_cols))

    def has(self, **kw):
        return tuple(str(kw[c]) for c in self.key_cols) in self.done

    def add(self, row):
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        self.done.add(tuple(str(row[c]) for c in self.key_cols))


def base_row(family, label, seed, p, synthesis, stats, k):
    M = len(p.solutions())
    return {"family": family, "size": label, "seed": seed, "problem": p.name, "synthesis": synthesis,
            "n_vars": p.n_vars, "constraints": len(p.constraints), "qubits": stats["qubits"],
            "cx": stats["cx"], "depth": stats["depth"], "N": p.N, "M": M, "iterations": k,
            "ideal": csp.ideal_success_probability(p.N, M, k),
            "classical": csp.brute_force_expected(p.N, M)}


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def run_threshold(out, shots, quick):
    log = Log(f"{out}/study_threshold.csv", ["problem", "synthesis", "p2"])
    noise = NOISE if not quick else [1e-3, 5e-3]
    for family, label, seed, p in instance_suite(quick=quick):
        M = len(p.solutions())
        k = csp.optimal_iterations(p.N, M)
        for syn in SYNTH:
            if all(log.has(problem=p.name, synthesis=syn, p2=p2) for p2 in noise):
                continue
            tqc = csp.transpile_grover(p, k, synthesis=syn)
            stats = csp.circuit_stats(tqc)
            for p2 in noise:
                if log.has(problem=p.name, synthesis=syn, p2=p2):
                    continue
                t0 = time.time()
                r = csp.run_grover(p, k, shots=shots, p2=p2, tqc=tqc, seed=17 + seed)
                hits = round(r["success"] * shots)
                log.add({**base_row(family, label, seed, p, syn, stats, k),
                         "p2": p2, "shots": shots, "hits": hits})
                print(f"  {p.name:26s} {syn:8s} cx={stats['cx']:5d} p2={p2:<7g} "
                      f"success={hits / shots:.3f} ({time.time() - t0:.1f}s)", flush=True)


def device_instances(quick):
    """One instance (seed 0) per size."""
    return [x for x in instance_suite(quick=quick) if x[2] == 0]


def load_device(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def run_device(out, shots, quick):
    from qiskit_aer import AerSimulator
    log = Log(f"{out}/study_device.csv", ["problem", "synthesis", "device"])
    devices = DEVICES if not quick else ["FakeTorino"]
    insts = device_instances(quick)
    for dev_name in devices:
        if all(log.has(problem=p.name, synthesis=s, device=dev_name) for _, _, _, p in insts for s in SYNTH):
            continue
        backend = load_device(dev_name)
        for family, label, seed, p in insts:
            M = len(p.solutions())
            k = csp.optimal_iterations(p.N, M)
            for syn in SYNTH:
                if log.has(problem=p.name, synthesis=syn, device=dev_name):
                    continue
                t0 = time.time()
                a2a = csp.circuit_stats(csp.transpile_grover(p, k, synthesis=syn))
                tqc = csp.transpile_grover(p, k, synthesis=syn, backend=backend, optimization_level=3)
                stats = csp.circuit_stats(tqc)
                small, nm = csp.compact_device_circuit(tqc, backend)
                r = csp.run_grover(p, k, shots=shots, tqc=small,
                                   simulator=AerSimulator(noise_model=nm, seed_simulator=29 + seed))
                hits = round(r["success"] * shots)
                log.add({**base_row(family, label, seed, p, syn, stats, k), "device": dev_name,
                         "cx_all_to_all": a2a["cx"], "shots": shots, "hits": hits})
                print(f"  {dev_name:13s} {p.name:26s} {syn:8s} 2q: {a2a['cx']}->{stats['cx']} "
                      f"success={hits / shots:.3f} ({time.time() - t0:.1f}s)", flush=True)
                del tqc, small, nm
                gc.collect()
        del backend
        gc.collect()


def run_zne(out, shots, quick):
    from qiskit_aer import AerSimulator
    log = Log(f"{out}/study_zne.csv", ["problem", "noise", "factor"])
    picks = [x for x in instance_suite(quick=quick) if x[2] == 0]
    wanted = {"n=6", "n=3", "2 blanks"} if not quick else {x[1] for x in picks}
    picks = [x for x in picks if x[1] in wanted]
    levels = [1e-3, 3e-3] if not quick else [1e-3]
    factors = [1, 3, 5]
    backend = None
    for family, label, seed, p in picks:
        M = len(p.solutions())
        k = csp.optimal_iterations(p.N, M)
        tqc = csp.transpile_grover(p, k, synthesis="aux+rp")
        stats = csp.circuit_stats(tqc)
        jobs = [(f"depolarizing p2={p2:g}", tqc, dict(p2=p2)) for p2 in levels]
        if not quick:
            if backend is None:
                backend = load_device("FakeTorino")
            dtqc = csp.transpile_grover(p, k, synthesis="aux+rp", backend=backend, optimization_level=3)
            small, nm = csp.compact_device_circuit(dtqc, backend)
            jobs.append(("FakeTorino calibration", small, dict(noise_model=nm)))
        for noise_name, circ, cfg in jobs:
            for f in factors:
                if log.has(problem=p.name, noise=noise_name, factor=f):
                    continue
                t0 = time.time()
                folded = csp.fold_circuit(circ, f)
                if "noise_model" in cfg:
                    sim = AerSimulator(noise_model=cfg["noise_model"], seed_simulator=41)
                    r = csp.run_grover(p, k, shots=shots, tqc=folded, simulator=sim)
                else:
                    r = csp.run_grover(p, k, shots=shots, tqc=folded, p2=cfg["p2"], seed=41)
                hits = round(r["success"] * shots)
                log.add({**base_row(family, label, seed, p, "aux+rp", stats, k), "noise": noise_name,
                         "factor": f, "shots": shots, "hits": hits})
                print(f"  {p.name:26s} {noise_name:24s} x{f} success={hits / shots:.3f} "
                      f"({time.time() - t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def threshold_from_success(levels, success, iterations, classical, ideal):
    """Advantage threshold with p2 = 0 taken from the exact ideal success."""
    from experiments import advantage_threshold
    lv = [0.0] + list(levels)
    ratios = []
    for s in [ideal] + list(success):
        q = csp.grover_expected_queries(iterations, s)
        ratios.append(q / classical if np.isfinite(q) else 1e6)
    return advantage_threshold(lv, ratios)


def analyze_threshold(df, n_boot=300, seed=0):
    """Per instance and design: threshold + 95% parametric-bootstrap CI over shot noise."""
    rng = np.random.default_rng(seed)
    rows = []
    n_levels = df["p2"].nunique()
    for (prob, syn), g in df.groupby(["problem", "synthesis"]):
        if len(g) < n_levels:  # still running
            continue
        g = g.sort_values("p2")
        levels, shots = list(g["p2"]), int(g["shots"].iloc[0])
        succ = list(g["hits"] / shots)
        first = g.iloc[0]
        args = (int(first["iterations"]), float(first["classical"]), float(first["ideal"]))
        thr, status = threshold_from_success(levels, succ, *args)
        boots = []
        for _ in range(n_boot):
            s_b = rng.binomial(shots, np.clip(succ, 0, 1)) / shots
            t_b, _ = threshold_from_success(levels, s_b, *args)
            if np.isfinite(t_b):
                boots.append(t_b)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        rows.append({"family": first["family"], "size": first["size"], "seed": first["seed"],
                     "problem": prob, "synthesis": syn, "qubits": first["qubits"], "cx": first["cx"],
                     "N": first["N"], "M": first["M"], "iterations": first["iterations"],
                     "ideal": first["ideal"], "classical": first["classical"],
                     "threshold": thr, "ci_low": lo, "ci_high": hi, "status": status})
    return pd.DataFrame(rows)


def fit_decay_model(thr, fit_families=None):
    """One-parameter model of the threshold.

    Grover beats brute force while P_succ > P_req = (k+1)/classical. If noise scales
    success as P_ideal * exp(-alpha * p2 * CX), the crossing is at
    p* = ln(P_ideal / P_req) / (alpha * CX). alpha is fitted (geometric mean) on
    `fit_families` (all families if None); every instance gets a prediction.
    """
    t = thr[thr["status"] == "ok"].copy()
    t["p_req"] = (t["iterations"] + 1) / t["classical"]
    t["log_margin"] = np.log(t["ideal"] / t["p_req"])
    t["alpha_i"] = t["log_margin"] / (t["threshold"] * t["cx"])
    fit = t if fit_families is None else t[t["family"].str.startswith(tuple(fit_families))]
    alpha = float(np.exp(np.log(fit["alpha_i"]).mean()))
    t["predicted"] = t["log_margin"] / (alpha * t["cx"])
    t["ratio"] = t["threshold"] / t["predicted"]

    def r2(g):
        y, yhat = np.log(g["threshold"]), np.log(g["predicted"])
        return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))

    summary = {"alpha": alpha, "alpha_min": fit["alpha_i"].min(), "alpha_max": fit["alpha_i"].max(),
               "r2_all": r2(t), "ratio_min": t["ratio"].min(), "ratio_max": t["ratio"].max(),
               "n_fit": len(fit), "n_all": len(t)}
    held = t.drop(fit.index)
    if len(held) > 2:
        summary.update(r2_held_out=r2(held), held_ratio_min=held["ratio"].min(),
                       held_ratio_max=held["ratio"].max(), n_held_out=len(held))
    return t, summary


def fit_power_law(thr, n_boot=2000, seed=0):
    """log p* = log A - b log CX, OLS + bootstrap over instances. Per synthesis mode."""
    rng = np.random.default_rng(seed)
    out = []
    for syn, g in thr[thr["status"] == "ok"].groupby("synthesis"):
        x, y = np.log(g["cx"].to_numpy(float)), np.log(g["threshold"].to_numpy(float))
        if len(x) < 3:
            continue
        slope, icpt = np.polyfit(x, y, 1)
        bs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(x), len(x))
            if np.ptp(x[idx]) == 0:
                continue
            bs.append(np.polyfit(x[idx], y[idx], 1)[0])
        r2 = 1 - np.sum((y - (slope * x + icpt)) ** 2) / np.sum((y - y.mean()) ** 2)
        out.append({"synthesis": syn, "b": -slope, "b_ci_low": -np.percentile(bs, 97.5),
                     "b_ci_high": -np.percentile(bs, 2.5), "A": math.exp(icpt), "r2": r2, "n": len(x)})
    return pd.DataFrame(out)


def analyze(out, fig_dir=FIG):
    os.makedirs(fig_dir, exist_ok=True)
    report = []
    path = f"{out}/study_threshold.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        thr = analyze_threshold(df)
        thr.to_csv(f"{out}/study_thresholds.csv", index=False)
        fit = fit_power_law(thr)
        fit.to_csv(f"{out}/study_powerlaw.csv", index=False)
        agg = (thr.groupby(["family", "size", "synthesis"])
               .agg(cx=("cx", "mean"), qubits=("qubits", "max"), n=("threshold", "size"),
                    thr_median=("threshold", "median"), thr_min=("threshold", "min"),
                    thr_max=("threshold", "max"))
               .reset_index())
        agg.to_csv(f"{out}/study_thresholds_by_size.csv", index=False)
        piv = thr.pivot_table(index="problem", columns="synthesis", values=["threshold", "cx"])
        gain = (piv["threshold"]["aux+rp"] / piv["threshold"]["baseline"]).dropna()
        cxr = (piv["cx"]["aux+rp"] / piv["cx"]["baseline"]).dropna()
        report += ["## Threshold study", agg.round(5).to_string(index=False), "",
                   "## Power law  p* = A * CX^-b", fit.round(3).to_string(index=False), "",
                   f"aux+rp vs baseline: CX ratio median {cxr.median():.2f} "
                   f"[{cxr.min():.2f}, {cxr.max():.2f}], threshold gain median {gain.median():.2f}x "
                   f"[{gain.min():.2f}, {gain.max():.2f}] over {len(gain)} instances", ""]
        # Decay model: alpha fitted on the families listed in MODEL_FIT_FAMILIES, the rest held out.
        pred, summ = fit_decay_model(thr, MODEL_FIT_FAMILIES)
        pred.to_csv(f"{out}/study_decay_model.csv", index=False)
        pd.DataFrame([summ]).to_csv(f"{out}/study_decay_model_summary.csv", index=False)
        report += ["## Decay model  p* = ln(P_ideal / P_req) / (alpha * CX)",
                   f"fitted on {MODEL_FIT_FAMILIES}: " + ", ".join(f"{k}={v:.4g}" for k, v in summ.items()), ""]
        csp_plots.plot_decay_model(pred, summ["alpha"], summ["r2_all"], MODEL_FIT_FAMILIES).savefig(
            f"{fig_dir}/study_decay_model.png", dpi=170)
        # Heatmaps of mean success per size, one per synthesis mode.
        for syn in SYNTH:
            d = df[df["synthesis"] == syn].copy()
            d["s"] = d["hits"] / d["shots"]
            rows = []
            for (fam, size), g in d.groupby(["family", "size"], sort=False):
                pv = g.pivot_table(index="problem", columns="p2", values="s")
                mean_s = pv.mean(axis=0)
                ideal = g.groupby("problem")["ideal"].first().mean()
                k = int(g["iterations"].iloc[0])
                cl = float(g["classical"].iloc[0])
                s_all = [ideal] + list(mean_s)
                adv = [csp.grover_expected_queries(k, s) < cl for s in s_all]
                rows.append({"label": f"{fam.split(' (')[0]} {size} · {int(g['qubits'].max())}q, "
                                      f"~{int(g['cx'].mean())} CX, {g['problem'].nunique()} inst.",
                             "success": s_all, "advantage": adv, "cx": g["cx"].mean()})
            rows.sort(key=lambda r: (r["label"].split(" ")[0], r["cx"]))
            fig = csp_plots.plot_noise_heatmap(rows, [0.0] + sorted(d["p2"].unique()),
                                               title=f"Noise-threshold heatmap · {csp.SYNTHESIS_LABELS[syn]}")
            fig.savefig(f"{fig_dir}/study_heatmap_{syn.replace('+', '_')}.png", dpi=170)
        csp_plots.plot_threshold_scatter(thr, fit).savefig(f"{fig_dir}/study_threshold_vs_cx.png", dpi=170)

    path = f"{out}/study_device.csv"
    if os.path.exists(path):
        dv = pd.read_csv(path)
        dv["success"] = dv["hits"] / dv["shots"]
        dv["ci_low"], dv["ci_high"] = zip(*[csp.wilson_interval(h, n) for h, n in zip(dv["hits"], dv["shots"])])
        dv["random_guess"] = dv["M"] / dv["N"]
        dv["grover_queries"] = [csp.grover_expected_queries(k, s) for k, s in zip(dv["iterations"], dv["success"])]
        dv["advantage"] = dv["grover_queries"] < dv["classical"]
        dv["routing_overhead"] = dv["cx"] / dv["cx_all_to_all"]
        dv.to_csv(f"{out}/study_device_analyzed.csv", index=False)
        summ = (dv.groupby(["device", "synthesis"])
                .agg(instances=("problem", "size"), routing_overhead=("routing_overhead", "median"),
                     success_median=("success", "median"), advantage=("advantage", "sum"))
                .reset_index())
        report += ["## Device study", summ.round(3).to_string(index=False), "",
                   dv[["device", "problem", "synthesis", "cx_all_to_all", "cx", "ideal", "success",
                       "random_guess", "advantage"]].round(3).to_string(index=False), ""]
        csp_plots.plot_device_results(dv).savefig(f"{fig_dir}/study_devices.png", dpi=170)

    path = f"{out}/study_zne.csv"
    if os.path.exists(path):
        z = pd.read_csv(path)
        z["success"] = z["hits"] / z["shots"]
        rows = []
        for (prob, noise), g in z.groupby(["problem", "noise"], sort=False):
            g = g.sort_values("factor")
            if len(g) < 3:
                continue
            fs, ss = list(g["factor"]), list(g["success"])
            rows.append({"problem": prob, "noise": noise, "ideal": g["ideal"].iloc[0],
                         "raw": ss[0], "linear": float(np.clip(np.polyval(np.polyfit(fs[:2], ss[:2], 1), 0), 0, 1)),
                         "richardson": float(np.clip(csp.richardson_extrapolate(fs, ss), 0, 1)),
                         "classical": g["classical"].iloc[0], "iterations": g["iterations"].iloc[0]})
        zr = pd.DataFrame(rows)
        zr["raw_error"] = (zr["raw"] - zr["ideal"]).abs()
        zr["richardson_error"] = (zr["richardson"] - zr["ideal"]).abs()
        zr.to_csv(f"{out}/study_zne_analyzed.csv", index=False)
        report += ["## Zero-noise extrapolation (estimate of P(success))", zr.round(3).to_string(index=False), ""]
        csp_plots.plot_zne(zr).savefig(f"{fig_dir}/study_zne.png", dpi=170)

    text = "\n".join(report)
    open(f"{out}/study_report.txt", "w", encoding="utf-8").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["threshold", "device", "zne", "analyze", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shots", type=int, default=None)
    args = ap.parse_args()
    out = OUT if not args.quick else f"{OUT}/quick"
    os.makedirs(out, exist_ok=True)
    parts = ["threshold", "device", "zne", "analyze"] if args.part == "all" else [args.part]
    if "threshold" in parts:
        print("Threshold study", flush=True)
        run_threshold(out, args.shots or 1024, args.quick)
    if "device" in parts:
        print("Device study", flush=True)
        run_device(out, args.shots or 1024, args.quick)
    if "zne" in parts:
        print("ZNE study", flush=True)
        run_zne(out, args.shots or 2048, args.quick)
    if "analyze" in parts:
        analyze(out, FIG if not args.quick else f"{FIG}/quick")


if __name__ == "__main__":
    sys.exit(main())
