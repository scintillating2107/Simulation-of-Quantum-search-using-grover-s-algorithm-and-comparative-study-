"""Write the paper's tables and in-text numbers from the result CSVs.

    python study.py analyze            # refresh results/study_*.csv first
    python paper/make_tables.py        # -> paper/generated/*.tex

Anything whose data does not exist yet is written as a red \\todo, so a
draft built before the study finishes cannot be mistaken for a final one.
"""

import argparse
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def sci(x, digits=2):
    """siunitx number in scientific notation, e.g. 1.4e-3 -> \\num{1.4e-3}."""
    if x is None or not math.isfinite(x):
        return "--"
    e = math.floor(math.log10(abs(x)))
    m = round(x / 10 ** e, digits - 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return rf"\num{{{m:g}e{e}}}"


def fixed(x, nd=2):
    return "--" if x is None or not math.isfinite(x) else f"{x:.{nd}f}"


class Numbers:
    def __init__(self):
        self.lines = []

    def __setitem__(self, name, value):
        self.lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    def pending(self, *names):
        for n in names:
            self[n] = r"\todo{pending}"


def threshold_table(res, nums):
    path = f"{res}/study_thresholds.csv"
    if not os.path.exists(path):
        nums.pending("thrLowest", "thrHighest", "cxRatioMed", "cxRatioMin", "cxRatioMax", "cxSavingMax",
                     "gainMed", "gainMax",
                     "gainSatMed", "nInstances")
        return r"\todo{threshold table pending}"
    thr = pd.read_csv(path)
    thr["fam"] = thr["family"].str.split(" (", regex=False).str[0]
    order = {"2-colouring": 0, "3-SAT": 1, "4x4 Sudoku": 2}
    rows = []
    for (fam, size), g in sorted(thr.groupby(["fam", "size"]), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])):
        cells = {}
        for syn in ("baseline", "aux+rp"):
            s = g[g["synthesis"] == syn]
            above = (s["status"] == "above_grid").all()
            med = s["threshold"].median()
            cells[syn] = dict(cx=int(round(s["cx"].mean())), med=med, above=above, n=len(s))
        pb, pa = cells["baseline"], cells["aux+rp"]
        fmt = lambda c: r"$>$\num{1e-2}" if c["above"] else sci(c["med"])
        gain = "--" if (pb["above"] or pa["above"]) else f"{pa['med'] / pb['med']:.2f}"
        label = {"2-colouring": "2-col.", "3-SAT": "3-SAT", "4x4 Sudoku": "Sudoku"}[fam]
        size_s = size.replace(" blanks", " bl.").replace("n=", "$n{=}$")
        rows.append(f"{label} {size_s} & {int(g['qubits'].max())} & {int(g['N'].iloc[0])} & {int(g['M'].iloc[0])} "
                    f"& {pb['cx']} / {pa['cx']} & {fmt(pb)} & {fmt(pa)} & {gain} \\\\")
    ok = thr[thr["status"] == "ok"]
    base = ok[ok["synthesis"] == "baseline"]
    nums["thrLowest"] = sci(base["threshold"].min())
    nums["thrHighest"] = sci(base["threshold"].max())
    piv = thr.pivot_table(index="problem", columns="synthesis", values=["threshold", "cx"])
    cxr = (piv["cx"]["aux+rp"] / piv["cx"]["baseline"]).dropna()
    both_ok = ok.pivot_table(index="problem", columns="synthesis", values="threshold").dropna()
    gain = both_ok["aux+rp"] / both_ok["baseline"]
    sat = gain[gain.index.str.startswith("3-SAT")]
    nums["cxRatioMed"] = fixed(cxr.median())
    nums["cxRatioMin"] = fixed(cxr.min())
    nums["cxRatioMax"] = fixed(cxr.max())
    nums["cxSavingMax"] = str(int(round(100 * (1 - cxr.min()))))
    nums["gainMed"] = fixed(gain.median())
    nums["gainMax"] = fixed(gain.max())
    nums["gainSatMed"] = fixed(sat.median()) if len(sat) else "--"
    nums["nInstances"] = str(thr["problem"].nunique())
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Advantage threshold $p_2^*$ (median over random instances). CX counts are for the
baseline / ancilla-aware + relative-phase (aux+rp) oracle after all-to-all transpilation.
Gain is the median ratio $p_2^*(\text{{aux+rp}})/p_2^*(\text{{baseline}})$.}}
\label{{tab:threshold}}
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}lrrrcccc@{{}}}}
\toprule
Instance & $q$ & $N$ & $M$ & CX & $p_2^*$ base & $p_2^*$ aux+rp & gain \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def model_numbers(res, nums):
    path = f"{res}/study_decay_model_summary.csv"
    names = ("alphaFit", "alphaMin", "alphaMax", "modelRsq", "ratioMin", "ratioMax", "nModel")
    if not os.path.exists(path):
        nums.pending(*names, "heldRsq", "heldRatioMin", "heldRatioMax", "nHeld")
        return
    s = pd.read_csv(path).iloc[0]
    nums["alphaFit"] = fixed(s["alpha"])
    nums["alphaMin"] = fixed(s["alpha_min"])
    nums["alphaMax"] = fixed(s["alpha_max"])
    nums["modelRsq"] = fixed(s["r2_all"], 3)
    nums["ratioMin"] = fixed(s["ratio_min"])
    nums["ratioMax"] = fixed(s["ratio_max"])
    nums["nModel"] = str(int(s["n_all"]))
    if "r2_held_out" in s and pd.notna(s.get("r2_held_out")):
        nums["heldRsq"] = fixed(s["r2_held_out"], 3)
        nums["heldRatioMin"] = fixed(s["held_ratio_min"])
        nums["heldRatioMax"] = fixed(s["held_ratio_max"])
        nums["nHeld"] = str(int(s["n_held_out"]))
    else:
        nums.pending("heldRsq", "heldRatioMin", "heldRatioMax", "nHeld")


def power_law_numbers(res, nums):
    path = f"{res}/study_powerlaw.csv"
    fit = pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) > 2 else pd.DataFrame()
    for syn, tag in (("baseline", "Base"), ("aux+rp", "Aux")):
        f = fit[fit["synthesis"] == syn] if not fit.empty else fit
        if f.empty:
            nums.pending(f"pow{tag}", f"pow{tag}Lo", f"pow{tag}Hi", f"pow{tag}Rsq")
            continue
        f = f.iloc[0]
        nums[f"pow{tag}"] = fixed(f["b"])
        nums[f"pow{tag}Lo"] = fixed(f["b_ci_low"])
        nums[f"pow{tag}Hi"] = fixed(f["b_ci_high"])
        nums[f"pow{tag}Rsq"] = fixed(f["r2"])


def multi_table(res, nums):
    m = pd.read_csv(f"{res}/multi_solution.csv")
    rows = []
    for r in m.itertuples():
        noisy = fixed(r.grover_noisy_queries, 1) if pd.notna(r.grover_noisy_queries) else f"({r.qubits}\\,q)"
        rows.append(f"{r.n_vars} & {r.M} & {r.iterations} & {r.brute_force_expected:.1f} & {r.dpll_nodes:.1f} & "
                    f"{r.walksat_mean_evals:.1f} & {r.grover_ideal_queries:.1f} & {r.bbht_ideal_queries:.1f} & {noisy} \\\\")
    sim = m.dropna(subset=["grover_noisy_queries"])
    nums["multiNoisyP"] = sci(float(m["noisy_p2"].iloc[0]), 1)
    nums["bbhtRatioMin"] = fixed((m["bbht_ideal_queries"] / m["grover_ideal_queries"]).min(), 1)
    nums["bbhtRatioMax"] = fixed((m["bbht_ideal_queries"] / m["grover_ideal_queries"]).max(), 1)
    one = m[(m["n_vars"] == 4) & (m["M"] == 1)].iloc[0]
    four = m[(m["n_vars"] == 4) & (m["M"] == 4)].iloc[0]
    nums["succMOneIdeal"] = fixed(one["grover_ideal_success"])
    nums["succMOneNoisy"] = fixed(one["grover_noisy_success"])
    nums["succMFourIdeal"] = fixed(four["grover_ideal_success"])
    nums["succMFourNoisy"] = fixed(four["grover_noisy_success"])
    nums["nMultiSim"] = str(len(sim))
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Expected cost to obtain a verified solution of random 3-SAT with exactly $M$ solutions
(3 instances per row; classical columns averaged). Grover and BBHT count oracle calls plus one
classical check per measurement; DPLL counts search nodes and WalkSAT assignments evaluated.
The noisy column uses $p_2=\num{{1e-3}}$ on the first instance; rows marked with a qubit count
were too large to simulate with noise.}}
\label{{tab:multi}}
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}rrrrrrrrr@{{}}}}
\toprule
$n$ & $M$ & $k$ & brute & DPLL & WalkSAT & Grover & BBHT & noisy \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def device_table(res, nums):
    path = f"{res}/study_device_analyzed.csv"
    if not os.path.exists(path):
        nums.pending("routeOverhead", "deviceAdvantage", "deviceRuns")
        return r"\todo{device table pending: run \texttt{python study.py device} then \texttt{analyze}}"
    dv = pd.read_csv(path)
    nums["routeOverhead"] = fixed(dv["routing_overhead"].median(), 1)
    nums["deviceAdvantage"] = str(int(dv["advantage"].sum()))
    nums["deviceRuns"] = str(len(dv))
    rows = []
    for (dev, syn), g in dv.groupby(["device", "synthesis"], sort=False):
        rows.append(f"{dev.replace('Fake', '')} & {syn} & {g['routing_overhead'].median():.1f} & "
                    f"{g['success'].median():.2f} & {int(g['advantage'].sum())}/{len(g)} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Grover on calibrated IBM device models (one instance per size). Routing overhead is the
median ratio of routed to all-to-all two-qubit gates; ``adv.'' counts instances where Grover's
expected cost is below brute force.}}
\label{{tab:device}}
\begin{{tabular}}{{@{{}}llccc@{{}}}}
\toprule
Device & Oracle & routing & median $P_\text{{succ}}$ & adv. \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def zne_table(res, nums):
    path = f"{res}/study_zne_analyzed.csv"
    if not os.path.exists(path):
        nums.pending("zneRawErr", "zneRichErr")
        return r"\todo{ZNE table pending: run \texttt{python study.py zne} then \texttt{analyze}}"
    z = pd.read_csv(path)
    nums["zneRawErr"] = fixed(z["raw_error"].median(), 3)
    nums["zneRichErr"] = fixed(z["richardson_error"].median(), 3)
    rows = [f"{r.problem} & {r.noise.replace('depolarizing ', '').replace('p2', '$p_2$').replace(' calibration', '')} & "
            f"{r.ideal:.2f} & {r.raw:.2f} & {r.richardson:.2f} \\\\" for r in z.itertuples()]
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Zero-noise extrapolation of $P_\text{{succ}}$ by global folding (scale factors 1, 3, 5) and
Richardson extrapolation, aux+rp oracle.}}
\label{{tab:zne}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}llccc@{{}}}}
\toprule
Instance & Noise & ideal & raw & ZNE \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--study-results", default=None,
                    help="directory with the study_*.csv files (default: --results)")
    ap.add_argument("--out", default=os.path.join(HERE, "generated"))
    args = ap.parse_args()
    res, sres = args.results, args.study_results or args.results
    os.makedirs(args.out, exist_ok=True)
    nums = Numbers()
    tables = {
        "tab_threshold": threshold_table(sres, nums),
        "tab_multi": multi_table(res, nums),
        "tab_device": device_table(sres, nums),
        "tab_zne": zne_table(sres, nums),
    }
    model_numbers(sres, nums)
    power_law_numbers(sres, nums)
    for name, tex in tables.items():
        open(os.path.join(args.out, f"{name}.tex"), "w", encoding="utf-8").write(tex)
    open(os.path.join(args.out, "numbers.tex"), "w", encoding="utf-8").write(
        "% Generated by make_tables.py -- do not edit.\n" + "\n".join(nums.lines) + "\n")
    print("\n".join(nums.lines))


if __name__ == "__main__":
    main()
