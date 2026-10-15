import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import csp
import csp_plots

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
NOISE_OPTIONS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2]
MAX_QUBITS = 16          # ideal simulation stays quick up to here
MAX_NOISY_QUBITS = 14    # noisy shots get slow past this on a laptop

st.set_page_config(page_title="Grover for Constraint Satisfaction", page_icon="⚛", layout="wide")
st.markdown("<style>#MainMenu { visibility: hidden; } footer { visibility: hidden; }</style>",
            unsafe_allow_html=True)


def show(fig):
    if fig is None:
        return
    st.pyplot(fig, width="stretch", clear_figure=False)
    plt.close(fig)


def fmt_p(p):
    if isinstance(p, str):
        return p
    return "0 (ideal)" if p == 0 else f"{p:g}"


# ---------------------------------------------------------------------------
# Cached computation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=64)
def cached_grover(_problem, key, iterations, shots, p2, readout, synthesis):
    tqc = csp.transpile_grover(_problem, iterations, synthesis=synthesis)
    ideal = csp.run_grover(_problem, iterations, shots=shots, tqc=tqc)
    noisy = None
    if p2 > 0 or readout > 0:
        noisy = csp.run_grover(_problem, iterations, shots=shots, p2=p2, readout=readout, tqc=tqc)
    return ideal, noisy


@st.cache_data(show_spinner=False, max_entries=64)
def cached_classical(_problem, key):
    return csp.classical_summary(_problem, walksat_runs=100)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_noise_sweep(_problem, key, iterations, shots, levels, synthesis):
    tqc = csp.transpile_grover(_problem, iterations, synthesis=synthesis)
    return [csp.run_grover(_problem, iterations, shots=shots, p2=p, tqc=tqc)["success"] for p in levels]


@st.cache_data(show_spinner=False)
def cached_bbht(N, M):
    return csp.bbht_expected_queries(N, M, trials=20000)


def load_csv(name):
    path = os.path.join(RESULTS, name)
    return pd.read_csv(path) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Sidebar: choose a problem
# ---------------------------------------------------------------------------

st.title("Grover's Algorithm for Constraint Satisfaction")
st.caption("SAT, graph colouring and Sudoku, compared with classical brute force, DPLL and WalkSAT "
           "under realistic noise")

with st.expander("How this differs from the password demo"):
    st.markdown(
        """
        In the password demo the oracle is built from the answer, so it already knows what it is
        looking for. Here the oracle only knows the **rules**. It evaluates every constraint on a
        superposition of all assignments, computing one ancilla qubit per constraint, and flips the
        phase of the assignments that satisfy all of them. Then it uncomputes the ancillas. Nobody
        has to know a solution in advance, which makes this a genuine use of Grover.

        * **Qubits** = variables + one ancilla per constraint + one phase qubit.
        * **Iterations**: the optimal count ⌊π/(4θ)⌋ with sin θ = √(M/N) needs the number of
          solutions M. The *Unknown M* tab shows the BBHT variant, which works without it.
        * **Noise**: the circuit is transpiled to IBM-style gates (rz, sx, x, cx). Depolarizing error
          p₂ is applied to CX and p₂/10 to sx and x, with optional readout error.
        * **Cost** is counted in oracle calls or assignment checks, so the classical and quantum
          numbers are in the same unit.
        """
    )

presets = csp.presets()
sb = st.sidebar
sb.header("Problem")
kind = sb.radio("Type", ["SAT", "Graph colouring", "Sudoku"], horizontal=True)
source = sb.radio("Instance", ["Preset", "Custom"], horizontal=True)

problem, err = None, None
try:
    if source == "Preset":
        prefix = {"SAT": "SAT", "Graph colouring": "Colouring", "Sudoku": "Sudoku"}[kind]
        names = [n for n in presets if n.startswith(prefix)]
        choice = sb.selectbox("Preset", names)
        problem = presets[choice]
    elif kind == "SAT":
        mode = sb.radio("Input", ["Random 3-SAT", "DIMACS / clause list"], horizontal=True)
        if mode == "Random 3-SAT":
            n = sb.slider("Variables", 3, 6, 4)
            m = sb.slider("Clauses", 1, 10, 5)
            seed = sb.number_input("Seed", 0, 10_000, 1)
            problem = csp.sat_problem(csp.random_ksat(n, m, seed=int(seed)), n,
                                      name=f"Random 3-SAT n={n} m={m} seed={seed}")
        else:
            text = sb.text_area("Clauses (one per line, 1-indexed, '-' for NOT; DIMACS also works)",
                                "1 2 -3\n-1 -2\n2 3\n-2 -3 4")
            clauses, n_decl = csp.parse_dimacs(text)
            problem = csp.sat_problem(clauses, n_decl, name="Custom SAT")
    elif kind == "Graph colouring":
        nv = sb.slider("Vertices", 2, 7, 4)
        edges_txt = sb.text_input("Edges (u-v, comma separated)", "0-1, 1-2, 2-3, 3-0")
        k = sb.select_slider("Colours", [2, 3, 4], value=2)
        edges = []
        for tok in edges_txt.split(","):
            tok = tok.strip()
            if tok:
                u, v = (int(t) for t in tok.split("-"))
                if not (0 <= u < nv and 0 <= v < nv) or u == v:
                    raise ValueError(f"Bad edge {tok} for {nv} vertices")
                edges.append((u, v))
        problem = csp.colouring_problem(nv, edges, k)
    else:
        txt = sb.text_area("4x4 grid (digits 1-4, 0 = blank)", "0234\n3012\n2143\n4321")
        grid = [[int(ch) for ch in line.strip()] for line in txt.strip().splitlines()]
        if len(grid) != 4 or any(len(r) != 4 for r in grid):
            raise ValueError("Grid must be 4 lines of 4 digits")
        problem = csp.sudoku_4x4(grid)
except Exception as exc:  # bad user input
    err = str(exc)

if err:
    st.error(f"Could not build the problem: {err}")
    st.stop()

key = (problem.name, problem.n_vars, tuple(problem.constraints))
M = len(problem.solutions())
# With M = 0 there is no optimum; run the M = 1 count so the failure is visible.
k_opt = csp.optimal_iterations(problem.N, max(M, 1))

sb.caption(f"{problem.n_vars} variables · {len(problem.constraints)} constraints · "
           f"**{problem.n_qubits} qubits**")
if problem.n_qubits > MAX_QUBITS:
    st.error(f"This instance needs {problem.n_qubits} qubits. Keep it to {MAX_QUBITS} or fewer "
             "so the simulation finishes: use fewer clauses, vertices or blanks.")
    st.stop()

sb.header("Simulation")
synthesis = sb.radio("Oracle design", list(csp.SYNTHESIS_MODES), index=2,
                     format_func=lambda m: csp.SYNTHESIS_LABELS[m],
                     help="Baseline: Qiskit's default multi-controlled gates. Ancilla-aware: borrow idle qubits. "
                          "Relative-phase: cheaper Toffolis inside compute/uncompute blocks, where the "
                          "leftover phases cancel exactly.")
auto_iter = sb.checkbox("Use optimal iteration count", value=True)
iterations = k_opt if auto_iter else sb.slider("Grover iterations", 0, max(8, 2 * k_opt + 2), k_opt)
p2 = sb.select_slider("Two-qubit error p₂", NOISE_OPTIONS, value=1e-3, format_func=fmt_p)
readout = sb.select_slider("Readout error", [0.0, 0.005, 0.01, 0.02, 0.05], value=0.0, format_func=fmt_p)
shots = sb.select_slider("Shots", [256, 512, 1024, 2048], value=1024)
if (p2 > 0 or readout > 0) and problem.n_qubits > MAX_NOISY_QUBITS:
    sb.warning(f"Noisy simulation of {problem.n_qubits} qubits can take a minute or more.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

t_problem, t_grover, t_compare, t_unknown, t_noise = st.tabs(
    ["Problem", "Grover run", "Classical vs quantum", "Unknown M (BBHT)", "Noise study"])

with t_problem:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader(problem.name)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Search space N", problem.N)
        m2.metric("Solutions M", M)
        m3.metric("Qubits", problem.n_qubits)
        m4.metric("Optimal iterations" if M else "Iterations (as if M = 1)", k_opt)
        st.markdown("**Constraints the oracle checks**")
        st.code("\n".join(c.describe(problem.labels) for c in problem.constraints), language=None)
        if M == 0:
            st.warning("This instance is unsatisfiable. Grover has nothing to amplify, so every "
                       "measurement fails the classical check. The Unknown M tab shows how BBHT behaves.")
        elif M > problem.N / 2:
            st.info("More than half of the assignments are solutions, so random guessing already wins "
                    "and Grover uses 0 iterations.")
        st.caption("M is counted by classical enumeration and used only to grade results and choose "
                   "the iteration count. The oracle never sees it.")
    with c2:
        show(csp_plots.draw_problem(problem))
        if problem.kind in ("colouring", "sudoku") and M:
            st.caption("Blue cells / coloured vertices are the variables Grover searches over.")
    if M and M <= 16:
        with st.expander(f"All {M} solutions (classical enumeration)"):
            st.dataframe(pd.DataFrame([{"assignment x0x1…": problem.bitstring(x),
                                        "decoded": str(problem.decode(x))} for x in problem.solutions()]),
                         width="stretch", hide_index=True)

with t_grover:
    with st.spinner("Simulating Grover's algorithm…"):
        ideal, noisy = cached_grover(problem, key, iterations, shots, p2, readout, synthesis)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Iterations", iterations)
    m2.metric("CX gates", ideal["cx"])
    m3.metric("Depth", ideal["depth"])
    m4.metric("Ideal success", f"{ideal['success']:.1%}", help=f"Theory: {ideal['ideal_theory']:.1%}")
    m5.metric("Noisy success", f"{noisy['success']:.1%}" if noisy else "noise off",
              delta=f"{(noisy['success'] - ideal['success']) * 100:+.1f} pts" if noisy else None)
    st.caption(f"Random guessing succeeds with probability M/N = {M / problem.N:.1%}.")

    counts_list, labels = [ideal["counts"]], [f"Ideal simulator ({shots} shots)"]
    if noisy:
        counts_list.append(noisy["counts"])
        labels.append(f"Noisy simulator, p₂={fmt_p(p2)}, readout={fmt_p(readout)}")
    show(csp_plots.plot_counts(problem, counts_list, labels, top=40))

    best_counts = noisy["counts"] if noisy else ideal["counts"]
    best = max(best_counts, key=best_counts.get)
    x_best = int(best, 2)
    ok = problem.check(x_best)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"**Most frequent outcome:** `{problem.bitstring(x_best)}`, which "
                    f"{'satisfies every constraint ✅' if ok else 'violates a constraint ❌'}")
        if problem.kind == "sat":
            st.json(problem.decode(x_best))
    with c2:
        show(csp_plots.draw_problem(problem, x_best))

    with st.expander("Circuit (one Grover iteration, oracle and diffuser as blocks)"):
        try:
            fig = csp.build_grover_circuit(problem, 1, as_gates=True, synthesis=synthesis).draw(output="mpl", fold=-1)
            show(fig)
        except Exception as exc:
            st.text(csp.build_grover_circuit(problem, 1, as_gates=True).draw(output="text"))
            st.caption(f"(mpl drawer unavailable: {exc})")
        with st.spinner("Decomposing oracle…"):
            st.text(csp.build_oracle(problem, synthesis).draw(output="text", fold=120))

with t_compare:
    with st.spinner("Running classical solvers…"):
        cl = cached_classical(problem, key)
    ideal, noisy = cached_grover(problem, key, iterations, shots, p2, readout, synthesis)
    g_ideal = csp.grover_expected_queries(iterations, ideal["ideal_theory"])
    entries = [
        ("Brute force, sequential (this instance)", cl["brute_force_sequential"], "classical"),
        ("Brute force, random order (expected)", cl["brute_force_expected"], "classical"),
        ("DPLL search nodes", cl["dpll_nodes"], "classical"),
        ("WalkSAT assignments tried (mean)", cl["walksat_mean_evals"], "classical"),
        ("Grover, ideal", g_ideal, "quantum"),
    ]
    if noisy:
        entries.append((f"Grover, noisy p₂={fmt_p(p2)}", csp.grover_expected_queries(iterations, noisy["success"]),
                        "quantum"))
    if M == 0:
        st.warning("Unsatisfiable: brute force needs all N checks to prove it. Grover cannot prove "
                   "unsatisfiability; it can only fail repeatedly.")
    else:
        show(csp_plots.plot_cost_bars(entries))
    st.markdown(
        """
        **How to read this.** Brute-force and WalkSAT costs count assignments checked against the
        constraints. Grover's cost is the expected oracle calls when you repeat
        *k iterations + 1 classical check* until a valid answer comes out, which is (k+1)/P(success).
        DPLL counts search nodes (decisions plus unit propagations). That is a different unit, and
        on instances this small clever classical search is very hard to beat.
        """
    )
    st.dataframe(pd.DataFrame([{"method": e[0], "cost": e[1]} for e in entries]),
                 hide_index=True, width="stretch")

with t_unknown:
    st.markdown(
        """
        In practice the number of solutions M is unknown. **BBHT** (Boyer, Brassard, Høyer and Tapp)
        handles this. It runs a random number of iterations j ∈ [0, m), checks the result, and grows
        m by 6/5 after each failure. Its expected cost is still O(√(N/M)).
        """
    )
    if M == 0:
        st.info("With M = 0, BBHT never succeeds. In practice you stop after about √N rounds and "
                "report the instance as *probably* unsatisfiable.")
    else:
        bb = cached_bbht(problem.N, M)
        c1, c2, c3 = st.columns(3)
        c1.metric("Grover, M known (ideal)", f"{csp.grover_expected_queries(k_opt, csp.ideal_success_probability(problem.N, M, k_opt)):.2f}")
        c2.metric("BBHT, M unknown (ideal)", f"{bb:.2f}")
        c3.metric("Brute force (expected)", f"{csp.brute_force_expected(problem.N, M):.2f}")
        Ms = sorted({1, M, min(problem.N // 4, 2 * M) or 1, min(problem.N // 4, 4 * M) or 1})
        show(csp_plots.plot_success_vs_iterations(problem.N, Ms, max_iter=max(5, 2 * csp.optimal_iterations(problem.N, 1) + 1)))

with t_noise:
    st.subheader("Noise sweep for this problem")
    st.caption("Runs the current instance at each noise level and finds where Grover stops beating brute force.")
    if M == 0 or iterations == 0:
        st.info("Needs a satisfiable instance and at least one Grover iteration.")
    elif st.button("Run noise sweep", type="primary"):
        levels = tuple(NOISE_OPTIONS)
        with st.spinner(f"Simulating {len(levels)} noise levels…"):
            succ = cached_noise_sweep(problem, key, iterations, min(shots, 512), levels, synthesis)
        classical = csp.brute_force_expected(problem.N, M)
        adv = [csp.grover_expected_queries(iterations, s) < classical for s in succ]
        show(csp_plots.plot_noise_heatmap(
            [{"label": problem.name, "success": succ, "advantage": adv}], list(levels),
            title="Noise sweep"))
        last = [p for p, a in zip(levels, adv) if a]
        if all(adv):
            st.success(f"Grover keeps its advantage at every tested level, up to p₂ = {levels[-1]:g}.")
        elif last:
            st.info(f"Advantage holds up to about p₂ = {last[-1]:g} and is gone by "
                    f"{levels[len(last)]:g}.")
        else:
            st.warning("No advantage, even without noise.")

    st.divider()
    st.subheader("Precomputed study (from experiments.py)")
    hm = load_csv("noise_heatmap.csv")
    thr = load_csv("noise_thresholds.csv")
    ms = load_csv("multi_solution.csv")
    sw = load_csv("iteration_sweep.csv")
    if hm is None:
        st.info("No saved results yet. Run `python experiments.py` (or `--quick`) to generate them.")
    else:
        import experiments
        rows, levels = experiments.heatmap_rows(hm)
        show(csp_plots.plot_noise_heatmap(rows, levels))
        if thr is not None:
            show(csp_plots.plot_threshold_vs_cx(thr.to_dict("records")))
            with st.expander("Threshold table"):
                st.dataframe(thr, hide_index=True, width="stretch")
    if sw is not None:
        noisy_pts = {M_: (list(g["iterations"]), list(g["success"])) for M_, g in sw.groupby("M")}
        N_sw = 2 ** int(sw["n_vars"].iloc[0])
        show(csp_plots.plot_success_vs_iterations(N_sw, sorted(noisy_pts), max_iter=5, noisy=noisy_pts))
    if ms is not None:
        for n in sorted(ms["n_vars"].unique()):
            show(csp_plots.plot_queries_vs_solutions(ms.to_dict("records"), int(n),
                                                     noisy_label=f"{ms['noisy_p2'].iloc[0]:g}"))
        with st.expander("Multiple-solution table"):
            st.dataframe(ms.drop(columns=["clauses_json"]), hide_index=True, width="stretch")
