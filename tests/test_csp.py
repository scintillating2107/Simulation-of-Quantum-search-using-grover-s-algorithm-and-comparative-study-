"""Correctness tests for csp.py.  Run:  python -m pytest tests   (or  python tests/test_csp.py)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csp  # noqa: E402


def small_presets():
    return {k: p for k, p in csp.presets().items() if p.n_qubits <= 11}


def test_oracle_marks_exactly_the_solutions():
    # Full unitary check: phase kickback on solutions only, ancillas restored.
    for name, p in small_presets().items():
        assert csp.oracle_statevector_check(p), name


def test_every_oracle_design_is_exact():
    # Relative-phase Toffolis are only correct up to phases that must cancel; check that they do.
    for name, p in small_presets().items():
        for mode in csp.SYNTHESIS_MODES:
            assert csp.oracle_statevector_check(p, csp.build_oracle(p, mode)), (name, mode)


def test_cheaper_designs_use_fewer_cx():
    p = csp.presets()["SAT · 3 vars, 7 clauses (1 solution)"]
    cx = {m: csp.circuit_stats(csp.transpile_grover(p, 2, synthesis=m))["cx"] for m in csp.SYNTHESIS_MODES}
    assert cx["aux+rp"] < cx["aux"] <= cx["baseline"]


def test_folding_preserves_ideal_output():
    p = csp.presets()["Sudoku · 2x2 binary"]
    t = csp.transpile_grover(p, 2)
    base = csp.run_grover(p, 2, shots=3000, tqc=t, seed=2)["success"]
    folded = csp.fold_circuit(t, 3)
    assert csp.circuit_stats(folded)["cx"] == 3 * csp.circuit_stats(t)["cx"]
    assert abs(csp.run_grover(p, 2, shots=3000, tqc=folded, seed=2)["success"] - base) < 0.03


def test_richardson_and_wilson():
    assert abs(csp.richardson_extrapolate([1, 3, 5], [0.9, 0.7, 0.5]) - 1.0) < 1e-9
    lo, hi = csp.wilson_interval(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.2


def test_random_generators():
    for s in range(5):
        edges = csp.random_tree(6, seed=s)
        assert len(edges) == 5
        assert len(csp.colouring_problem(6, edges, 2).solutions()) == 2
        p = csp.random_sudoku_4x4_puzzle(3, seed=s)
        assert p is not None and len(p.solutions()) == 1


def test_checker_rejects_a_wrong_oracle():
    # The checker must be able to fail: grade problem a against problem b's oracle.
    a = csp.sat_problem([[1, 2], [-1, 2]], 2)      # solutions: x1 = 1
    b = csp.sat_problem([[1, 2], [1, -2]], 2)      # solutions: x0 = 1
    assert not csp.oracle_statevector_check(a, oracle=csp.build_oracle(b))


def test_known_solution_counts():
    P = csp.presets()
    assert len(P["Sudoku · 2x2 binary"].solutions()) == 2
    assert len(P["Colouring · triangle, 3 colours"].solutions()) == 6   # 3! colourings
    assert len(P["Colouring · square C4, 2 colours"].solutions()) == 2
    assert len(P["Colouring · pentagon C5, 2 colours (unsatisfiable)"].solutions()) == 0
    assert len(P["Sudoku · 4x4, 3 blanks"].solutions()) == 1
    assert len(P["SAT · 4 vars, 6 clauses (4 solutions)"].solutions()) == 4


def test_cnf_encoding_matches_constraints():
    for name, p in csp.presets().items():
        cnf = p.to_cnf()
        for x in range(p.N):
            cnf_ok = all(any(((x >> v) & 1) != neg for v, neg in c) for c in cnf)
            assert cnf_ok == p.check(x), (name, x)


def test_classical_solvers_agree_with_enumeration():
    for name, p in csp.presets().items():
        M = len(p.solutions())
        x, _ = csp.dpll(p.to_cnf(), p.n_vars)
        assert (x is not None) == (M > 0), name
        if x is not None:
            assert p.check(x), name
        xb, _ = csp.brute_force(p)
        assert (xb is not None) == (M > 0), name
        if M:
            xw, _ = csp.walksat(p.to_cnf(), p.n_vars, seed=1)
            assert xw is None or p.check(xw), name


def test_ideal_grover_matches_theory():
    p = csp.presets()["Sudoku · 4x4, 2 blanks"]
    r = csp.run_grover(p, shots=4000, seed=3)
    assert abs(r["success"] - r["ideal_theory"]) < 0.03
    assert r["ideal_theory"] > 0.9


def test_noise_lowers_success():
    p = csp.presets()["Colouring · square C4, 2 colours"]
    ideal = csp.run_grover(p, shots=2000, seed=1)["success"]
    noisy = csp.run_grover(p, shots=2000, seed=1, p2=0.03)["success"]
    assert noisy < ideal - 0.1


def test_parse_dimacs_formats():
    clauses, n = csp.parse_dimacs("c comment\np cnf 3 2\n1 -2 0\n2 3 0\n")
    assert clauses == [[1, -2], [2, 3]] and n == 3
    clauses, n = csp.parse_dimacs("1 -2\n2 3")
    assert clauses == [[1, -2], [2, 3]] and n is None
    clauses, _ = csp.parse_dimacs("1 -2 0 2 3 0")
    assert clauses == [[1, -2], [2, 3]]


def test_tautological_clause_is_dropped():
    p = csp.sat_problem([[1, -1, 2], [2, 3]], 3)
    assert len(p.constraints) == 1


def test_optimal_iterations():
    assert csp.optimal_iterations(16, 1) == 3
    assert csp.optimal_iterations(64, 1) == 6
    assert csp.optimal_iterations(16, 4) == 1
    assert csp.optimal_iterations(16, 0) == 0


def test_bbht_close_to_theory():
    # BBHT costs more than known-M Grover but less than brute force for large enough N.
    N, M = 1024, 1
    known = csp.grover_expected_queries(csp.optimal_iterations(N, M),
                                        csp.ideal_success_probability(N, M, csp.optimal_iterations(N, M)))
    bbht = csp.bbht_expected_queries(N, M, trials=3000)
    assert known < bbht < csp.brute_force_expected(N, M)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
