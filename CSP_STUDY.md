# Simulation of Grover's Algorithm for Constraint Satisfaction Problems: A Comparative Study with Classical Search under Realistic Noise

## Summary

Grover's algorithm is implemented here for three families of constraint satisfaction problems: Boolean satisfiability (3-SAT), graph colouring and 4x4 Sudoku. The oracle evaluates the constraints on a superposition of all assignments and never sees a solution, which separates this study from the textbook "find the hidden bitstring" demonstrations. Grover is compared with classical brute force, DPLL and WalkSAT, with all costs counted in oracle calls or assignment checks. Three new contributions:

1. **A noise-threshold heatmap.** This is the success probability across 9 instances (6 to 13 qubits, 20 to 1,488 CX gates) and 7 depolarizing error rates. It also gives the error rate p₂* above which Grover stops beating brute force.
2. **Inputs with several solutions.** These cover the M-dependence of the iteration count, the cost against classical solvers, and the BBHT variant for when M is unknown.
3. **An interactive app** (`pages/1_Constraint_Satisfaction.py`). It runs any preset or custom SAT, colouring or Sudoku instance and repeats the noise sweep on it.

**Main finding.** On every instance studied, Grover's advantage over brute force disappears for two-qubit error rates between about **7×10⁻⁴ and 1×10⁻²**. This is close to or below the error rates of current superconducting hardware, which are around 10⁻³ to 10⁻². The threshold falls as the oracle gets deeper. Ideal Grover needs the fewest oracle calls, but at p₂ = 10⁻³ noisy Grover is no cheaper than WalkSAT, and on a unique-solution formula it needs more steps than DPLL.

## 1. Method

### 1.1 Encodings (`csp.py`)

Every problem is a list of constraints over n binary variables. Each constraint gets one ancilla qubit.

| Constraint | Used for | Quantum circuit | Classical CNF |
|---|---|---|---|
| `Clause` (OR of literals) | SAT | X on positive literals, MCX into the ancilla, X on the ancilla | the clause |
| `NotEqual(a, b)` | colouring edges, Sudoku peers | 1 bit: two CX. Wider: CX bitwise, OR the differences, uncompute | forbid each shared value |
| `InSet(reg, S)` | Sudoku givens, "only 3 of 4 codes" for 3-colouring | one MCX per allowed (or forbidden) value | forbid each value outside S |

* **Graph k-colouring** uses ⌈log₂k⌉ qubits per vertex.
* **4x4 Sudoku** uses 2 qubits per blank cell. The givens become one `InSet` per blank, and blank-blank peers become `NotEqual`.

**Oracle:** compute all constraint ancillas, then MCX(ancillas → output qubit in |−⟩), then uncompute. Total qubits = n + (number of constraints) + 1.

**Diffuser:** the standard inversion about the mean, applied to the n variable qubits.

**Correctness.** `tests/test_csp.py` runs every basis input through every preset oracle up to 11 qubits. It checks that the oracle flips exactly the solutions and returns every ancilla to |0⟩. The same exhaustive check also passes on the 13-qubit presets when run by hand.

### 1.2 Noise model

* Circuits are transpiled to the IBM-style basis {rz, sx, x, cx} with `optimization_level=1`.
* Depolarizing error is p₂ on `cx` and p₂/10 on `sx` and `x`. `rz` is virtual and error-free.
* The app also offers a symmetric readout error.
* The simulator is Qiskit Aer (statevector with noise sampling), with 512 shots per heatmap cell.
* All-to-all connectivity is assumed: no SWAP routing for a real coupling map (see Limitations).

### 1.3 Cost model

| Method | Cost counted |
|---|---|
| Brute force | assignments checked. Expected value for random order: (N+1)/(M+1) |
| DPLL (unit propagation, pure literals, most-frequent-variable branching) | search nodes = decisions + propagations |
| WalkSAT (p = 0.5, 100 restarts) | assignments evaluated = 1 + flips |
| Grover, M known | k = ⌊π/(4θ)⌋, sin θ = √(M/N). Repeat until a measured answer passes the classical check: (k+1)/P(success) |
| BBHT, M unknown | Monte Carlo (20,000 trials) of Boyer–Brassard–Høyer–Tapp with λ = 6/5, counting j iterations + 1 check per round |

The **advantage threshold** p₂* is the largest p₂ at which Grover's expected cost is still below brute force. It is interpolated in log–log between the grid points.

## 2. Results

Run `python experiments.py` to regenerate every number below. The raw data is in `results/` and the figures in `figures/csp_*.png`. The app's *Noise study* tab shows the same figures.

### 2.1 Noise-threshold heatmap

P(measured assignment is a solution), 512 shots per cell:

| Instance | Qubits | CX | M | ideal | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | **p₂\*** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2-colouring P3 | 6 | 20 | 2 | 1.00 | 1.00 | .99 | .98 | .96 | .89 | .68 | > 3e-2 |
| 2-colouring P4 | 8 | 80 | 2 | .94 | .94 | .93 | .91 | .81 | .54 | .26 | 1.0e-2 |
| 2-colouring P5 | 10 | 204 | 2 | .96 | .95 | .93 | .83 | .61 | .25 | .07 | 6.0e-3 |
| 2-colouring P6 | 12 | 352 | 2 | 1.00 | .98 | .92 | .76 | .43 | .09 | .04 | 4.8e-3 |
| 4x4 Sudoku, 2 blanks | 8 | 204 | 1 | .96 | .96 | .92 | .80 | .60 | .26 | .09 | 4.2e-3 |
| 4x4 Sudoku, 3 blanks | 11 | 672 | 1 | 1.00 | .94 | .84 | .58 | .22 | .02 | .01 | 3.0e-3 |
| 4x4 Sudoku, 4 blanks | 13 | 1488 | 1 | 1.00 | .91 | .71 | .35 | .04 | .00 | .00 | 1.8e-3 |
| 3-SAT n=3, m=7 | 11 | 522 | 1 | .95 | .90 | .84 | .62 | .32 | .11 | .10 | 7.4e-4 |
| 3-SAT n=4, m=8 | 13 | 932 | 1 | .96 | .90 | .74 | .44 | .13 | .04 | .04 | 8.5e-4 |

**Observations:**

* **Deeper oracles break first.** Within each family, p₂* falls as the CX count grows. Sudoku goes from 4.2e-3 to 3.0e-3 to 1.8e-3 as blanks are added. 1/(CX count), the error rate at which the circuit expects about one two-qubit error, is a useful rough guide: every measured threshold is within about 3× of it.
* **3-SAT is the most fragile family.** Each 3-literal clause costs a 3-control MCX, and m ≈ 2n clauses are needed to pin down a single solution. That makes a deep oracle relative to the size of the search space, so the threshold is below 10⁻³ even for n = 3.
* **The advantage is relative.** The threshold compares with brute force, whose cost grows with N, so a larger instance can tolerate slightly more noise per gate than a smaller one of similar depth. This is why 3-SAT n=4 (8.5e-4) sits just above n=3 (7.4e-4).
* **Noise pulls success toward the M/N floor.** At p₂ = 3×10⁻², most instances fall to roughly what random guessing gives (M/N = 1/8 for 3-SAT n=3, 1/64 for Sudoku with 3 blanks).

### 2.2 Inputs with several solutions

Random 3-SAT instances were generated with exactly M solutions, 3 instances per (n, M); the classical columns are averaged. The noisy column uses p₂ = 10⁻³.

| n | M | k | brute force | DPLL nodes | WalkSAT evals | Grover ideal | BBHT ideal | Grover noisy |
|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 3 | 8.5 | 4.7 | 9.2 | 4.2 | 5.8 | 8.2 |
| 4 | 2 | 2 | 5.7 | 6.7 | 5.5 | 3.2 | 3.9 | 4.9 |
| 4 | 4 | 1 | 3.4 | 3.3 | 2.3 | 2.0 | 2.9 | 2.3 |
| 4 | 8 | 0 | 1.9 | 3.0 | 1.7 | 2.0 | 2.6 | 2.0 |
| 5 | 1 | 4 | 16.5 | 11.3 | 15.8 | 5.0 | 8.6 | (17 qubits, not simulated) |
| 5 | 2 | 3 | 11.0 | 5.3 | 9.1 | 4.2 | 5.8 | (16 qubits, not simulated) |
| 5 | 4 | 2 | 6.6 | 8.0 | 5.7 | 3.2 | 3.9 | 5.4 |
| 5 | 8 | 1 | 3.7 | 5.0 | 2.5 | 2.0 | 2.9 | 2.4 |

**Observations:**

* **More solutions mean fewer iterations**, roughly (π/4)√(N/M), so the circuit is shallower and noise does less damage. For n=4, ideal-to-noisy success drops from 0.96 → 0.49 at M=1 and only from 1.00 → 0.88 at M=4.
* **Not knowing M costs about 1.2–1.7×.** BBHT is about 1.2–1.7× the cost of the known-M optimum, and still beats brute force for every instance with M ≤ N/4.
* **Near M = N/2, Grover's edge is gone.** The optimal k drops to 0, so Grover is just random sampling. At n=4, M=8 it costs 2.0 against 1.9 for brute force.
* **DPLL is very hard to beat at this scale.** Its node count is not the same unit as an oracle call, but unit propagation alone solves many of these tiny instances in a few steps. A quadratic speedup over brute force is not a speedup over the best classical solver.

### 2.3 Figures

| File | Content |
|---|---|
| `figures/csp_noise_heatmap.png` | heatmap from §2.1, with the advantaged cells outlined |
| `figures/csp_threshold_vs_cx.png` | p₂* against CX count, per family |
| `figures/csp_success_vs_iterations.png` | P(success) against k for M = 1, 2, 4 (N = 16): theory plus noisy simulation |
| `figures/csp_queries_vs_solutions_n4.png`, `_n5.png` | cost against M for every method |

## 3. Limitations

* **No qubit routing.** Mapping onto a real heavy-hex coupling map would add SWAPs, multiplying the CX count by roughly 2–3×. The thresholds above are therefore optimistic.
* **Noise is depolarizing only.** There is no T1/T2 relaxation, crosstalk or leakage.
* **Iteration counts use M** from classical enumeration. BBHT (§2.2) removes this assumption at a constant-factor cost.
* **Instances are small** (N ≤ 256) because noisy simulation slows sharply beyond about 15 qubits. Asymptotic claims rest on the formulas, not on these runs.
* **DPLL and WalkSAT costs use different units.** Their costs are not oracle calls, so read them for trends, not as exact ratios.

## 4. Reproducing

```bash
pip install -r requirements.txt
python tests/test_csp.py           # exhaustive oracle checks + solver tests (~1 min)
python experiments.py --quick      # smoke run (~1-2 min)
python experiments.py              # full study (~30 min on a 4-core laptop)
streamlit run app.py               # then open the "Constraint Satisfaction" page
```

## 5. Related work (base papers)

These are the base papers listed for this study. Their identifiers are reproduced as given and have not been re-checked here.

* Constraint satisfiability with Grover's algorithm: Qiskit, IBM hardware, and a custom noise model used to find the noise threshold. arXiv:2106.09976. *This is the closest match to the approach here.*
* Grover's algorithm for graph colouring, using Qiskit with the FakeVigo noise model. arXiv:2312.00832.
* Benchmarking Grover's algorithm for 3-, 4- and 5-SAT (thesis).
* Magic-square constraints with classical benchmarking. arXiv:2604.04786.
* Boyer, Brassard, Høyer and Tapp, *Tight bounds on quantum searching* (1998). This is the source of the unknown-M algorithm used in §2.2.
