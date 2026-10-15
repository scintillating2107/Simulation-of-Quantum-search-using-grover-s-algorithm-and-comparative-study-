"""Grover's algorithm for constraint satisfaction problems (SAT, graph colouring, Sudoku).

Unlike the password demo in grovers.py, the oracle here never sees the answer.
It evaluates the constraints on a superposition of assignments and flips the
phase of every assignment that satisfies all of them:

    compute each constraint into its own ancilla  ->  MCX(all ancillas -> |->)
    ->  uncompute the ancillas

Classical baselines (brute force, DPLL, WalkSAT) run on the same problems, and
everything is measured in oracle queries / assignment checks so the numbers
are comparable.

Bit convention: an assignment is an int ``x`` whose bit ``i`` is variable ``i``.
Variable ``i`` lives on qubit ``i`` and is measured into clbit ``i``, so
``int(counts_key, 2) == x``.
"""

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

# IBM-style native gate set. rz is virtual (error-free) on IBM hardware.
BASIS_GATES = ["rz", "sx", "x", "cx"]


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def _bits_of(x, reg):
    """Value of the register ``reg`` (list of var indices, LSB first) in assignment x."""
    return sum(((x >> v) & 1) << i for i, v in enumerate(reg))


@dataclass(frozen=True)
class Clause:
    """OR of literals. Literal = (var, negated)."""
    literals: tuple

    def satisfied(self, x):
        return any(((x >> v) & 1) != neg for v, neg in self.literals)

    def to_cnf(self):
        return [list(self.literals)]

    def compute(self, qc, var, anc, mcx=None):
        mcx = mcx or qc.mcx
        # Make each qubit 1 exactly when its literal is false, AND them, then negate.
        pos = [v for v, neg in self.literals if not neg]
        for v in pos:
            qc.x(var[v])
        mcx([var[v] for v, _ in self.literals], anc)
        qc.x(anc)
        for v in pos:
            qc.x(var[v])

    def describe(self, labels):
        return " ∨ ".join(("¬" if n else "") + labels[v] for v, n in self.literals)


@dataclass(frozen=True)
class NotEqual:
    """Two equal-width registers must hold different values (colouring / Sudoku)."""
    a: tuple
    b: tuple

    def satisfied(self, x):
        return _bits_of(x, self.a) != _bits_of(x, self.b)

    def to_cnf(self):
        # Forbid every value the two registers could share. No auxiliary variables.
        clauses = []
        for val in range(2 ** len(self.a)):
            clause = []
            for reg in (self.a, self.b):
                for i, v in enumerate(reg):
                    clause.append((v, bool((val >> i) & 1)))
            clauses.append(clause)
        return clauses

    def compute(self, qc, var, anc, mcx=None):
        mcx = mcx or qc.mcx
        if len(self.a) == 1:
            qc.cx(var[self.a[0]], anc)
            qc.cx(var[self.b[0]], anc)
            return
        # b_i <- a_i XOR b_i, then anc = OR(b_i), then restore b.
        for va, vb in zip(self.a, self.b):
            qc.cx(var[va], var[vb])
        diff = [var[v] for v in self.b]
        qc.x(diff)
        mcx(diff, anc)
        qc.x(diff)
        qc.x(anc)
        for va, vb in zip(self.a, self.b):
            qc.cx(var[va], var[vb])

    def describe(self, labels):
        return f"{_reg_label(self.a, labels)} ≠ {_reg_label(self.b, labels)}"


@dataclass(frozen=True)
class InSet:
    """A register's value must be one of ``allowed`` (e.g. Sudoku givens, k=3 colours)."""
    reg: tuple
    allowed: frozenset

    def satisfied(self, x):
        return _bits_of(x, self.reg) in self.allowed

    def to_cnf(self):
        clauses = []
        for val in range(2 ** len(self.reg)):
            if val not in self.allowed:
                clauses.append([(v, bool((val >> i) & 1)) for i, v in enumerate(self.reg)])
        return clauses

    def compute(self, qc, var, anc, mcx=None):
        mcx = mcx or qc.mcx
        size = 2 ** len(self.reg)
        values = sorted(self.allowed)
        flip = len(values) > size // 2
        if flip:
            values = [v for v in range(size) if v not in self.allowed]
        qubits = [var[v] for v in self.reg]
        # The value indicators are mutually exclusive, so XOR-ing them into anc is an OR.
        for val in values:
            zeros = [q for i, q in enumerate(qubits) if not (val >> i) & 1]
            if zeros:
                qc.x(zeros)
            mcx(qubits, anc)
            if zeros:
                qc.x(zeros)
        if flip:
            qc.x(anc)

    def describe(self, labels):
        return f"{_reg_label(self.reg, labels)} ∈ {sorted(self.allowed)}"


def _reg_label(reg, labels):
    if len(reg) == 1:
        return labels[reg[0]]
    name = labels[reg[0]].rsplit(".", 1)[0]
    return name


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    name: str
    kind: str                 # "sat" | "colouring" | "sudoku"
    n_vars: int
    constraints: list
    labels: list = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.labels is None:
            self.labels = [f"x{i}" for i in range(self.n_vars)]
        if not self.constraints:
            raise ValueError("A problem needs at least one constraint.")

    @property
    def N(self):
        return 2 ** self.n_vars

    @property
    def n_qubits(self):
        return self.n_vars + len(self.constraints) + 1

    def check(self, x):
        return all(c.satisfied(x) for c in self.constraints)

    def solutions(self):
        return _solutions_cached(self)

    def to_cnf(self):
        cnf = []
        for c in self.constraints:
            cnf.extend(c.to_cnf())
        return cnf

    def decode(self, x):
        """Human-readable view of an assignment."""
        if self.kind == "colouring":
            b = self.meta["bits"]
            names = self.meta.get("colour_names", COLOUR_NAMES)
            return {f"v{v}": names[_bits_of(x, range(v * b, v * b + b))]
                    for v in range(self.meta["n_vertices"])}
        if self.kind == "sudoku":
            return sudoku_grid(self, x)
        return {self.labels[i]: (x >> i) & 1 for i in range(self.n_vars)}

    def bitstring(self, x):
        """Assignment written x0 x1 x2 ... (variable order, left to right)."""
        return "".join(str((x >> i) & 1) for i in range(self.n_vars))

    def __hash__(self):
        return hash((self.name, self.n_vars, tuple(self.constraints)))

    def __eq__(self, other):
        return (isinstance(other, Problem) and self.n_vars == other.n_vars
                and self.constraints == other.constraints)


@lru_cache(maxsize=256)
def _solutions_cached(problem):
    return tuple(x for x in range(problem.N) if problem.check(x))


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------

COLOUR_NAMES = ["red", "green", "blue", "yellow"]


def sat_problem(clauses, n_vars=None, name="SAT"):
    """Build from DIMACS-style clauses, e.g. [[1, -2, 3], [-1, 2]] (1-indexed)."""
    if n_vars is None:
        n_vars = max(abs(l) for c in clauses for l in c)
    constraints = []
    for c in clauses:
        lits = sorted({(abs(l) - 1, l < 0) for l in c})
        vars_ = [v for v, _ in lits]
        if len(set(vars_)) < len(vars_):
            continue  # contains x and ¬x: always true, needs no ancilla
        constraints.append(Clause(tuple(lits)))
    return Problem(name, "sat", n_vars, constraints,
                   meta={"clauses": [list(c) for c in clauses]})


def parse_dimacs(text):
    """Parse DIMACS CNF (clauses ended by 0), or one clause per line of ints.

    Returns (clauses, n_vars or None).
    """
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l and not l.startswith(("c", "%"))]
    n_vars = None
    body = []
    for l in lines:
        if l.startswith("p"):
            n_vars = int(l.split()[2])
        else:
            body.append([int(t) for t in l.replace(",", " ").split()])
    dimacs = n_vars is not None or any(row and row[-1] == 0 for row in body)
    clauses = []
    if dimacs:
        current = []
        for lit in (t for row in body for t in row):
            if lit == 0:
                if current:
                    clauses.append(current)
                current = []
            else:
                current.append(lit)
        if current:
            clauses.append(current)
    else:
        clauses = [row for row in body if row]
    if not clauses:
        raise ValueError("No clauses found.")
    return clauses, n_vars


def random_ksat(n_vars, n_clauses, k=3, seed=None):
    rng = random.Random(seed)
    clauses = []
    for _ in range(n_clauses):
        vars_ = rng.sample(range(1, n_vars + 1), k)
        clauses.append([v if rng.random() < 0.5 else -v for v in vars_])
    return clauses


def random_ksat_with_solutions(n_vars, n_clauses, n_solutions, k=3, seed=0, max_tries=20000):
    """Sample random k-SAT instances until one has exactly ``n_solutions`` models."""
    rng = random.Random(seed)
    for _ in range(max_tries):
        clauses = random_ksat(n_vars, n_clauses, k, seed=rng.random())
        p = sat_problem(clauses, n_vars, name=f"3-SAT n={n_vars} m={n_clauses}")
        if len(p.constraints) == n_clauses and len(p.solutions()) == n_solutions:
            return p
    return None


def colouring_problem(n_vertices, edges, k=3, name=None):
    """Graph k-colouring, k in {2, 3, 4}. Each vertex gets ceil(log2 k) qubits."""
    if k not in (2, 3, 4):
        raise ValueError("k must be 2, 3 or 4")
    b = 1 if k == 2 else 2
    regs = [tuple(range(v * b, v * b + b)) for v in range(n_vertices)]
    constraints = [NotEqual(regs[u], regs[v]) for u, v in edges]
    if k == 3:
        constraints += [InSet(regs[v], frozenset({0, 1, 2})) for v in range(n_vertices)]
    labels = [f"v{v}.b{i}" if b > 1 else f"v{v}" for v in range(n_vertices) for i in range(b)]
    return Problem(name or f"{k}-colouring ({n_vertices} vertices, {len(edges)} edges)",
                   "colouring", n_vertices * b, constraints, labels,
                   meta={"k": k, "bits": b, "n_vertices": n_vertices, "edges": list(edges)})


def binary_sudoku_2x2():
    """The 2x2 binary Sudoku: no row or column may repeat a bit. Two solutions."""
    # Cells v0 v1 / v2 v3
    cons = [NotEqual((0,), (1,)), NotEqual((2,), (3,)),
            NotEqual((0,), (2,)), NotEqual((1,), (3,))]
    return Problem("2x2 binary Sudoku", "sudoku", 4, cons, ["c00", "c01", "c10", "c11"],
                   meta={"size": 2, "binary": True, "grid": [[None, None], [None, None]],
                         "blanks": [(0, 0), (0, 1), (1, 0), (1, 1)]})


def _peers4(r, c):
    peers = set()
    for i in range(4):
        peers.add((r, i))
        peers.add((i, c))
    br, bc = 2 * (r // 2), 2 * (c // 2)
    for i in range(br, br + 2):
        for j in range(bc, bc + 2):
            peers.add((i, j))
    peers.discard((r, c))
    return peers


def sudoku_4x4(grid):
    """4x4 Sudoku. ``grid`` is 4 lists of 4, digits 1-4 or 0/None for blank.

    Each blank uses 2 qubits (digit-1 in binary). Givens become one InSet
    constraint per blank cell; blank-blank peers become NotEqual constraints.
    """
    grid = [[(g or 0) for g in row] for row in grid]
    blanks = [(r, c) for r in range(4) for c in range(4) if grid[r][c] == 0]
    if not blanks:
        raise ValueError("Grid has no blank cells.")
    index = {cell: i for i, cell in enumerate(blanks)}
    regs = {cell: (2 * i, 2 * i + 1) for cell, i in index.items()}
    constraints = []
    for cell in blanks:
        used = {grid[r][c] - 1 for r, c in _peers4(*cell) if grid[r][c]}
        allowed = frozenset(set(range(4)) - used)
        if not allowed:
            raise ValueError(f"Cell {cell} has no legal digit.")
        if len(allowed) < 4:
            constraints.append(InSet(regs[cell], allowed))
    for i, a in enumerate(blanks):
        for b in blanks[i + 1:]:
            if b in _peers4(*a):
                constraints.append(NotEqual(regs[a], regs[b]))
    labels = [f"r{r}c{c}.b{i}" for (r, c) in blanks for i in range(2)]
    return Problem(f"4x4 Sudoku ({len(blanks)} blanks)", "sudoku", 2 * len(blanks),
                   constraints, labels,
                   meta={"size": 4, "binary": False, "grid": grid, "blanks": blanks})


def sudoku_grid(problem, x):
    meta = problem.meta
    grid = [list(row) for row in meta["grid"]]
    for i, (r, c) in enumerate(meta["blanks"]):
        if meta["binary"]:
            grid[r][c] = (x >> i) & 1
        else:
            grid[r][c] = _bits_of(x, (2 * i, 2 * i + 1)) + 1
    return grid


SUDOKU_4x4_SOLUTION = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]


def sudoku_4x4_with_blanks(cells):
    grid = [row[:] for row in SUDOKU_4x4_SOLUTION]
    for r, c in cells:
        grid[r][c] = 0
    return sudoku_4x4(grid)


def random_tree(n_vertices, seed=0):
    """Edges of a uniformly random labelled tree (Prufer sequence). Always 2-colourable, M = 2."""
    rng = random.Random(seed)
    if n_vertices == 2:
        return [(0, 1)]
    prufer = [rng.randrange(n_vertices) for _ in range(n_vertices - 2)]
    degree = [1] * n_vertices
    for v in prufer:
        degree[v] += 1
    edges = []
    for v in prufer:
        leaf = min(i for i in range(n_vertices) if degree[i] == 1)
        edges.append((leaf, v))
        degree[leaf] -= 1
        degree[v] -= 1
    u, w = [i for i in range(n_vertices) if degree[i] == 1]
    edges.append((u, w))
    return edges


def random_sudoku_4x4_grid(seed=0):
    """A random valid 4x4 Sudoku solution (relabel digits, shuffle rows/cols within bands, transpose)."""
    rng = random.Random(seed)
    g = [row[:] for row in SUDOKU_4x4_SOLUTION]
    perm = list(range(1, 5))
    rng.shuffle(perm)
    g = [[perm[v - 1] for v in row] for row in g]
    bands = [[0, 1], [2, 3]]
    rng.shuffle(bands)
    rows = [r for band in bands for r in rng.sample(band, 2)]
    g = [g[r] for r in rows]
    stacks = [[0, 1], [2, 3]]
    rng.shuffle(stacks)
    cols = [c for st in stacks for c in rng.sample(st, 2)]
    g = [[row[c] for c in cols] for row in g]
    if rng.random() < 0.5:
        g = [list(r) for r in zip(*g)]
    return g


def random_sudoku_4x4_puzzle(n_blanks, seed=0, unique=True, max_tries=500, max_qubits=None):
    """Random 4x4 puzzle with ``n_blanks`` blanks (and a unique solution if asked)."""
    rng = random.Random(seed)
    cells = [(r, c) for r in range(4) for c in range(4)]
    for _ in range(max_tries):
        grid = random_sudoku_4x4_grid(rng.random())
        for r, c in rng.sample(cells, n_blanks):
            grid[r][c] = 0
        try:
            p = sudoku_4x4(grid)
        except ValueError:
            continue
        if max_qubits is not None and p.n_qubits > max_qubits:
            continue
        if not unique or len(p.solutions()) == 1:
            return p
    return None


def presets():
    """Small instances that simulate in seconds. name -> Problem."""
    return {
        "SAT · 3 vars, 7 clauses (1 solution)": sat_problem(
            [[-2, -1, 3], [2, 1, 3], [1, -3, 2], [-1, 3, 2], [3, -2, 1], [-2, -1, -3], [-1, 2, -3]],
            name="3-SAT n=3 m=7"),
        "SAT · 4 vars, 8 clauses (1 solution)": sat_problem(
            [[2, -3, 4], [-4, -3, -1], [3, 1, -4], [1, 4, 3], [-3, 1, -4], [3, 4, -1], [3, -4, -2], [4, -2, -3]],
            name="3-SAT n=4 m=8"),
        "SAT · 4 vars, 6 clauses (4 solutions)": sat_problem(
            [[2, -4, 3], [3, 2, 4], [-4, -3, 2], [-1, 3, -2], [4, 2, -3], [1, -2, 3]],
            name="3-SAT n=4 m=6"),
        "SAT · 5 vars, 6 clauses (8 solutions)": sat_problem(
            [[-1, -5, 3], [3, 5, -1], [-3, -1, -4], [-1, -3, 4], [1, 2, -5], [1, 3, 5]],
            name="3-SAT n=5 m=6"),
        "Colouring · square C4, 2 colours": colouring_problem(
            4, [(0, 1), (1, 2), (2, 3), (3, 0)], k=2, name="2-colouring of C4"),
        "Colouring · path P5, 2 colours": colouring_problem(
            5, [(0, 1), (1, 2), (2, 3), (3, 4)], k=2, name="2-colouring of P5"),
        "Colouring · triangle, 3 colours": colouring_problem(
            3, [(0, 1), (1, 2), (2, 0)], k=3, name="3-colouring of K3"),
        "Colouring · pentagon C5, 2 colours (unsatisfiable)": colouring_problem(
            5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], k=2, name="2-colouring of C5"),
        "Sudoku · 2x2 binary": binary_sudoku_2x2(),
        "Sudoku · 4x4, 2 blanks": sudoku_4x4_with_blanks([(0, 0), (1, 1)]),
        "Sudoku · 4x4, 3 blanks": sudoku_4x4_with_blanks([(0, 0), (0, 1), (2, 2)]),
    }


# ---------------------------------------------------------------------------
# Grover circuit
# ---------------------------------------------------------------------------

def optimal_iterations(N, M):
    """floor(pi / (4 theta)) with sin(theta) = sqrt(M/N); 0 when there is nothing to amplify."""
    if M <= 0 or M >= N:
        return 0
    theta = math.asin(math.sqrt(M / N))
    return max(0, int(math.floor(math.pi / (4 * theta))))


def ideal_success_probability(N, M, k):
    if M <= 0:
        return 0.0
    theta = math.asin(math.sqrt(M / N))
    return math.sin((2 * k + 1) * theta) ** 2


SYNTHESIS_MODES = ("baseline", "aux", "aux+rp")
SYNTHESIS_LABELS = {
    "baseline": "Baseline (ancilla-free MCX)",
    "aux": "Ancilla-aware MCX",
    "aux+rp": "Ancilla-aware + relative-phase compute",
}


def _mcx_factory(qc, mode, clean=(), relative_phase=False):
    """Return mcx(controls, target) for ``qc``.

    baseline: Qiskit's default ancilla-free decomposition.
    aux:      borrow idle qubits. With >= k-2 clean qubits (known |0>) use the
              Maslov v-chain; with one, the Khattar-Gidney 1-clean-ancilla MCX
              (6k-6 CX); otherwise a dirty qubit (any state, restored) for k > 4.
    relative_phase (aux+rp, compute blocks only): use relative-phase Toffolis
              (rccx: 3 CX, rcccx: 6 CX). They are correct only up to a diagonal
              phase, which cancels because every compute block is later undone by
              its exact inverse and the phase kick in between is diagonal too.
    """
    if mode == "baseline":
        return qc.mcx
    from qiskit.synthesis import (synth_mcx_1_clean_kg24, synth_mcx_1_dirty_kg24,
                                  synth_mcx_n_clean_m15)

    def mcx(controls, target):
        controls = list(controls)
        k = len(controls)
        if relative_phase and k == 2:
            qc.rccx(controls[0], controls[1], target)
            return
        if relative_phase and k == 3:
            qc.rcccx(controls[0], controls[1], controls[2], target)
            return
        if k <= 2:
            qc.mcx(controls, target)
            return
        busy = set(controls) | {target}
        free_clean = [q for q in clean if q not in busy]
        if k >= 4 and len(free_clean) >= k - 2:
            qc.compose(synth_mcx_n_clean_m15(k), qubits=controls + [target] + free_clean[:k - 2],
                       inplace=True)
            return
        if free_clean:
            qc.compose(synth_mcx_1_clean_kg24(k), qubits=controls + [target, free_clean[0]], inplace=True)
            return
        dirty = [q for q in qc.qubits if q not in busy]
        if dirty and k > 4:
            qc.compose(synth_mcx_1_dirty_kg24(k), qubits=controls + [target, dirty[0]], inplace=True)
            return
        qc.mcx(controls, target)

    return mcx


def _registers(problem):
    return (QuantumRegister(problem.n_vars, "x"),
            QuantumRegister(len(problem.constraints), "c"),
            QuantumRegister(1, "out"))


def build_oracle(problem, synthesis="baseline"):
    """Phase oracle on (var, anc, out). out must hold |->; ancillas return to |0>."""
    var, anc, out = _registers(problem)
    compute = QuantumCircuit(var, anc, out)
    for i, (c, a) in enumerate(zip(problem.constraints, anc)):
        # Ancillas of later constraints are still |0> here (and again when this one is uncomputed).
        c.compute(compute, var, a, _mcx_factory(compute, synthesis, clean=list(anc[i + 1:]),
                                                relative_phase=synthesis == "aux+rp"))
    oracle = QuantumCircuit(var, anc, out, name="Oracle")
    oracle.compose(compute, inplace=True)
    _mcx_factory(oracle, synthesis)(list(anc), out[0])
    oracle.compose(compute.inverse(), inplace=True)
    return oracle


def build_diffuser(n, synthesis="baseline", n_clean=0):
    """Inversion about the mean on qubits 0..n-1; qubits n..n+n_clean-1 are clean helpers."""
    qc = QuantumCircuit(n + n_clean, name="Diffusion")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        mcx = _mcx_factory(qc, synthesis, clean=[qc.qubits[i] for i in range(n, n + n_clean)])
        mcx([qc.qubits[i] for i in range(n - 1)], qc.qubits[n - 1])
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(problem, iterations, measure=True, as_gates=False, synthesis="baseline"):
    var, anc, out = _registers(problem)
    qc = QuantumCircuit(var, anc, out)
    if measure:
        qc.add_register(ClassicalRegister(problem.n_vars, "m"))
    qc.h(var)
    qc.x(out)
    qc.h(out)
    oracle = build_oracle(problem, synthesis)
    # Between oracle calls every constraint ancilla is |0>: the diffuser may use them.
    diffuser = build_diffuser(problem.n_vars, synthesis, n_clean=len(anc))
    diff_qubits = list(var) + list(anc)
    if as_gates:
        oracle, diffuser = oracle.to_gate(label="Oracle"), diffuser.to_gate(label="Diffusion")
    for _ in range(iterations):
        if as_gates:
            qc.append(oracle, list(var) + list(anc) + list(out))
            qc.append(diffuser, diff_qubits)
        else:
            qc.compose(oracle, inplace=True)
            qc.compose(diffuser, qubits=diff_qubits, inplace=True)
    if measure:
        qc.measure(var, qc.cregs[0])
    return qc


def transpile_grover(problem, iterations, synthesis="baseline", coupling_map=None, backend=None,
                     seed=7, optimization_level=1):
    """Transpile the Grover circuit.

    default: all-to-all connectivity, basis {rz, sx, x, cx}.
    coupling_map: same basis, but routed onto that connectivity (SWAPs become CX).
    backend: a (fake) IBM backend; its own native gates, layout and routing.
    """
    qc = build_grover_circuit(problem, iterations, synthesis=synthesis)
    if backend is not None:
        return transpile(qc, backend=backend, optimization_level=optimization_level, seed_transpiler=seed)
    return transpile(qc, basis_gates=BASIS_GATES, coupling_map=coupling_map,
                     optimization_level=optimization_level, seed_transpiler=seed)


def circuit_stats(tqc):
    ops = tqc.count_ops()
    two_q = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
    active = {q for inst in tqc.data for q in inst.qubits}
    return {
        "qubits": len(active),
        "depth": tqc.depth(),
        "cx": int(two_q),  # all two-qubit gates (cx / ecr / cz)
        "one_qubit_gates": int(sum(v for k, v in ops.items() if k in ("sx", "x", "rz"))),
    }


def make_noise_model(p2, p1=None, readout=0.0):
    """Depolarizing noise on sx/x (p1) and cx (p2), optional symmetric readout error.

    p1 defaults to p2/10, the usual 1q:2q error ratio on superconducting devices.
    """
    if p1 is None:
        p1 = p2 / 10
    nm = NoiseModel(basis_gates=BASIS_GATES)
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    if readout > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return nm


def compact_device_circuit(tqc, backend):
    """Shrink a circuit transpiled for a large device to the physical qubits it uses.

    Returns (small circuit, noise model for those qubits). The noise model is
    built from the backend's calibration snapshot: per-gate depolarizing error
    plus T1/T2 thermal relaxation over the gate duration, and per-qubit readout
    error. Simulating the full 127/133/156-qubit register is never needed.
    """
    from qiskit_aer.noise import thermal_relaxation_error

    active = sorted({tqc.find_bit(q).index for inst in tqc.data for q in inst.qubits})
    index = {p: i for i, p in enumerate(active)}
    small = QuantumCircuit(len(active), tqc.num_clbits)
    for inst in tqc.data:
        if inst.operation.name == "barrier":
            continue
        qs = [index[tqc.find_bit(q).index] for q in inst.qubits]
        cs = [tqc.find_bit(c).index for c in inst.clbits]
        small.append(inst.operation, qs, cs)

    target = backend.target
    props = {p: target.qubit_properties[p] for p in active} if target.qubit_properties else {}
    used_gates = {inst.operation.name for inst in small.data} - {"measure", "rz", "delay", "barrier"}
    nm = NoiseModel(basis_gates=sorted(used_gates | {"rz", "measure"}))

    def relax(p, duration):
        qp = props.get(p)
        if qp is None or not duration or qp.t1 is None:
            return None
        t2 = min(qp.t2 or qp.t1, 2 * qp.t1)
        return thermal_relaxation_error(qp.t1, t2, duration)

    for name in used_gates:
        for qargs, ip in target[name].items():
            if ip is None or qargs is None or not all(q in index for q in qargs):
                continue
            err = depolarizing_error(min(ip.error, 0.75), len(qargs)) if ip.error else None
            relaxes = [relax(q, ip.duration) for q in qargs]
            if all(r is not None for r in relaxes):
                # tensor puts its argument on the lower qubit index: qargs[0] is qubit 0.
                r = relaxes[0] if len(qargs) == 1 else relaxes[1].tensor(relaxes[0])
                err = r if err is None else err.compose(r)
            if err is not None:
                nm.add_quantum_error(err, name, [index[q] for q in qargs])
    for q in active:
        ip = target["measure"].get((q,)) if "measure" in target else None
        if ip is not None and ip.error:
            e = ip.error
            nm.add_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]), [index[q]])
    return small, nm


def fold_circuit(tqc, factor):
    """Global unitary folding for zero-noise extrapolation: U -> U (U^dag U)^j, factor = 2j+1."""
    if factor < 1 or factor % 2 != 1:
        raise ValueError("fold factor must be an odd integer >= 1")
    body = tqc.remove_final_measurements(inplace=False)
    folded = body.copy()
    for _ in range((factor - 1) // 2):
        folded.compose(body.inverse(), inplace=True)
        folded.compose(body, inplace=True)
    # Re-attach the original measurements (same qubit -> clbit map).
    folded.add_bits([c for c in tqc.clbits if c not in folded.clbits])
    for inst in tqc.data:
        if inst.operation.name == "measure":
            folded.measure(tqc.find_bit(inst.qubits[0]).index, tqc.find_bit(inst.clbits[0]).index)
    return folded


def richardson_extrapolate(factors, values):
    """Extrapolate to zero noise with the polynomial through all points (Richardson)."""
    coeffs = np.polyfit(np.asarray(factors, float), np.asarray(values, float), len(factors) - 1)
    return float(np.polyval(coeffs, 0.0))


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def success_from_counts(problem, counts):
    shots = sum(counts.values())
    sols = set(problem.solutions())
    hits = sum(c for k, c in counts.items() if int(k, 2) in sols)
    return hits / shots if shots else 0.0


def run_grover(problem, iterations=None, shots=1024, p2=0.0, p1=None, readout=0.0,
               seed=11, tqc=None, synthesis="baseline", simulator=None):
    """Simulate Grover on ``problem``. Returns success probability, counts and stats.

    ``simulator`` overrides the noise settings (e.g. AerSimulator.from_backend(fake)).
    """
    M = len(problem.solutions())
    if iterations is None:
        iterations = optimal_iterations(problem.N, M)
    if tqc is None:
        tqc = transpile_grover(problem, iterations, synthesis=synthesis)
    noisy = p2 > 0 or (p1 or 0) > 0 or readout > 0
    sim = simulator or AerSimulator(noise_model=make_noise_model(p2, p1, readout) if noisy else None,
                                    seed_simulator=seed)
    counts = sim.run(tqc, shots=shots).result().get_counts()
    counts = {k.replace(" ", ""): v for k, v in counts.items()}
    return {
        "iterations": iterations,
        "M": M,
        "N": problem.N,
        "success": success_from_counts(problem, counts),
        "ideal_theory": ideal_success_probability(problem.N, M, iterations),
        "counts": counts,
        **circuit_stats(tqc),
    }


def oracle_statevector_check(problem, oracle=None):
    """Exact check of the oracle on every basis input. True if it is correct.

    Correct means |x, 0..0, o> -> |x, 0..0, o XOR f(x)> for both values of the
    output bit o, i.e. a phase flip on solutions once out holds |->, with every
    ancilla returned to |0>.
    """
    from qiskit.quantum_info import Statevector
    if problem.n_qubits > 16:
        raise ValueError("Too many qubits for an exhaustive oracle check.")
    if oracle is None:
        oracle = build_oracle(problem)
    dim = 2 ** problem.n_qubits
    sols = set(problem.solutions())
    out_bit = problem.n_qubits - 1
    for x in range(problem.N):
        f = 1 if x in sols else 0
        for o in (0, 1):
            sv = Statevector.from_int(x | (o << out_bit), dim).evolve(oracle)
            if not np.isclose(sv.data[x | ((o ^ f) << out_bit)], 1.0):
                return False
    return True


# ---------------------------------------------------------------------------
# Unknown number of solutions: Boyer-Brassard-Hoyer-Tapp (BBHT)
# ---------------------------------------------------------------------------

def bbht_expected_queries(N, M, trials=4000, seed=0, success_fn=None, max_rounds=200):
    """Monte Carlo of BBHT, which needs no knowledge of M.

    Each round picks j uniformly in [0, m), runs j Grover iterations (j oracle
    calls), measures, and checks the result classically (1 more call). On failure
    m <- min(6/5 m, sqrt N). ``success_fn(j)`` gives P(success | j iterations);
    defaults to the ideal formula. Returns mean oracle calls (incl. checks).
    """
    if M == 0:
        return float("inf")
    rng = np.random.default_rng(seed)
    if success_fn is None:
        success_fn = lambda j: ideal_success_probability(N, M, j)  # noqa: E731
    total = 0
    sqrtN = math.sqrt(N)
    for _ in range(trials):
        m, q = 1.0, 0
        for _ in range(max_rounds):
            j = int(rng.integers(0, max(1, math.ceil(m))))
            q += j + 1
            if rng.random() < success_fn(j):
                break
            m = min(1.2 * m, sqrtN)
        total += q
    return total / trials


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

def brute_force(problem, order="sequential", seed=0):
    """Check assignments one by one. Returns (solution or None, checks used)."""
    xs = list(range(problem.N))
    if order == "random":
        random.Random(seed).shuffle(xs)
    for i, x in enumerate(xs, 1):
        if problem.check(x):
            return x, i
    return None, problem.N


def brute_force_expected(N, M):
    """Expected checks for random-order brute force: (N+1)/(M+1)."""
    return (N + 1) / (M + 1) if M > 0 else float(N)


def _lit_true(lit, assign):
    v, neg = lit
    val = assign.get(v)
    return None if val is None else (val != neg)


def dpll(cnf, n_vars):
    """DPLL with unit propagation and pure-literal elimination.

    Returns (assignment int or None, stats) where stats counts decisions,
    propagations, and nodes (decisions + propagations, a rough work measure).
    """
    stats = {"decisions": 0, "propagations": 0, "backtracks": 0}

    def simplify(clauses, assign):
        out = []
        for c in clauses:
            vals = [_lit_true(l, assign) for l in c]
            if any(v is True for v in vals):
                continue
            rest = [l for l, v in zip(c, vals) if v is None]
            if not rest:
                return None
            out.append(rest)
        return out

    def solve(clauses, assign):
        clauses = simplify(clauses, assign)
        if clauses is None:
            return None
        while True:
            unit = next((c[0] for c in clauses if len(c) == 1), None)
            if unit is None:
                break
            assign = {**assign, unit[0]: not unit[1]}
            stats["propagations"] += 1
            clauses = simplify(clauses, assign)
            if clauses is None:
                return None
        lits = {l for c in clauses for l in c}
        for v, neg in lits:
            if (v, not neg) not in lits and v not in assign:
                assign = {**assign, v: not neg}
                stats["propagations"] += 1
        clauses = simplify(clauses, assign)
        if clauses is None:
            return None
        if not clauses:
            return assign
        # Branch on the most frequent variable.
        freq = {}
        for c in clauses:
            for v, _ in c:
                freq[v] = freq.get(v, 0) + 1
        v = max(freq, key=freq.get)
        for val in (True, False):
            stats["decisions"] += 1
            res = solve(clauses, {**assign, v: val})
            if res is not None:
                return res
            stats["backtracks"] += 1
        return None

    res = solve([list(c) for c in cnf], {})
    stats["nodes"] = stats["decisions"] + stats["propagations"]
    if res is None:
        return None, stats
    x = sum(1 << v for v in range(n_vars) if res.get(v, False))
    return x, stats


def walksat(cnf, n_vars, p=0.5, max_flips=1000, seed=0):
    """WalkSAT. Returns (assignment int or None, flips used)."""
    rng = random.Random(seed)
    assign = [rng.random() < 0.5 for _ in range(n_vars)]

    def sat(c):
        return any(assign[v] != neg for v, neg in c)

    for flip in range(max_flips + 1):
        unsat = [c for c in cnf if not sat(c)]
        if not unsat:
            return sum(1 << i for i, b in enumerate(assign) if b), flip
        if flip == max_flips:
            break
        c = rng.choice(unsat)
        if rng.random() < p:
            v = rng.choice(c)[0]
        else:
            def breaks(v):
                assign[v] = not assign[v]
                n = sum(1 for cc in cnf if not sat(cc))
                assign[v] = not assign[v]
                return n
            v = min((l[0] for l in c), key=breaks)
        assign[v] = not assign[v]
    return None, max_flips


def classical_summary(problem, walksat_runs=50):
    """All classical baselines on one problem."""
    M = len(problem.solutions())
    cnf = problem.to_cnf()
    _, checks_seq = brute_force(problem)
    x_dpll, dstats = dpll(cnf, problem.n_vars)
    flips = []
    # WalkSAT is incomplete: on an unsatisfiable instance every run burns max_flips.
    if M == 0:
        walksat_runs = 1
    for s in range(walksat_runs):
        x, f = walksat(cnf, problem.n_vars, seed=s)
        if x is not None:
            flips.append(f)
    return {
        "M": M,
        "N": problem.N,
        "cnf_clauses": len(cnf),
        "brute_force_sequential": checks_seq,
        "brute_force_expected": brute_force_expected(problem.N, M),
        "dpll_nodes": dstats["nodes"],
        "dpll_decisions": dstats["decisions"],
        "dpll_found": x_dpll is not None and problem.check(x_dpll),
        "walksat_mean_flips": float(np.mean(flips)) if flips else float("nan"),
        # Initial random assignment + one per flip: same unit as brute-force checks.
        "walksat_mean_evals": float(np.mean(flips)) + 1 if flips else float("nan"),
        "walksat_success_rate": len(flips) / walksat_runs,
    }


def grover_expected_queries(iterations, p_success):
    """Repeat-until-verified cost: each attempt = k oracle calls + 1 classical check."""
    if p_success <= 0:
        return float("inf")
    return (iterations + 1) / p_success
