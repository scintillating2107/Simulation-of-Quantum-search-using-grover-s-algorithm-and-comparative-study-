"""Matplotlib figures for the constraint-satisfaction study (used by experiments.py and the app)."""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle

from csp import COLOUR_NAMES, ideal_success_probability, optimal_iterations

# Validated categorical order (blue, orange, aqua, yellow, magenta, green) and a one-hue blue ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#c3c2b7"
GRID = "#e6e5e0"
SURFACE = "#ffffff"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

VERTEX_FILL = {"red": "#e34948", "green": "#008300", "blue": "#2a78d6", "yellow": "#eda100"}


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _title(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold",
                 pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def _fmt_p(p):
    if p == 0:
        return "0 (ideal)"
    exp = math.floor(math.log10(p))
    mant = p / 10 ** exp
    return f"{mant:g}e{exp}" if mant != 1 else f"1e{exp}"


# ---------------------------------------------------------------------------
# Study figures
# ---------------------------------------------------------------------------

def plot_noise_heatmap(rows, noise_levels, title="Noise-threshold heatmap"):
    """rows: list of dicts with label, success (list per noise level), advantage (list of bool).

    Colour = probability that one Grover run returns a valid solution.
    A dark outline marks cells where Grover still needs fewer expected queries
    than classical brute force.
    """
    data = np.array([r["success"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(1.15 * len(noise_levels) + 4.5, 0.55 * len(rows) + 2.2))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(data, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i, r in enumerate(rows):
        for j in range(len(noise_levels)):
            v = data[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 else INK)
            if r["advantage"][j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=INK, linewidth=1.6))
    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([_fmt_p(p) for p in noise_levels], fontsize=9, color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9, color=INK)
    ax.set_xlabel("Two-qubit depolarizing error p₂  (one-qubit error = p₂/10)", color=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("P(measured assignment is a solution)", color=INK_2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    cb.outline.set_visible(False)
    _title(ax, title, "Outlined cells: Grover still beats classical brute force on expected queries")
    fig.tight_layout()
    return fig


def plot_threshold_vs_cx(rows):
    """Advantage threshold p* against the circuit's CX count (log-log)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, grid_axis="both")
    families = sorted({r["family"] for r in rows})
    hollow_drawn = False
    for fam_i, fam in enumerate(families):
        pts = [(r["cx"], r["threshold"], r.get("status", "ok")) for r in rows
               if r["family"] == fam and r["threshold"] and np.isfinite(r["threshold"])]
        if not pts:
            continue
        pts.sort()
        exact = [(x, y) for x, y, s in pts if s == "ok"]
        if exact:
            xs, ys = zip(*exact)
            ax.plot(xs, ys, marker="o", color=SERIES[fam_i], lw=2, ms=8, label=fam,
                    markeredgecolor=SURFACE, markeredgewidth=2)
        for x, y, s in pts:
            if s == "above_grid":
                ax.plot([x], [y], marker="^", ms=9, color=SERIES[fam_i], markerfacecolor=SURFACE,
                        markeredgewidth=1.8, ls="none",
                        label=None if hollow_drawn else "lower bound (advantage at every tested p₂)")
                hollow_drawn = True
    cx_all = [r["cx"] for r in rows if r["threshold"] and np.isfinite(r["threshold"])]
    if cx_all:
        xs = np.geomspace(min(cx_all), max(cx_all), 50)
        ax.plot(xs, 1 / xs, ls="--", color=INK_2, lw=1, label="1 / CX count (reference)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CX gates in the transpiled circuit")
    ax.set_ylabel("Noise threshold p₂*")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _title(ax, "Where the quantum advantage disappears",
           "Largest p₂ at which Grover still needs fewer expected queries than brute force")
    fig.tight_layout()
    return fig


def plot_success_vs_iterations(N, M_values, max_iter=None, noisy=None):
    """Ideal success probability vs iterations for several solution counts.

    noisy: optional {M: (iterations list, success list)} of simulated points.
    """
    if max_iter is None:
        max_iter = max(optimal_iterations(N, 1) * 2 + 1, 4)
    ks = np.arange(0, max_iter + 1)
    kc = np.linspace(0, max_iter, 400)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for i, M in enumerate(M_values):
        c = SERIES[i % len(SERIES)]
        theta = math.asin(math.sqrt(M / N))
        ax.plot(kc, np.sin((2 * kc + 1) * theta) ** 2, color=c, lw=1.2, alpha=0.45)
        ys = [ideal_success_probability(N, M, k) for k in ks]
        ax.plot(ks, ys, "o", color=c, ms=6, label=f"M = {M}")
        k_opt = optimal_iterations(N, M)
        ax.plot([k_opt], [ideal_success_probability(N, M, k_opt)], "o", ms=11, color=c,
                markeredgecolor=INK, markeredgewidth=1.5)
        if noisy and M in noisy:
            nk, ns = noisy[M]
            ax.plot(nk, ns, "s", ms=7, color=c, markerfacecolor=SURFACE, markeredgewidth=1.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Grover iterations k")
    ax.set_ylabel("P(success)")
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=9, ncol=min(4, len(M_values)), loc="upper center",
              bbox_to_anchor=(0.5, -0.16), labelcolor=INK)
    sub = f"N = {N}. Dots: ideal; outlined dot: optimal k ≈ (π/4)√(N/M)"
    if noisy:
        sub += "; hollow squares: noisy simulation"
    _title(ax, "More solutions, faster search", sub)
    fig.tight_layout()
    return fig


QUERY_SERIES = [
    ("brute_force_expected", "Classical brute force (expected)"),
    ("dpll_nodes", "DPLL (search nodes)"),
    ("walksat_mean_evals", "WalkSAT (assignments tried)"),
    ("grover_ideal_queries", "Grover, M known (ideal)"),
    ("bbht_ideal_queries", "Grover/BBHT, M unknown (ideal)"),
    ("grover_noisy_queries", "Grover, M known (noisy)"),
]


def plot_queries_vs_solutions(rows, n_vars, noisy_label=""):
    """Expected oracle calls / checks to find one solution as the solution count grows."""
    rows = sorted((r for r in rows if r["n_vars"] == n_vars), key=lambda r: r["M"])
    Ms = [r["M"] for r in rows]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, grid_axis="both")
    for i, (key, label) in enumerate(QUERY_SERIES):
        ys = [r.get(key, float("nan")) for r in rows]
        if all(not np.isfinite(y) for y in ys):
            continue
        lab = label + (f", p₂={noisy_label}" if key == "grover_noisy_queries" and noisy_label else "")
        ax.plot(Ms, ys, marker="o", color=SERIES[i], lw=2, ms=8, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=2,
                ls="--" if key.startswith("dpll") or key.startswith("walksat") else "-")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(m) for m in Ms])
    ax.set_xlabel("Number of solutions M")
    ax.set_ylabel("Oracle calls / checks to find a solution")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1))
    _title(ax, f"Cost to find a solution: random 3-SAT, {n_vars} variables (N = {2**n_vars})",
           "Dashed: classical solvers whose work unit differs (search nodes, local-search steps)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Single-problem figures (app)
# ---------------------------------------------------------------------------

def plot_cost_bars(entries):
    """entries: list of (label, value, 'classical'|'quantum')."""
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(entries) + 1.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, grid_axis="x")
    labels = [e[0] for e in entries][::-1]
    vals = [e[1] for e in entries][::-1]
    colors = [SERIES[0] if e[2] == "classical" else SERIES[1] for e in entries][::-1]
    finite = [v for v in vals if np.isfinite(v)]
    cap = max(finite) * 1.15 if finite else 1
    bars = ax.barh(labels, [v if np.isfinite(v) else cap for v in vals], color=colors, height=0.6,
                   edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + cap * 0.01, b.get_y() + b.get_height() / 2,
                "∞ (never succeeds)" if not np.isfinite(v) else f"{v:.1f}",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, cap * 1.18)
    ax.set_xlabel("Oracle calls / checks to find a solution (lower is better)")
    ax.tick_params(axis="y", labelsize=9, colors=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="Classical"), Patch(color=SERIES[1], label="Quantum (Grover)")],
              frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    _title(ax, "Classical vs quantum cost on this problem")
    fig.tight_layout()
    return fig


def plot_counts(problem, counts_list, labels, top=None):
    """Measurement histogram. Solutions in blue, other assignments in gray."""
    sols = set(problem.solutions())
    keys = set()
    for c in counts_list:
        keys |= {int(k, 2) for k in c}
    xs = sorted(keys)
    if top and len(xs) > top:
        total = {x: sum(c.get(format(x, f"0{problem.n_vars}b"), 0) for c in counts_list) for x in xs}
        xs = sorted(sorted(xs, key=lambda x: -total[x])[:top])
    fig, axes = plt.subplots(len(counts_list), 1, figsize=(max(7, 0.32 * len(xs) + 2), 2.6 * len(counts_list)),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, counts, lab in zip(axes[:, 0], counts_list, labels):
        _style(ax)
        shots = sum(counts.values()) or 1
        ys = [counts.get(format(x, f"0{problem.n_vars}b"), 0) / shots for x in xs]
        ax.bar(range(len(xs)), ys, color=[SERIES[0] if x in sols else MUTED for x in xs],
               width=0.8, edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([problem.bitstring(x) for x in xs], rotation=90, fontsize=7.5,
                           family="monospace", color=INK_2)
        ax.set_ylabel("Probability")
        ax.set_title(lab, loc="left", fontsize=10, color=INK)
    from matplotlib.patches import Patch
    axes[0, 0].legend(handles=[Patch(color=SERIES[0], label="satisfies all constraints"),
                               Patch(color=MUTED, label="violates a constraint")],
                      frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
    axes[-1, 0].set_xlabel("Assignment (variables x0 x1 x2 … left to right)")
    fig.tight_layout()
    return fig


def draw_problem(problem, x=None):
    """Picture of the instance; if x is given, the assignment drawn on it."""
    if problem.kind == "colouring":
        return _draw_graph(problem, x)
    if problem.kind == "sudoku":
        return _draw_sudoku(problem, x)
    return None


def _draw_graph(problem, x):
    n = problem.meta["n_vertices"]
    edges = problem.meta["edges"]
    ang = [math.pi / 2 - 2 * math.pi * i / n for i in range(n)]
    pos = [(math.cos(a), math.sin(a)) for a in ang]
    colours = problem.decode(x) if x is not None else None
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.axis("off")
    for u, v in edges:
        bad = colours is not None and colours[f"v{u}"] == colours[f"v{v}"]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#e34948" if bad else INK_2, lw=3 if bad else 1.8, zorder=1)
    for i, (px, py) in enumerate(pos):
        name = colours[f"v{i}"] if colours else None
        fill = VERTEX_FILL.get(name, "#f0efec") if name in COLOUR_NAMES[:problem.meta["k"]] else "#f0efec"
        ax.add_patch(Circle((px, py), 0.2, facecolor=fill, edgecolor=INK, lw=1.5, zorder=2))
        ax.text(px, py, f"v{i}", ha="center", va="center", fontsize=10,
                color="white" if colours else INK, fontweight="bold", zorder=3)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    fig.tight_layout()
    return fig


def _draw_sudoku(problem, x):
    size = problem.meta["size"]
    grid = problem.decode(x) if x is not None else problem.meta["grid"]
    blanks = set(problem.meta["blanks"])
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.axis("off")
    for r in range(size):
        for c in range(size):
            is_blank = (r, c) in blanks
            ax.add_patch(Rectangle((c, size - 1 - r), 1, 1, facecolor="#cde2fb" if is_blank else SURFACE,
                                   edgecolor=MUTED, lw=1))
            v = grid[r][c]
            if v is not None and not (v == 0 and not problem.meta["binary"]):
                ax.text(c + 0.5, size - 0.5 - r, str(v), ha="center", va="center", fontsize=16,
                        color=SERIES[0] if is_blank else INK, fontweight="bold" if is_blank else "normal")
    box = 2 if size == 4 else 1
    for i in range(0, size + 1, box):
        ax.plot([i, i], [0, size], color=INK, lw=2)
        ax.plot([0, size], [i, i], color=INK, lw=2)
    ax.set_xlim(-0.1, size + 0.1)
    ax.set_ylim(-0.1, size + 0.1)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Study figures (study.py)
# ---------------------------------------------------------------------------

SYNTH_STYLE = {"baseline": (SERIES[0], "o", "Baseline oracle"),
               "aux+rp": (SERIES[1], "D", "Ancilla-aware + relative-phase oracle")}


def plot_threshold_scatter(thr, fit=None):
    """Per-instance thresholds (95% bootstrap CI) against 2-qubit gate count, with power-law fits."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, grid_axis="both")
    for syn, (color, marker, label) in SYNTH_STYLE.items():
        g = thr[(thr["synthesis"] == syn) & (thr["status"] == "ok")]
        if g.empty:
            continue
        yerr = np.vstack([(g["threshold"] - g["ci_low"]).clip(lower=0).fillna(0),
                          (g["ci_high"] - g["threshold"]).clip(lower=0).fillna(0)])
        ax.errorbar(g["cx"], g["threshold"], yerr=yerr, fmt=marker, color=color, ms=6.5,
                    mec=SURFACE, mew=1.2, elinewidth=1, capsize=0, alpha=0.9, label=label)
        if fit is not None and not fit.empty and syn in set(fit["synthesis"]):
            f = fit[fit["synthesis"] == syn].iloc[0]
            xs = np.geomspace(g["cx"].min(), g["cx"].max(), 50)
            ax.plot(xs, f["A"] * xs ** (-f["b"]), color=color, lw=1.6,
                    label=f"fit: p* ∝ CX^-{f['b']:.2f} [{f['b_ci_low']:.2f}, {f['b_ci_high']:.2f}], R²={f['r2']:.2f}")
    allx = thr["cx"]
    xs = np.geomspace(allx.min(), allx.max(), 50)
    ax.plot(xs, 1 / xs, ls="--", color=INK_2, lw=1, label="1 / CX (one expected error)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Two-qubit gates in the transpiled circuit (all-to-all)")
    ax.set_ylabel("Advantage threshold p₂*")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    _title(ax, "Noise threshold vs circuit size",
           "Each point is one random instance; bars are 95% bootstrap intervals over shot noise")
    fig.tight_layout()
    return fig


FAMILY_MARKERS = {"2-colouring": "o", "3-SAT": "s", "4x4 Sudoku": "^"}


def plot_decay_model(thr, alpha, r2, fit_families=None):
    """Measured threshold against the one-parameter prediction ln(P_ideal / P_req) / (alpha * CX)."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, grid_axis="both")
    for syn, (color, _, syn_label) in SYNTH_STYLE.items():
        for fam, marker in FAMILY_MARKERS.items():
            g = thr[(thr["synthesis"] == syn) & thr["family"].str.startswith(fam)]
            if g.empty:
                continue
            held_out = fit_families is not None and fam not in fit_families
            ax.scatter(g["predicted"], g["threshold"], marker=marker, s=46, edgecolor=color,
                       facecolor=SURFACE if held_out else color, lw=1.4, zorder=3,
                       label=f"{fam} · {syn_label.split(' oracle')[0].lower()}" + (" (held out)" if held_out else ""))
    lo = min(thr["predicted"].min(), thr["threshold"].min()) / 1.3
    hi = max(thr["predicted"].max(), thr["threshold"].max()) * 1.3
    xs = np.geomspace(lo, hi, 20)
    ax.plot(xs, xs, color=INK_2, lw=1, ls="--", label="measured = predicted", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted p₂* = ln(P_ideal / P_req) / (α · CX)")
    ax.set_ylabel("Measured advantage threshold p₂*")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    _title(ax, "One-parameter threshold model", f"α = {alpha:.2f}, R² = {r2:.3f} on log p₂*")
    fig.tight_layout()
    return fig


def plot_device_results(dv):
    """Success on calibrated fake devices vs ideal and random guessing, per instance."""
    devices = list(dict.fromkeys(dv["device"]))
    fig, axes = plt.subplots(1, len(devices), figsize=(max(8.5, 4.3 * len(devices)), 0.55 * dv["problem"].nunique() + 2.6),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    probs = list(dict.fromkeys(dv.sort_values("cx_all_to_all")["problem"]))
    for ax, dev in zip(axes[0], devices):
        _style(ax, grid_axis="x")
        d = dv[dv["device"] == dev]
        for j, (syn, (color, marker, label)) in enumerate(SYNTH_STYLE.items()):
            g = d[d["synthesis"] == syn].set_index("problem").reindex(probs)
            ys = np.arange(len(probs)) + (j - 0.5) * 0.28
            ax.errorbar(g["success"], ys, xerr=[g["success"] - g["ci_low"], g["ci_high"] - g["success"]],
                        fmt=marker, color=color, ms=6, mec=SURFACE, mew=1, elinewidth=1, label=label)
        g0 = d.drop_duplicates("problem").set_index("problem").reindex(probs)
        ys = np.arange(len(probs))
        ax.scatter(g0["ideal"], ys, marker="|", s=160, color=INK, label="ideal (noise-free)", zorder=3)
        ax.scatter(g0["random_guess"], ys, marker="x", s=30, color=INK_2, label="random guess M/N", zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels(probs, fontsize=8, color=INK)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("P(success)")
        ax.set_title(dev.replace("Fake", "Fake "), loc="left", fontsize=10.5, color=INK, fontweight="bold")
    axes[0][0].invert_yaxis()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Grover on calibrated IBM device models", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    return fig


def plot_zne(zr):
    """Raw vs ZNE-mitigated estimate of P(success) against the ideal value."""
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(zr) + 2.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, grid_axis="x")
    labels = [f"{r.problem} · {r.noise}" for r in zr.itertuples()]
    ys = np.arange(len(zr))
    ax.hlines(ys, zr["raw"], zr["richardson"], color=MUTED, lw=1.5, zorder=1)
    ax.scatter(zr["raw"], ys, color=SERIES[0], s=40, label="raw (fold ×1)", zorder=2)
    ax.scatter(zr["richardson"], ys, color=SERIES[1], s=40, marker="D", label="ZNE (Richardson ×1,3,5)", zorder=2)
    ax.scatter(zr["ideal"], ys, marker="|", s=200, color=INK, label="ideal", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.05)
    ax.set_xlabel("Estimated P(success)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    _title(ax, "Zero-noise extrapolation", "Recovers the estimate of P(success), not better samples")
    fig.tight_layout()
    return fig
