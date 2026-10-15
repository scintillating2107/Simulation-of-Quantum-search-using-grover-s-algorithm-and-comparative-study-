# Grover's Algorithm: Quantum vs Classical Search

Simulation of Grover's quantum search with classical brute-force comparison. Built with Qiskit. Run locally or deploy as a web app.

The project has two parts:

1. **Password search** (`app.py`, `grovers.py`): the textbook demo, where the oracle is built from a known target.
2. **Constraint satisfaction** (`csp.py`, the *Constraint Satisfaction* page in the app, `experiments.py`): Grover on SAT, graph colouring and Sudoku. Here the oracle checks the rules and never sees an answer. Results are compared with brute force, DPLL and WalkSAT under realistic noise. See [CSP_STUDY.md](CSP_STUDY.md).

**Requirements:** Python 3.7+, `pip install -r requirements.txt`

## Run

**Web app (recommended):**
```bash
streamlit run app.py
```
Open the URL in the terminal (e.g. http://localhost:8501). Use the sidebar to set qubits, target state(s), and click Run simulation. Pick **Constraint Satisfaction** in the page list for the SAT / colouring / Sudoku study.

**Constraint-satisfaction benchmarks:**
```bash
python experiments.py --quick   # ~1-2 min smoke run
python experiments.py           # full study, ~30 min; writes results/*.csv and figures/csp_*.png
python tests/test_csp.py        # or: python -m pytest tests
```

**CLI:** `python grovers.py` for an interactive prompt (qubit count, single vs compare mode, target selection). Plots open in windows.

## Deploy (e.g. for resume)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. New app → choose repo, branch `main`, main file `app.py`. Optionally set a custom subdomain.
4. Deploy. First build can take a few minutes. Share the `*.streamlit.app` link.

No server or payment required. Pushing to the repo redeploys the app.

## Project layout

- `app.py` – Streamlit web UI (password search)
- `pages/1_Constraint_Satisfaction.py` – Streamlit page for SAT / graph colouring / Sudoku
- `grovers.py` – Grover circuit, classical search, state evolution, and all plots
- `csp.py` – CSP encodings, structural oracles, noise model, Grover runner, BBHT, brute force / DPLL / WalkSAT
- `csp_plots.py` – figures for the CSP study
- `experiments.py` – benchmark suite (noise-threshold heatmap, multiple-solution study)
- `results/` – CSV output of `experiments.py`, read by the app
- `tests/test_csp.py` – exhaustive oracle checks and solver tests
- `requirements.txt` – numpy, matplotlib, qiskit, qiskit-aer, streamlit
- `.streamlit/config.toml` – theme/settings for the web app

See `REPORT.md` and `ABOUT.md` for algorithm and visualization notes.

## Result summary

| Qubits | Classical (worst) | Grover iters | Speedup |
|--------|-------------------|--------------|---------|
| 2      | 4                 | 1            | 4x      |
| 3      | 8                 | 2            | 4x      |
| 4      | 16                | 3            | ~5.3x   |
| 5      | 32                | 4            | 8x      |

Grover gives O(√N) queries vs O(N) for classical search. The app also shows ideal vs noisy success probability (depolarizing noise model).

## License

MIT.
