# Grover's Algorithm: Quantum vs Classical Search

Simulation of Grover's quantum search with classical brute-force comparison. Built with Qiskit. Run locally or deploy as a web app.

**Requirements:** Python 3.7+, `pip install -r requirements.txt`

## Run

**Web app (recommended):**
```bash
streamlit run app.py
```
Open the URL in the terminal (e.g. http://localhost:8501). Use the sidebar to set qubits, target state(s), and click Run simulation.

**CLI:** `python grovers.py` for an interactive prompt (qubit count, single vs compare mode, target selection). Plots open in windows.

## Deploy (e.g. for resume)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. New app → choose repo, branch `main`, main file `app.py`. Optionally set a custom subdomain.
4. Deploy. First build can take a few minutes. Share the `*.streamlit.app` link.

No server or payment required. Pushing to the repo redeploys the app.

## Project layout

- `app.py` – Streamlit web UI
- `grovers.py` – Grover circuit, classical search, state evolution, and all plots
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
