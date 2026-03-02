import streamlit as st

from grovers import (
    generate_passwords,
    classical_search,
    run_quantum,
    plot_superposition_state,
    plot_advanced_grover_steps,
    plot_grover_simulation,
    plot_grover_evolution_animated,
    plot_comparative_study,
    plot_classical_vs_quantum,
    compare_target_states,
    plot_scaling,
    plot_success_probability,
    plot_grover_circuit_diagram,
    get_circuit_stats,
)


@st.cache_data(show_spinner=False)
def compute_scaling_data(qubit_range, shots, noise_enabled, error_1q, error_2q):
    classical_list = []
    quantum_list = []
    ideal_probs = []
    noisy_probs = []

    for q in qubit_range:
        target_q = "1" * q
        passwords_q = generate_passwords(q)
        classical_q = classical_search(passwords_q, target_q)
        grover_q, ideal_q, noisy_q, _, _ = run_quantum(
            q,
            target_q,
            shots=shots,
            noise_enabled=noise_enabled,
            error_1q=error_1q,
            error_2q=error_2q,
        )
        classical_list.append(classical_q)
        quantum_list.append(grover_q)
        ideal_probs.append(ideal_q)
        noisy_probs.append(noisy_q)

    return classical_list, quantum_list, ideal_probs, noisy_probs

st.set_page_config(
    page_title="Grover's Algorithm | Quantum vs Classical Search",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>#MainMenu { visibility: hidden; } footer { visibility: hidden; }</style>
""", unsafe_allow_html=True)

st.title("Simulation of Quantum Search Using Grover's Algorithm")
st.subheader("Comparative Study of Quantum Search vs Classical Search")

with st.expander("What this app demonstrates"):
    st.markdown(
        """
        - **Goal**: compare classical exhaustive search \(O(N)\) with Grover's quantum search \(O(\sqrt{N})\) on a password‑style search problem.
        - **Grover's loop**: repeated application of an **oracle** (marks the target state) and **diffusion** (amplitude amplification) to boost the target's measurement probability.
        - **Noise model**: configurable depolarizing noise on 1‑ and 2‑qubit gates to mimic real hardware and study robustness.
        - **Visuals**:
          - State amplitudes/probabilities across iterations.
          - Classical vs quantum scaling as qubits grow.
          - The actual quantum circuit used for the current configuration.
        """
    )

# Sidebar
st.sidebar.header("Experiment setup")

n_qubits = st.sidebar.slider(
    "Number of qubits",
    min_value=1,
    max_value=10,
    value=3,
)

st.sidebar.caption(
    "1–6 qubits are fast and fully visualized. Higher values grow exponentially and may be slow."
)
if n_qubits > 8:
    st.sidebar.warning(
        f"{n_qubits} qubits → {2**n_qubits} basis states. Some plots may be simplified."
    )

st.sidebar.subheader("Search mode & targets")
mode = st.sidebar.radio(
    "Mode",
    ["Single target", "Compare multiple targets"],
    index=0,
)

all_states = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]

if mode == "Single target":
    target = st.sidebar.selectbox(
        "Target state",
        options=all_states,
        index=len(all_states) - 1,
        format_func=lambda x: f"{x} (index {all_states.index(x)})",
    )
    target_list = None
else:
    target_list = st.sidebar.multiselect(
        "Target states to compare",
        options=all_states,
        default=[all_states[0], all_states[-1]],
        format_func=lambda x: x,
    )
    if len(target_list) < 2:
        st.sidebar.warning("Select at least 2 targets for comparison.")
    target = None

st.sidebar.divider()
st.sidebar.subheader("Quantum backend & sampling")

noise_enabled = st.sidebar.checkbox(
    "Include noisy simulator",
    value=True,
    help="If disabled, only the ideal (noise‑free) simulator is run.",
)

if noise_enabled:
    error_1q = st.sidebar.slider(
        "1‑qubit depolarizing error",
        min_value=0.0,
        max_value=0.05,
        value=0.01,
        step=0.005,
        format="%0.3f",
    )
    error_2q = st.sidebar.slider(
        "2‑qubit depolarizing error",
        min_value=0.0,
        max_value=0.10,
        value=0.02,
        step=0.005,
        format="%0.3f",
    )
else:
    error_1q = 0.0
    error_2q = 0.0

shots = st.sidebar.slider(
    "Number of shots",
    min_value=256,
    max_value=4096,
    step=256,
    value=1024,
    help="Number of measurement repetitions used when estimating probabilities.",
)

preset = st.sidebar.selectbox(
    "Experiment preset",
    [
        "Custom",
        "Noiseless baseline",
        "Realistic noisy hardware",
        "High noise (failure regime)",
    ],
    index=2,
)

if preset == "Noiseless baseline":
    noise_enabled = False
    error_1q = 0.0
    error_2q = 0.0
    shots = max(shots, 1024)
    st.sidebar.caption("Ideal simulator only, no noise, moderate shot count.")
elif preset == "Realistic noisy hardware":
    noise_enabled = True
    error_1q = 0.01
    error_2q = 0.02
    st.sidebar.caption(
        "Typical small‑scale device noise: 1% single‑qubit, 2% two‑qubit errors."
    )
elif preset == "High noise (failure regime)":
    noise_enabled = True
    error_1q = max(error_1q, 0.03)
    error_2q = max(error_2q, 0.06)
    shots = max(shots, 2048)
    st.sidebar.caption(
        "Stress test with strong noise and more shots to study how Grover's advantage degrades."
    )

run = st.sidebar.button("Run simulation")

if run:
    show = False

    if mode == "Compare multiple targets" and target_list and len(target_list) >= 2:
        st.header("Comparing multiple target states")
        fig = compare_target_states(n_qubits, target_list, show=show)
        if fig is not None:
            st.pyplot(fig)

    else:
        if target is None:
            target = "1" * n_qubits

        passwords = generate_passwords(n_qubits)
        classical_steps = classical_search(passwords, target)
        grover_iters, ideal_p, noisy_p, ideal_counts, noisy_counts = run_quantum(
            n_qubits,
            target,
            shots=shots,
            noise_enabled=noise_enabled,
            error_1q=error_1q,
            error_2q=error_2q,
        )

        circuit_stats = get_circuit_stats(n_qubits, target)

        st.header("Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Classical steps (worst case)", classical_steps)
        with col2:
            st.metric("Grover iterations", grover_iters)
        with col3:
            st.metric("Ideal success probability", f"{ideal_p*100:.2f}%")
        with col4:
            noisy_label = (
                f"{noisy_p*100:.2f}%" if noise_enabled else "N/A (noise disabled)"
            )
            st.metric("Noisy success probability", noisy_label)

        speedup = classical_steps / grover_iters if grover_iters > 0 else 0
        st.info(f"Quantum speedup: **{speedup:.2f}x** faster than classical search.")

        st.subheader("Circuit‑level characteristics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Circuit depth", circuit_stats["depth"])
        with c2:
            st.metric("Gate count", circuit_stats["size"])
        with c3:
            st.metric("Qubits / classical bits", f"{circuit_stats['num_qubits']} / {circuit_stats['num_clbits']}")

        st.subheader("Visualizations and analytics")
        st.caption("Generating plots for this configuration. For larger qubit counts this can take a few seconds.")

        with st.spinner("Building Grover visualizations..."):
            # Precompute all figures once so we can organize them into tabs.
            if n_qubits <= 8:
                fig_superposition = plot_superposition_state(n_qubits, show=show)
                fig_steps = plot_advanced_grover_steps(n_qubits, target, show=show)
                fig_simulation = plot_grover_simulation(n_qubits, target, show=show)
                fig_evolution = plot_grover_evolution_animated(n_qubits, target, show=show)
            else:
                fig_superposition = None
                fig_steps = None
                fig_simulation = None
                fig_evolution = None

            fig_comparative = plot_comparative_study(
                n_qubits, classical_steps, grover_iters, ideal_p, noisy_p, show=show
            )
            fig_bar = plot_classical_vs_quantum(
                classical_steps, grover_iters, n_qubits, show=show
            )
            fig_circuit = plot_grover_circuit_diagram(n_qubits, target, show=show)

            # Compute scaling data across qubits to show richer analytics.
            max_scaling_n = min(max(n_qubits, 3) + 2, 8)
            qubit_range = list(range(1, max_scaling_n + 1))
            (
                classical_list,
                quantum_list,
                ideal_probs,
                noisy_probs,
            ) = compute_scaling_data(
                qubit_range,
                shots=shots,
                noise_enabled=noise_enabled,
                error_1q=error_1q,
                error_2q=error_2q,
            )

            fig_scaling = plot_scaling(classical_list, quantum_list, qubit_range, show=show)
            fig_success = plot_success_probability(
                ideal_probs, noisy_probs, qubit_range, show=show
            )

        (
            overview_tab,
            steps_tab,
            evolution_tab,
            comparison_tab,
            scaling_tab,
            circuit_tab,
        ) = st.tabs(
            [
                "Overview",
                "Grover steps",
                "Iteration evolution",
                "Classical vs Quantum",
                "Scaling across qubits",
                "Circuit diagram",
            ]
        )

        with overview_tab:
            st.subheader("Overview for current configuration")
            if fig_superposition is not None:
                st.pyplot(fig_superposition)
            if fig_bar is not None:
                st.pyplot(fig_bar)

        with steps_tab:
            st.subheader("Grover algorithm key steps")
            if fig_steps is not None:
                st.pyplot(fig_steps)

        with evolution_tab:
            st.subheader("State distribution and target probability over iterations")
            if fig_simulation is not None:
                st.pyplot(fig_simulation)
            if fig_evolution is not None:
                st.pyplot(fig_evolution)

        with comparison_tab:
            st.subheader("Detailed classical vs quantum comparison")
            if fig_comparative is not None:
                st.pyplot(fig_comparative)

        with scaling_tab:
            st.subheader("How scaling behaves across qubits")
            st.caption(
                f"Computed for {qubit_range[0]}–{qubit_range[-1]} qubits using worst-case targets."
            )
            if fig_scaling is not None:
                st.pyplot(fig_scaling)
            if fig_success is not None:
                st.pyplot(fig_success)

        with circuit_tab:
            st.subheader("Grover circuit for this configuration")
            st.caption(
                "Actual quantum circuit sent to the simulators, including oracle and diffusion operators."
            )
            if fig_circuit is not None:
                st.pyplot(fig_circuit)

        with st.expander(f"Measurement results ({shots} shots)"):
            st.write("**Ideal quantum:**")
            st.write(
                f"Target '{target}': {ideal_counts.get(target, 0)} measurements ({ideal_p*100:.2f}%)"
            )
            st.write("**Noisy quantum:**")
            if noise_enabled:
                st.write(
                    f"Target '{target}': {noisy_counts.get(target, 0)} measurements ({noisy_p*100:.2f}%)"
                )
            else:
                st.write("Noisy simulator disabled for this run.")

else:
    st.info("Set options in the sidebar and click **Run simulation**.")

st.sidebar.divider()
st.sidebar.caption("Grover: O(√N) vs classical O(N).")
