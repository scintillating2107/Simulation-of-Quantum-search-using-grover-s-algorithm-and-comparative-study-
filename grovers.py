"""Grover's algorithm: quantum vs classical search simulation and plots."""

# Removed banner:
# Classical vs Quantum Search using Grover’s Algorithm
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Statevector


def classical_search(search_space, target):
    steps = 0
    for candidate in search_space:
        steps += 1
        if candidate == target:
            return steps
    return steps


def generate_passwords(n):
    return [format(i, f'0{n}b') for i in range(2**n)]


def get_target_state(n_qubits, allow_custom=True):
    all_states = generate_passwords(n_qubits)
    
    print(f"\n{'='*60}")
    print(f"TARGET STATE SELECTION")
    print(f"{'='*60}")
    print(f"\nAvailable states for {n_qubits} qubit{'s' if n_qubits > 1 else ''}:")
    for i, state in enumerate(all_states):
        print(f"  {i}: {state}", end="  ")
        if (i + 1) % 4 == 0:  # New line every 4 states
            print()
    if len(all_states) % 4 != 0:
        print()
    
    while True:
        try:
            if allow_custom:
                print(f"\nOptions:")
                print(f"  [1] Enter target state directly (e.g., '101' for 3 qubits)")
                print(f"  [2] Enter index number (0-{len(all_states)-1})")
                print(f"  [3] Use worst-case (all 1s: {'1'*n_qubits})")
                print(f"  Or enter target state directly as binary string")
                choice = input("\nEnter option [1/2/3] or target state: ").strip()
            else:
                choice = input(f"\nEnter target state (binary string, {n_qubits} bits) or index (0-{len(all_states)-1}): ").strip()
            
            if allow_custom and choice in ['1', '2', '3']:
                if choice == '1':
                    target = input(f"Enter {n_qubits}-bit binary string: ").strip()
                    if all(c in '01' for c in target) and len(target) == n_qubits:
                        if target in all_states:
                            print(f"\n✓ Selected target state: {target}")
                            return target
                        else:
                            print(f"✗ Invalid: '{target}' is not a valid {n_qubits}-bit state")
                    else:
                        print(f"✗ Invalid format. Must be {n_qubits} bits (0s and 1s only)")
                elif choice == '2':
                    idx_str = input(f"Enter index (0-{len(all_states)-1}): ").strip()
                    if idx_str.isdigit():
                        idx = int(idx_str)
                        if 0 <= idx < len(all_states):
                            target = all_states[idx]
                            print(f"\n✓ Selected target state: {target} (index {idx})")
                            return target
                        else:
                            print(f"✗ Invalid index. Please enter 0-{len(all_states)-1}")
                    else:
                        print(f"✗ Invalid input. Please enter a number between 0-{len(all_states)-1}")
                elif choice == '3':
                    target = "1" * n_qubits
                    print(f"\n✓ Selected worst-case target: {target}")
                    return target
            
            elif all(c in '01' for c in choice) and len(choice) == n_qubits:
                if choice in all_states:
                    print(f"\n✓ Selected target state: {choice}")
                    return choice
                else:
                    print(f"✗ Invalid: '{choice}' is not a valid {n_qubits}-bit state")
            
            elif choice.isdigit() and not (allow_custom and choice in ['1', '2', '3']):
                idx = int(choice)
                if 0 <= idx < len(all_states):
                    target = all_states[idx]
                    print(f"\n✓ Selected target state: {target} (index {idx})")
                    return target
                else:
                    print(f"✗ Invalid index. Please enter 0-{len(all_states)-1}")
            
            else:
                if allow_custom:
                    print(f"✗ Invalid input. Please enter option [1/2/3] or a valid {n_qubits}-bit binary string")
                else:
                    print(f"✗ Invalid input. Please enter a valid {n_qubits}-bit binary string or index (0-{len(all_states)-1})")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            exit(0)
        except Exception as e:
            print(f"✗ Error: {e}. Please try again.")


def compare_target_states(n_qubits, target_list, show=True):
    results = []
    passwords = generate_passwords(n_qubits)

    if show:
        print(f"\n{'='*70}\nCOMPARING MULTIPLE TARGET STATES\n{'='*70}\n")

    for target in target_list:
        if show:
            print(f"Analyzing target: {target}...")
        classical_steps = classical_search(passwords, target)
        grover_iters, ideal_p, noisy_p, _, _ = run_quantum(n_qubits, target)
        
        results.append({
            'target': target,
            'classical_steps': classical_steps,
            'grover_iters': grover_iters,
            'ideal_prob': ideal_p,
            'noisy_prob': noisy_p,
            'speedup': classical_steps / grover_iters if grover_iters > 0 else 0
        })
    
    # Create comparison visualization - smaller size for better text visibility
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Steps/Iterations Comparison
    ax1 = axes[0, 0]
    targets = [r['target'] for r in results]
    classical_steps_list = [r['classical_steps'] for r in results]
    grover_iters_list = [r['grover_iters'] for r in results]
    
    x = np.arange(len(targets))
    width = 0.35
    bars1 = ax1.bar(x - width/2, classical_steps_list, width, label='Classical', 
                    color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, grover_iters_list, width, label='Grover', 
                    color='#4ECDC4', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Target State', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Steps/Iterations', fontsize=10, fontweight='bold')
    ax1.set_title('Steps Comparison by Target State', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 2. Success Probability Comparison
    ax2 = axes[0, 1]
    ideal_probs = [r['ideal_prob'] for r in results]
    noisy_probs = [r['noisy_prob'] for r in results]
    
    x = np.arange(len(targets))
    bars1 = ax2.bar(x - width/2, ideal_probs, width, label='Ideal Quantum', 
                    color='#95E1D3', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, noisy_probs, width, label='Noisy Quantum', 
                    color='#F38181', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Target State', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Success Probability', fontsize=10, fontweight='bold')
    ax2.set_title('Success Rate by Target State', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets, fontsize=9)
    ax2.set_ylim([0, 1.1])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Speedup Comparison
    ax3 = axes[1, 0]
    speedups = [r['speedup'] for r in results]
    bars = ax3.bar(targets, speedups, color='#FFD93D', alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Target State', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Speedup Factor', fontsize=10, fontweight='bold')
    ax3.set_title('Quantum Speedup by Target State', fontsize=11, fontweight='bold')
    ax3.set_xticklabels(targets, fontsize=9, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [['Target', 'Classical', 'Grover', 'Ideal %', 'Noisy %', 'Speedup']]
    for r in results:
        table_data.append([
            r['target'],
            str(r['classical_steps']),
            str(r['grover_iters']),
            f"{r['ideal_prob']*100:.1f}%",
            f"{r['noisy_prob']*100:.1f}%",
            f"{r['speedup']:.2f}x"
        ])
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax4.set_title('Summary Table', fontsize=11, fontweight='bold', pad=15)
    
    plt.suptitle(f'Target State Comparison (n={n_qubits} qubits)', 
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    if show:
        plt.show()
        best_speedup = max(results, key=lambda x: x['speedup'])
        worst_speedup = min(results, key=lambda x: x['speedup'])
        print(f"\n{'='*70}\nDETAILED COMPARISON RESULTS\n{'='*70}\n")
        print(f"{'Target':<10} {'Classical':<12} {'Grover':<10} {'Ideal %':<10} {'Noisy %':<10} {'Speedup':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['target']:<10} {r['classical_steps']:<12} {r['grover_iters']:<10} "
                  f"{r['ideal_prob']*100:>6.1f}%    {r['noisy_prob']*100:>6.1f}%    {r['speedup']:>6.2f}x")
        print(f"\nBest speedup: {best_speedup['target']} ({best_speedup['speedup']:.2f}x), "
              f"worst: {worst_speedup['target']} ({worst_speedup['speedup']:.2f}x)\n{'='*70}\n")
    else:
        return plt.gcf()


def grover_oracle(n_qubits, marked_state):
    oracle = QuantumCircuit(n_qubits)

    # Map marked state to |111...>
    for i, bit in enumerate(marked_state):
        if bit == '0':
            oracle.x(i)

    # Phase flip based on qubit count
    if n_qubits == 1:
        oracle.z(0)

    elif n_qubits == 2:
        oracle.cz(0, 1)

    else:
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)

    # Undo X mapping
    for i, bit in enumerate(marked_state):
        if bit == '0':
            oracle.x(i)

    gate = oracle.to_gate()
    gate.name = "Oracle"
    return gate


def diffusion_operator(n):
    diff = QuantumCircuit(n)

    diff.h(range(n))
    diff.x(range(n))

    if n == 1:
        diff.z(0)

    elif n == 2:
        diff.h(1)
        diff.cz(0, 1)
        diff.h(1)

    else:
        diff.h(n - 1)
        diff.mcx(list(range(n - 1)), n - 1)
        diff.h(n - 1)

    diff.x(range(n))
    diff.h(range(n))

    gate = diff.to_gate()
    gate.name = "Diffusion"
    return gate


def grover_circuit(n_qubits, marked_state):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = grover_oracle(n_qubits, marked_state)
    diffuser = diffusion_operator(n_qubits)

    iterations = int(math.floor(math.pi / 4 * math.sqrt(2**n_qubits)))

    for _ in range(iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser, range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def get_circuit_stats(n_qubits, marked_state):
    """Return basic structural statistics for the Grover circuit."""
    qc, iterations = grover_circuit(n_qubits, marked_state)
    return {
        "iterations": iterations,
        "depth": qc.depth(),
        "size": qc.size(),
        "num_qubits": qc.num_qubits,
        "num_clbits": qc.num_clbits,
    }


def plot_grover_circuit_diagram(n_qubits, marked_state, show=True):
    """Return a matplotlib figure with the Grover circuit diagram."""
    qc, _ = grover_circuit(n_qubits, marked_state)
    fig = qc.draw(output="mpl")
    if show:
        plt.show()
    else:
        return fig


def create_noise_model(error_1q=0.01, error_2q=0.02):
    noise = NoiseModel()

    error_1q = depolarizing_error(error_1q, 1)
    error_2q = depolarizing_error(error_2q, 2)

    noise.add_all_qubit_quantum_error(error_1q, ['h', 'x'])
    noise.add_all_qubit_quantum_error(error_2q, ['cx'])

    return noise


def run_quantum(
    n_qubits,
    target,
    shots=1024,
    noise_enabled=True,
    error_1q=0.01,
    error_2q=0.02,
):
    """Run Grover's algorithm on ideal and (optionally) noisy simulators.

    Returns:
        iterations: number of Grover iterations applied
        ideal_prob: success probability on the ideal simulator
        noisy_prob: success probability on the noisy simulator (0 if disabled)
        ideal_counts: raw counts from the ideal simulator
        noisy_counts: raw counts from the noisy simulator (empty if disabled)
    """
    qc, iterations = grover_circuit(n_qubits, target)

    # Ideal simulator
    ideal_sim = AerSimulator()
    qc_ideal = transpile(qc, ideal_sim)
    ideal_counts = ideal_sim.run(qc_ideal, shots=shots).result().get_counts()
    ideal_prob = ideal_counts.get(target, 0) / shots

    # Optional noisy simulator
    if noise_enabled:
        noisy_sim = AerSimulator(noise_model=create_noise_model(error_1q, error_2q))
        qc_noisy = transpile(qc, noisy_sim)
        noisy_counts = noisy_sim.run(qc_noisy, shots=shots).result().get_counts()
        noisy_prob = noisy_counts.get(target, 0) / shots
    else:
        noisy_counts = {}
        noisy_prob = 0.0

    return iterations, ideal_prob, noisy_prob, ideal_counts, noisy_counts


def simulate_grover_evolution(n_qubits, marked_state, max_iterations=None):
    if max_iterations is None:
        max_iterations = int(math.floor(math.pi / 4 * math.sqrt(2**n_qubits)))
    
    oracle = grover_oracle(n_qubits, marked_state)
    diffuser = diffusion_operator(n_qubits)
    
    # Initialize state - equal superposition
    qc_init = QuantumCircuit(n_qubits)
    qc_init.h(range(n_qubits))
    state = Statevector.from_instruction(qc_init)
    
    # Store probabilities at each step
    probabilities = []
    states = []
    iteration_numbers = []
    
    # Initial state (after Hadamard)
    probs = np.abs(state.data) ** 2
    # Normalize to ensure probabilities sum to 1 (numerical precision)
    probs = probs / np.sum(probs)
    probabilities.append(probs.copy())
    states.append("Initial (Hadamard)")
    iteration_numbers.append(0)
    
    # Apply oracle and diffuser iteratively
    for i in range(max_iterations):
        # Apply oracle
        qc_oracle = QuantumCircuit(n_qubits)
        qc_oracle.append(oracle, range(n_qubits))
        state = state.evolve(qc_oracle)
        probs = np.abs(state.data) ** 2
        # Normalize to ensure probabilities sum to 1 (numerical precision)
        probs = probs / np.sum(probs)
        probabilities.append(probs.copy())
        states.append(f"After Oracle {i+1}")
        iteration_numbers.append(i + 0.5)  # Half iteration (after oracle, before diffusion)
        
        # Apply diffuser
        qc_diff = QuantumCircuit(n_qubits)
        qc_diff.append(diffuser, range(n_qubits))
        state = state.evolve(qc_diff)
        probs = np.abs(state.data) ** 2
        # Normalize to ensure probabilities sum to 1 (numerical precision)
        probs = probs / np.sum(probs)
        probabilities.append(probs.copy())
        states.append(f"After Diffusion {i+1}")
        iteration_numbers.append(i + 1)  # Full iteration (after diffusion)
    
    return probabilities, states, iteration_numbers


def plot_superposition_state(n_qubits, show=True):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    state = Statevector.from_instruction(qc)
    
    # Get amplitudes (not probabilities)
    amplitudes = np.abs(state.data)
    
    # Generate all possible states
    all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    
    # Create the plot - smaller size for better text visibility
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(range(len(all_states)), amplitudes, color='blue', alpha=0.7, width=0.5)
    
    # Set labels and title
    plt.xlabel('All states equally likely', fontsize=10, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=10, fontweight='bold')
    plt.title(f'Step 1: Superposition ({n_qubits} qubit{"s" if n_qubits > 1 else ""})', 
              fontsize=11, fontweight='bold', pad=12)
    
    # Set x-axis ticks and labels
    plt.xticks(range(len(all_states)), all_states, rotation=0, fontsize=9)
    
    # Set y-axis range (amplitude for equal superposition is 1/√N)
    max_amplitude = 1.0 / np.sqrt(2**n_qubits)
    plt.ylim([0, max_amplitude * 1.25])  # Reduced space for compact view
    
    # Set y-axis ticks for better readability
    y_ticks = np.arange(0, max_amplitude * 1.25 + 0.05, 0.05)
    plt.yticks(y_ticks, [f'{tick:.2f}' for tick in y_ticks], fontsize=8)
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars - positioned above with spacing
    for i, (bar, amp) in enumerate(zip(bars, amplitudes)):
        height = bar.get_height()
        y_pos = height + max_amplitude * 0.04
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{amp:.3f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Add info box at bottom - smaller
    info_text = f'Amplitude: {max_amplitude:.4f} (1/√{2**n_qubits})'
    plt.text(0.5, 0.03, info_text, transform=plt.gca().transAxes, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black'),
             fontsize=8, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    if show:
        plt.show()
        print(f"  Superposition ({n_qubits} qubit{'s' if n_qubits > 1 else ''}): {2**n_qubits} states, amplitude {max_amplitude:.4f}\n")
    else:
        return plt.gcf()


def plot_advanced_grover_steps(n_qubits, marked_state, show=True):
    oracle = grover_oracle(n_qubits, marked_state)
    diffuser = diffusion_operator(n_qubits)
    
    # Generate all possible states
    all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    target_idx = int(marked_state, 2)
    
    # Step 1: Initial Superposition
    qc_init = QuantumCircuit(n_qubits)
    qc_init.h(range(n_qubits))
    state_init = Statevector.from_instruction(qc_init)
    amplitudes_init = np.abs(state_init.data)
    probs_init = amplitudes_init ** 2
    
    # Step 2: After Oracle
    qc_oracle = QuantumCircuit(n_qubits)
    qc_oracle.append(oracle, range(n_qubits))
    state_oracle = state_init.evolve(qc_oracle)
    amplitudes_oracle = np.abs(state_oracle.data)
    probs_oracle = amplitudes_oracle ** 2
    # Get phase information for visualization
    phases_oracle = np.angle(state_oracle.data)
    
    # Step 3: After Diffusion
    qc_diff = QuantumCircuit(n_qubits)
    qc_diff.append(diffuser, range(n_qubits))
    state_diff = state_oracle.evolve(qc_diff)
    amplitudes_diff = np.abs(state_diff.data)
    probs_diff = amplitudes_diff ** 2
    
    # Create figure with 3 subplots - smaller size for better text visibility
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Calculate max values for consistent y-axis scaling
    max_amp = max(max(amplitudes_init), max(amplitudes_oracle))
    max_prob = max(probs_diff)
    
    # Step 1: Superposition
    ax1 = axes[0]
    colors1 = ['red' if i == target_idx else 'blue' for i in range(len(all_states))]
    bars1 = ax1.bar(range(len(all_states)), amplitudes_init, color=colors1, alpha=0.7, width=0.5)
    bars1[target_idx].set_color('red')
    bars1[target_idx].set_alpha(1.0)
    ax1.set_xticks(range(len(all_states)))
    ax1.set_xticklabels(all_states, rotation=0, fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax1.set_title('Step 1: Superposition\n(Equal Superposition)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylim([0, max_amp * 1.25])  # Reduced space for compact view
    ax1.grid(True, alpha=0.3, axis='y')
    # Add value labels - positioned above bars
    for i, (bar, amp) in enumerate(zip(bars1, amplitudes_init)):
        height = bar.get_height()
        y_pos = height + max_amp * 0.04
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{amp:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    # Info box at bottom - smaller
    info_text1 = f'Amp: {amplitudes_init[0]:.4f} (1/√{2**n_qubits})'
    ax1.text(0.5, 0.03, info_text1, transform=ax1.transAxes, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black'),
             fontsize=9, fontweight='bold')
    
    # Step 2: After Oracle
    ax2 = axes[1]
    colors2 = ['red' if i == target_idx else 'blue' for i in range(len(all_states))]
    bars2 = ax2.bar(range(len(all_states)), amplitudes_oracle, color=colors2, alpha=0.7, width=0.5)
    bars2[target_idx].set_color('red')
    bars2[target_idx].set_alpha(1.0)
    ax2.set_xticks(range(len(all_states)))
    ax2.set_xticklabels(all_states, rotation=0, fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax2.set_title('Step 2: Oracle\n(Phase Flip)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_ylim([0, max_amp * 1.25])
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for i, (bar, amp) in enumerate(zip(bars2, amplitudes_oracle)):
        height = bar.get_height()
        y_pos = height + max_amp * 0.04
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{amp:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Add phase indicator below x-axis for target
        if i == target_idx:
            ax2.text(bar.get_x() + bar.get_width()/2., -max_amp * 0.06,
                    f'({"−" if phases_oracle[i] < 0 else "+"})', ha='center', va='top', 
                    fontsize=9, color='red', fontweight='bold')
    # Info box at bottom - smaller
    info_text2 = f'Target: {amplitudes_oracle[target_idx]:.4f}'
    ax2.text(0.5, 0.03, info_text2, transform=ax2.transAxes, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7, edgecolor='black'),
             fontsize=9, fontweight='bold')
    
    # Step 3: After Diffusion
    ax3 = axes[2]
    colors3 = ['red' if i == target_idx else 'blue' for i in range(len(all_states))]
    bars3 = ax3.bar(range(len(all_states)), probs_diff, color=colors3, alpha=0.7, width=0.5)
    bars3[target_idx].set_color('red')
    bars3[target_idx].set_alpha(1.0)
    ax3.set_xticks(range(len(all_states)))
    ax3.set_xticklabels(all_states, rotation=0, fontsize=10)
    ax3.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax3.set_title('Step 3: Diffusion\n(Amplification)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax3.set_ylim([0, max_prob * 1.25])
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels - only for significant probabilities
    for i, (bar, prob) in enumerate(zip(bars3, probs_diff)):
        height = bar.get_height()
        if prob > 0.05:  # Only show labels for significant probabilities
            y_pos = height + max_prob * 0.04
            ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    # Info box at bottom - smaller
    info_text3 = f'Target: {probs_diff[target_idx]:.4f} ({probs_diff[target_idx]*100:.1f}%)'
    ax3.text(0.5, 0.03, info_text3, transform=ax3.transAxes, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='black'),
             fontsize=9, fontweight='bold')
    
    # Add overall title - smaller
    plt.suptitle(f"Grover's Algorithm: Three Key Steps (n={n_qubits} qubits, target={marked_state})", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    if show:
        plt.show()
    else:
        return plt.gcf()


def plot_grover_simulation(n_qubits, marked_state, show=True):
    probabilities, state_labels, _ = simulate_grover_evolution(n_qubits, marked_state)
    
    # Generate all possible states
    all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    target_idx = int(marked_state, 2)
    
    # Create subplots for key iterations
    num_steps = len(probabilities)
    # Calculate grid dimensions
    cols = min(3, num_steps)  # Max 3 columns
    rows = (num_steps + cols - 1) // cols  # Ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))  # Smaller size for better text visibility
    
    # Handle different subplot configurations
    # plt.subplots can return: single Axes (1,1), 1D array (1,n), or 2D array (n,m)
    if not hasattr(axes, 'flatten'):
        # Single subplot case - axes is a single Axes object
        axes = np.array([axes])
    else:
        # Array case - flatten to 1D (works for both 1D and 2D arrays)
        axes = axes.flatten()
    
    for step, (prob, label) in enumerate(zip(probabilities, state_labels)):
        ax = axes[step]
        colors = ['red' if i == target_idx else 'blue' for i in range(len(all_states))]
        bars = ax.bar(range(len(all_states)), prob, color=colors, alpha=0.7, width=0.5)
        ax.set_xticks(range(len(all_states)))
        ax.set_xticklabels(all_states, rotation=45 if len(all_states) > 4 else 0, 
                           ha='right' if len(all_states) > 4 else 'center', fontsize=8)
        ax.set_ylabel('Probability', fontsize=9, fontweight='bold')
        ax.set_title(label, fontsize=10, fontweight='bold', pad=8)
        ax.set_ylim([0, 1.12])  # Reduced space for compact view
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight target state
        bars[target_idx].set_color('red')
        bars[target_idx].set_alpha(1.0)
        
        # Add value labels only for significant probabilities to avoid clutter
        for i, (bar, p) in enumerate(zip(bars, prob)):
            if p > 0.15:  # Only label significant probabilities
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                       f'{p:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Hide unused subplots
    for i in range(num_steps, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"Grover's Algorithm Simulation (n={n_qubits}, target={marked_state})", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return plt.gcf()


def plot_grover_evolution_animated(n_qubits, marked_state, show=True):
    probabilities, state_labels, iteration_numbers = simulate_grover_evolution(n_qubits, marked_state)
    target_idx = int(marked_state, 2)
    
    # Extract target probability at each step
    target_probs = [prob[target_idx] for prob in probabilities]
    
    # Use the iteration numbers from simulation
    iterations = iteration_numbers
    
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, target_probs, marker='o', linewidth=2, markersize=6, 
             label=f'Target state: {marked_state}', color='red')
    plt.xlabel('Iteration Number', fontsize=10, fontweight='bold')
    plt.ylabel('Probability', fontsize=10, fontweight='bold')
    plt.title(f"Target State Probability Evolution (n={n_qubits})", 
              fontsize=11, fontweight='bold', pad=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    plt.xlim([-0.1, max(iterations) + 0.1])
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Maximum (1.0)')
    plt.legend(fontsize=9)
    plt.tick_params(labelsize=9)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return plt.gcf()


def plot_comparative_study(n_qubits, classical_steps, grover_iters, ideal_prob, noisy_prob, show=True):
    fig = plt.figure(figsize=(12, 7))
    
    # Calculate speedup
    speedup = classical_steps / grover_iters if grover_iters > 0 else 0
    
    # 1. Direct Comparison Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    methods = ["Classical\nSearch", "Grover's\nAlgorithm"]
    steps = [classical_steps, grover_iters]
    colors = ['#FF6B6B', '#4ECDC4']
    bars1 = ax1.bar(methods, steps, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2, width=0.6)
    ax1.set_ylabel('Steps/Iterations', fontsize=9, fontweight='bold')
    ax1.set_title(f'Direct Comparison\n(n={n_qubits} qubits)', 
                  fontsize=10, fontweight='bold', pad=8)
    ax1.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, step in zip(bars1, steps):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(steps) * 0.03,
                f'{step}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    # Add speedup annotation - smaller
    ax1.text(0.5, 0.92, f'Speedup: {speedup:.2f}x', transform=ax1.transAxes,
             ha='center', va='top', fontsize=8, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=2))
    
    # 2. Time Complexity Comparison
    ax2 = plt.subplot(2, 3, 2)
    qubit_range = list(range(1, min(n_qubits + 4, 8)))
    classical_complexity = [2**n for n in qubit_range]
    quantum_complexity = [int(math.floor(math.pi / 4 * math.sqrt(2**n))) for n in qubit_range]
    ax2.plot(qubit_range, classical_complexity, marker='o', linewidth=2, 
             markersize=6, label='Classical O(N)', color='#FF6B6B')
    ax2.plot(qubit_range, quantum_complexity, marker='s', linewidth=2, 
             markersize=6, label='Grover O(√N)', color='#4ECDC4')
    ax2.set_xlabel('Number of Qubits', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Steps Required', fontsize=9, fontweight='bold')
    ax2.set_title('Time Complexity Scaling', fontsize=10, fontweight='bold', pad=8)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale to show exponential difference
    ax2.tick_params(labelsize=8)
    
    # Highlight current qubit count - smaller markers
    if n_qubits in qubit_range:
        idx = qubit_range.index(n_qubits)
        ax2.plot(n_qubits, classical_complexity[idx], 'ro', markersize=8, 
                markeredgecolor='black', markeredgewidth=1.5)
        ax2.plot(n_qubits, quantum_complexity[idx], 'bs', markersize=8, 
                markeredgecolor='black', markeredgewidth=1.5)
    
    # 3. Success Rate Comparison
    ax3 = plt.subplot(2, 3, 3)
    success_data = {
        'Classical': 1.0,
        'Ideal Q': ideal_prob,
        'Noisy Q': noisy_prob
    }
    colors3 = ['#95E1D3', '#F38181', '#AA96DA']
    bars3 = ax3.bar(success_data.keys(), success_data.values(), color=colors3, 
                    alpha=0.8, edgecolor='black', linewidth=1.2, width=0.6)
    ax3.set_ylabel('Success Probability', fontsize=9, fontweight='bold')
    ax3.set_title('Success Rate', fontsize=10, fontweight='bold', pad=8)
    ax3.set_ylim([0, 1.15])
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels - simplified
    for bar, (method, prob) in zip(bars3, success_data.items()):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{prob:.2f}\n({prob*100:.0f}%)', ha='center', va='bottom', 
                fontsize=8, fontweight='bold')
    ax3.tick_params(labelsize=8)
    
    # 4. Efficiency Ratio
    ax4 = plt.subplot(2, 3, 4)
    efficiency_ratio = classical_steps / grover_iters if grover_iters > 0 else 0
    categories = ['Efficiency\nRatio']
    ax4.barh(categories, [efficiency_ratio], color='#FFD93D', alpha=0.8, 
             edgecolor='black', linewidth=1.2, height=0.4)
    ax4.set_xlabel('Speedup Factor', fontsize=9, fontweight='bold')
    ax4.set_title(f'Quantum Advantage\n({efficiency_ratio:.2f}x faster)', 
                  fontsize=10, fontweight='bold', pad=8)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.text(efficiency_ratio/2, 0, f'{efficiency_ratio:.2f}x', 
             ha='center', va='center', fontsize=10, fontweight='bold')
    ax4.tick_params(labelsize=8)
    
    # 5. Search Space Visualization
    ax5 = plt.subplot(2, 3, 5)
    search_space = 2**n_qubits
    classical_checks = list(range(1, min(classical_steps + 1, search_space + 1)))
    
    ax5.plot(classical_checks, classical_checks, 'o-', linewidth=1.5, markersize=4,
             label='Classical', color='#FF6B6B', alpha=0.7)
    if grover_iters > 0:
        ax5.axhline(y=grover_iters, color='#4ECDC4', linewidth=2.5, 
                   label=f'Grover ({grover_iters})', linestyle='--')
    ax5.set_xlabel('Items Checked', fontsize=9, fontweight='bold')
    ax5.set_ylabel('Steps Required', fontsize=9, fontweight='bold')
    ax5.set_title('Search Strategy', fontsize=10, fontweight='bold', pad=8)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, min(search_space, 20)])
    ax5.set_ylim([0, max(classical_steps, grover_iters) * 1.15])
    ax5.tick_params(labelsize=8)
    
    # 6. Summary Statistics Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table - more compact
    summary_data = [
        ['Metric', 'Classical', 'Quantum'],
        ['Search Space', f'{2**n_qubits}', f'{2**n_qubits}'],
        ['Steps', f'{classical_steps}', f'{grover_iters}'],
        ['Complexity', 'O(N)', 'O(√N)'],
        ['Success', '100%', f'{ideal_prob*100:.0f}%'],
        ['Speedup', '-', f'{speedup:.1f}x']
    ]
    
    table = ax6.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.35, 0.32, 0.33])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Style the table
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white', size=8)
    
    for i in range(1, len(summary_data)):
        if i % 2 == 0:
            table[(i, 0)].set_facecolor('#F0F0F0')
            table[(i, 1)].set_facecolor('#FFE5E5')
            table[(i, 2)].set_facecolor('#E5F5F5')
    
    ax6.set_title('Summary', fontsize=10, fontweight='bold', pad=15)
    
    # Overall title - smaller
    plt.suptitle(f'Comparative Study: Classical vs Quantum (n={n_qubits} qubits)', 
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    if show:
        plt.show()
        print(f"\nComparative study (n={n_qubits}): classical {classical_steps} steps, Grover {grover_iters} iters, speedup {speedup:.2f}x")
    else:
        return plt.gcf()


def plot_classical_vs_quantum(classical, quantum, n, show=True):
    plt.figure(figsize=(6, 4))
    methods = ["Classical Search", "Grover Search"]
    steps = [classical, quantum]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = plt.bar(methods, steps, color=colors, alpha=0.8, edgecolor='black', 
                   linewidth=1.2, width=0.6)
    plt.ylabel("Attempts / Iterations", fontsize=10, fontweight='bold')
    plt.title(f"Classical vs Quantum Search (n = {n} qubits)", 
              fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tick_params(labelsize=9)
    
    # Add value labels
    for bar, step in zip(bars, steps):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(steps) * 0.03,
                f'{step}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add speedup annotation - smaller
    speedup = classical / quantum if quantum > 0 else 0
    plt.text(0.5, 0.92, f'Speedup: {speedup:.2f}x', transform=plt.gca().transAxes,
             ha='center', va='top', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=3))
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return plt.gcf()


def plot_scaling(classical_list, quantum_list, qubits, show=True):
    plt.figure(figsize=(7, 4.5))
    plt.plot(qubits, classical_list, marker='o', linewidth=2, markersize=6, 
             label="Classical O(N)", color='#FF6B6B')
    plt.plot(qubits, quantum_list, marker='s', linewidth=2, markersize=6, 
             label="Grover O(√N)", color='#4ECDC4')
    plt.xlabel("Number of Qubits", fontsize=10, fontweight='bold')
    plt.ylabel("Steps / Iterations", fontsize=10, fontweight='bold')
    plt.title("Scaling: Classical vs Quantum Search", fontsize=11, fontweight='bold', pad=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=9)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return plt.gcf()


def plot_success_probability(ideal_probs, noisy_probs, qubits, show=True):
    plt.figure(figsize=(7, 4.5))
    plt.plot(qubits, ideal_probs, marker='o', linewidth=2, markersize=6, 
             label="Ideal Quantum", color='#95E1D3')
    plt.plot(qubits, noisy_probs, marker='s', linewidth=2, markersize=6, 
             label="Noisy Quantum", color='#F38181')
    plt.xlabel("Number of Qubits", fontsize=10, fontweight='bold')
    plt.ylabel("Success Probability", fontsize=10, fontweight='bold')
    plt.title("Grover Success Probability vs Qubits", fontsize=11, fontweight='bold', pad=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    plt.tick_params(labelsize=9)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return plt.gcf()


if __name__ == "__main__":
    while True:
        try:
            n_qubits = int(input("How many qubits would you like to simulate? (1-5 recommended): "))
            if n_qubits < 1:
                print("Please enter a positive number (1 or greater).")
                continue
            if n_qubits > 6:
                response = input(f"Warning: {n_qubits} qubits will create {2**n_qubits} states. This may take a while. Continue? (y/n): ")
                if response.lower() != 'y':
                    continue
            break
        except ValueError:
            print("Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            exit(0)
    
    while True:
        try:
            compare_choice = input("Do you want to compare multiple target states? (y/n): ").strip().lower()
            if compare_choice in ['y', 'yes']:
                compare_mode = True
                break
            elif compare_choice in ['n', 'no']:
                compare_mode = False
                break
            else:
                print("Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            exit(0)
    
    if compare_mode:
        target_list = []
        print("Enter target states to compare (enter 'done' when finished):")
        while True:
            try:
                target = get_target_state(n_qubits, allow_custom=True)
                if target not in target_list:
                    target_list.append(target)
                    print(f"✓ Added '{target}' to comparison list ({len(target_list)} target{'s' if len(target_list) > 1 else ''})")
                else:
                    print(f"✗ '{target}' already in list")
                
                if len(target_list) >= 2**n_qubits:
                    print("\nAll possible states added!")
                    break
                
                more = input("\nAdd another target? (y/n): ").strip().lower()
                if more not in ['y', 'yes']:
                    break
            except KeyboardInterrupt:
                print("\n\nExiting...")
                exit(0)
        
        if len(target_list) < 2:
            print("\nNeed at least 2 targets to compare. Using default targets...")
            target_list = ["0" * n_qubits, "1" * n_qubits]
        
        # Run comparison
        compare_target_states(n_qubits, target_list)
        
        # Ask if user wants to see detailed analysis for one target
        while True:
            try:
                detail_choice = input("\nDo you want detailed analysis for a specific target? (y/n): ").strip().lower()
                if detail_choice in ['y', 'yes']:
                    target = get_target_state(n_qubits, allow_custom=True)
                    if target not in target_list:
                        print(f"Note: '{target}' was not in comparison list, but will be analyzed.")
                    break
                elif detail_choice in ['n', 'no']:
                    print("\nComparison complete!")
                    exit(0)
                else:
                    print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                exit(0)
    else:
        # Single target mode
        target = get_target_state(n_qubits, allow_custom=True)
    
    passwords = generate_passwords(n_qubits)
    
    # Run classical search
    classical_steps = classical_search(passwords, target)
    
    # Run quantum search
    grover_iters, ideal_p, noisy_p, ideal_counts, noisy_counts = run_quantum(n_qubits, target)
    
    print(f"\nResults: target={target}, classical steps={classical_steps}, Grover iters={grover_iters}, "
          f"ideal p={ideal_p:.2%}, noisy p={noisy_p:.2%}\n")

    plot_superposition_state(n_qubits)
    plot_advanced_grover_steps(n_qubits, target)
    plot_grover_simulation(n_qubits, target)
    plot_grover_evolution_animated(n_qubits, target)
    plot_comparative_study(n_qubits, classical_steps, grover_iters, ideal_p, noisy_p)
    plot_classical_vs_quantum(classical_steps, grover_iters, n_qubits)
    print(f"Measurements (1024 shots): ideal target={ideal_counts.get(target, 0)}, noisy target={noisy_counts.get(target, 0)}\n")
