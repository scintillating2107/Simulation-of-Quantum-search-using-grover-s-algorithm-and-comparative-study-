# Simulation of Quantum Search Model Using Grover's Algorithm and Comparative Study of Quantum Search with Classical Search

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Latest-green.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive simulation and analysis tool for Grover's Quantum Search Algorithm, featuring interactive visualizations, comparative studies, and detailed performance analysis comparing quantum and classical search methods.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Visualizations](#visualizations)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project demonstrates **Grover's Quantum Search Algorithm** and provides a comprehensive comparison with classical brute-force search methods. The implementation includes:

- **Interactive quantum search simulation** with customizable qubit counts and target states
- **Step-by-step visualization** of quantum state evolution
- **Comparative analysis** between classical and quantum search performance
- **Noise modeling** to simulate realistic quantum hardware
- **Multiple visualization tools** for understanding quantum algorithms

### Key Achievements

- ✅ Complete Grover's algorithm implementation
- ✅ Quadratic speedup demonstration (O(√N) vs O(N))
- ✅ Interactive target state selection
- ✅ Comprehensive comparative study
- ✅ 8+ different visualization types
- ✅ Noise impact analysis

## ✨ Features

### Core Features

1. **Interactive Qubit Selection**
   - Choose any number of qubits (1-6+ recommended)
   - Dynamic search space generation
   - Validation and warnings for large configurations

2. **Custom Target State Selection**
   - Direct binary string input
   - Index-based selection
   - Quick worst-case option
   - Multiple target comparison mode

3. **Classical vs Quantum Comparison**
   - Brute-force classical search implementation
   - Grover's quantum search algorithm
   - Speedup factor calculation
   - Performance metrics analysis

4. **Noise Modeling**
   - Ideal quantum computer (noiseless)
   - Realistic noisy quantum computer
   - Depolarizing noise model (1% single-qubit, 2% two-qubit errors)
   - Success rate comparison

5. **State Evolution Tracking**
   - Step-by-step probability distribution
   - Phase information tracking
   - Amplitude amplification visualization

### Visualization Features

- **Step 1: Superposition** - Initial equal superposition state
- **Advanced 3-Step View** - Superposition → Oracle → Diffusion
- **Complete Evolution** - Full iteration-by-iteration breakdown
- **Probability Evolution Curve** - Target state probability over iterations
- **6-Panel Comparative Study** - Comprehensive analysis dashboard
- **Target State Comparison** - Multi-target performance analysis
- **Scaling Analysis** - Complexity growth visualization
- **Success Probability Analysis** - Noise impact visualization

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/scintillating2107/Simulation-of-Quantum-search-and-comparative-study-of-quantum-search-with-classical-search.git
   cd Simulation-of-Quantum-search-and-comparative-study-of-quantum-search-with-classical-search
   ```

2. **Install dependencies:**
   ```bash
   pip install qiskit qiskit-aer numpy matplotlib
   ```

   Or install from requirements.txt (if available):
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Basic Usage

Run the main simulation:

```bash
python grovers.py
```

### Interactive Flow

1. **Select Qubit Count**
   - Enter number of qubits (1-5 recommended)
   - Confirm for larger configurations

2. **Choose Mode**
   - Single target: Detailed analysis
   - Multiple targets: Comparison study

3. **Select Target State(s)**
   - Option 1: Direct binary string (e.g., "101")
   - Option 2: Index number (0 to 2^n-1)
   - Option 3: Worst-case (all 1s)

4. **View Results**
   - Multiple visualizations appear
   - Console shows detailed statistics
   - Close plots to continue

### Example Session

```
How many qubits would you like to simulate? (1-5 recommended): 3

Do you want to compare multiple target states? (y/n): n

Options:
  [1] Enter target state directly (e.g., '101' for 3 qubits)
  [2] Enter index number (0-7)
  [3] Use worst-case (all 1s: 111)
  Or enter target state directly as binary string

Enter option [1/2/3] or target state: 3
✓ Selected worst-case target: 111
```

## 📁 Project Structure

```
.
├── grovers.py          # Main simulation code
├── ABOUT.md            # Visualization explanations
├── REPORT.md           # Comprehensive project report
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## 📊 Visualizations

### 1. Step 1: Superposition
Shows initial equal superposition of all quantum states with amplitudes.

### 2. Advanced 3-Step Visualization
Side-by-side comparison of:
- **Step 1:** Superposition (equal amplitudes)
- **Step 2:** Oracle (phase flip on target)
- **Step 3:** Diffusion (amplitude amplification)

### 3. Complete Step-by-Step Evolution
Multi-panel visualization showing probability distribution at each iteration.

### 4. Probability Evolution Curve
Line plot showing how target state probability changes with iterations.

### 5. Comprehensive Comparative Study
6-panel dashboard including:
- Direct comparison bar chart
- Time complexity scaling
- Success rate comparison
- Efficiency ratio
- Search strategy comparison
- Summary statistics table

### 6. Target State Comparison
4-panel analysis comparing multiple target states.

## 📈 Results

### Performance Comparison

| Qubits | States | Classical (Worst) | Grover | Speedup |
|--------|--------|-------------------|--------|---------|
| 1      | 2      | 2                 | 1      | 2.0x    |
| 2      | 4      | 4                 | 1      | 4.0x    |
| 3      | 8      | 8                 | 2      | 4.0x    |
| 4      | 16     | 16                | 3      | 5.3x    |
| 5      | 32     | 32                | 4      | 8.0x    |
| 10     | 1024   | 1024              | 25     | 41.0x   |

### Key Findings

- ✅ **Quadratic Speedup:** Grover's algorithm provides O(√N) vs O(N) for classical
- ✅ **Target Independence:** Grover iterations constant regardless of target position
- ✅ **Optimal Iterations:** Formula ⌊π/4 × √N⌋ provides near-optimal results
- ✅ **Noise Impact:** Ideal quantum ~95-100% success, Noisy ~85-98% success

## 📚 Documentation

- **[REPORT.md](REPORT.md)** - Comprehensive project report with detailed explanations
- **[ABOUT.md](ABOUT.md)** - Detailed explanation of all visualizations and graphs

## 🔬 Technical Details

### Algorithm Implementation

- **Oracle:** Phase flip on target state using controlled gates
- **Diffusion:** Inversion about the mean using Hadamard and controlled gates
- **Optimal Iterations:** ⌊π/4 × √(2^n)⌋

### Noise Model

- Single-qubit gate errors: 1% depolarizing noise
- Two-qubit gate errors: 2% depolarizing noise

## 🎓 Applications

- Password cracking demonstration
- Database search algorithms
- Optimization problems
- Machine learning (feature selection)
- Cryptography (key search)
- Educational purposes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**scintillating2107**

- GitHub: [@scintillating2107](https://github.com/scintillating2107)

## 🙏 Acknowledgments

- Qiskit team for the quantum computing framework
- IBM Quantum for quantum computing resources
- Grover's algorithm original paper by Lov K. Grover

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⭐ If you find this project useful, please consider giving it a star!**

