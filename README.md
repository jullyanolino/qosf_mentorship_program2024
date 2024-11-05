# QOSF Mentorship Program2024
Screening task to QOSF Mentorship Program 2024

## Quantum Circuit Simulator

Welcome to the Quantum Circuit Simulator, a Python-based project that allows for the simulation of quantum circuits using both matrix-based and tensor-based approaches. This project is designed to demonstrate fundamental principles in quantum computing, including gate operations, state vector evolution, and quantum measurement. It also offers benchmarking capabilities to evaluate performance and scalability of different simulation techniques.

### Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Getting Started](#getting-started)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Examples](#examples)
8. [Benchmarking](#benchmarking)
9. [Questions and Research](#questions-and-research)
10. [Contributing](#contributing)
11. [License](#license)

---

### Project Overview

Quantum Circuit Simulator is a project created to emulate quantum circuits on classical hardware. It serves as both a learning tool and a testing environment for simulating quantum algorithms, gate applications, and measurement outcomes. The simulator offers two methods for gate application: a **naive matrix multiplication** approach and an **advanced tensor multiplication** approach, providing an opportunity to compare efficiency and scalability between these methods.

This simulator is designed for educational purposes, targeting users who want to explore and learn about quantum computing concepts without requiring access to quantum hardware.

### Features

- **Quantum Gates**: Supports various gates like Identity, Pauli-X, Hadamard, and CNOT.
- **Matrix-based Simulation**: Uses the Kronecker product to construct full gate matrices.
- **Tensor-based Simulation**: Uses tensor operations to improve simulation efficiency.
- **Measurement Function**: Samples from the final state vector to simulate measurement outcomes.
- **Expectation Value Calculation**: Computes expectation values for operators on quantum states.
- **Benchmarking**: Provides runtime benchmarking for both simulation methods.

### Getting Started

#### Prerequisites

To run this project, you will need:

- Python 3.8 or higher
- Libraries listed in `requirements.txt`

#### Installation

Clone the repository and install dependencies.

```bash
git clone https://github.com/your-username/quantum-circuit-simulator.git
cd quantum-circuit-simulator
pip install -r requirements.txt
```

#### Usage
##### Example Usage in Python
This project includes sample scripts, unit tests, and a Jupyter notebook demonstrating the simulator’s features.
1. Run the Jupyter Notebook:
  - Open example_usage.ipynb in Jupyter Notebook to see code examples and visualizations.
2. Run the Benchmarking Script:
  - Execute plot_benchmarks() to benchmark the simulator's performance with increasing qubit counts.
3. Run Unit Tests:
  - Execute the tests in test_quantum_simulator.py to validate functionality.

```bash
python -m unittest test_quantum_simulator.py
```

##### Quick Code Example
```python
from quantum_simulator import QuantumSimulator, QuantumGate

# Initialize the simulator for 2 qubits
sim = QuantumSimulator(2)

# Define gates
h_gate = QuantumGate(sim.H, [0], "Hadamard")
cnot_gate = QuantumGate(sim.CNOT, [0, 1], "CNOT")

# Apply gates and measure the outcome
sim.apply_gate_matrix(h_gate)
sim.apply_gate_matrix(cnot_gate)
results = sim.measure(samples=1000)

print("Measurement results:", results)
```

### Project Structure
```plaintext
QuantumCircuitSimulator/
├── src/
│   └── quantum_simulator.py       # Main code file
├── tests/
│   └── test_quantum_simulator.py  # Unit tests for simulator functionality
├── docs/
│   └── README.md                  # Overview and explanation of the project
├── examples/
│   └── example_usage.ipynb        # Jupyter notebook showing code use and results
└── requirements.txt               # List of dependencies
```

### Examples
1. Gate Operations and Measurement: The simulator allows you to apply quantum gates (e.g., Hadamard, CNOT) and perform measurements on qubits to observe the probabilistic nature of quantum states.
2. Expectation Value Calculation: Using the simulator, you can calculate expectation values of operators. For example, you can measure the expectation of the Z operator on the first qubit, a key calculation in many quantum algorithms.
3. Benchmarking: You can benchmark the matrix and tensor approaches to compare their performance as the qubit count increases. This is particularly useful for understanding how classical simulators handle the exponential growth of quantum state spaces.

### Benchmarking
The plot_benchmarks() function evaluates simulation times for different qubit counts. By comparing the matrix and tensor approaches, you can observe performance trends and limitations as the number of qubits increases.

#### Running Benchmarks
```python
from quantum_simulator import plot_benchmarks

plot_benchmarks(max_qubits=10)
```
This function will generate a plot displaying the runtime for each approach, helping users see which method performs better as the system grows.

#### Questions and Research
##### Research Questions
1. What are the benefits of matrix vs. tensor simulation?
  - The matrix approach is simple but scales poorly with qubit count. The tensor approach optimizes by reducing state space overhead but still faces exponential complexity.

2. Why is the Kronecker product critical in quantum simulation?
  - It enables correct matrix construction for applying single-qubit gates across multiple qubits.

3. How are quantum measurements simulated?
  - Measurements are computed by squaring amplitude probabilities and randomly sampling to match quantum probabilistic outcomes.

### Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request.
#### Guidelines
  - Follow Python best practices.
  - Add tests for new features.
  - Ensure code is well-documented and readable.

### License
This project is licensed under the MIT License.




