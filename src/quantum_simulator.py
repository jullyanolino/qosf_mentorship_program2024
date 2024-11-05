import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Union
from dataclasses import dataclass

@dataclass
class QuantumGate:
    """Represents a quantum gate with its matrix and target qubits.

    Attributes:
        matrix (np.ndarray): The matrix representation of the quantum gate.
        target_qubits (List[int]): The list of qubit indices that the gate acts upon.
        name (str): The name of the gate for identification.
    """
    matrix: np.ndarray
    target_qubits: List[int]
    name: str

class QuantumSimulator:
    def __init__(self, n_qubits: int):
        """Initialize the quantum simulator with n qubits in |0> state.

        Args:
            n_qubits (int): The number of qubits in the simulator.
        """
        self.n_qubits = n_qubits
        # Initialize state vector as |0...0>
        self.state_vector = np.zeros(2**n_qubits)
        self.state_vector[0] = 1  # Set initial state to |0...0>

        # Initialize tensor state
        self.state_tensor = np.zeros([2] * n_qubits)
        self.state_tensor[tuple([0] * n_qubits)] = 1  # Set tensor state to |0...0>

        # Define basic gates
        self.I = np.array([[1, 0], [0, 1]])  # Identity gate
        self.X = np.array([[0, 1], [1, 0]])  # Pauli-X gate
        self.H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard gate
        self.CNOT = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]]).reshape(2, 2, 2, 2)  # CNOT gate

    def apply_gate_matrix(self, gate: QuantumGate) -> None:
        """Apply a quantum gate using matrix multiplication approach.

        Args:
            gate (QuantumGate): The quantum gate to apply.
        """
        # Build the full matrix for the n-qubit system
        full_matrix = 1  # Start with the identity matrix
        for i in range(self.n_qubits):
            if i in gate.target_qubits:
                idx = gate.target_qubits.index(i)
                if idx == 0 and len(gate.target_qubits) > 1:
                    # For the first qubit of a multi-qubit gate
                    current_matrix = gate.matrix.reshape(2, 2, 2, 2)
                    current_matrix = np.moveaxis(current_matrix, [0, 1, 2, 3], [0, 2, 1, 3])
                    current_matrix = current_matrix.reshape(4, 4)
                elif idx > 0:
                    continue  # Skip other qubits of multi-qubit gate
                else:
                    current_matrix = gate.matrix
            else:
                current_matrix = self.I  # Use identity for non-target qubits
            full_matrix = np.kron(full_matrix, current_matrix)  # Build full matrix using Kronecker product
        
        # Apply the gate to the state vector
        self.state_vector = full_matrix @ self.state_vector

    def apply_gate_tensor(self, gate: QuantumGate) -> None:
        """Apply a quantum gate using tensor multiplication approach.

        Args:
            gate (QuantumGate): The quantum gate to apply.
        """
        if len(gate.target_qubits) == 1:
            # Handle single qubit gate
            self.state_tensor = np.tensordot(
                gate.matrix,
                self.state_tensor,
                axes=([1], [gate.target_qubits[0]])  # Dot along the target qubit axis
            )
            self.state_tensor = np.moveaxis(
                self.state_tensor,
                0,
                gate.target_qubits[0]  # Move the tensor to the target qubit's position
            )
        elif len(gate.target_qubits) == 2:
            # Handle two qubit gates (like CNOT)
            temp = np.tensordot(
                gate.matrix,
                self.state_tensor,
                axes=([2, 3], gate.target_qubits)  # Dot along both target qubit axes
            )
            for i, axis in enumerate(gate.target_qubits):
                temp = np.moveaxis(temp, i, axis)  # Move tensor axes to match target qubit positions
            self.state_tensor = temp

    def measure(self, samples: int = 1000) -> np.ndarray:
        """Sample from the final state vector.

        Args:
            samples (int): The number of samples to take from the state vector.

        Returns:
            np.ndarray: An array of measured outcomes based on probabilities.
        """
        probabilities = np.abs(self.state_vector) ** 2  # Calculate probabilities of outcomes
        outcomes = np.arange(len(probabilities))  # Possible outcomes
        # Sample from outcomes based on their probabilities
        samples = np.random.choice(outcomes, size=samples, p=probabilities)
        return samples

    def expectation_value(self, operator: np.ndarray) -> complex:
        """Compute the expectation value <Ψ|Op|Ψ>.

        Args:
            operator (np.ndarray): The operator for which to compute the expectation value.

        Returns:
            complex: The computed expectation value.
        """
        return np.vdot(self.state_vector, operator @ self.state_vector)  # Use the bra-ket notation for expectation value

def benchmark_simulation(max_qubits: int, method: str = 'matrix') -> Tuple[List[int], List[float]]:
    """Benchmark simulation time vs number of qubits.

    Args:
        max_qubits (int): The maximum number of qubits to test.
        method (str): The method of simulation ('matrix' or 'tensor').

    Returns:
        Tuple[List[int], List[float]]: A tuple containing a list of qubit counts and their corresponding execution times.
    """
    qubit_range = range(2, max_qubits + 1)  # Range of qubit counts to benchmark
    times = []
    
    for n_qubits in qubit_range:
        simulator = QuantumSimulator(n_qubits)  # Create a new simulator instance for each qubit count
        # Simple test circuit: Apply Hadamard on the first qubit, CNOT on the first two qubits
        h_gate = QuantumGate(simulator.H, [0], "H")  # Hadamard gate on qubit 0
        cnot_gate = QuantumGate(simulator.CNOT, [0, 1], "CNOT")  # CNOT gate on qubits 0 and 1
        
        start_time = time.time()  # Record start time for benchmarking
        if method == 'matrix':
            simulator.apply_gate_matrix(h_gate)  # Apply Hadamard gate using matrix method
            simulator.apply_gate_matrix(cnot_gate)  # Apply CNOT gate using matrix method
        else:  # tensor method
            simulator.apply_gate_tensor(h_gate)  # Apply Hadamard gate using tensor method
            simulator.apply_gate_tensor(cnot_gate)  # Apply CNOT gate using tensor method
        times.append(time.time() - start_time)  # Record time taken for this simulation
    
    return list(qubit_range), times  # Return the list of qubit counts and their execution times

# Run benchmarks and plot results
def plot_benchmarks(max_qubits: int = 8) -> None:
    """Run benchmarks for both matrix and tensor simulation methods and plot the results.

    Args:
        max_qubits (int): The maximum number of qubits to benchmark.
    """
    # Get benchmarks for matrix and tensor methods
    qubits_matrix, times_matrix = benchmark_simulation(max_qubits, 'matrix')
    qubits_tensor, times_tensor = benchmark_simulation(max_qubits, 'tensor')
    
    # Create a plot to compare execution times
    plt.figure(figsize=(10, 6))
    plt.semilogy(qubits_matrix, times_matrix, 'o-', label='Matrix multiplication')
    plt.semilogy(qubits_tensor, times_tensor, 's-', label='Tensor multiplication')
    plt.xlabel('Number of qubits')  # Label for x-axis
    plt.ylabel('Execution time (seconds)')  # Label for y-axis
    plt.title('Quantum Circuit Simulation Performance')  # Title of the plot
    plt.grid(True)  # Add grid to the plot
    plt.legend()  # Add legend to the plot
    plt.show()  # Show the plot

# Example usage
if __name__ == "__main__":
    plot_benchmarks()  # Run the benchmark plotting function
    
    # Demonstrate sampling and expectation value computation
    sim = QuantumSimulator(2)  # Create a simulator with 2 qubits
    h_gate = QuantumGate(sim.H, [0], "H")  # Define Hadamard gate
    cnot_gate = QuantumGate(sim.CNOT, [0, 1], "CNOT")  # Define CNOT gate
    
    # Apply gates to the simulator
    sim.apply_gate_matrix(h_gate)  # Apply Hadamard gate
    sim.apply_gate_matrix(cnot_gate)  # Apply CNOT gate
    
    # Sample from the final state
    samples = sim.measure(samples=1000)  # Take 1000 samples from the final state vector
    print("Sampled outcomes:", samples)  # Print the sampled outcomes

    # Compute expectation value for the Pauli-Z operator on the first qubit
    pauli_z = np.array([[1, 0], [0, -1]])  # Pauli-Z operator
    exp_value = sim.expectation_value(pauli_z)  # Compute expectation value
    print("Expectation value of Pauli-Z:", exp_value)  # Print the expectation value
