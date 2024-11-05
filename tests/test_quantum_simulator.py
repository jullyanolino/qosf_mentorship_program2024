import unittest
import numpy as np
from src.quantum_simulator import QuantumSimulator, QuantumGate

class TestQuantumSimulator(unittest.TestCase):
    def setUp(self):
        # Set up a 2-qubit simulator and define some common gates
        self.simulator = QuantumSimulator(n_qubits=2)
        self.h_gate = QuantumGate(matrix=self.simulator.H, target_qubits=[0], name="H")
        self.x_gate = QuantumGate(matrix=self.simulator.X, target_qubits=[1], name="X")
        self.cnot_gate = QuantumGate(matrix=self.simulator.CNOT, target_qubits=[0, 1], name="CNOT")

    def test_initial_state(self):
        """Test if the simulator initializes the state vector and tensor correctly in |0...0> state."""
        expected_state_vector = np.zeros(2**self.simulator.n_qubits)
        expected_state_vector[0] = 1  # Only the |00> state should be 1
        np.testing.assert_array_almost_equal(self.simulator.state_vector, expected_state_vector, 
                                             err_msg="Initial state vector is incorrect.")

        expected_state_tensor = np.zeros([2] * self.simulator.n_qubits)
        expected_state_tensor[tuple([0] * self.simulator.n_qubits)] = 1  # Only the |00> state should be 1
        np.testing.assert_array_almost_equal(self.simulator.state_tensor, expected_state_tensor,
                                             err_msg="Initial state tensor is incorrect.")

    def test_apply_gate_matrix_h(self):
        """Test if applying an H gate on the first qubit works correctly in the matrix method."""
        self.simulator.apply_gate_matrix(self.h_gate)
        expected_state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        np.testing.assert_array_almost_equal(self.simulator.state_vector, expected_state, 
                                             err_msg="H gate application (matrix method) is incorrect.")

    def test_apply_gate_matrix_cnot(self):
        """Test if applying H on the first qubit and CNOT on both qubits works correctly in the matrix method."""
        self.simulator.apply_gate_matrix(self.h_gate)
        self.simulator.apply_gate_matrix(self.cnot_gate)
        expected_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(self.simulator.state_vector, expected_state, 
                                             err_msg="CNOT gate application (matrix method) is incorrect.")

    def test_apply_gate_tensor_h(self):
        """Test if applying an H gate on the first qubit works correctly in the tensor method."""
        self.simulator.apply_gate_tensor(self.h_gate)
        expected_state_tensor = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0]).reshape(2, 2)
        np.testing.assert_array_almost_equal(self.simulator.state_tensor.flatten(), expected_state_tensor.flatten(),
                                             err_msg="H gate application (tensor method) is incorrect.")

    def test_apply_gate_tensor_cnot(self):
        """Test if applying H on the first qubit and CNOT on both qubits works correctly in the tensor method."""
        self.simulator.apply_gate_tensor(self.h_gate)
        self.simulator.apply_gate_tensor(self.cnot_gate)
        expected_state_tensor = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]).reshape(2, 2)
        np.testing.assert_array_almost_equal(self.simulator.state_tensor.flatten(), expected_state_tensor.flatten(),
                                             err_msg="CNOT gate application (tensor method) is incorrect.")

    def test_measure(self):
        """Test if measurement results have the expected probabilities."""
        self.simulator.apply_gate_matrix(self.h_gate)
        self.simulator.apply_gate_matrix(self.cnot_gate)
        
        measurements = self.simulator.measure(samples=1000)
        # Expect roughly equal measurements of |00> and |11> states
        unique, counts = np.unique(measurements, return_counts=True)
        measured_probs = dict(zip(unique, counts / 1000))
        
        self.assertAlmostEqual(measured_probs.get(0, 0), 0.5, delta=0.1, 
                               msg="Probability of measuring |00> is incorrect.")
        self.assertAlmostEqual(measured_probs.get(3, 0), 0.5, delta=0.1, 
                               msg="Probability of measuring |11> is incorrect.")

    def test_expectation_value(self):
        """Test if expectation value calculation is correct for Z ⊗ I on |0> state."""
        # Initial state is |00> so <Z ⊗ I> should equal 1
        Z = np.array([[1, 0], [0, -1]])
        ZI = np.kron(Z, self.simulator.I)
        expected_value = 1
        computed_value = self.simulator.expectation_value(ZI)
        self.assertAlmostEqual(computed_value, expected_value, 
                               msg="Expectation value calculation is incorrect.")

if __name__ == "__main__":
    unittest.main()
