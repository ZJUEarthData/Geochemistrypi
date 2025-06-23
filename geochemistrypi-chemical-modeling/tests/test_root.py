import unittest
import numpy as np
from root import equations, jacobi_matrix
from scipy.optimize import root


class TestRoot(unittest.TestCase):
    """Test cases for root implementation."""

    def setUp(self):
        """Set up test cases."""
        self.test_params = (0.5, 0.5, 2.0)
        self.tolerance = 1e-6

    def test_equations(self):
        """Test the equations function."""
        # Test if equations returns correct number of equations
        result = equations(self.test_params)
        self.assertEqual(len(result), 3, "Should return 3 equations")

        # Test if equations returns float values
        self.assertTrue(all(isinstance(x, float) for x in result),
                        "All returned values should be float")

    def test_jacobi_matrix(self):
        """Test the Jacobian matrix function."""
        # Test matrix shape
        result = jacobi_matrix(self.test_params)
        self.assertEqual(result.shape, (3, 3),
                         "Jacobian matrix should be 3x3")

        # Test if matrix contains float values
        self.assertTrue(isinstance(result[0, 0], float),
                        "Matrix elements should be float")

    def test_root_solver(self):
        """Test the root solver."""
        # Test with default initial guess
        initial_guess = (0.5, 0.5, 2.0)
        solution = root(equations, initial_guess, jac=jacobi_matrix, method='hybr')

        # Check if solver converged
        self.assertTrue(solution.success,
                        "Root solver should converge")

        # Check solution dimension
        self.assertEqual(len(solution.x), 3,
                         "Solution should have 3 components")

    def test_solution_validity(self):
        """Test if the solution satisfies the equations."""
        initial_guess = (0.5, 0.5, 2.0)
        solution = root(equations, initial_guess, jac=jacobi_matrix, method='hybr')
        residuals = equations(solution.x)

        # Check if residuals are close to zero
        self.assertTrue(all(abs(r) < self.tolerance for r in residuals),
                        "Solution should satisfy equations")

    def test_input_validation(self):
        """Test input parameter validation."""
        # Test with invalid input types
        with self.assertRaises(TypeError):
            equations(["not", "a", "tuple"])

        with self.assertRaises(TypeError):
            jacobi_matrix(["not", "a", "tuple"])


if __name__ == '__main__':
    unittest.main()