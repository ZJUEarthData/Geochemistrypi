import unittest
import numpy as np
from fsolve import equations, jacobi_matrix, solve_equations


class TestFSolve(unittest.TestCase):
    """Test cases for fsolve implementation."""

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

    def test_solve_equations(self):
        """Test the solve_equations function."""
        # Test with default initial guess
        solution = solve_equations()
        self.assertEqual(len(solution), 3,
                         "Solution should have 3 components")

        # Test with custom initial guess
        custom_guess = (0.6, 0.4, 1.5)
        solution = solve_equations(custom_guess)
        self.assertEqual(len(solution), 3,
                         "Solution should have 3 components")

    def test_solution_validity(self):
        """Test if the solution satisfies the equations."""
        solution = solve_equations()
        residuals = equations(solution)

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