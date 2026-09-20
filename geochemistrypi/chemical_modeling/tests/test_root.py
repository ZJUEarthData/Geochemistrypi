import unittest

import numpy as np
from scipy.optimize import root

from geochemistrypi.chemical_modeling.process.mo_double_spike import _equations


class TestRoot(unittest.TestCase):
    """Tests for the root path used by the current Mo double-spike process."""

    def setUp(self):
        self.masses = (100, 98, 97)
        self.spike = (0.5, 2.0, 0.7)
        self.standard = (0.1, 1.5, 0.6)
        self.expected = (0.35, 0.2, -0.15)
        phi_ref, beta_sample, beta_mix = self.expected
        self.mixture = tuple(
            (
                phi_ref * spike_ratio
                + (1 - phi_ref)
                * standard_ratio
                * (95 / mass) ** beta_sample
            )
            / (95 / mass) ** beta_mix
            for mass, spike_ratio, standard_ratio in zip(
                self.masses,
                self.spike,
                self.standard,
                strict=True,
            )
        )

    def test_equations_return_three_finite_residuals(self):
        residuals = _equations(
            self.expected,
            self.spike,
            self.standard,
            self.mixture,
        )

        self.assertEqual(len(residuals), 3)
        self.assertTrue(all(np.isfinite(value) for value in residuals))
        self.assertTrue(np.allclose(residuals, 0.0, atol=1e-12))

    def test_root_recovers_known_parameters(self):
        solution = root(
            _equations,
            (0.5, 0.5, 2.0),
            args=(self.spike, self.standard, self.mixture),
            method="hybr",
        )

        self.assertTrue(solution.success, solution.message)
        self.assertTrue(np.allclose(solution.x, self.expected, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
