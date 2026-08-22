"""One-reaction equilibrium solver based on the law of mass action."""

from __future__ import annotations

import math
from typing import Mapping


def _validate_inputs(
    K: float,
    stoich: Mapping[str, float],
    init_conc: Mapping[str, float],
) -> tuple[list[str], list[float], list[float]]:
    if not math.isfinite(K) or K <= 0:
        raise ValueError("K must be a finite number greater than 0")
    if not stoich:
        raise ValueError("stoich must not be empty")
    if set(stoich) != set(init_conc):
        raise ValueError("stoich and initial_concentrations must contain the same species")

    species = list(stoich)
    coefficients: list[float] = []
    concentrations: list[float] = []
    for name in species:
        coefficient = float(stoich[name])
        concentration = float(init_conc[name])
        if not math.isfinite(coefficient) or coefficient == 0:
            raise ValueError(f"Stoichiometric coefficient for '{name}' must be finite and non-zero")
        if not math.isfinite(concentration) or concentration < 0:
            raise ValueError(f"Initial concentration for '{name}' must be finite and non-negative")
        coefficients.append(coefficient)
        concentrations.append(concentration)
    if not any(value < 0 for value in coefficients) or not any(value > 0 for value in coefficients):
        raise ValueError("stoich must contain at least one reactant and one product")
    return species, coefficients, concentrations


def _log_reaction_quotient(
    extent: float,
    coefficients: list[float],
    concentrations: list[float],
) -> float:
    total = 0.0
    for coefficient, initial in zip(coefficients, concentrations):
        concentration = initial + coefficient * extent
        if concentration <= 0:
            return -math.inf if coefficient > 0 else math.inf
        total += coefficient * math.log(concentration)
    return total


def law_of_mass_action(
    K: float,
    stoich: Mapping[str, float],
    init_conc: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> dict[str, float]:
    """Solve a single ideal reaction by bisection on its reaction extent.

    Positive stoichiometric coefficients denote products and negative
    coefficients denote reactants.  For positive concentrations,
    ``log(Q)`` is strictly increasing with reaction extent, so bisection gives
    a deterministic solution without requiring SciPy.
    """

    species, coefficients, concentrations = _validate_inputs(K, stoich, init_conc)
    lower = max(
        (-initial / coefficient for coefficient, initial in zip(coefficients, concentrations) if coefficient > 0),
        default=-math.inf,
    )
    upper = min(
        (initial / -coefficient for coefficient, initial in zip(coefficients, concentrations) if coefficient < 0),
        default=math.inf,
    )
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("The reaction has no finite feasible concentration interval")

    span = upper - lower
    margin = max(span * 1e-14, 1e-15)
    left = lower + margin
    right = upper - margin
    target = math.log(K)

    left_value = _log_reaction_quotient(left, coefficients, concentrations) - target
    right_value = _log_reaction_quotient(right, coefficients, concentrations) - target
    if left_value > 0 or right_value < 0:
        raise ValueError("The supplied reaction cannot reach the requested equilibrium constant")

    midpoint = (left + right) / 2
    for _ in range(max_iterations):
        midpoint = (left + right) / 2
        value = _log_reaction_quotient(midpoint, coefficients, concentrations) - target
        if abs(value) <= tolerance or (right - left) <= tolerance * max(1.0, abs(midpoint)):
            break
        if value < 0:
            left = midpoint
        else:
            right = midpoint
    else:
        raise RuntimeError("Mass-action solver did not converge")

    result: dict[str, float] = {}
    for name, coefficient, initial in zip(species, coefficients, concentrations):
        value = initial + coefficient * midpoint
        result[name] = max(0.0, float(value))
    return result
