"""Constrained Gibbs-energy minimization for ideal pure species/phases."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import linprog


def _as_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _validate_named_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty JSON object")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValueError(f"{label} keys must be non-empty strings")
    return value


def gibbs_minimization(
    gibbs_energies: Mapping[str, float],
    stoichiometry: Mapping[str, Mapping[str, float]],
    component_totals: Mapping[str, float],
) -> dict[str, Any]:
    """Minimize ``sum(g_i * n_i)`` subject to component balance and ``n_i >= 0``.

    The supplied molar Gibbs energies are treated as constants at the user's
    chosen pressure and temperature. This is an ideal pure-species/phase model;
    activity, solution-mixing, and non-ideal interaction terms are not included.
    """

    energies_raw = _validate_named_mapping(gibbs_energies, "gibbs_energies")
    stoich_raw = _validate_named_mapping(stoichiometry, "stoichiometry")
    totals_raw = _validate_named_mapping(component_totals, "component_totals")

    species = list(energies_raw)
    if set(stoich_raw) != set(species):
        raise ValueError("gibbs_energies and stoichiometry must contain the same species")

    energies = [
        _as_finite_number(energies_raw[name], f"gibbs_energies['{name}']")
        for name in species
    ]
    components = list(totals_raw)
    totals = [
        _as_finite_number(totals_raw[name], f"component_totals['{name}']")
        for name in components
    ]
    if any(total < 0 for total in totals):
        raise ValueError("component_totals values must be greater than or equal to 0")
    if not any(total > 0 for total in totals):
        raise ValueError("component_totals must contain at least one positive value")

    normalized_stoich: dict[str, dict[str, float]] = {}
    seen_components: set[str] = set()
    for species_name in species:
        composition_raw = _validate_named_mapping(
            stoich_raw[species_name],
            f"stoichiometry['{species_name}']",
        )
        composition: dict[str, float] = {}
        for component, raw_coefficient in composition_raw.items():
            coefficient = _as_finite_number(
                raw_coefficient,
                f"stoichiometry['{species_name}']['{component}']",
            )
            if coefficient < 0:
                raise ValueError("stoichiometry coefficients must be greater than or equal to 0")
            composition[component] = coefficient
            seen_components.add(component)
        if not any(coefficient > 0 for coefficient in composition.values()):
            raise ValueError(
                f"stoichiometry['{species_name}'] must contain at least one positive coefficient"
            )
        normalized_stoich[species_name] = composition

    if seen_components != set(components):
        raise ValueError(
            "stoichiometry components and component_totals must contain the same components"
        )

    balance_matrix = np.array(
        [
            [normalized_stoich[name].get(component, 0.0) for name in species]
            for component in components
        ],
        dtype=float,
    )
    result = linprog(
        c=np.array(energies, dtype=float),
        A_eq=balance_matrix,
        b_eq=np.array(totals, dtype=float),
        bounds=[(0, None)] * len(species),
        method="highs",
    )
    if not result.success:
        raise ValueError(f"Gibbs minimization failed: {result.message}")

    solution = np.asarray(result.x, dtype=float)
    solution[np.abs(solution) < 1e-12] = 0.0
    residuals = balance_matrix @ solution - np.array(totals, dtype=float)
    max_residual = float(np.max(np.abs(residuals)))
    tolerance = 1e-8 * max(1.0, max(abs(total) for total in totals))
    if max_residual > tolerance:
        raise ValueError(
            f"Gibbs minimization balance residual is too large: {max_residual:.3g}"
        )

    return {
        "minimum_gibbs": float(result.fun),
        "equilibrium_moles": {
            name: float(amount) for name, amount in zip(species, solution)
        },
        "max_balance_residual": max_residual,
    }
