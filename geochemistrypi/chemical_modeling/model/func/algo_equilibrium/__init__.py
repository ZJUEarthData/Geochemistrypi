"""Workbook adapters for geochemical equilibrium calculations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def list_methods() -> dict[str, str]:
    return {
        "mass_balance": "质量平衡检查 (Mass balance)",
        "precipitation_dissolution": "溶解-沉淀平衡 (Precipitation/Dissolution)",
        "ion_exchange": "离子交换平衡 (Ion exchange)",
        "mass_action": "质量作用定律 (Law of mass action)",
    }


def list_elements(method: str) -> list[str]:
    if method in {"mass_balance", "precipitation_dissolution", "ion_exchange", "mass_action"}:
        return ["Any"]
    return []


def _write_result(dataframe: pd.DataFrame, out_dir: str, filename: str) -> dict[str, str]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    dataframe.to_excel(out_path, index=False)
    return {"status": "success", "out_path": str(out_path)}


def _run_mass_balance(dataframe: pd.DataFrame) -> pd.DataFrame:
    from .mass_balance import mass_balance

    species_columns = [column for column in dataframe.columns if column != "total_mass"]
    result = dataframe.copy()
    result["species_sum"] = dataframe[species_columns].sum(axis=1)
    result["mass_difference"] = result["species_sum"] - dataframe["total_mass"]
    result["is_balanced"] = dataframe.apply(
        lambda row: mass_balance(
            {column: float(row[column]) for column in species_columns},
            float(row["total_mass"]),
        ),
        axis=1,
    )
    return result


def _run_precipitation_dissolution(dataframe: pd.DataFrame) -> pd.DataFrame:
    from .precipitation_dissolution import calc_saturation_index, is_precipitation

    result = dataframe.copy()
    result["saturation_index"] = dataframe.apply(
        lambda row: calc_saturation_index(
            float(row["ion_activity_product"]),
            float(row["ksp"]),
        ),
        axis=1,
    )
    result["state"] = result["saturation_index"].map(
        lambda value: "precipitation" if is_precipitation(value) else ("equilibrium" if value == 0 else "dissolution")
    )
    return result


def _run_ion_exchange(dataframe: pd.DataFrame) -> pd.DataFrame:
    from .ion_exchange import gaines_thomas_exchange

    result = dataframe.copy()
    result["exchange_ratio"] = dataframe.apply(
        lambda row: gaines_thomas_exchange(
            float(row["eq_conc_a"]),
            float(row["eq_conc_b"]),
            float(row["selectivity"]),
        ),
        axis=1,
    )
    return result


def _parse_mapping(value: object, column_name: str) -> dict[str, float | int]:
    if not isinstance(value, str):
        raise ValueError(f"Column '{column_name}' must contain a JSON object")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Column '{column_name}' must contain valid JSON") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(f"Column '{column_name}' must contain a non-empty JSON object")
    return parsed


def _run_mass_action(dataframe: pd.DataFrame) -> pd.DataFrame:
    from .mass_action import law_of_mass_action

    result = dataframe.copy()
    equilibrium_values: list[str] = []
    for row in dataframe.itertuples(index=False):
        stoich = _parse_mapping(row.stoich, "stoich")
        initial = _parse_mapping(row.initial_concentrations, "initial_concentrations")
        equilibrium = law_of_mass_action(float(row.K), stoich, initial)
        equilibrium_values.append(json.dumps(equilibrium, ensure_ascii=False, sort_keys=True))
    result["equilibrium_concentrations"] = equilibrium_values
    return result


def run(method: str, element: str, input_path: str, out_dir: str, **kwargs):
    """Run one equilibrium method against an Online workbook."""

    dataframe = pd.read_excel(input_path)
    if method == "mass_balance":
        return _write_result(_run_mass_balance(dataframe), out_dir, "mass_balance_results.xlsx")
    if method == "precipitation_dissolution":
        return _write_result(
            _run_precipitation_dissolution(dataframe),
            out_dir,
            "precipitation_dissolution_results.xlsx",
        )
    if method == "ion_exchange":
        return _write_result(_run_ion_exchange(dataframe), out_dir, "ion_exchange_results.xlsx")
    if method == "mass_action":
        return _write_result(_run_mass_action(dataframe), out_dir, "mass_action_results.xlsx")
    raise NotImplementedError(f"Method {method} not implemented in algo_equilibrium.")
