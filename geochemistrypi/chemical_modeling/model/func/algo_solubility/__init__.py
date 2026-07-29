"""
algo_solubility（硫溶解度/溶解度算法）模型包，集成多种经典与机器学习模型。
"""

import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# from .nn_model import NN, NNP

# ==== helper functions ====

METHOD_COLUMNS = {
    "rubie": ["Pressure", "T"],
    "ding": [
        "Pressure",
        "T",
        "SiO2",
        "TiO2",
        "Al2O3",
        "FeO",
        "MgO",
        "CaO",
        "Na2O",
        "K2O",
        "sulfide_Ni",
    ],
    "blanchard": [
        "Pressure",
        "T",
        "SiO2",
        "TiO2",
        "Al2O3",
        "FeO",
        "MgO",
        "CaO",
        "Na2O",
        "K2O",
        "H2O",
        "Fe",
        "Ni",
        "Cu",
    ],
    "hybrid": [
        "Pressure",
        "T",
        "SiO2",
        "TiO2",
        "Al2O3",
        "FeO",
        "MgO",
        "CaO",
        "NiO",
        "Na2O",
        "K2O",
        "H2O",
        "Fe",
        "Ni+Cu+Co",
        "S",
        "O",
    ],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "SCSS Measurement" in df.columns and "SCSS" not in df.columns:
        rename_map["SCSS Measurement"] = "SCSS"
    if "P" in df.columns and "Pressure" not in df.columns:
        rename_map["P"] = "Pressure"
    if "NiO*" in df.columns and "NiO" not in df.columns:
        rename_map["NiO*"] = "NiO"
    if "NaO" in df.columns and "Na2O" not in df.columns:
        rename_map["NaO"] = "Na2O"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def oxide_wt_at(SiO2, TiO2, Al2O3, MgO, FeO, CaO, Na2O, K2O, Cr2O3):
    # Formula masses are divided by the number of cations for Al2O3,
    # Na2O, K2O, and Cr2O3 so that the normalized values are cation
    # mole fractions, as required by the Ding and Blanchard equations.
    cation_moles = (
        SiO2 / 60.0843,
        TiO2 / 79.866,
        Al2O3 / (101.9613 / 2),
        MgO / 40.3044,
        FeO / 71.844,
        CaO / 56.0774,
        Na2O / (61.9789 / 2),
        K2O / (94.196 / 2),
        Cr2O3 / (151.9904 / 2),
    )
    total_oxide_atom = sum(cation_moles)
    if total_oxide_atom <= 0:
        raise ValueError("Oxide composition must contain a positive cation total")
    (
        X_SiO2,
        X_TiO2,
        X_Al2O3,
        X_MgO,
        X_FeO,
        X_CaO,
        X_Na2O,
        X_K2O,
        X_Cr2O3,
    ) = (value / total_oxide_atom for value in cation_moles)
    return X_SiO2, X_TiO2, X_Al2O3, X_MgO, X_FeO, X_CaO, X_Na2O, X_K2O, X_Cr2O3


def hydrous_cation_fractions(row: pd.Series) -> dict[str, float]:
    """Convert oxide wt.% to hydrous single-cation mole fractions."""

    oxide_basis = {
        "SiO2": (60.0843, 1),
        "TiO2": (79.866, 1),
        "Al2O3": (101.9613, 2),
        "MgO": (40.3044, 1),
        "FeO": (71.844, 1),
        "CaO": (56.0774, 1),
        "Na2O": (61.9789, 2),
        "K2O": (94.196, 2),
        "H2O": (18.01528, 2),
        # These components have zero regression coefficients in the
        # Blanchard equations, but still contribute to the normalization.
        "MnO": (70.9374, 1),
        "P2O5": (141.9445, 2),
        "Cr2O3": (151.9904, 2),
    }
    cation_moles = {
        oxide: float(row.get(oxide, 0.0)) * cations / formula_mass
        for oxide, (formula_mass, cations) in oxide_basis.items()
    }
    total = sum(cation_moles.values())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Oxide composition must contain a positive finite cation total")
    return {oxide: value / total for oxide, value in cation_moles.items()}


def sulfide_fe_fraction(Fe, Ni, Cu):
    """Return the atomic Fe/(Fe+Ni+Cu) ratio used as ideal a_FeS."""

    cation_moles = np.asarray(
        [float(Fe) / 55.845, float(Ni) / 58.6934, float(Cu) / 63.546],
        dtype=float,
    )
    if not np.isfinite(cation_moles).all() or (cation_moles < 0).any():
        raise ValueError("Sulfide Fe, Ni, and Cu must be finite and non-negative")
    if cation_moles[0] <= 0:
        raise ValueError("Sulfide Fe must be greater than 0")
    total = float(cation_moles.sum())
    if total <= 0:
        raise ValueError("Sulfide Fe, Ni, and Cu must have a positive atomic total")
    return float(cation_moles[0] / total)


def Mole_Sum(X_SiO2, A_SiO2, X_TiO2, A_TiO2, X_Al2O3, A_Al2O3, X_MgO, A_MgO, X_FeO, A_FeO, X_CaO, A_CaO, X_Na2O, A_Na2O, X_K2O, A_K2O, X_H2O, A_H2O, A_SiFe):
    return X_SiO2 * A_SiO2 + X_TiO2 * A_TiO2 + X_Al2O3 * A_Al2O3 + X_MgO * A_MgO + X_FeO * A_FeO + X_CaO * A_CaO + X_Na2O * A_Na2O + X_K2O * A_K2O + X_H2O * A_H2O + X_SiO2 * X_FeO * A_SiFe


def CiXm_sum(C_Ti, X_TiO2, C_Ca, X_CaO, C_Si, X_SiO2, C_Al, X_Al2O3, C_Fe, X_FeO):
    return C_Ti * X_TiO2 + C_Ca * X_CaO + C_Si * X_SiO2 + C_Al * X_Al2O3 + C_Fe * X_FeO


def scss_rubie(pressure, temperature):
    return np.exp(14.2 - 11032 / temperature - 379 * pressure / temperature)


def scss_ding(A, B, CiXm, D, E, T, P, X_FeO, X_TiO2):
    return np.exp(A + B / T + CiXm + D * X_FeO * X_TiO2 + E * P / T)


def ding_ni_correction_factor(sulfide_ni_wt):
    """Return the Ding et al. (2018) empirical Ni correction denominator."""

    sulfide_ni_wt = np.asarray(sulfide_ni_wt, dtype=float)
    return np.where(
        sulfide_ni_wt > 8.5,
        0.0013 * sulfide_ni_wt**2 - 0.0109 * sulfide_ni_wt + 1,
        1.0,
    )


def scss_blanchard(a, b, c, XmAm, T, P, X_FeS, X_FeO):
    """Evaluate Blanchard et al. (2021) Equation 12 (model 2)."""

    return np.exp(a + b / T + c * P / T + XmAm / T + np.log(X_FeS) - np.log(X_FeO))


def scss_blanchard_model1(a, b, c, XmAm, T, P, X_FeS, X_FeO):
    """Evaluate Blanchard et al. (2021) Equation 11 (model 1)."""

    return np.exp(a + b / T + c * P / T + XmAm + np.log(X_FeS) - np.log(X_FeO))


def _prepare_solubility_dataframe(input_path: Union[str, Path], method: str) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    df = _normalize_columns(df)
    try:
        expected = METHOD_COLUMNS[method]
    except KeyError as exc:
        raise NotImplementedError(f"Unknown method: {method}") from exc
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for solubility {method}: {missing}")
    return df


def _get_hybrid_features(df: pd.DataFrame) -> pd.DataFrame:
    features = ["Pressure", "T", "SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "NiO", "Na2O", "K2O", "H2O", "Fe", "Ni+Cu+Co", "S", "O"]
    return df[features]


def _compute_ding(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_Ding = 12.10023817
    B_Ding = -4951.220517
    D_Ding = -40.67763841
    E_Ding = -273.4844764
    C_Ti = 4.02527185
    C_Ca = 4.173632609
    C_Si = -3.643865073
    C_Al = -3.936000202
    C_Fe = 5.574892678
    ni_free_results = []
    for _, row in df.iterrows():
        X_SiO2, X_TiO2, X_Al2O3, X_MgO, X_FeO, X_CaO, X_Na2O, X_K2O, _ = oxide_wt_at(row["SiO2"], row["TiO2"], row["Al2O3"], row["MgO"], row["FeO"], row["CaO"], row["Na2O"], row["K2O"], 0.0)
        CiXm = CiXm_sum(C_Ti, X_TiO2, C_Ca, X_CaO, C_Si, X_SiO2, C_Al, X_Al2O3, C_Fe, X_FeO)
        scss = scss_ding(A_Ding, B_Ding, CiXm, D_Ding, E_Ding, row["T"], row["Pressure"], X_FeO, X_TiO2)
        ni_free_results.append(scss)
    ni_free = np.array(ni_free_results, dtype=float)
    correction_factor = ding_ni_correction_factor(df["sulfide_Ni"].to_numpy(dtype=float))
    corrected = ni_free / correction_factor
    return ni_free, correction_factor, corrected


def _blanchard_composition_term(
    fractions: dict[str, float],
    coefficients: dict[str, float],
) -> float:
    term = sum(
        fractions[oxide] * coefficient
        for oxide, coefficient in coefficients.items()
        if oxide != "SiFe"
    )
    return term + fractions["SiO2"] * fractions["FeO"] * coefficients["SiFe"]


def _compute_blanchard(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model1 = {
        "a": 27.0,
        "b": -4621.0,
        "c": -193.0,
        "SiO2": -25.0,
        "TiO2": -13.0,
        "Al2O3": -18.0,
        "MgO": -16.0,
        "FeO": -32.0,
        "CaO": -14.0,
        "Na2O": -17.0,
        "K2O": -27.0,
        "H2O": -19.0,
        "SiFe": 76.0,
    }
    model2 = {
        "a": 7.95,
        "b": 18159.0,
        "c": -190.0,
        "SiO2": -32677.0,
        "TiO2": -15014.0,
        "Al2O3": -23071.0,
        "MgO": -18258.0,
        "FeO": -41706.0,
        "CaO": -14668.0,
        "Na2O": -19529.0,
        "K2O": -34641.0,
        "H2O": -22677.0,
        "SiFe": 120662.0,
    }
    composition_keys = (
        "SiO2",
        "TiO2",
        "Al2O3",
        "MgO",
        "FeO",
        "CaO",
        "Na2O",
        "K2O",
        "H2O",
        "SiFe",
    )
    model1_composition = {key: model1[key] for key in composition_keys}
    model2_composition = {key: model2[key] for key in composition_keys}
    eq11_results = []
    eq12_results = []
    sulfide_fe_fractions = []
    for _, row in df.iterrows():
        fractions = hydrous_cation_fractions(row)
        x_fes = sulfide_fe_fraction(row["Fe"], row["Ni"], row["Cu"])
        composition1 = _blanchard_composition_term(fractions, model1_composition)
        composition2 = _blanchard_composition_term(fractions, model2_composition)
        eq11 = scss_blanchard_model1(
            model1["a"],
            model1["b"],
            model1["c"],
            composition1,
            row["T"],
            row["Pressure"],
            x_fes,
            fractions["FeO"],
        )
        eq12 = scss_blanchard(
            model2["a"],
            model2["b"],
            model2["c"],
            composition2,
            row["T"],
            row["Pressure"],
            x_fes,
            fractions["FeO"],
        )
        if not np.isfinite(eq11) or not np.isfinite(eq12) or eq11 <= 0 or eq12 <= 0:
            raise ValueError("Blanchard parameters produce a non-finite SCSS result")
        eq11_results.append(eq11)
        eq12_results.append(eq12)
        sulfide_fe_fractions.append(x_fes)
    return (
        np.array(eq11_results, dtype=float),
        np.array(eq12_results, dtype=float),
        np.array(sulfide_fe_fractions, dtype=float),
    )


def run(method: str, element: str, input_path: Union[str, Path], out_dir: Union[str, Path], **kwargs):
    df = _prepare_solubility_dataframe(input_path, method)
    if method == "rubie":
        df["SCSS_pred"] = scss_rubie(df["Pressure"], df["T"])
    elif method == "ding":
        ni_free, correction_factor, corrected = _compute_ding(df)
        df["SCSS_Ni_free"] = ni_free
        df["Ni_correction_factor"] = correction_factor
        df["SCSS_pred"] = corrected
    elif method == "blanchard":
        eq11, eq12, sulfide_fe = _compute_blanchard(df)
        df["sulfide_Fe_fraction"] = sulfide_fe
        df["SCSS_eq11_ppm"] = eq11
        df["SCSS_eq12_ppm"] = eq12
        # Blanchard et al. use model 1 (Equation 11) in their discussion.
        df["SCSS_pred"] = eq11
    elif method == "hybrid":
        # Load the versioned artifact only when this method is requested so the
        # empirical equations do not pay the scikit-learn import cost.
        from .hybrid_model import MODEL_VERSION, predict_hybrid_scss

        features = _get_hybrid_features(df)
        rf_prediction, correction, prediction = predict_hybrid_scss(features)
        df["hybrid_model_version"] = MODEL_VERSION
        df["RF_base_pred_ppm"] = rf_prediction
        df["PT_correction_factor"] = correction
        df["SCSS_pred"] = prediction
    else:
        raise NotImplementedError(f"Unknown method: {method}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir) / f"solubility_{method}_results.xlsx"
    df.to_excel(out_path, index=False)
    return {"status": "success", "out_path": str(out_path)}


def list_methods():
    return {
        "rubie": "Laurenz et al. (2016) 经验公式（保留旧方法键 rubie）",
        "ding": "Ding et al. (2018) 经验公式",
        "blanchard": "Blanchard et al. (2021) 经验公式",
        "hybrid": "集成机器学习模型 (Zhang et al. 2024)",
    }


def list_elements(method: str):
    return ["S"]
