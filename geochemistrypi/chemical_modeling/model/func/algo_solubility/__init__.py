"""
algo_solubility（硫溶解度/溶解度算法）模型包，集成多种经典与机器学习模型。
"""

import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .dataset import train_test_dfs
from .emodels import EModels

# from .nn_model import NN, NNP

# ==== helper functions ====


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
    total_oxide_atom = (SiO2 / 60) + (TiO2 / 89.8) + (Al2O3 / 51) + (MgO / 40) + (FeO / 73.8) + (CaO / 56) + (Na2O / 31) + (K2O / 47) + (Cr2O3 / 76)
    X_SiO2 = (SiO2 / 60) / total_oxide_atom
    X_TiO2 = (TiO2 / 89.8) / total_oxide_atom
    X_Al2O3 = (Al2O3 / 51) / total_oxide_atom
    X_MgO = (MgO / 40) / total_oxide_atom
    X_FeO = (FeO / 73.8) / total_oxide_atom
    X_CaO = (CaO / 56) / total_oxide_atom
    X_Na2O = (Na2O / 31) / total_oxide_atom
    X_K2O = (K2O / 47) / total_oxide_atom
    X_Cr2O3 = (Cr2O3 / 76) / total_oxide_atom
    return X_SiO2, X_TiO2, X_Al2O3, X_MgO, X_FeO, X_CaO, X_Na2O, X_K2O, X_Cr2O3


def sulfide_wt_at(Fe, Ni, S, oxygen):
    total_sulfide_atom = (Fe / 55.8) + (Ni / 58.5) + (S / 32) + (oxygen / 16)
    X_Fe = (Fe / 55.8) / total_sulfide_atom
    X_Ni = (Ni / 58.5) / total_sulfide_atom
    X_S = (S / 32) / total_sulfide_atom
    X_O = (oxygen / 16) / total_sulfide_atom
    X_FeS = X_Fe / (X_Fe + X_Ni + X_O)
    return X_Fe, X_Ni, X_S, X_O, X_FeS


def Mole_Sum(X_SiO2, A_SiO2, X_TiO2, A_TiO2, X_Al2O3, A_Al2O3, X_MgO, A_MgO, X_FeO, A_FeO, X_CaO, A_CaO, X_Na2O, A_Na2O, X_K2O, A_K2O, X_H2O, A_H2O, A_SiFe):
    return X_SiO2 * A_SiO2 + X_TiO2 * A_TiO2 + X_Al2O3 * A_Al2O3 + X_MgO * A_MgO + X_FeO * A_FeO + X_CaO * A_CaO + X_Na2O * A_Na2O + X_K2O * A_K2O + X_H2O * A_H2O + X_SiO2 * X_FeO * A_SiFe


def CiXm_sum(C_Ti, X_TiO2, C_Ca, X_CaO, C_Si, X_SiO2, C_Al, X_Al2O3, C_Fe, X_FeO):
    return C_Ti * X_TiO2 + C_Ca * X_CaO + C_Si * X_SiO2 + C_Al * X_Al2O3 + C_Fe * X_FeO


def scss_rubie(pressure, temperature):
    return np.exp(14.2 - 11032 / temperature - 379 * pressure / temperature)


def scss_ding(A, B, CiXm, D, E, T, P, X_FeO, X_TiO2):
    return np.exp(A + B / T + CiXm + D * X_FeO * X_TiO2 + E * P / T)


def scss_blanchard(a, b, c, XmAm, T, P, X_FeS, X_FeO):
    return np.exp(a + b / T + c * P / T + XmAm / T + np.log(X_FeS) - np.log(X_FeO))


def _prepare_solubility_dataframe(input_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    df = _normalize_columns(df)
    expected = ["Pressure", "T", "SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "NiO", "Na2O", "K2O", "H2O", "Fe", "Ni+Cu+Co", "S", "O"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for solubility: {missing}")
    if "SCSS" not in df.columns:
        if "SCSS Measurement" in df.columns:
            df["SCSS"] = df["SCSS Measurement"]
    return df


def _get_hybrid_features(df: pd.DataFrame) -> pd.DataFrame:
    features = ["Pressure", "T", "SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "NiO", "Na2O", "K2O", "H2O", "Fe", "Ni+Cu+Co", "S", "O"]
    return df[features]


def _compute_ding(df: pd.DataFrame) -> np.ndarray:
    A_Ding = 12.10023817
    B_Ding = -4951.220517
    D_Ding = -40.67763841
    E_Ding = -273.4844764
    C_Ti = 4.02527185
    C_Ca = 4.173632609
    C_Si = -3.643865073
    C_Al = -3.936000202
    C_Fe = 5.574892678
    results = []
    for _, row in df.iterrows():
        X_SiO2, X_TiO2, X_Al2O3, X_MgO, X_FeO, X_CaO, X_Na2O, X_K2O, _ = oxide_wt_at(row["SiO2"], row["TiO2"], row["Al2O3"], row["MgO"], row["FeO"], row["CaO"], row["Na2O"], row["K2O"], 0.0)
        CiXm = CiXm_sum(C_Ti, X_TiO2, C_Ca, X_CaO, C_Si, X_SiO2, C_Al, X_Al2O3, C_Fe, X_FeO)
        scss = scss_ding(A_Ding, B_Ding, CiXm, D_Ding, E_Ding, row["T"], row["Pressure"], X_FeO, X_TiO2)
        results.append(scss)
    return np.array(results, dtype=float)


def _compute_blanchard(df: pd.DataFrame) -> np.ndarray:
    a_blanchard = 7.95
    b_blanchard = 18159
    c_blanchard = -190
    A_blanchard_SiO2 = -32677
    A_blanchard_TiO2 = -15014
    A_blanchard_Al2O3 = -23071
    A_blanchard_MgO = -18258
    A_blanchard_FeO = -41706
    A_blanchard_CaO = -14668
    A_blanchard_Na2O = -19529
    A_blanchard_K2O = -34641
    A_blanchard_H2O = -22677
    A_blanchard_SiFe = 120662
    results = []
    for _, row in df.iterrows():
        X_SiO2, X_TiO2, X_Al2O3, X_MgO, X_FeO, X_CaO, X_Na2O, X_K2O, _ = oxide_wt_at(row["SiO2"], row["TiO2"], row["Al2O3"], row["MgO"], row["FeO"], row["CaO"], row["Na2O"], row["K2O"], 0.0)
        X_Fe, X_Ni, X_S, X_O, X_FeS = sulfide_wt_at(row["Fe"], row.get("Ni+Cu+Co", row.get("Ni", 0.0)), row["S"], row["O"])
        XmAm = Mole_Sum(
            X_SiO2,
            A_blanchard_SiO2,
            X_TiO2,
            A_blanchard_TiO2,
            X_Al2O3,
            A_blanchard_Al2O3,
            X_MgO,
            A_blanchard_MgO,
            X_FeO,
            A_blanchard_FeO,
            X_CaO,
            A_blanchard_CaO,
            X_Na2O,
            A_blanchard_Na2O,
            X_K2O,
            A_blanchard_K2O,
            row["H2O"],
            A_blanchard_H2O,
            A_blanchard_SiFe,
        )
        scss = scss_blanchard(a_blanchard, b_blanchard, c_blanchard, XmAm, row["T"], row["Pressure"], X_FeS, X_FeO)
        results.append(scss)
    return np.array(results, dtype=float)


def run(method: str, element: str, input_path: Union[str, Path], out_dir: Union[str, Path], **kwargs):
    df = _prepare_solubility_dataframe(input_path)
    if method == "rubie":
        df["SCSS_pred"] = scss_rubie(df["Pressure"], df["T"])
    elif method == "ding":
        df["SCSS_pred"] = _compute_ding(df)
    elif method == "blanchard":
        df["SCSS_pred"] = _compute_blanchard(df)
    elif method == "hybrid":
        if "SCSS" not in df.columns:
            raise ValueError("Hybrid model requires a target column named 'SCSS' or 'SCSS Measurement'.")
        features = _get_hybrid_features(df)
        train_df = df[features.columns.tolist() + ["SCSS"]].copy()
        x_train, x_test, y_train, y_test, scaler = train_test_dfs(train_df, test_size=0.30)
        emp = EModels(x_train, x_test, y_train, y_test)
        _, y_pred_emp = emp.predict_em(scaler.transform(features))
        df["SCSS_pred"] = y_pred_emp[0]
    else:
        raise NotImplementedError(f"Unknown method: {method}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir) / f"solubility_{method}_results.xlsx"
    df.to_excel(out_path, index=False)
    return {"status": "success", "out_path": str(out_path)}


def list_methods():
    return {"rubie": "Rubie et al. (2016) 经验公式", "ding": "Ding et al. (2018) 经验公式", "blanchard": "Blanchard et al. (2021) 经验公式", "hybrid": "集成机器学习模型 (Zhang et al. 2024)"}


def list_elements(method: str):
    return ["S"]
