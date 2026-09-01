# -*- coding: utf-8 -*-
"""
Process wrapper for Mo double-spike method: read constants from excel, solve equations,
save results to results dir, save to GEOPI output (with fallback), and optionally log to MLflow.
"""
import math
import os
from typing import Tuple

import pandas as pd
from scipy.optimize import fsolve, root

# Try to import repository helpers/constants; if not available, provide fallbacks.
try:
    # preferred locations in the repo
    from geochemistrypi.utils.base import save_data as _repo_save_data  # type: ignore
except Exception:
    _repo_save_data = None

# constants fallback
try:
    from geochemistrypi.constants import GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH  # type: ignore
except Exception:
    GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.path.join(os.getcwd(), "geopi_output")
    MLFLOW_ARTIFACT_DATA_PATH = None


def _read_constants(excel_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    df = pd.read_excel(excel_path, sheet_name="3程序处理_输入常数")
    R_sp = (df.loc[0, "R_100_sp"], df.loc[0, "R_98_sp"], df.loc[0, "R_97_sp"])
    R_std = (df.loc[0, "R_100_std"], df.loc[0, "R_98_std"], df.loc[0, "R_97_std"])
    r_mix = (df.loc[0, "r_100_mix"], df.loc[0, "r_98_mix"], df.loc[0, "r_97_mix"])
    return R_sp, R_std, r_mix


def _equations(params, R_sp, R_std, r_mix):
    φ_ref, β_sple, β_mix = params
    R_100_sp, R_98_sp, R_97_sp = R_sp
    R_100_std, R_98_std, R_97_std = R_std
    r_100_mix, r_98_mix, r_97_mix = r_mix

    f1 = φ_ref * R_100_sp + (1 - φ_ref) * R_100_std * (95 / 100) ** β_sple - r_100_mix * (95 / 100) ** β_mix
    f2 = φ_ref * R_98_sp + (1 - φ_ref) * R_98_std * (95 / 98) ** β_sple - r_98_mix * (95 / 98) ** β_mix
    f3 = φ_ref * R_97_sp + (1 - φ_ref) * R_97_std * (95 / 97) ** β_sple - r_97_mix * (95 / 97) ** β_mix
    return [f1, f2, f3]


def _save_via_repo_or_fallback(df: pd.DataFrame, input_path: str, filename: str) -> str:
    """
    Try repository save_data helper; if unavailable, write to GEOPI_OUTPUT_ARTIFACTS_DATA_PATH fallback.
    Returns out_path.
    """
    name_all = os.path.splitext(os.path.basename(input_path))[0]
    # prefer repo helper if exists
    if _repo_save_data:
        try:
            _repo_save_data(df, name_all, filename.replace(".", "_"), GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        except Exception as e:
            print(f"[warning] repo save_data raised: {e}; falling back to direct write.")
    # fallback direct write to GEOPI_OUTPUT_ARTIFACTS_DATA_PATH/<name_all>/
    try:
        dest_dir = os.path.join(GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, name_all)
        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, filename)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path
    except Exception as e:
        print(f"[warning] fallback save failed: {e}")
        # last resort: write to current working results folder
        fallback = os.path.join(os.getcwd(), filename)
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def _maybe_log_mlflow(out_path: str, solver: str, res: dict) -> None:
    try:
        import mlflow

        mlflow_store_uri = os.environ.get("MLFLOW_STORE_PATH")
        if mlflow_store_uri:
            mlflow.set_tracking_uri(mlflow_store_uri)
        mlflow.set_experiment("chemical_modeling")
        with mlflow.start_run():
            mlflow.log_param("solver", solver)
            for k, v in res.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
            mlflow.log_artifact(out_path, artifact_path="chemical_modeling/Mo")
    except Exception as e:
        # Don't fail pipeline for mlflow issues
        print(f"[warning] mlflow logging failed: {e}")


def run(input_path: str, out_dir: str, solver: str = "fsolve"):
    """
    Execute Mo double-spike calculation using data from input_path, save results to out_dir.
    Returns path to output CSV (primary written location).
    """
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Mo data file not found: {input_path}")

    R_sp, R_std, r_mix = _read_constants(input_path)
    initial_guess = (0.5, 0.5, 2.0)

    if solver == "fsolve":
        sol, info, ier, message = fsolve(
            _equations,
            initial_guess,
            args=(R_sp, R_std, r_mix),
            full_output=True,
        )
        if ier != 1:
            raise ValueError(f"Mo double-spike solver did not converge: {message}")
        res = {"phi_ref": float(sol[0]), "beta_sple": float(sol[1]), "beta_mix": float(sol[2])}
    elif solver == "root":
        sol = root(_equations, initial_guess, args=(R_sp, R_std, r_mix), method="hybr")
        if not sol.success:
            raise ValueError(f"Mo double-spike solver did not converge: {sol.message}")
        res = {"phi_ref": float(sol.x[0]), "beta_sple": float(sol.x[1]), "beta_mix": float(sol.x[2])}
    else:
        raise ValueError("solver must be either 'fsolve' or 'root'")

    if not all(math.isfinite(value) for value in res.values()):
        raise ValueError("Mo double-spike solver returned non-finite parameters")
    if not 0 <= res["phi_ref"] <= 1:
        raise ValueError("Mo double-spike solution is non-physical: phi_ref must be between 0 and 1")
    residual = max(abs(value) for value in _equations(tuple(res.values()), R_sp, R_std, r_mix))
    if residual > 1e-8:
        raise ValueError(f"Mo double-spike solution residual is too large: {residual:.3g}")

    # write canonical results to requested out_dir
    df = pd.DataFrame([{"method": solver, **res}])
    os.makedirs(out_dir, exist_ok=True)
    primary_out_path = os.path.join(out_dir, "Mo_results.csv")
    df.to_csv(primary_out_path, index=False, encoding="utf-8-sig")

    # also save to geopi output (repo helper or fallback)
    saved_artifact_path = _save_via_repo_or_fallback(df, input_path, "Mo_results.csv")

    # optional MLflow logging (non-fatal)
    _maybe_log_mlflow(saved_artifact_path, solver, res)

    return primary_out_path
