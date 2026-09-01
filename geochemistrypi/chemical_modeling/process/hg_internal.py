# -*- coding: utf-8 -*-
"""
Refactored Hg internal standard method.
Provides:
- load_hg_data(path) -> DataFrame
- compute_fractionation(df) -> DataFrame (adds computed columns)
- run(input_path, out_dir) -> saves the result, writes to GEOPI output (with fallback), and optionally logs to MLflow.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd

from geochemistrypi.chemical_modeling.data.data_readiness import ensure_numeric, read_excel

# Try to import repository helpers/constants; if not available provide fallbacks.
try:
    from geochemistrypi.utils.base import save_data as _repo_save_data  # type: ignore
except Exception:
    _repo_save_data = None

try:
    from geochemistrypi.constants import GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH  # type: ignore
except Exception:
    GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.path.join(os.getcwd(), "geopi_output")
    MLFLOW_ARTIFACT_DATA_PATH = None


COL_MAP = {
    "202Hg": "THg(%)",
    "202Hg/198Hg": "d202(‰)",
    "201Hg/198Hg": "d201(‰)",
    "200Hg/198Hg": "d200(‰)",
    "199Hg/198Hg": "d199(‰)",
}


def load_hg_data(path: str) -> pd.DataFrame:
    """
    Read Excel input. pandas.read_excel may return a DataFrame or a dict (if multiple sheets).
    If a dict is returned, choose the first sensible sheet (prefer one with 'Label' or a source column).
    """
    df = read_excel(path)
    if isinstance(df, dict):
        candidate = None
        for sheet_name, sheet_df in df.items():
            if not isinstance(sheet_df, pd.DataFrame):
                continue
            cols = set(map(str, sheet_df.columns))
            if "Label" in cols or any(col in cols for col in COL_MAP.keys()):
                candidate = sheet_df
                break
            if candidate is None:
                candidate = sheet_df
        if candidate is None:
            raise ValueError(f"No suitable sheet found in Excel file: {path}")
        df = candidate

    df.columns = df.columns.astype(str).str.strip()
    if "Label" in df.columns:
        def normalize_label(value):
            if isinstance(value, (int, float)) and not pd.isna(value):
                numeric = float(value)
                if numeric.is_integer():
                    return str(int(numeric))
            return str(value).strip()

        df["Label"] = df["Label"].map(normalize_label)
    ensure_numeric(df, list(COL_MAP.keys()))
    return df


def _find_nearest_3133_indices(df: pd.DataFrame):
    idx_3133 = df.index[df.get("Label") == "3133"].tolist()
    return idx_3133


def compute_fractionation(df: pd.DataFrame) -> pd.DataFrame:
    idx_3133 = _find_nearest_3133_indices(df)

    def find_nearest_3133(idx):
        prev = [i for i in idx_3133 if i < idx]
        nxt = [i for i in idx_3133 if i > idx]
        return (prev[-1] if prev else None, nxt[0] if nxt else None)

    def calc_fractionation(row, src_col):
        idx = row.name
        cur_val = row.get(src_col)
        prev_idx, next_idx = find_nearest_3133(idx)
        if prev_idx is None or next_idx is None or pd.isna(cur_val):
            return np.nan
        prev_val = df.loc[prev_idx, src_col]
        next_val = df.loc[next_idx, src_col]
        if pd.isna(prev_val) or pd.isna(next_val) or (prev_val + next_val) == 0:
            return np.nan
        return ((2 * cur_val) / (prev_val + next_val) - 1) * 1000

    for src_col, tgt_col in COL_MAP.items():
        df[tgt_col] = df.apply(lambda row: calc_fractionation(row, src_col), axis=1)

    if "THg(%)" in df.columns:
        df["THg(%)"] = pd.to_numeric(df["THg(%)"], errors="coerce") / 1000 * 100

    if all(k in df.columns for k in ("d199(‰)", "d202(‰)")):
        df["D199"] = df["d199(‰)"] - df["d202(‰)"] * 0.252
    if all(k in df.columns for k in ("d200(‰)", "d202(‰)")):
        df["D200"] = df["d200(‰)"] - df["d202(‰)"] * 0.5024
    if all(k in df.columns for k in ("d201(‰)", "d202(‰)")):
        df["D201"] = df["d201(‰)"] - df["d202(‰)"] * 0.752

    return df


def _save_via_repo_or_fallback(df: pd.DataFrame, input_path: str, filename: str) -> str:
    name_all = os.path.splitext(os.path.basename(input_path))[0]
    if _repo_save_data:
        try:
            _repo_save_data(df, name_all, filename.replace(".", "_"), GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        except Exception as e:
            print(f"[warning] repo save_data raised: {e}; falling back to direct write.")
    try:
        dest_dir = os.path.join(GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, name_all)
        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, filename)
        df.to_excel(out_path, index=False)
        return out_path
    except Exception as e:
        print(f"[warning] fallback save failed: {e}")
        fallback = os.path.join(os.getcwd(), filename)
        df.to_excel(fallback, index=False)
        return fallback


def _maybe_log_mlflow(out_path: str, df: pd.DataFrame) -> None:
    try:
        import os
        import re

        import mlflow

        mlflow_store_uri = os.environ.get("MLFLOW_STORE_PATH")
        if mlflow_store_uri:
            mlflow.set_tracking_uri(mlflow_store_uri)
        mlflow.set_experiment("chemical_modeling")
        with mlflow.start_run():
            non_null_counts = df.notna().sum().to_dict()
            for raw_k, raw_v in non_null_counts.items():
                try:
                    v = int(raw_v)
                except Exception:
                    continue
                # sanitize metric name: keep letters, numbers, underscore, dot, hyphen
                safe_k = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(raw_k))
                metric_name = f"non_null_{safe_k}"
                try:
                    mlflow.log_metric(metric_name, float(v))
                except Exception as e:
                    print(f"[warning] failed to log metric {metric_name}: {e}")
            try:
                mlflow.log_artifact(out_path, artifact_path="chemical_modeling/Hg")
            except Exception as e:
                print(f"[warning] failed to log artifact to mlflow: {e}")
    except Exception as e:
        print(f"[warning] mlflow logging failed: {e}")


def run(input_path: str, out_dir: Optional[str] = None) -> str:
    df = load_hg_data(input_path)
    df_out = compute_fractionation(df)
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)
    primary_out = os.path.join(out_dir, "Hg_results.xlsx")
    df_out.to_excel(primary_out, index=False)

    # save to geopi output or fallback
    saved_path = _save_via_repo_or_fallback(df_out, input_path, "Hg_results.xlsx")
    _maybe_log_mlflow(saved_path, df_out)
    return primary_out
