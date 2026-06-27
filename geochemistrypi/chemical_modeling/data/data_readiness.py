# -*- coding: utf-8 -*-
"""
Data utilities for chemical_modeling: reading and basic validation.
"""
import os
from typing import Optional

import pandas as pd


def read_excel(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_name)


def ensure_numeric(df: pd.DataFrame, columns):
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
