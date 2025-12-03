import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. Automatically locate the input file:  ../data/Hg_data.xlsx
# ------------------------------------------------------------------
excel_path = os.path.join(os.path.dirname(__file__), "..", "data", "Hg_data.xlsx")
df = pd.read_excel(excel_path)

# Basic cleaning
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

# ------------------------------------------------------------------
# 2. Column mapping
# ------------------------------------------------------------------
col_map = {"202Hg": "THg(%)", "202Hg/198Hg": "d202(‰)", "201Hg/198Hg": "d201(‰)", "200Hg/198Hg": "d200(‰)", "199Hg/198Hg": "d199(‰)"}

# Ensure source columns are numeric
for src_col in col_map.keys():
    df[src_col] = pd.to_numeric(df[src_col], errors="coerce")

# ------------------------------------------------------------------
# 3. Calculate fractionation values
# ------------------------------------------------------------------
idx_3133 = df.index[df["Label"] == "3133"].tolist()


def find_nearest_3133(idx):
    prev = [i for i in idx_3133 if i < idx]
    next = [i for i in idx_3133 if i > idx]
    return (prev[-1] if prev else None, next[0] if next else None)


def calc_fractionation(row, col):
    idx = row.name
    cur_val = row[col]
    prev_idx, next_idx = find_nearest_3133(idx)
    if prev_idx is None or next_idx is None or pd.isna(cur_val):
        return np.nan
    prev_val = df.loc[prev_idx, col]
    next_val = df.loc[next_idx, col]
    if pd.isna(prev_val) or pd.isna(next_val) or (prev_val + next_val) == 0:
        return np.nan
    return ((2 * cur_val) / (prev_val + next_val) - 1) * 1000


# Apply calculation
for src_col, tgt_col in col_map.items():
    df[tgt_col] = df.apply(lambda row: calc_fractionation(row, src_col), axis=1)

# ------------------------------------------------------------------
# 4. Divide THg(%) by 1000 and convert to percent
# ------------------------------------------------------------------
df["THg(%)"] = df["THg(%)"] / 1000 * 100

# ------------------------------------------------------------------
# 5. Add three new columns
# ------------------------------------------------------------------
df["D199"] = df["d199(‰)"] - df["d202(‰)"] * 0.252
df["D200"] = df["d200(‰)"] - df["d202(‰)"] * 0.5024
df["D201"] = df["d201(‰)"] - df["d202(‰)"] * 0.752

# ------------------------------------------------------------------
# 6. Save result to ../results/Hg_results.xlsx
# ------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "Hg_results.xlsx")

df.to_excel(out_path, index=False)
print(f"Results saved to: {out_path}")
