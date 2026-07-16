"""Command-line runner for time series analysis (subaerial proportion).

Usage example:
  python run_time_series.py --input /path/to/data.xlsx --bin-width 10 --iter 100 --out-dir results/
"""
import argparse
import os

import pandas as pd

from .process.time_series import compute_subaerial_proportion, plot_and_save


def main():
    p = argparse.ArgumentParser(description="Run time series subaerial proportion analysis")
    p.add_argument("--input", required=True, help="Input Excel/CSV file path")
    p.add_argument("--sheet", default=0, help="Excel sheet name or index (default 0)")
    p.add_argument("--bin-width", type=float, required=True, help="Bin width in Ma (e.g. 10)")
    p.add_argument("--iter", type=int, default=100, help="Bootstrap iterations (default 100)")
    p.add_argument("--out-dir", default=".", help="Output directory")
    p.add_argument("--age-col", default="R_AGE")
    p.add_argument("--age-max-col", default="R_MAX_AGE")
    p.add_argument("--prob-col", default="SBAP")
    p.add_argument("--lat-col", default="LATITUDE")
    p.add_argument("--lon-col", default="LONGITUDE")

    args = p.parse_args()

    input_path = args.input
    if input_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path, sheet_name=args.sheet)
    else:
        df = pd.read_csv(input_path)

    age_x, ave_bin, std_bin = compute_subaerial_proportion(
        df,
        bin_width=args.bin_width,
        n_iter=args.iter,
        age_col=args.age_col,
        age_max_col=args.age_max_col,
        prob_col=args.prob_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    base = plot_and_save(age_x, ave_bin, std_bin, out_dir=args.out_dir)
    print(f"Saved outputs to {args.out_dir} (base name: {base})")


if __name__ == "__main__":
    main()
