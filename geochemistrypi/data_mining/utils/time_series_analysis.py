"""Backward-compatible wrapper around the new time series module.

This file preserves the previous interactive behaviour but delegates
the computation to `geochemistrypi.data_mining.process.time_series`.
"""
import os
import tkinter as tk
from tkinter import filedialog

import pandas as pd

from ...data_mining.process.time_series import compute_subaerial_proportion, plot_and_save


def interactive_main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Choose your Excel dataset", filetypes=[("Excel", "*.xlsx *.xls")])
    if not file_path:
        print("finished")
        return

    bin_size = input("Time interval (Ma)=")
    bin_width = float(bin_size)

    df = pd.read_excel(file_path, sheet_name=0)

    age_x, ave_bin, std_bin = compute_subaerial_proportion(df, bin_width=bin_width)

    out_dir = os.path.dirname(file_path)
    plot_and_save(age_x, ave_bin, std_bin, out_dir=out_dir)


if __name__ == "__main__":
    interactive_main()
