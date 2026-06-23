"""Time series utilities for data_mining.

Provides functions to compute and plot subaerial proportion time series.
"""
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_subaerial_proportion(
    df: pd.DataFrame,
    bin_width: float,
    n_iter: int = 100,
    age_col: str = "R_AGE",
    age_max_col: str = "R_MAX_AGE",
    prob_col: str = "SBAP",
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and 2*std subaerial proportion per age bin.

    Returns (age_x, ave_bin, std_bin).
    """
    age = df[age_col].values
    ageMax = df[age_max_col].values
    age_error = np.abs(ageMax - age) / 2

    x = df[prob_col].values
    Lat = df[lat_col].values
    Lon = df[lon_col].values

    np.random.seed(2025)

    # all_idx = np.arange(age.size)

    # weights
    # wei = np.ones((age.size, 1))
    WEI = np.ones((age.size, 1))
    batch_size = 2000

    # compute WEI in batches (same formula as original)
    for i in range(0, age.size, batch_size):
        end = min(i + batch_size, age.size)
        outlat = Lat[i:end][:, np.newaxis]
        outlon = Lon[i:end][:, np.newaxis]
        outage = age[i:end][:, np.newaxis]

        ka = 1 / (((Lat - outlat) / 2) ** 2 + ((Lon - outlon) / 2) ** 2 + 1)
        kb = 1 / (((age - outage) / 38) ** 2 + 1)

        a = np.nansum(ka + kb, axis=1)
        WEI[i:end, 0] = 1 / (a / 0.2)

        nan_mask = np.isnan(x[i:end])
        WEI[i:end, 0][nan_mask] = 0

    # filter
    index_wei = np.where((np.isinf(WEI[:, 0])) | (np.isnan(x)))[0]
    mask = (~np.isinf(WEI[:, 0])) & (~np.isnan(x)) & (WEI[:, 0] > 0)
    del_age = age[mask]
    del_age_error = age_error[mask]
    del_p = x[mask]
    del_WEI = WEI[mask, 0]
    if del_WEI.sum() == 0:
        del_WEIP = np.ones_like(del_WEI) / max(1, del_WEI.size)
    else:
        del_WEIP = del_WEI / np.nansum(del_WEI)

    data_max = np.nanmax(age)
    total_age_limit = np.ceil(data_max / bin_width) * bin_width
    num_bins = int(total_age_limit / bin_width)

    boot6 = np.ones((num_bins, n_iter)) * np.nan

    bootfixa = np.zeros((index_wei.size, 1))
    bootfixy = np.zeros((index_wei.size, 1))
    if index_wei.size > 0:
        bootfixy[:, 0] = x[index_wei]

    for i in range(n_iter):
        if index_wei.size > 0:
            bootfixa[:, 0] = np.random.normal(loc=age[index_wei], scale=age_error[index_wei])

        if del_age.size == 0:
            break

        bootstrapSamples = np.random.choice(np.arange(del_age.size), size=del_age.size, p=del_WEIP)
        boot1 = np.random.normal(loc=del_age[bootstrapSamples], scale=del_age_error[bootstrapSamples]).reshape(-1, 1)
        boot2 = del_p[bootstrapSamples].reshape(-1, 1)

        bootage_cmb = np.vstack((bootfixa, boot1))
        booty_cmb = np.vstack((bootfixy, boot2))
        boot3 = np.hstack((bootage_cmb, booty_cmb))
        boot4 = boot3[boot3[:, 0].argsort()]

        boot5_list = []
        for L in range(1, num_bins + 1):
            condition = (boot4[:, 0] >= (L - 1) * bin_width) & (boot4[:, 0] <= L * bin_width)
            Bin = boot4[condition, 1]
            if Bin.size > 0:
                Ave = np.sum(Bin >= 0.5) / Bin.size * 100
            else:
                Ave = np.nan
            boot5_list.append(Ave)

        boot6[:, i] = boot5_list

    ave_bin = np.nanmean(boot6, axis=1)[:num_bins]
    std_bin = 2 * np.nanstd(boot6, axis=1)[:num_bins]
    age_x = np.arange(bin_width / 2, total_age_limit, bin_width)

    return age_x, ave_bin, std_bin


def plot_and_save(
    age_x: np.ndarray,
    ave_bin: np.ndarray,
    std_bin: np.ndarray,
    out_dir: Optional[str] = None,
    out_name: str = "Subaerial_proportion",
) -> str:
    """Plot the result and save PDF and CSV. Returns base path saved."""
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.errorbar(age_x, ave_bin, yerr=std_bin, ecolor="r", capsize=4)
    plt.xlabel("Age (Ma)")
    plt.ylabel("Subaerial proportion (%)")
    plt.xlim((0, max(age_x) if age_x.size > 0 else 4000))
    plt.ylim((0, 100))

    pdf_path = os.path.join(out_dir, f"{out_name}.pdf")
    csv_path = os.path.join(out_dir, f"{out_name}.csv")
    plt.savefig(pdf_path, dpi=300)

    # save csv with columns age, mean, std
    df_out = pd.DataFrame({"age": age_x, "mean": ave_bin, "std2": std_bin})
    df_out.to_csv(csv_path, index=False)

    return os.path.join(out_dir, out_name)


__all__ = ["compute_subaerial_proportion", "plot_and_save"]
