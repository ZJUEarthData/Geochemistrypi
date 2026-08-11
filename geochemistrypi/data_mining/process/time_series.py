"""Time series utilities for data_mining.

Provides functions to compute and plot subaerial proportion time series.
"""
import os
from typing import Optional, Tuple

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
    age_unit: str = "Ma",
    title: Optional[str] = None,
    fit_curve: bool = True,
) -> str:
    """
    Plot the result and save PDF and CSV.

    Parameters
    ----------
    age_x : np.ndarray
        Array of bin center ages.
    ave_bin : np.ndarray
        Array of mean subaerial proportions per bin.
    std_bin : np.ndarray
        Array of 2-sigma standard deviations per bin.
    out_dir : Optional[str], default=None
        Output directory for saving files. If None, uses current working directory.
    out_name : str, default="Subaerial_proportion"
        Base name for output files (without extension).
    age_unit : str, default="Ma"
        Unit for age axis. Either "Ma" (million years) or "Ga" (billion years).
    title : Optional[str], default=None
        Title for the plot. If None, no title is added.

    Returns
    -------
    str
        Base path of saved files.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Time Series plotting requires matplotlib. Install the v0.8 core "
            "dependencies before creating PDF output."
        ) from exc

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # Convert age unit if needed
    if age_unit == "Ga":
        plot_age = age_x / 1000.0
    else:
        plot_age = age_x

    # ---- Remove NaN values for automatic range detection ----
    valid_mask = ~np.isnan(ave_bin)
    if np.sum(valid_mask) == 0:
        print("Warning: No valid data to plot.")
        return os.path.join(out_dir, out_name)

    plot_age_valid = plot_age[valid_mask]
    ave_bin_valid = ave_bin[valid_mask]
    std_bin_valid = std_bin[valid_mask]

    # ============================================================
    # 1. Set publication-quality plotting style
    # ============================================================
    # Use a built-in style that is available in most Matplotlib versions
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            # Fallback: use default style with manual settings
            plt.style.use("default")
            print("Info: Using default Matplotlib style (seaborn styles not available)")

    # Set Times New Roman font (standard in geoscience publications)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"  # Math font matching Times

    # Set font sizes
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    # ============================================================
    # 2. Create figure
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 5))  # 10" x 5", close to golden ratio

    if fit_curve:
        # ---- 2a. Draw gray error band (2-sigma, semi-transparent) ----
        ax.fill_between(plot_age_valid, ave_bin_valid - std_bin_valid, ave_bin_valid + std_bin_valid, color="gray", alpha=0.35, label=r"$\pm 2\sigma$")
        # ---- 2b. Draw main curve ----
        ax.plot(plot_age_valid, ave_bin_valid, color="#1f77b4", linewidth=2.5, label="Mean proportion")
    else:
        # ---- 2b. Draw scatter points with error bars ----
        ax.errorbar(
            plot_age_valid,
            ave_bin_valid,
            yerr=std_bin_valid,
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.5,
            capsize=4,
            markerfacecolor="#1f77b4",
            markeredgecolor="black",
            markersize=6,
            label="Mean proportion",
        )

    # ============================================================
    # 3. Configure axes
    # ============================================================
    # ---- 3a. Automatic x-axis range detection ----
    x_min = np.min(plot_age_valid) - 0.02 * (np.max(plot_age_valid) - np.min(plot_age_valid))
    x_max = np.max(plot_age_valid) + 0.02 * (np.max(plot_age_valid) - np.min(plot_age_valid))
    if x_max - x_min < 0.1:
        x_min = 0
        x_max = 4.0 if age_unit == "Ga" else 4000
    ax.set_xlim((x_min, x_max))

    # ---- 3b. Reverse x-axis (older ages on the left) ----
    ax.invert_xaxis()

    # ---- 3c. Automatic y-axis range detection ----
    y_min = 0
    y_max = np.nanmax(ave_bin_valid) + 5
    if y_max < 20:
        y_max = 100
    ax.set_ylim((y_min, y_max))

    # ---- 3d. Axis labels ----
    ax.set_xlabel(f"Age ({age_unit})", fontsize=14)
    ax.set_ylabel("Estimated Proportion of Subaerial Basalts (%)", fontsize=14)

    # ---- 3e. Tick control (fine-grained) ----
    # x-axis ticks: 0.5 Ga interval for Ga, 500 Ma interval for Ma
    if age_unit == "Ga":
        ax.set_xticks(np.arange(0, 4.5, 0.5))
    else:
        ax.set_xticks(np.arange(0, 4500, 500))

    # y-axis ticks: 0, 20, 40, 60, 80, 100
    ax.set_yticks(np.arange(0, 101, 20))

    # ============================================================
    # 4. Add grid lines (light gray, dashed)
    # ============================================================
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)

    # ============================================================
    # 5. Legend (best location)
    # ============================================================
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black", framealpha=0.9)

    # ============================================================
    # 6. Title (optional, usually omitted in papers)
    # ============================================================
    if title:
        ax.set_title(title, fontsize=14)

    # ============================================================
    # 7. Adjust layout and save
    # ============================================================
    plt.tight_layout()

    pdf_path = os.path.join(out_dir, f"{out_name}.pdf")
    csv_path = os.path.join(out_dir, f"{out_name}.csv")
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close()

    # Save CSV with columns: age, mean, std
    df_out = pd.DataFrame({"age": plot_age, "mean": ave_bin, "std2": std_bin})
    df_out.to_csv(csv_path, index=False)

    return os.path.join(out_dir, out_name)


__all__ = ["compute_subaerial_proportion", "plot_and_save"]
