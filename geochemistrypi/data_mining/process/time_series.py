"""Time series utilities for data_mining.

Provides functions to compute and plot subaerial proportion time series.
"""
import math
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIME_SERIES_RANDOM_SEED = 2025
MAX_BOOTSTRAP_ITERATIONS = 10_000
MAX_TIME_BINS = 10_000


class TimeSeriesValidationError(ValueError):
    """Raised before numerical work when Time Series inputs are unsafe."""


def _validated_time_series_arrays(
    df: pd.DataFrame,
    bin_width: float,
    n_iter: int,
    age_col: str,
    age_max_col: str,
    prob_col: str,
    lat_col: str,
    lon_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise TimeSeriesValidationError("Time Series input must contain at least one data row.")
    if not math.isfinite(bin_width) or bin_width <= 0:
        raise TimeSeriesValidationError("bin_width must be a finite positive number.")
    if isinstance(n_iter, bool) or not isinstance(n_iter, int):
        raise TimeSeriesValidationError("n_iter must be an integer.")
    if n_iter < 1 or n_iter > MAX_BOOTSTRAP_ITERATIONS:
        raise TimeSeriesValidationError(f"n_iter must be between 1 and {MAX_BOOTSTRAP_ITERATIONS}.")
    roles = {
        "age": age_col,
        "maximum age": age_max_col,
        "probability": prob_col,
        "latitude": lat_col,
        "longitude": lon_col,
    }
    if len(set(roles.values())) != len(roles):
        raise TimeSeriesValidationError("Time Series column roles must identify five different columns.")
    missing = sorted(set(roles.values()) - set(df.columns))
    if missing:
        raise TimeSeriesValidationError(f"Time Series input is missing required columns: {missing}.")
    arrays = {}
    for role, column in roles.items():
        try:
            values = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise TimeSeriesValidationError(f"Time Series {role} column must contain only numeric values: {column!r}.") from exc
        finite = np.isfinite(values)
        if not bool(finite.all()):
            rows = [int(index) + 2 for index in np.flatnonzero(~finite)[:10]]
            raise TimeSeriesValidationError(f"Time Series {role} column contains missing or non-finite values at data rows {rows}: {column!r}.")
        arrays[role] = values
    age = arrays["age"]
    age_max = arrays["maximum age"]
    probability = arrays["probability"]
    latitude = arrays["latitude"]
    longitude = arrays["longitude"]
    if bool((age < 0).any()):
        raise TimeSeriesValidationError("Time Series ages must be non-negative.")
    if bool((age_max < 0).any()):
        raise TimeSeriesValidationError("Time Series comparison ages must be non-negative.")
    if bool(((probability < 0) | (probability > 1)).any()):
        raise TimeSeriesValidationError("Time Series probability values must be between 0 and 1.")
    if bool(((latitude < -90) | (latitude > 90)).any()):
        raise TimeSeriesValidationError("Time Series latitude values must be between -90 and 90 degrees.")
    if bool(((longitude < -180) | (longitude > 180)).any()):
        raise TimeSeriesValidationError("Time Series longitude values must be between -180 and 180 degrees.")
    # Preserve the published workflow: the central reconstructed age defines
    # the plotted time span; the comparison age only defines uncertainty.
    data_max = float(np.max(age))
    if data_max <= 0:
        raise TimeSeriesValidationError("Time Series input must contain at least one positive age.")
    num_bins = int(math.ceil(data_max / bin_width))
    if num_bins < 1 or num_bins > MAX_TIME_BINS:
        raise TimeSeriesValidationError(f"bin_width creates {num_bins} bins; the safety limit is {MAX_TIME_BINS}.")
    return age, age_max, probability, latitude, longitude, num_bins


def compute_subaerial_proportion(
    df: pd.DataFrame,
    bin_width: float,
    n_iter: int = 100,
    age_col: str = "R_AGE",
    age_max_col: str = "R_MAX_AGE",
    prob_col: str = "SBAP",
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
    seed: int = TIME_SERIES_RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and 2*std subaerial proportion per age bin.

    Returns (age_x, ave_bin, std_bin).
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TimeSeriesValidationError("seed must be a non-negative integer.")
    age, age_max, x, latitude, longitude, num_bins = _validated_time_series_arrays(
        df,
        bin_width,
        n_iter,
        age_col,
        age_max_col,
        prob_col,
        lat_col,
        lon_col,
    )
    # The Liu et al. workflow treats the distance between the reconstructed
    # age and its comparison age as an unsigned uncertainty.  In the bundled
    # workbook R_AGE is rounded more coarsely than R_MAX_AGE, so either value
    # can be the larger one for otherwise valid rows.
    age_error = np.abs(age_max - age) / 2
    random = np.random.RandomState(seed)

    WEI = np.ones((age.size, 1))
    batch_size = 2000

    # compute WEI in batches (same formula as original)
    for i in range(0, age.size, batch_size):
        end = min(i + batch_size, age.size)
        outlat = latitude[i:end][:, np.newaxis]
        outlon = longitude[i:end][:, np.newaxis]
        outage = age[i:end][:, np.newaxis]

        ka = 1 / (((latitude - outlat) / 2) ** 2 + ((longitude - outlon) / 2) ** 2 + 1)
        kb = 1 / (((age - outage) / 38) ** 2 + 1)

        a = np.nansum(ka + kb, axis=1)
        WEI[i:end, 0] = 1 / (a / 0.2)

    if not bool(np.isfinite(WEI[:, 0]).all()) or bool((WEI[:, 0] <= 0).any()):
        raise TimeSeriesValidationError("Time Series spatial-temporal weights are not finite and positive.")
    probabilities = WEI[:, 0] / WEI[:, 0].sum()

    boot6 = np.ones((num_bins, n_iter)) * np.nan

    for i in range(n_iter):
        bootstrap_samples = random.choice(np.arange(age.size), size=age.size, p=probabilities)
        boot1 = random.normal(loc=age[bootstrap_samples], scale=age_error[bootstrap_samples]).reshape(-1, 1)
        boot2 = x[bootstrap_samples].reshape(-1, 1)
        boot3 = np.hstack((boot1, boot2))
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

    ave_bin = np.asarray([np.mean(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in boot6])
    std_bin = np.asarray([2 * np.std(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in boot6])
    if not bool(np.isfinite(ave_bin).any()):
        raise TimeSeriesValidationError("Time Series computation produced no populated age bins.")
    age_x = (np.arange(num_bins, dtype=float) + 0.5) * bin_width

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
    csv_out_dir: Optional[str] = None,
    pdf_out_dir: Optional[str] = None,
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
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    csv_directory = csv_out_dir or out_dir
    pdf_directory = pdf_out_dir or out_dir
    os.makedirs(csv_directory, exist_ok=True)
    os.makedirs(pdf_directory, exist_ok=True)

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

    pdf_path = os.path.join(pdf_directory, f"{out_name}.pdf")
    csv_path = os.path.join(csv_directory, f"{out_name}.csv")
    plt.savefig(
        pdf_path,
        dpi=600,
        bbox_inches="tight",
        metadata={
            "Creator": "GeochemistryPi",
            "Producer": "GeochemistryPi",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close()

    # Save CSV with columns: age, mean, std
    df_out = pd.DataFrame(
        {
            f"age_{age_unit}": plot_age,
            "mean_percent": ave_bin,
            "two_sigma_percent": std_bin,
        }
    )
    df_out.to_csv(
        csv_path,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )

    return os.path.join(out_dir, out_name)


__all__ = ["compute_subaerial_proportion", "plot_and_save"]
