# -*- coding: utf-8 -*-
import logging
import os
import platform

import pandas as pd
from matplotlib import pyplot as plt
from rich import print


def clear_output() -> None:
    # TODO(sany hecan@mail2.sysu.edu.cn): Incite exception capture mechanism
    flag = input("(Press Enter key to move forward.)")
    my_os = platform.system()
    if flag == "":
        if my_os == "Windows":
            os.system("cls")  # for Windows
        else:
            os.system("clear")  # for Linux and macOS

    print("")


def save_fig(fig_name: str, image_path: str, tight_layout: bool = True) -> None:
    """Save the figure.

    Parameters
    ----------
    image_path : str
        The path to store the image.

    fig_name : str
        Figure name.

    tight_layout : bool, default=True
        Automatically adjust subplot parameters to give specified padding.
    """
    # TODO: seperate the stored path outside
    path = os.path.join(image_path, fig_name + ".png")
    print(f"Save figure '{fig_name}' in {image_path}.")
    if tight_layout:
        plt.tight_layout()

    # Check that the original file exists,
    # and if it does, add a number after the filename to distinguish
    i = 1
    dir = path[:-4]
    while os.path.isfile(path):
        path = dir + str(i) + ".png"
        i = i + 1
    plt.savefig(path, format="png", dpi=300)
    plt.close()


def save_data(df: pd.DataFrame, df_name: str, path: str) -> None:
    """Save the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to store.

    df_name : str
        The name of the data sheet.

    path : str
        The path to store the data sheet
    """
    try:
        # drop the index in case that the dimensions change
        # store the result in the directory "results"
        df.to_excel(os.path.join(path, "{}.xlsx".format(df_name)), index=False)
        print(f"Successfully store '{df_name}' in '{df_name}.xlsx' in {path}.")
    except ModuleNotFoundError:
        print("** Please download openpyxl by pip3 **")
        print("** The data will be stored in .csv file **")
        # store the result in the directory "results"
        df.to_csv(os.path.join(path, "{}.csv".format(df_name)))
        print(f"Successfully store '{df_name}' in '{df_name}.csv' in {path}.")


def log(log_path, log_name):
    # Create and configure logger
    # LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s"
    DATE_FORMAT = "%Y-%m-%d  %H:%M:%S %a "
    logging.basicConfig(filename=os.path.join(log_path, log_name), level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, filemode="w")
    logger = logging.getLogger()
    return logger


def show_warning(is_show: bool = True) -> None:
    """Overriding Python's default filter to control whether to display warning information."""
    import sys

    if not is_show:
        if not sys.warnoptions:
            import os

            os.environ["PYTHONWARNINGS"] = "ignore"
            # os.environ["PYTHONWARNINGS"] = "default"
