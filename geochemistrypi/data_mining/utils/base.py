# -*- coding: utf-8 -*-
import datetime
import logging
import os
import platform

import mlflow
import pandas as pd
from matplotlib import pyplot as plt
from rich import print

from ..constants import OUTPUT_PATH


def create_geopi_output_dir(experiment_name: str, run_name: str) -> None:
    """Create the output directory for the current run and store the related pathes as environment variable.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment.

    run_name : str
        The name of the run.
    """
    # Set the output path for the current run
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    geopi_output_path = os.path.join(OUTPUT_PATH, experiment_name, f"{run_name} {timestamp}")
    os.environ["GEOPI_OUTPUT_PATH"] = geopi_output_path
    os.makedirs(geopi_output_path, exist_ok=True)

    # Set the output artifacts path for the current run
    geopi_output_artifacts_path = os.path.join(geopi_output_path, "artifacts")
    os.environ["GEOPI_OUTPUT_ARTIFACTS_PATH"] = geopi_output_artifacts_path
    os.makedirs(geopi_output_artifacts_path, exist_ok=True)

    # Set the output artifacts data path for the current run
    geopi_output_artifacts_data_path = os.path.join(geopi_output_artifacts_path, "data")
    os.environ["GEOPI_OUTPUT_ARTIFACTS_DATA_PATH"] = geopi_output_artifacts_data_path
    os.makedirs(geopi_output_artifacts_data_path, exist_ok=True)

    # Set the output artifacts model path for the current run
    geopi_output_artifacts_model_path = os.path.join(geopi_output_artifacts_path, "model")
    os.environ["GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH"] = geopi_output_artifacts_model_path
    os.makedirs(geopi_output_artifacts_model_path, exist_ok=True)

    # Set the output artifacts image path for the current run
    geopi_output_artifacts_image_path = os.path.join(geopi_output_artifacts_path, "image")
    os.environ["GEOPI_OUTPUT_ARTIFACTS_IMAGE_PATH"] = geopi_output_artifacts_image_path
    os.makedirs(geopi_output_artifacts_image_path, exist_ok=True)

    # Set the output parameters path for the current run
    geopi_output_parameters_path = os.path.join(geopi_output_path, "parameters")
    os.environ["GEOPI_OUTPUT_PARAMETERS_PATH"] = geopi_output_parameters_path
    os.makedirs(geopi_output_parameters_path, exist_ok=True)

    # Set the outout metrics path for the current run
    geopi_output_metrics_path = os.path.join(geopi_output_path, "metrics")
    os.environ["GEOPI_OUTPUT_METRICS_PATH"] = geopi_output_metrics_path
    os.makedirs(geopi_output_metrics_path, exist_ok=True)


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


def save_data(df: pd.DataFrame, df_name: str, path: str, mlflow_artifact_data_path: str = None) -> None:
    """Save the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to store.

    df_name : str
        The name of the data sheet.

    path : str
        The path to store the data sheet

    mlflow_artifact_data_path : str, default=None
        The path to store the data sheet in mlflow.
    """
    try:
        # drop the index in case that the dimensions change
        # store the result in the directory "results"
        df.to_excel(os.path.join(path, "{}.xlsx".format(df_name)), index=False)
        if mlflow_artifact_data_path:
            mlflow.log_artifact(os.path.join(path, "{}.xlsx".format(df_name)), artifact_path=mlflow_artifact_data_path)
        else:
            mlflow.log_artifact(os.path.join(path, "{}.xlsx".format(df_name)))
        print(f"Successfully store '{df_name}' in '{df_name}.xlsx' in {path}.")
    except ModuleNotFoundError:
        print("** Please download openpyxl by pip3 **")
        print("** The data will be stored in .csv file **")
        # store the result in the directory "results"
        df.to_csv(os.path.join(path, "{}.csv".format(df_name)))
        print(f"Successfully store '{df_name}' in '{df_name}.csv' in {path}.")


def save_text(string: str, text_name: str, path: str) -> None:
    """Save the text.

    Parameters
    ----------
    string : str
        The text to store.

    text_name : str
        The name of the text.

    path : str
        The path to store the text.
    """
    file_path = os.path.join(path, text_name + ".txt")
    with open(file_path, "w") as f:
        f.write(string)
    print(f"Successfully store '{text_name}' in '{text_name}.txt' in {path}.")


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
