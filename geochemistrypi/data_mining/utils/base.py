# -*- coding: utf-8 -*-
import logging
import os
import pickle
import platform
import shutil
from typing import Optional

import joblib
import mlflow
import pandas as pd
from matplotlib import pyplot as plt
from rich import print


def create_geopi_output_dir(output_path: str, experiment_name: str, run_name: str, sub_run_name: Optional[str] = None) -> None:
    """Create the output directory for the current run and store the related pathes as environment variable.

    Parameters
    ----------
    output_path : str
        The root path to store the output.

    experiment_name : str
        The name of the experiment.

    run_name : str
        The name of the run.

    sub_run_name : str, default=None
        The name of the sub run.
    """
    # Set the output path for the current run
    # timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    if sub_run_name:
        geopi_output_path = os.path.join(output_path, experiment_name, f"{run_name}", sub_run_name)
    else:
        geopi_output_path = os.path.join(output_path, experiment_name, f"{run_name}")
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

    # Set the output artifacts image model output path for the current run
    geopi_output_artifacts_image_model_output_path = os.path.join(geopi_output_artifacts_image_path, "model_output")
    os.environ["GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH"] = geopi_output_artifacts_image_model_output_path
    os.makedirs(geopi_output_artifacts_image_model_output_path, exist_ok=True)

    # Set the output artifacts image statistic path for the current run
    geopi_output_artifacts_image_statistic_path = os.path.join(geopi_output_artifacts_image_path, "statistic")
    os.environ["GEOPI_OUTPUT_ARTIFACTS_IMAGE_STATISTIC_PATH"] = geopi_output_artifacts_image_statistic_path
    os.makedirs(geopi_output_artifacts_image_statistic_path, exist_ok=True)

    # Set the output artifacts image map path for the current run
    geopi_output_artifacts_image_map_path = os.path.join(geopi_output_artifacts_image_path, "map")
    os.environ["GEOPI_OUTPUT_ARTIFACTS_IMAGE_MAP_PATH"] = geopi_output_artifacts_image_map_path
    os.makedirs(geopi_output_artifacts_image_map_path, exist_ok=True)

    # Set the output parameters path for the current run
    geopi_output_parameters_path = os.path.join(geopi_output_path, "parameters")
    os.environ["GEOPI_OUTPUT_PARAMETERS_PATH"] = geopi_output_parameters_path
    os.makedirs(geopi_output_parameters_path, exist_ok=True)

    # Set the outout metrics path for the current run
    geopi_output_metrics_path = os.path.join(geopi_output_path, "metrics")
    os.environ["GEOPI_OUTPUT_METRICS_PATH"] = geopi_output_metrics_path
    os.makedirs(geopi_output_metrics_path, exist_ok=True)

    # Set the summary outout path for the current run
    geopi_output_summary_path = os.path.join(geopi_output_path, "summary")
    os.environ["GEOPI_OUTPUT_SUMMARY_PATH"] = geopi_output_summary_path
    os.makedirs(geopi_output_summary_path, exist_ok=True)


def get_os() -> str:
    """Get the operating system.

    Returns
    -------
    str
        The operating system.
    """
    my_os = platform.system()
    if my_os == "Windows":
        return "Windows"
    elif my_os == "Linux":
        return "Linux"
    elif my_os == "Darwin":
        return "macOS"
    else:
        return "Unknown"


def check_package(package_name: str) -> bool:
    """Check whether the package is installed.

    Parameters
    ----------
    package_name : str
        The name of the package.

    Returns
    -------
    bool
        Whether the package is installed.
    """
    import importlib

    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name: str) -> None:
    """Install the package.

    Parameters
    ----------
    package_name : str
        The name of the package.
    """
    import subprocess

    subprocess.check_call(["python", "-m", "pip", "install", "--quiet", package_name])


def clear_output(text: str = None) -> None:
    """Clear the console output."""
    if text:
        flag = input(text)
    else:
        flag = input("(Press Enter key to move forward)")
    my_os = platform.system()
    if flag == "":
        if my_os == "Windows":
            os.system("cls")  # for Windows
        else:
            os.system("clear")  # for Linux and macOS
    print("")


def save_fig(fig_name: str, local_image_path: str, mlflow_artifact_image_path: str = None, tight_layout: bool = True) -> None:
    """Save the figure in the local directory and in mlflow specialized directory.

    Parameters
    ----------
    fig_name : str
        Figure name.

    local_image_path : str
        The path to store the image.

    mlflow_artifact_image_path : str, default=None
        The path to store the image in mlflow.

    tight_layout : bool, default=True
        Automatically adjust subplot parameters to give specified padding.
    """
    full_path = os.path.join(local_image_path, fig_name + ".png")
    print(f"Save figure '{fig_name}' in {local_image_path}.")
    if tight_layout:
        plt.tight_layout()

    # Check that the original file exists,
    # and if it does, add a number after the filename to distinguish
    i = 1
    dir = full_path[:-4]
    while os.path.isfile(full_path):
        full_path = dir + str(i) + ".png"
        i = i + 1
    plt.savefig(full_path, format="png", dpi=300)
    plt.close()
    if mlflow_artifact_image_path:
        mlflow.log_artifact(full_path, artifact_path=mlflow_artifact_image_path)
    else:
        mlflow.log_artifact(full_path)


def save_data(df: pd.DataFrame, name_column: str, df_name: str, local_data_path: str, mlflow_artifact_data_path: str = None, index: bool = False) -> None:
    """Save the dataset in the local directory and in mlflow specialized directory.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to store.

    name_column:
        The name of the data.

    df_name : str
        The name of the data sheet.

    local_data_path : str
        The path to store the data sheet

    mlflow_artifact_data_path : str, default=None
        The path to store the data sheet in mlflow.

    index : bool, default=False
        Whether to write the index.
    """
    if name_column is not None and len(df) == len(name_column):
        if not df.index.empty and len(name_column.index) == len(df.index) and set(df.index) == set(name_column.index):
            name_column = name_column.reindex(df.index)
        df.reset_index(drop=True, inplace=True)
        name_column.reset_index(drop=True, inplace=True)
        df = pd.concat([name_column, df], axis=1)
    try:
        # drop the index in case that the dimensions change
        full_path = os.path.join(local_data_path, "{}.xlsx".format(df_name))
        df.to_excel(full_path, index=index)
        if mlflow_artifact_data_path:
            mlflow.log_artifact(full_path, artifact_path=mlflow_artifact_data_path)
        else:
            mlflow.log_artifact(full_path)
        print(f"Successfully store '{df_name}' in '{df_name}.xlsx' in {local_data_path}.")
    except ModuleNotFoundError:
        print("** Please download openpyxl by pip3 **")
        print("** The data will be stored in .csv file **")
        full_path = os.path.join(local_data_path, "{}.csv".format(df_name))
        df.to_csv(full_path, index=index)
        print(f"Successfully store '{df_name}' in '{df_name}.csv' in {local_data_path}.")


def save_data_without_data_identifier(df: pd.DataFrame, df_name: str, local_data_path: str, mlflow_artifact_data_path: str = None, index: bool = False) -> None:
    """Save the dataset in the local directory and in mlflow specialized directory.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to store.

    df_name : str
        The name of the data sheet.

    local_data_path : str
        The path to store the data sheet

    mlflow_artifact_data_path : str, default=None
        The path to store the data sheet in mlflow.

    index : bool, default=False
        Whether to write the index.
    """
    try:
        # drop the index in case that the dimensions change
        full_path = os.path.join(local_data_path, "{}.xlsx".format(df_name))
        df.to_excel(full_path, index=index)
        if mlflow_artifact_data_path:
            mlflow.log_artifact(full_path, artifact_path=mlflow_artifact_data_path)
        else:
            mlflow.log_artifact(full_path)
        print(f"Successfully store '{df_name}' in '{df_name}.xlsx' in {local_data_path}.")
    except ModuleNotFoundError:
        print("** Please download openpyxl by pip3 **")
        print("** The data will be stored in .csv file **")
        full_path = os.path.join(local_data_path, "{}.csv".format(df_name))
        df.to_csv(full_path, index=index)
        print(f"Successfully store '{df_name}' in '{df_name}.csv' in {local_data_path}.")


def save_text(string: str, text_name: str, local_text_path: str, mlflow_artifact_text_path: str = None) -> None:
    """Save the text.

    Parameters
    ----------
    string : str
        The text to store.

    text_name : str
        The name of the text.

    local_text_path : str
        The path to store the text.

    mlflow_artifact_text_path : str, default=None
        The path to store the text in mlflow.
    """
    full_path = os.path.join(local_text_path, text_name + ".txt")
    with open(full_path, "w") as f:
        f.write(string)
    print(f"Successfully store '{text_name}' in '{text_name}.txt' in {local_text_path}.")
    if not mlflow_artifact_text_path:
        pass
    elif mlflow_artifact_text_path == "root":
        # If artifact_path is "root", the artifact will be placed in the root artifacts directory.
        mlflow.log_artifact(full_path)
    else:
        mlflow.log_artifact(full_path, artifact_path=mlflow_artifact_text_path)


def save_model(model: object, model_name: str, data_sample: pd.DataFrame, local_model_path: str, mlflow_artifact_model_path: str = None) -> None:
    """Save the model in the local directory and in mlflow specialized directory.

    Parameters
    ----------
    model : object
        The model to store.

    model_name : str
        The name of the model.

    data_sample : pd.DataFrame
        The sample of the dataset.

    local_model_path : str
        The path to store the model.

    mlflow_artifact_model_path : str, default=None
        The path to store the model in mlflow.
    """
    pickle_filename = model_name + ".pkl"
    pickle_path = os.path.join(local_model_path, pickle_filename)
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Successfully store '{model_name}' in '{pickle_filename}' in {local_model_path}.")
    joblib_filename = model_name + ".joblib"
    joblib_path = os.path.join(local_model_path, joblib_filename)
    with open(joblib_path, "wb") as f:
        joblib.dump(model, f)
    print(f"Successfully store '{model_name}' in '{joblib_filename}' in {local_model_path}.")
    if not mlflow_artifact_model_path:
        mlflow.sklearn.log_model(model, model_name, input_example=data_sample)
    else:
        mlflow.sklearn.log_model(model, mlflow_artifact_model_path, input_example=data_sample)


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


def copy_files(GEOPI_OUTPUT_ARTIFACTS_PATH: str, GEOPI_OUTPUT_METRICS_PATH: str, GEOPI_OUTPUT_PARAMETERS_PATH: str, GEOPI_OUTPUT_SUMMARY_PATH: str) -> None:
    """Copy all files from the source folder to the destination folder.

    Parameters
    ----------
    GEOPI_OUTPUT_ARTIFACTS_PATH: str
        Source folder path.

    GEOPI_OUTPUT_METRICS_PATH: str
        Source folder path.

    GEOPI_OUTPUT_PARAMETERS_PATH: str
        Source folder path.

    GEOPI_OUTPUT_SUMMARY_PATH: str
        Destination folder path
    """
    source_paths = [GEOPI_OUTPUT_ARTIFACTS_PATH, GEOPI_OUTPUT_METRICS_PATH, GEOPI_OUTPUT_PARAMETERS_PATH]
    for source_path in source_paths:
        for root, dirs, files in os.walk(source_path):
            for file in files:
                source_file_path = os.path.join(root, file)
                shutil.copy2(source_file_path, GEOPI_OUTPUT_SUMMARY_PATH)


def copy_files_from_source_dir_to_dest_dir(source_dir: str, dest_dir: str) -> None:
    """Copy all files from the source folder to the destination folder.

    Parameters
    ----------
    source_dir: str
        Source folder path.

    dest_dir: str
        Destination folder path
    """
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            source_file_path = os.path.join(root, file)
            shutil.copy2(source_file_path, dest_dir)


def list_excel_files(directory: str) -> list:
    """Recursively lists all Excel files (including .xlsx, .xls, and .csv) in the specified directory and its subdirectories.

    Parameters
    ----------
    directory : str
        The path to the directory to search for Excel files.

    Returns
    -------
    excel_files : list
        A list of file paths for all Excel files found.

    Notes
    -----
    (1) The function uses `os.walk` to traverse the directory and its subdirectories.
    (2) Only files with extensions .xlsx, .xls, and .csv are considered as Excel files.
    """
    excel_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".xls") or file.endswith(".csv"):
                excel_files.append(os.path.join(root, file))
    return excel_files
