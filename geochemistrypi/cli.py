# -*- coding: utf-8 -*-
import os

# import platform
import subprocess
import threading
from typing import Optional

import typer
from rich import print

from ._version import __version__
from .data_mining.cli_pipeline import cli_pipeline
from .data_mining.enum import DataSource

app = typer.Typer()

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
FRONTEND_PATH = os.path.join(CURRENT_PATH, "frontend")
BACKEND_PATH = os.path.join(CURRENT_PATH, "start_dash_pipeline.py")
PIPELINE_PATH = os.path.join(CURRENT_PATH, "start_cli_pipeline.py")


def _version_callback(value: bool) -> None:
    """Show Geochemistry Pi version."""
    if value:
        typer.echo(f"Geochemistry π {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: Optional[bool] = typer.Option(None, "--version", "-v", help="Show version.", callback=_version_callback, is_eager=True)) -> None:
    """
    Geochemistry π is an open-sourced highly automated machine learning Python framework for data-driven geochemistry discovery.
    It has the cores components of continous training, machine learning lifecycle management and model inference.
    """
    return


@app.command()
def data_mining(
    data: str = typer.Option("", help="The path of the training data without model inference."),
    desktop: bool = typer.Option(False, help="Use the data in the directory 'geopi_input' on the desktop for model training and model inference."),
    training: str = typer.Option("", help="The path of the training data."),
    application: str = typer.Option("", help="The path of the inference data."),
    mlflow: bool = typer.Option(False, help="Start the mlflow server."),
    # web: bool = False,
) -> None:
    """Implement the customized automated machine learning pipeline for geochemistry data mining."""

    def start_backend():
        """Start the backend server."""
        start_backend_command = f"python {BACKEND_PATH}"
        subprocess.run(start_backend_command, shell=True)

    def start_frontend():
        """Start the frontend server."""
        start_frontend_command = f"cd {FRONTEND_PATH} && yarn start"
        subprocess.run(start_frontend_command, shell=True)

    def start_mlflow():
        """Start the mlflow server."""
        # Check if the current working directory has the 'geopi_tracking' directory to store the tracking data for mlflow
        # If yes, set the MLFLOW_STORE_PATH to the current working directory
        # If no, set the MLFLOW_STORE_PATH to the desktop
        cur_working_dir = os.getcwd()
        geopi_tracking_dir = os.path.join(cur_working_dir, "geopi_tracking")
        if not os.path.exists(geopi_tracking_dir):
            print(f"[bold red]The 'geopi_tracking' directory is not found in the current working directory '{cur_working_dir}'.[bold red]")
            geopi_tracking_dir = os.path.join(os.path.expanduser("~"), "Desktop", "geopi_tracking")
            if not os.path.exists(geopi_tracking_dir):
                print("[bold red]The 'geopi_tracking' directory is not found on the desktop.[bold red]")
                print("[bold green]Creating the 'geopi_tracking' directory ...[/bold green]")
                print("[bold green]Successfully create 'geopi_tracking' directory on the desktop to store the tracking data for mlflow.[/bold green]")
            else:
                print("[bold green]The 'geopi_tracking' directory is found on the desktop.[bold green]")
                print("[bold green]Our software will use the 'geopi_tracking' directory on the desktop to store the tracking data for mlflow.[bold green]")
        else:
            print(f"[bold green]The 'geopi_tracking' directory is found in the current working directory '{cur_working_dir}'.[bold green]")
            print("[bold green]Our software will use the 'geopi_tracking' directory in the current working directory to store the tracking data for mlflow.[bold green]")
        MLFLOW_STORE_PATH = "file:///" + geopi_tracking_dir
        print("[bold green]Press [bold magenta]Ctrl + C[/bold magenta] to close mlflow server at any time.[bold green]")
        start_mlflow_command = f"mlflow ui --backend-store-uri {MLFLOW_STORE_PATH} "
        subprocess.run(start_mlflow_command, shell=True)

    # TODO: Currently, the web application is not fully implemented. It is disabled by default.
    web = False
    if web:
        # Start the backend and frontend in parallel
        backend_thread = threading.Thread(target=start_backend)
        backend_thread.start()
        frontend_thread = threading.Thread(target=start_frontend)
        frontend_thread.start()
        # Wait for the threads to finish
        backend_thread.join()
        frontend_thread.join()
    else:
        if mlflow:
            # If mlflow is enabled, start the mlflow server, otherwise start the CLI pipeline
            mlflow_thread = threading.Thread(target=start_mlflow)
            mlflow_thread.start()
        elif desktop:
            # Start the CLI pipeline with the data in the directory 'geopi_input' on the desktop
            #   - Both continuous training and model inference
            #   - Continuous training only
            cli_pipeline(training_data_path="", application_data_path="", data_source=DataSource.DESKTOP)
        else:
            if data:
                # If the data is provided, start the CLI pipeline with continuous training
                cli_pipeline(training_data_path=data, application_data_path="", data_source=DataSource.ANY_PATH)
            elif training and application:
                # If the training data and inference data are provided, start the CLI pipeline with continuous training and inference
                cli_pipeline(training_data_path=training, application_data_path=application, data_source=DataSource.ANY_PATH)
            elif training and not application:
                # If the training data is provided, start the CLI pipeline with continuous training
                cli_pipeline(training_data_path=training, application_data_path="", data_source=DataSource.ANY_PATH)
            else:
                # If no data is provided, use built-in data to start the CLI pipeline with continuous training and inference
                cli_pipeline(training_data_path="", application_data_path="", data_source=DataSource.BUILT_IN)


# TODO: Currently, the web application is not fully implemented. It is disabled by default.
# @app.command()
# def web_setup() -> None:
#     """Set up the dependency of the web application."""
#     my_os = platform.system()
#     if my_os == "Windows":
#         # Define the command to download and install Yarn on Windows using Chocolatey package manager
#         download_yarn = "choco install yarn"
#         subprocess.run(download_yarn, shell=True)
#         # Define the command to download and install Node.js on Windows using Chocolatey package manager
#         download_node = "choco install nodejs"
#         subprocess.run(download_node, shell=True)
#     elif my_os == "Linux":
#         # Define the command to download and install Yarn on Linux using npm
#         download_yarn = "apt-get install -y yarn"
#         subprocess.run(download_yarn, shell=True)
#         # Define the command to download and install Node.js on Linux using npm
#         download_node = "apt-get install -y nodejs"
#         subprocess.run(download_node, shell=True)
#     elif my_os == "Darwin":
#         try:
#             check_node = "node --version"
#             subprocess.run(check_node, shell=True)
#             print("Node.js is already installed.")
#         except subprocess.CalledProcessError:
#             # Define the command to download and install Node.js on macOS using Homebrew
#             download_node = "brew install node"
#             subprocess.run(download_node, shell=True)
#         try:
#             # Define the command to check if Yarn is installed
#             check_yarn = "yarn --version"
#             subprocess.run(check_yarn, shell=True)
#             print("Yarn is already installed.")
#         except subprocess.CalledProcessError:
#             # Define the command to download and install Yarn on macOS using Homebrew
#             download_yarn = "brew install yarn"
#             subprocess.run(download_yarn, shell=True)

#         # Define the command to install the frontend dependencies
#         install_frontend_dependency_cmd = f"cd {FRONTEND_PATH} && yarn install"
#         subprocess.run(install_frontend_dependency_cmd, shell=True)
