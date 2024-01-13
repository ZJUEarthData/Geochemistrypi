# -*- coding: utf-8 -*-
import os
import platform
import subprocess
import threading
from typing import Optional

import typer

from ._version import __version__
from .data_mining.cli_pipeline import cli_pipeline
from .data_mining.constants import WORKING_PATH

app = typer.Typer()

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
FRONTEND_PATH = os.path.join(CURRENT_PATH, "frontend")
BACKEND_PATH = os.path.join(CURRENT_PATH, "start_dash_pipeline.py")
PIPELINE_PATH = os.path.join(CURRENT_PATH, "start_cli_pipeline.py")
MLFLOW_STORE_PATH = os.path.join(f"file:{WORKING_PATH}", "geopi_tracking")


def _version_callback(value: bool) -> None:
    """Show Geochemistry Pi version."""
    if value:
        typer.echo(f"Geochemistry π {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: Optional[bool] = typer.Option(None, "--version", "-v", help="Show version.", callback=_version_callback, is_eager=True)) -> None:
    """
    Geochemistry π is an open-sourced highly automated machine learning Python framework for data-driven geochemistry discovery.
    It has the cores components of continous training, machine learning lifecycle management and model serving.
    """
    return


@app.command()
def data_mining(
    data: str = typer.Option("", help="The path of the training data without model inference."),
    training: str = typer.Option("", help="The path of the training data."),
    inference: str = typer.Option("", help="The path of the inference data."),
    mlflow: bool = typer.Option(False, help="Start the mlflow server."),
    web: bool = False,
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
        start_mlflow_command = f"mlflow ui --backend-store-uri {MLFLOW_STORE_PATH} "
        subprocess.run(start_mlflow_command, shell=True)

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
        # If mlflow is enabled, start the mlflow server, otherwise start the CLI pipeline
        if mlflow:
            # Start mlflow server to track the experiment
            mlflow_thread = threading.Thread(target=start_mlflow)
            mlflow_thread.start()
        else:
            # If the data is provided, start the CLI pipeline with continuous training
            if data:
                cli_pipeline(data)
            # If the training data and inference data are provided, start the CLI pipeline with continuous training and inference
            elif training and inference:
                cli_pipeline(training, inference)
            # If no data is provided, use built-in data to start the CLI pipeline with continuous training and inference
            else:
                cli_pipeline(training, inference)


@app.command()
def web_setup() -> None:
    """Set up the dependency of the web application."""
    my_os = platform.system()
    if my_os == "Windows":
        # Define the command to download and install Yarn on Windows using Chocolatey package manager
        download_yarn = "choco install yarn"
        subprocess.run(download_yarn, shell=True)
        # Define the command to download and install Node.js on Windows using Chocolatey package manager
        download_node = "choco install nodejs"
        subprocess.run(download_node, shell=True)
    elif my_os == "Linux":
        # Define the command to download and install Yarn on Linux using npm
        download_yarn = "apt-get install -y yarn"
        subprocess.run(download_yarn, shell=True)
        # Define the command to download and install Node.js on Linux using npm
        download_node = "apt-get install -y nodejs"
        subprocess.run(download_node, shell=True)
    elif my_os == "Darwin":
        try:
            check_node = "node --version"
            subprocess.run(check_node, shell=True)
            print("Node.js is already installed.")
        except subprocess.CalledProcessError:
            # Define the command to download and install Node.js on macOS using Homebrew
            download_node = "brew install node"
            subprocess.run(download_node, shell=True)
        try:
            # Define the command to check if Yarn is installed
            check_yarn = "yarn --version"
            subprocess.run(check_yarn, shell=True)
            print("Yarn is already installed.")
        except subprocess.CalledProcessError:
            # Define the command to download and install Yarn on macOS using Homebrew
            download_yarn = "brew install yarn"
            subprocess.run(download_yarn, shell=True)

        # Define the command to install the frontend dependencies
        install_frontend_dependency_cmd = f"cd {FRONTEND_PATH} && yarn install"
        subprocess.run(install_frontend_dependency_cmd, shell=True)
