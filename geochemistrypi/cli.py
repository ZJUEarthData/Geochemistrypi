# -*- coding: utf-8 -*-
import inspect
import os
import subprocess
from functools import wraps
from typing import Optional

import click
import typer
import typer.core
from rich import print

from ._version import __version__


def _patch_typer_for_click() -> None:
    """Keep the v0.8 Typer CLI compatible with newer Click releases."""
    unset = getattr(click.core, "UNSET", None)
    flag_value = inspect.signature(click.Option.__init__).parameters.get("flag_value")
    if (
        unset is not None
        and flag_value is not None
        and flag_value.default is unset
        and not getattr(typer.core.TyperOption.__init__, "_geopi_click_unset_patch", False)
    ):
        original_init = typer.core.TyperOption.__init__

        @wraps(original_init)
        def patched_init(self, *args, **kwargs):
            if kwargs.get("flag_value") is None:
                kwargs["flag_value"] = unset
            return original_init(self, *args, **kwargs)

        patched_init._geopi_click_unset_patch = True
        typer.core.TyperOption.__init__ = patched_init

    ctx_parameter = inspect.signature(click.Parameter.make_metavar).parameters.get("ctx")
    if ctx_parameter is not None and ctx_parameter.default is inspect.Parameter.empty:
        typer.core.rich = None


_patch_typer_for_click()

app = typer.Typer()

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
FRONTEND_PATH = os.path.join(CURRENT_PATH, "frontend")
BACKEND_PATH = os.path.join(CURRENT_PATH, "start_dash_pipeline.py")
PIPELINE_PATH = os.path.join(CURRENT_PATH, "start_cli_pipeline.py")


def _run_cli_pipeline(
    training_data_path: str,
    application_data_path: str,
    data_source_name: str,
) -> None:
    """Import the heavy Data Mining stack only when a run is requested."""
    from .data_mining.cli_pipeline import cli_pipeline
    from .data_mining.enum_ import DataSource

    cli_pipeline(
        training_data_path=training_data_path,
        application_data_path=application_data_path,
        data_source=DataSource[data_source_name],
    )


def _version_callback(value: bool) -> None:
    """Show Geochemistry Pi version."""
    if value:
        typer.echo(f"Geochemistry Pi {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """
    Top-level callback. If no subcommand is invoked, present an interactive launcher
    that lets the user choose between data_mining and chemical_modeling.
    """
    # If a subcommand is invoked explicitly, do nothing here.
    if ctx.invoked_subcommand:
        return

    # Interactive launcher
    print("\n[bold blue]Welcome to Geochemistry Pi[/bold blue]")
    print("[bold]Please choose a module to run:[/bold]")
    print("1 - Data Mining (automated ML pipelines)")
    print("2 - Chemical Modeling (equilibrium / fractionation / etc.)")
    print("Q - Quit")
    choice = input("Enter 1 or 2 (or Q to quit): ").strip().lower()

    if choice in ("q", "quit", "exit", ""):
        print("Exit.")
        raise typer.Exit()

    if choice == "1":
        # Delay-import data_mining modules to avoid loading heavy deps unless needed
        try:
            from .data_mining.enum_ import DataSource  # type: ignore
        except Exception as e:
            print(f"[red]Failed to import data_mining modules: {e}[/red]")
            print()
            print("[yellow]Likely cause: missing system library for lightgbm (libomp).[/yellow]")
            print("[yellow]To run data_mining on macOS, install libomp (Homebrew):[/yellow]")
            print("  brew install libomp")
            print("[yellow]Or with conda (if you use conda):[/yellow]")
            print("  conda install -c conda-forge libomp")
            print()
            print("[green]After installing libomp, re-run the CLI and choose option 1 again.[/green]")
            raise typer.Exit(code=1)

        # Provide a couple of simple choices for data_mining usage
        print("\n[bold]Data Mining launcher[/bold]")
        use_mlflow = input("Start MLflow UI? (y/N): ").strip().lower() == "y"
        if use_mlflow:
            # Start mlflow ui
            cur_working_dir = os.getcwd()
            geopi_tracking_dir = os.path.join(cur_working_dir, "geopi_tracking")
            if not os.path.exists(geopi_tracking_dir):
                geopi_tracking_dir = os.path.join(os.path.expanduser("~"), "Desktop", "geopi_tracking")
                if not os.path.exists(geopi_tracking_dir):
                    os.makedirs(geopi_tracking_dir, exist_ok=True)
                    print(f"[green]Created geopi_tracking at {geopi_tracking_dir}[/green]")
            MLFLOW_STORE_PATH = "file:///" + geopi_tracking_dir
            print("[bold green]Starting MLflow UI... Press Ctrl+C to stop.[/bold green]")
            subprocess.run(f"mlflow ui --backend-store-uri {MLFLOW_STORE_PATH}", shell=True)
            raise typer.Exit()

        # Start interactive data_mining pipeline (uses built-in defaults)
        print("[green]Starting Data Mining CLI pipeline (interactive mode).[/green]")
        _run_cli_pipeline("", "", DataSource.BUILT_IN.name)
        raise typer.Exit()

    if choice == "2":
        # Delay-import chemical_modeling pipeline
        try:
            from .chemical_modeling.cli_pipeline import cli_pipeline as cm_cli_pipeline  # type: ignore
        except Exception as e:
            print(f"[red]Failed to import chemical_modeling modules: {e}[/red]")
            raise typer.Exit(code=1)

        print("\n[bold]Chemical Modeling launcher[/bold]")
        non_interactive = input("Run non-interactively with a file? (y/N): ").strip().lower() == "y"
        if non_interactive:
            input_path = input("Path to input data file (absolute path recommended): ").strip()
            task_in = input("Task name (e.g. algo_fractionation) [or number from the list]: ").strip()
            method_in = input("Method name (e.g. internal_standard) [or number from the list]: ").strip()
            element_in = input("Element (e.g. Hg, Mo) [or number from the list]: ").strip()
            out_dir_in = input("Output directory (leave blank for default 'results'): ").strip() or None

            # If user provided numbers, map them to names using dispatcher helpers
            try:
                # import discovery helpers only when needed
                from .chemical_modeling.dispatcher import discover_tasks, list_method_elements, list_task_methods
            except Exception as e:
                print(f"[red]Failed to import dispatcher helpers: {e}[/red]")
                raise typer.Exit(code=1)

            task = task_in
            method = method_in
            element = element_in

            # Map numeric task index -> task name
            if task and task.isdigit():
                tasks = discover_tasks()
                idx = int(task) - 1
                if 0 <= idx < len(tasks):
                    task = tasks[idx]
                else:
                    print(f"[red]Invalid task index: {task}[/red]")
                    raise typer.Exit(code=1)
            # If empty, fall back to interactive later
            if method and method.isdigit() and task:
                methods = list_task_methods(task)
                method_keys = list(methods.keys())
                idx = int(method) - 1
                if 0 <= idx < len(method_keys):
                    method = method_keys[idx]
                else:
                    print(f"[red]Invalid method index: {method}[/red]")
                    raise typer.Exit(code=1)
            if element and element.isdigit() and task and method:
                elements = list_method_elements(task, method)
                idx = int(element) - 1
                if 0 <= idx < len(elements):
                    element = elements[idx]
                else:
                    print(f"[red]Invalid element index: {element}[/red]")
                    raise typer.Exit(code=1)

            # If any of task/method/element is blank, fall back to interactive chemical_modeling
            if not (task and method and element):
                print("[yellow]Incomplete non-interactive parameters -> falling back to interactive chemical_modeling.[/yellow]")
                cm_cli_pipeline("", {"non_interactive": False})
                raise typer.Exit()

            config = {
                "task": task,
                "method": method,
                "element": element,
                "non_interactive": True,
            }
            if out_dir_in:
                config["out_dir"] = out_dir_in

            print("[green]Starting Chemical Modeling pipeline...[/green]")
            cm_cli_pipeline(input_path or "", config)
        else:
            # Interactive chemical_modeling: let chemical_modeling.cli_pipeline present its menus
            cm_cli_pipeline("", {"non_interactive": False})
        raise typer.Exit()

    print("Unknown choice. Exiting.")
    raise typer.Exit()


@app.command(name="data-mining")
def data_mining(
    data: str = typer.Option("", "--data", help="Training data path without model inference."),
    desktop: bool = typer.Option(False, "--desktop", help="Use the desktop geopi_input directory."),
    training: str = typer.Option("", "--training", help="Training data path."),
    application: str = typer.Option("", "--application", help="Inference data path."),
    mlflow: bool = typer.Option(False, "--mlflow", help="Start the MLflow server."),
) -> None:
    """Run the v0.8 automated Data Mining pipeline."""
    if mlflow:
        tracking_dir = os.path.join(os.getcwd(), "geopi_tracking")
        os.makedirs(tracking_dir, exist_ok=True)
        store_uri = "file:///" + tracking_dir.replace("\\", "/")
        subprocess.run(
            ["mlflow", "ui", "--backend-store-uri", store_uri],
            check=False,
        )
        return

    if desktop:
        _run_cli_pipeline("", "", "DESKTOP")
    elif data:
        _run_cli_pipeline(data, "", "ANY_PATH")
    elif training:
        _run_cli_pipeline(training, application, "ANY_PATH")
    else:
        _run_cli_pipeline("", "", "BUILT_IN")


if __name__ == "__main__":
    app()

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
