# -*- coding: utf-8 -*-
import inspect
import os

# import platform
import subprocess
import threading
from functools import wraps
from pathlib import Path
from typing import List, Optional

import click
import typer
import typer.core
from rich import print

from ._version import __version__


def _patch_typer_option_flag_value_for_click_unset() -> None:
    """Adapt Typer 0.7 option defaults to Click versions that use UNSET."""
    unset = getattr(click.core, "UNSET", None)
    flag_value_parameter = inspect.signature(click.Option.__init__).parameters.get("flag_value")
    if unset is None or flag_value_parameter is None or flag_value_parameter.default is not unset:
        return
    if getattr(typer.core.TyperOption.__init__, "_geopi_click_unset_patch", False):
        return

    original_init = typer.core.TyperOption.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        if kwargs.get("flag_value") is None:
            kwargs["flag_value"] = unset
        return original_init(self, *args, **kwargs)

    patched_init._geopi_click_unset_patch = True
    typer.core.TyperOption.__init__ = patched_init


def _patch_typer_make_metavar_for_click_context() -> None:
    """Allow Typer's help code to work with both old and new Click metavar APIs."""
    if getattr(click.core, "UNSET", None) is None:
        return

    if not getattr(typer.core.TyperOption.make_metavar, "_geopi_click_ctx_patch", False):
        original_option_make_metavar = typer.core.TyperOption.make_metavar

        @wraps(original_option_make_metavar)
        def patched_option_make_metavar(self, ctx=None):
            return original_option_make_metavar(self, ctx)

        patched_option_make_metavar._geopi_click_ctx_patch = True
        typer.core.TyperOption.make_metavar = patched_option_make_metavar

    if not getattr(typer.core.TyperArgument.make_metavar, "_geopi_click_ctx_patch", False):

        def patched_argument_make_metavar(self, ctx=None):
            if self.metavar is not None:
                return self.metavar
            var = (self.name or "").upper()
            if not self.required:
                var = f"[{var}]"
            type_var = self.type.get_metavar(param=self, ctx=ctx)
            if type_var:
                var += f":{type_var}"
            if self.nargs != 1:
                var += "..."
            return var

        patched_argument_make_metavar._geopi_click_ctx_patch = True
        typer.core.TyperArgument.make_metavar = patched_argument_make_metavar


def _use_click_help_when_typer_rich_help_is_incompatible() -> None:
    """Keep Typer 0.7 help usable with Click versions that require ctx."""
    ctx_parameter = inspect.signature(click.Parameter.make_metavar).parameters.get("ctx")
    if ctx_parameter is not None and ctx_parameter.default is inspect.Parameter.empty:
        typer.core.rich = None


_patch_typer_option_flag_value_for_click_unset()
_patch_typer_make_metavar_for_click_context()
_use_click_help_when_typer_rich_help_is_incompatible()

app = typer.Typer()

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
FRONTEND_PATH = os.path.join(CURRENT_PATH, "frontend")
BACKEND_PATH = os.path.join(CURRENT_PATH, "start_dash_pipeline.py")
PIPELINE_PATH = os.path.join(CURRENT_PATH, "start_cli_pipeline.py")


def _run_cli_pipeline(
    training_data_path: str,
    application_data_path: str,
    data_source_name: str,
    automation_plan: str = "",
    automation_events: str = "",
    scientific_config: str = "",
    world_map_config: str = "",
    tracking_root: str = "",
    existing_experiment_id: str = "",
) -> None:
    def execute() -> None:
        # Keep scientific imports inside the configured execution boundary. If a native
        # dependency is missing or incompatible, the CLI now emits a structured
        # failure document instead of leaving the MCP driver with only a missing
        # events-file symptom.
        from .data_mining.cli_pipeline import cli_pipeline
        from .data_mining.enum_ import DataSource
        from .data_mining.plot.map_plot import WorldMapConfiguration

        parsed_world_map_config = WorldMapConfiguration.from_json(world_map_config) if world_map_config else None
        cli_pipeline(
            training_data_path=training_data_path,
            application_data_path=application_data_path,
            data_source=DataSource[data_source_name],
            world_map_configuration=parsed_world_map_config,
            tracking_root=tracking_root or None,
            existing_experiment_id=existing_experiment_id or None,
        )

    if scientific_config:
        from .scientific_execution import scientific_execution_context

        with scientific_execution_context(Path(scientific_config)):
            if automation_plan:
                from .automation import automation_input_adapter

                with automation_input_adapter(Path(automation_plan), Path(automation_events)):
                    execute()
            else:
                execute()
    elif automation_plan:
        from .automation import automation_input_adapter

        with automation_input_adapter(Path(automation_plan), Path(automation_events)):
            execute()
    else:
        execute()


def _version_callback(value: bool) -> None:
    """Show Geochemistry Pi version."""
    if value:
        typer.echo(f"Geochemistry Pi {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", "-v", help="Show version.", callback=_version_callback, is_eager=True, is_flag=True)) -> None:
    """
    Geochemistry Pi is an open-sourced highly automated machine learning Python framework for data-driven geochemistry discovery.
    It has the cores components of continous training, machine learning lifecycle management and model inference.
    """
    return


@app.command()
def data_mining(
    data: str = typer.Option("", help="The path of the training data without model inference."),
    desktop: bool = typer.Option(False, "--desktop", help="Use the data in the directory 'geopi_input' on the desktop for model training and model inference.", is_flag=True),
    training: str = typer.Option("", help="The path of the training data."),
    application: str = typer.Option("", help="The path of the inference data."),
    mlflow: bool = typer.Option(False, "--mlflow", help="Start the mlflow server.", is_flag=True),
    automation_plan: str = typer.Option("", help="Absolute path to a versioned machine-input plan."),
    automation_events: str = typer.Option("", help="Absolute path for versioned machine-input events."),
    scientific_config: str = typer.Option(
        "",
        "--scientific-config",
        help=("Absolute path to versioned scientific execution controls. It can be used " "with either the normal interactive CLI or machine automation."),
    ),
    world_map_config: str = typer.Option(
        "",
        "--world-map-config",
        help="Versioned JSON world-map configuration; omit it for the human interactive map flow.",
    ),
    tracking_root: str = typer.Option("", "--tracking-root", help="Absolute local MLflow tracking directory."),
    existing_experiment_id: str = typer.Option("", "--existing-experiment-id", help="Stable MLflow experiment ID in --tracking-root."),
    # web: bool = False,
) -> None:
    """Implement the customized automated machine learning pipeline for geochemistry data mining."""

    if bool(automation_plan) != bool(automation_events):
        raise typer.BadParameter("--automation-plan and --automation-events must be provided together.")
    selected_sources = int(bool(data)) + int(bool(training)) + int(bool(desktop))
    if selected_sources > 1:
        raise typer.BadParameter("Use exactly one training source: --data, --training, or --desktop.")
    if application and not training:
        raise typer.BadParameter("--application requires --training.")
    if mlflow and (data or training or application or desktop or automation_plan or scientific_config or world_map_config):
        raise typer.BadParameter("--mlflow cannot be combined with analysis or automation options.")
    if existing_experiment_id and not tracking_root:
        raise typer.BadParameter("--existing-experiment-id requires --tracking-root.")
    if tracking_root and not Path(tracking_root).expanduser().is_absolute():
        raise typer.BadParameter("--tracking-root must be an absolute local path.")

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
            _run_cli_pipeline(
                training_data_path="",
                application_data_path="",
                data_source_name="DESKTOP",
                automation_plan=automation_plan,
                automation_events=automation_events,
                scientific_config=scientific_config,
                world_map_config=world_map_config,
                tracking_root=tracking_root,
                existing_experiment_id=existing_experiment_id,
            )
        else:
            if data:
                # If the data is provided, start the CLI pipeline with continuous training
                _run_cli_pipeline(
                    training_data_path=data,
                    application_data_path="",
                    data_source_name="ANY_PATH",
                    automation_plan=automation_plan,
                    automation_events=automation_events,
                    scientific_config=scientific_config,
                    world_map_config=world_map_config,
                    tracking_root=tracking_root,
                    existing_experiment_id=existing_experiment_id,
                )
            elif training and application:
                # If the training data and inference data are provided, start the CLI pipeline with continuous training and inference
                _run_cli_pipeline(
                    training_data_path=training,
                    application_data_path=application,
                    data_source_name="ANY_PATH",
                    automation_plan=automation_plan,
                    automation_events=automation_events,
                    scientific_config=scientific_config,
                    world_map_config=world_map_config,
                    tracking_root=tracking_root,
                    existing_experiment_id=existing_experiment_id,
                )
            elif training and not application:
                # If the training data is provided, start the CLI pipeline with continuous training
                _run_cli_pipeline(
                    training_data_path=training,
                    application_data_path="",
                    data_source_name="ANY_PATH",
                    automation_plan=automation_plan,
                    automation_events=automation_events,
                    scientific_config=scientific_config,
                    world_map_config=world_map_config,
                    tracking_root=tracking_root,
                    existing_experiment_id=existing_experiment_id,
                )
            else:
                # If no data is provided, use built-in data to start the CLI pipeline with continuous training and inference
                _run_cli_pipeline(
                    training_data_path="",
                    application_data_path="",
                    data_source_name="BUILT_IN",
                    automation_plan=automation_plan,
                    automation_events=automation_events,
                    scientific_config=scientific_config,
                    world_map_config=world_map_config,
                    tracking_root=tracking_root,
                    existing_experiment_id=existing_experiment_id,
                )


@app.command("scientific-config")
def scientific_config_command(
    kind: str = typer.Option(
        "schema",
        "--kind",
        help="Document to generate: schema, registry, template, or example.",
    ),
    workflow_family: str = typer.Option(
        "",
        "--workflow-family",
        help="Registered workflow family for --kind template.",
    ),
    workflow_mode: str = typer.Option(
        "",
        "--workflow-mode",
        help="Registered workflow mode for --kind template.",
    ),
    method: str = typer.Option(
        "",
        "--method",
        help="Registered native estimator method for --kind template.",
    ),
    example: str = typer.Option(
        "",
        "--example",
        help="Named complete example for --kind example; available: isolation_forest.",
    ),
    output: str = typer.Option(
        "",
        "--output",
        help="Optional absolute .json output path; stdout is used when omitted.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace an existing regular --output file intentionally.",
        is_flag=True,
    ),
) -> None:
    """Discover or generate versioned scientific execution controls."""

    from .scientific_config import (
        build_scientific_execution_template,
        scientific_config_document_json,
        scientific_config_registry_document,
        scientific_execution_example,
        scientific_execution_json_schema,
        write_scientific_config_document,
    )
    from .scientific_execution import ScientificExecutionContractError

    supported_kinds = {"schema", "registry", "template", "example"}
    if kind not in supported_kinds:
        raise typer.BadParameter("--kind must be one of: example, registry, schema, template.")
    identity_values = (workflow_family, workflow_mode, method)
    if kind in {"schema", "registry"} and (any(identity_values) or example):
        raise typer.BadParameter(f"--kind {kind} does not accept workflow or example selectors.")
    if kind == "template" and (not all(identity_values) or example):
        raise typer.BadParameter("--kind template requires --workflow-family, --workflow-mode, and " "--method, and does not accept --example.")
    if kind == "example" and (not example or any(identity_values)):
        raise typer.BadParameter("--kind example requires --example and does not accept workflow selectors.")
    if force and not output:
        raise typer.BadParameter("--force requires --output.")

    try:
        if kind == "schema":
            document = scientific_execution_json_schema()
        elif kind == "registry":
            document = scientific_config_registry_document()
        elif kind == "template":
            document = build_scientific_execution_template(
                workflow_family,
                workflow_mode,
                method,
            )
        else:
            document = scientific_execution_example(example)
        if output:
            written = write_scientific_config_document(
                Path(output),
                document,
                overwrite=force,
            )
            typer.echo(str(written))
        else:
            typer.echo(scientific_config_document_json(document))
    except ScientificExecutionContractError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def datasets(
    source: str = typer.Option("all", help="Dataset source to list: all, builtin, or desktop."),
    output: str = typer.Option("json", help="Machine-readable output format; currently json."),
    detail: str = typer.Option(
        "compact",
        help=("Dataset detail: compact metadata, or full ordered columns, dtypes, " "missing/nonfinite counts, numeric bounds, and low-cardinality values."),
    ),
    inspect_column: Optional[List[str]] = typer.Option(
        None,
        "--inspect-column",
        help=("For --detail full, calculate joint completeness for these exact " "columns; repeat the option to preserve task order."),
    ),
    dataset_id: Optional[List[str]] = typer.Option(
        None,
        "--dataset-id",
        help="Return only these exact built-in dataset IDs; repeat the option for more than one.",
    ),
    file_name: Optional[List[str]] = typer.Option(
        None,
        "--file-name",
        help="Return only these exact Desktop/geopi_input file names; repeat the option for more than one.",
    ),
) -> None:
    """List built-in and Desktop datasets without creating, copying, or modifying files."""
    if source not in {"all", "builtin", "desktop"}:
        raise typer.BadParameter("--source must be one of: all, builtin, desktop.")
    if output != "json":
        raise typer.BadParameter("--output currently supports only json.")
    if detail not in {"compact", "full"}:
        raise typer.BadParameter("--detail must be one of: compact, full.")
    if inspect_column and detail != "full":
        raise typer.BadParameter("--inspect-column requires --detail full.")
    from .data_mining.datasets import dataset_catalog_json

    typer.echo(
        dataset_catalog_json(
            source,
            dataset_ids=tuple(dataset_id or ()),
            file_names=tuple(file_name or ()),
            detail=detail,
            inspection_columns=tuple(inspect_column or ()),
        )
    )


@app.command("time-series")
def time_series(
    input_path: str = typer.Option(..., "--input", help="Path to a .csv or .xlsx Time Series dataset."),
    analysis_mode: str = typer.Option(
        "subaerial_proportion",
        "--analysis-mode",
        help="Scientific producer: subaerial_proportion or continuous.",
    ),
    bin_width: float = typer.Option(..., "--bin-width", help="Positive age-bin width in Ma."),
    iterations: int = typer.Option(100, "--iterations", min=1, max=10_000, help="Bootstrap iterations (1-10000)."),
    seed: int = typer.Option(2025, "--seed", min=0, help="Non-negative deterministic random seed."),
    output_root: str = typer.Option("geopi_output", "--output-root", help="Root directory for standard GeochemistryPi outputs."),
    experiment_name: str = typer.Option("Time Series", "--experiment-name"),
    run_name: str = typer.Option("Subaerial Proportion", "--run-name"),
    sheet: str = typer.Option("0", "--sheet", help="Excel sheet index or name; ignored for CSV."),
    age_column: str = typer.Option("R_AGE", "--age-column"),
    minimum_age_column: Optional[str] = typer.Option(None, "--minimum-age-column"),
    maximum_age_column: str = typer.Option("R_MAX_AGE", "--maximum-age-column"),
    probability_column: str = typer.Option("SBAP", "--probability-column"),
    value_column: Optional[str] = typer.Option(None, "--value-column"),
    latitude_column: str = typer.Option("LATITUDE", "--latitude-column"),
    longitude_column: str = typer.Option("LONGITUDE", "--longitude-column"),
    relative_value_two_sigma: float = typer.Option(0.0, "--relative-value-two-sigma", min=0),
    minimum_samples_per_bin: int = typer.Option(1, "--minimum-samples-per-bin", min=1),
    filter_column: Optional[str] = typer.Option(None, "--filter-column"),
    filter_minimum: Optional[float] = typer.Option(None, "--filter-minimum"),
    filter_maximum: Optional[float] = typer.Option(None, "--filter-maximum"),
    age_unit: str = typer.Option("Ma", "--age-unit", help="Output age unit: Ma or Ga."),
    fit_curve: bool = typer.Option(True, "--fit-curve/--no-fit-curve", help="Include the fitted trend curve in the PDF."),
    compact_y_axis: bool = typer.Option(False, "--compact-y-axis/--no-compact-y-axis"),
    identifier_column: Optional[str] = typer.Option(None, "--identifier-column", help="Optional sample-name column validated against the source data."),
    selected_column: Optional[List[str]] = typer.Option(None, "--selected-column", help="Repeat in source order to reproduce the interactive selected-data range."),
    missing_values: str = typer.Option("error", "--missing-values", help="Missing-value handling: error or drop_rows."),
    drop_missing_column: Optional[List[str]] = typer.Option(None, "--drop-missing-column", help="Repeat to drop on specific selected columns; omit to drop on every selected column."),
    feature_engineering: str = typer.Option("none", "--feature-engineering", help="Time Series currently supports only none."),
    automation_plan: str = typer.Option("", help="Absolute path to a versioned machine-input plan."),
    automation_events: str = typer.Option("", help="Absolute path for versioned machine-input events."),
) -> None:
    """Run a reproducible public Time Series producer."""
    if bool(automation_plan) != bool(automation_events):
        raise typer.BadParameter("--automation-plan and --automation-events must be provided together.")

    def execute() -> None:
        from pathlib import Path

        from .data_mining.run_time_series import run_time_series_analysis

        try:
            output_directory = run_time_series_analysis(
                input_path=Path(input_path),
                output_root=Path(output_root),
                experiment_name=experiment_name,
                run_name=run_name,
                bin_width=bin_width,
                iterations=iterations,
                seed=seed,
                sheet=sheet,
                analysis_mode=analysis_mode,
                age_col=age_column,
                age_min_col=minimum_age_column,
                age_max_col=maximum_age_column,
                probability_col=probability_column,
                value_col=value_column,
                latitude_col=latitude_column,
                longitude_col=longitude_column,
                relative_value_two_sigma=relative_value_two_sigma,
                minimum_samples_per_bin=minimum_samples_per_bin,
                filter_col=filter_column,
                filter_minimum=filter_minimum,
                filter_maximum=filter_maximum,
                age_unit=age_unit,
                fit_curve=fit_curve,
                compact_y_axis=compact_y_axis,
                identifier_column=identifier_column,
                selected_columns=tuple(selected_column or ()),
                missing_value_method=missing_values,
                drop_missing_columns=tuple(drop_missing_column or ()),
                feature_engineering=feature_engineering,
            )
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"Saved Time Series outputs to {output_directory}")

    if automation_plan:
        from pathlib import Path

        from .automation import automation_input_adapter

        with automation_input_adapter(Path(automation_plan), Path(automation_events)):
            execute()
    else:
        execute()


@app.command("reference-anomaly-time-series")
def reference_anomaly_time_series(
    input_path: str = typer.Option(..., "--input", help="Path to an observation .csv or .xlsx file."),
    time_column: str = typer.Option(..., "--time-column", help="Observation date/time column."),
    signal_column: List[str] = typer.Option(..., "--signal-column", help="Repeat for each numeric signal column."),
    reference_label_column: str = typer.Option(..., "--reference-label-column"),
    reference_positive_value: List[str] = typer.Option(..., "--reference-positive-value", help="Repeat for every value denoting a reference anomaly."),
    output_root: str = typer.Option("geopi_output", "--output-root"),
    experiment_name: str = typer.Option("Time Series", "--experiment-name"),
    run_name: str = typer.Option("Reference Anomaly Series", "--run-name"),
    sheet: str = typer.Option("0", "--sheet", help="Observation Excel sheet index or name; ignored for CSV."),
    reference_label_provenance: str = typer.Option("external_reference", "--reference-label-provenance"),
    comparison_label_column: Optional[str] = typer.Option(None, "--comparison-label-column"),
    comparison_positive_value: Optional[List[str]] = typer.Option(None, "--comparison-positive-value"),
    comparison_label_provenance: str = typer.Option("calculated", "--comparison-label-provenance"),
    event_path: Optional[str] = typer.Option(None, "--event-input", help="Optional event .csv or .xlsx file."),
    event_sheet: str = typer.Option("0", "--event-sheet"),
    event_time_column: Optional[str] = typer.Option(None, "--event-time-column"),
    event_identifier_column: Optional[str] = typer.Option(None, "--event-identifier-column"),
    event_filter_column: Optional[str] = typer.Option(None, "--event-filter-column"),
    event_filter_value: Optional[List[str]] = typer.Option(None, "--event-filter-value"),
    association_window_days: Optional[float] = typer.Option(None, "--association-window-days", min=0),
    association_direction: str = typer.Option("before_event", "--association-direction"),
    automation_plan: str = typer.Option("", help="Absolute path to a versioned machine-input plan."),
    automation_events: str = typer.Option("", help="Absolute path for versioned machine-input events."),
) -> None:
    """Render externally supplied anomaly labels and optional events on numeric time series."""
    if bool(automation_plan) != bool(automation_events):
        raise typer.BadParameter("--automation-plan and --automation-events must be provided together.")

    def execute() -> None:
        from pathlib import Path

        from .data_mining.run_reference_anomaly_series import run_reference_anomaly_series

        try:
            output_directory = run_reference_anomaly_series(
                input_path=Path(input_path),
                output_root=Path(output_root),
                experiment_name=experiment_name,
                run_name=run_name,
                sheet=sheet,
                time_column=time_column,
                signal_columns=tuple(signal_column),
                reference_label_column=reference_label_column,
                reference_positive_values=tuple(reference_positive_value),
                reference_label_provenance=reference_label_provenance,
                comparison_label_column=comparison_label_column,
                comparison_positive_values=tuple(comparison_positive_value or ()),
                comparison_label_provenance=comparison_label_provenance,
                event_path=Path(event_path) if event_path is not None else None,
                event_sheet=event_sheet,
                event_time_column=event_time_column,
                event_identifier_column=event_identifier_column,
                event_filter_column=event_filter_column,
                event_filter_values=tuple(event_filter_value or ()),
                association_window_days=association_window_days,
                association_direction=association_direction,
            )
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"Saved reference anomaly Time Series outputs to {output_directory}")

    if automation_plan:
        from pathlib import Path

        from .automation import automation_input_adapter

        with automation_input_adapter(Path(automation_plan), Path(automation_events)):
            execute()
    else:
        execute()


@app.command("embedding-label-overlay")
def embedding_label_overlay(
    coordinate_path: str = typer.Option(..., "--coordinates", help="Path to an embedding-coordinate .csv or .xlsx file."),
    label_path: str = typer.Option(..., "--labels", help="Path to an identifier-label .csv or .xlsx file."),
    coordinate_identifier_column: str = typer.Option(..., "--coordinate-identifier-column"),
    label_identifier_column: str = typer.Option(..., "--label-identifier-column"),
    x_column: str = typer.Option(..., "--x-column"),
    y_column: str = typer.Option(..., "--y-column"),
    label_column: str = typer.Option(..., "--label-column"),
    positive_label_value: List[str] = typer.Option(..., "--positive-label-value", help="Repeat for every value denoting an anomaly."),
    output_root: str = typer.Option("geopi_output", "--output-root"),
    experiment_name: str = typer.Option("Artifact Composition", "--experiment-name"),
    run_name: str = typer.Option("Embedding Label Overlay", "--run-name"),
    coordinate_sheet: str = typer.Option("0", "--coordinate-sheet"),
    label_sheet: str = typer.Option("0", "--label-sheet"),
    automation_plan: str = typer.Option("", help="Absolute path to a versioned machine-input plan."),
    automation_events: str = typer.Option("", help="Absolute path for versioned machine-input events."),
) -> None:
    """Join existing 2-D coordinates and labels by identifier and render an overlay."""
    if bool(automation_plan) != bool(automation_events):
        raise typer.BadParameter("--automation-plan and --automation-events must be provided together.")

    def execute() -> None:
        from pathlib import Path

        from .data_mining.run_embedding_label_overlay import run_embedding_label_overlay

        try:
            output_directory = run_embedding_label_overlay(
                coordinate_path=Path(coordinate_path),
                label_path=Path(label_path),
                output_root=Path(output_root),
                experiment_name=experiment_name,
                run_name=run_name,
                coordinate_sheet=coordinate_sheet,
                label_sheet=label_sheet,
                coordinate_identifier_column=coordinate_identifier_column,
                label_identifier_column=label_identifier_column,
                x_column=x_column,
                y_column=y_column,
                label_column=label_column,
                positive_label_values=tuple(positive_label_value),
            )
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"Saved embedding-label overlay outputs to {output_directory}")

    if automation_plan:
        from pathlib import Path

        from .automation import automation_input_adapter

        with automation_input_adapter(Path(automation_plan), Path(automation_events)):
            execute()
    else:
        execute()


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
