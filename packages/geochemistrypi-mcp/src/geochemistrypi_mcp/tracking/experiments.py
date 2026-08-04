"""Read-only bridge to MLflow in the isolated GeochemistryPi CLI environment."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..api.schemas import (
    GetExperimentRequest,
    GetExperimentResponse,
    ListExperimentsRequest,
    ListExperimentsResponse,
)
from ..config.constants import ISOLATED_CLI_ENVIRONMENT_VARIABLES
from ..config.settings import McpSettings, resolve_cli_interpreter


class ExperimentStoreError(ValueError):
    """Raised when the persistent tracking store cannot satisfy a request."""


class ExperimentManager:
    """Invoke the core package's bounded JSON bridge in its Python 3.9 environment."""

    def __init__(self, settings: McpSettings) -> None:
        self.settings = settings

    def _invoke(self, arguments: tuple[str, ...]) -> dict[str, Any]:
        executable, _ = self.settings.require_supported_cli()
        interpreter = resolve_cli_interpreter(executable)
        tracking_root = self.settings.tracking_root
        if tracking_root is None:
            raise ExperimentStoreError("The installer-owned MLflow tracking root is not configured.")
        tracking_root.mkdir(parents=True, exist_ok=True)
        command = (
            str(interpreter),
            "-m",
            "geochemistrypi.tracking_cli",
            *arguments,
            "--tracking-root",
            str(tracking_root),
        )
        environment = os.environ.copy()
        for name in ISOLATED_CLI_ENVIRONMENT_VARIABLES:
            environment.pop(name, None)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExperimentStoreError("Cannot query the persistent MLflow tracking store.") from exc
        if completed.returncode != 0:
            try:
                error_value = json.loads(completed.stderr.strip().splitlines()[-1])
                message = str(error_value.get("error", ""))
            except (IndexError, AttributeError, json.JSONDecodeError):
                message = ""
            raise ExperimentStoreError(message or "The MLflow tracking query failed in the CLI environment.")
        if len(completed.stdout) > 2_000_000:
            raise ExperimentStoreError("The MLflow tracking response exceeded the 2 MB safety limit.")
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ExperimentStoreError("The CLI returned an invalid MLflow tracking response.") from exc
        if not isinstance(value, dict):
            raise ExperimentStoreError("The CLI returned a non-object MLflow tracking response.")
        return value

    def list(self, request: ListExperimentsRequest) -> ListExperimentsResponse:
        value = self._invoke(("list", "--maximum-experiments", str(request.maximum_experiments)))
        return ListExperimentsResponse.model_validate(value)

    def get(self, request: GetExperimentRequest) -> GetExperimentResponse:
        value = self._invoke(
            (
                "get",
                "--experiment-id",
                request.experiment_id,
                "--maximum-runs",
                str(request.maximum_runs),
            )
        )
        return GetExperimentResponse.model_validate(value)

    def require_matching_name(self, experiment_id: str, expected_name: str) -> GetExperimentResponse:
        response = self.get(GetExperimentRequest(experiment_id=experiment_id, maximum_runs=0))
        if response.experiment.name != expected_name:
            raise ExperimentStoreError(
                f"existing_experiment_id {experiment_id!r} belongs to experiment "
                f"{response.experiment.name!r}; set experiment_name to that exact value or choose another ID."
            )
        return response
