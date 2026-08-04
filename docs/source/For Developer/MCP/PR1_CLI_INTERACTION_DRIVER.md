# PR1 CLI Interaction Driver

## Purpose

The interaction driver lets software operate the existing public
`geochemistrypi data-mining` command without changing the human CLI workflow.
It translates a strict semantic classification request into an ordered,
versioned prompt plan and sends each answer only after the matching prompt is
observed.

This is an automation boundary, not a second analysis implementation. The
driver does not import model, preprocessing, plotting, metric, or inference
code. All scientific work and all product outputs still come from the public
CLI process.

## PR1 supported workflow

PR1 deliberately supports only the classification branch frozen by the PR0
baseline:

- one training dataset in CSV or XLSX format;
- a named identifier, target, and one or more feature columns;
- original labels kept and encoded by the CLI;
- standardization;
- no feature engineering, feature selection, AutoML, or application data;
- logistic regression with L2 penalty, the `lbfgs` solver, and no class
  weighting;
- a configurable test ratio and regularization strength.

Unsupported fields, missing columns, conflicting column roles, duplicate
columns, unsafe run names, and unavailable files are rejected before the CLI
starts. Broader classification coverage belongs to PR3 and must be added with
new parity evidence.

## Components

- `schemas.py` defines the strict semantic request. Unknown fields are
  rejected.
- `interaction_plan.py` reads only the dataset header, maps column names to the CLI's
  one-based selections, resolves the public console command, and compiles the
  41-step versioned interaction plan.
- `cli_driver.py` starts that command in an isolated workspace, watches stdout and
  stderr concurrently, matches ordered output anchors, and writes each answer
  only after its prompt is present.

The public command is passed to `subprocess.Popen` as an argument list. No
shell command string is used, so spaces and platform-specific path separators
do not change the command meaning.

## Minimal developer example

```python
from pathlib import Path

from geochemistrypi_mcp import (
    ClassificationPlanCompiler,
    ClassificationRequest,
    CliInteractionDriver,
)

request = ClassificationRequest(
    training_dataset_path=Path("C:/data/classification.csv"),
    experiment_name="Local Classification",
    run_name="Logistic Regression V1",
    identifier_column="SampleID",
    feature_columns=("SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)"),
    target_column="Label",
)
plan = ClassificationPlanCompiler().compile(request)
result = CliInteractionDriver().run(plan)
print(result.output_root)
```

The driver now lives in the separate `geochemistrypi-mcp` distribution because
the official MCP SDK and the existing CLI require incompatible dependency
environments. It does not add or replace an end-user CLI command.

## Failure behavior

The driver fails closed:

- `UnexpectedPromptError`: a known later prompt arrives out of order;
- `PromptTimeoutError`: an expected prompt does not arrive or the process
  exceeds its total time limit;
- `UnusedResponsesError`: the CLI exits before consuming the complete plan;
- `CliProcessError`: all responses were consumed but the CLI exits non-zero;
- `WorkspacePathError`: a projected Windows output path is too long for the
  plotting dependencies used by the current CLI.

The Windows check is performed before model execution. Callers can resolve it
by choosing a shorter workspace parent rather than discovering the problem
after plots have already been calculated.

## Run evidence

Every completed or failed run has a private capture directory inside its
isolated workspace:

```text
wrapper/
  stdout.log
  stderr.log
  interaction-trace.json
```

The trace records the command, plan version, matched steps, responses, return
code, remaining steps, timestamps, and failure information. The original CLI
output remains under `geopi_output`; the driver does not rename, summarize, or
recreate those files.

## Verification

Run the deterministic subprocess checks on either Windows or Linux:

```text
python -m pytest tests/mcp_wrapper/interaction/test_driver_unit.py
```

Run the direct-versus-driven real CLI parity check:

```text
python -m pytest -m mcp_cli_parity tests/mcp_wrapper/parity
```

The parity test executes the same public CLI twice. One run receives the PR0
responses directly and one is started through stdio MCP and the
prompt-synchronized driver. Both runs must produce the PR0 124-file manifest,
matching metrics and parameters, and identical test membership and
predictions.

The parity CI job runs real direct-versus-MCP parity on Linux. A separate
Windows/Linux matrix runs deterministic subprocess, protocol, cancellation,
isolation, capture, and Windows path-budget checks.
