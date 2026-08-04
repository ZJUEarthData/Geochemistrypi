# PR2 MCP Package and Local Run Control

## Purpose

PR2 makes the validated CLI interaction driver callable through the Model
Context Protocol. A compatible local MCP client can inspect a dataset, start a
classification run, continue handling other requests while the CLI works,
check status, cancel the owned run, and retrieve references to the original
CLI outputs.

The package is an execution wrapper. It does not train models, calculate
metrics, create plots, or regenerate reports. Those operations remain inside
the existing `geochemistrypi data-mining` process.

## Environment boundary

The root `geochemistrypi` distribution keeps Python 3.9, Pydantic 1, and its
existing scientific dependencies. The independent
`packages/geochemistrypi-mcp` distribution uses Python 3.10 or newer,
Pydantic 2, and the official MCP Python SDK 2.0.0.

The MCP process never imports the GeochemistryPi package. It starts an
explicitly verified public `geochemistrypi` executable in its own environment.
This preserves the current CLI and avoids forcing incompatible dependencies
into either process.

## Tool contract

The stdio server exposes six client-neutral tools:

- `get_capabilities` reports only implemented workflows and safety limits;
- `inspect_dataset` reads an absolute CSV or XLSX path without modifying it
  and returns a bounded sample, inferred types, size, and SHA-256;
- `start_analysis` accepts scientific classification choices and queues the
  existing CLI workflow;
- `get_run_status` reads atomic local state without waiting for completion;
- `get_run_result` returns bounded CLI-reported metrics and original artifact
  references;
- `cancel_run` cancels only the live process tree owned by that run.

Every input schema rejects unknown fields. No tool accepts a shell command,
executable path, raw prompt answers, environment variables, or an output
directory.

## Managed run layout

Each run is isolated under the configured runs root:

```text
runs/<run-id>/
  wrapper/
    request.json
    status.json
    result.json
    interaction-trace.json
    stdout.log
    stderr.log
    artifact-index.json
  workspace/
    geopi_output/<experiment>/<run>/
      artifacts/
      metrics/
      parameters/
      summary/
    geopi_tracking/
```

Wrapper files and scientific output files are kept separate. JSON metadata and
capture files are published with same-directory atomic replacement. The
wrapper indexes existing CLI files but never moves, renames, loads, or
recreates models and images.

## Input and output safety

Dataset access follows the user's explicit absolute path and the current
operating-system account. No global trusted-data directory is required.

Before a run, the wrapper resolves the path, verifies a regular CSV/XLSX file,
enforces the configured size limit, and records its SHA-256. It checks the hash
again immediately before execution and after the CLI exits. A changed input
causes the wrapper result to fail integrity validation.

All outputs stay inside the managed workspace because the CLI subprocess uses
that workspace as its current directory. Artifact responses contain only
run-scoped references beneath the original four output directories.

## Cancellation and recovery

The default concurrency is one. `start_analysis` returns after validation and
queues the long task in a local worker thread, so the MCP protocol loop remains
available for status and cancellation requests.

The driver records the CLI PID and operating-system creation time from the
live process handle. Cancellation revalidates that identity and terminates the
root plus its descendants. It does not accept a PID from a client. After an
unclean server shutdown, stale `queued` or `running` metadata is marked failed;
the wrapper deliberately does not terminate a process from stale PID metadata.

## Current scope

PR2 supports only the PR1 training-only classification reference branch:
standardization and logistic regression with the characterized settings. Full
classification menu coverage, application data, additional models, feature
engineering, feature selection, balancing, and AutoML belong to PR3.

The development package is not published. `GEOCHEMISTRYPI_CLI_EXECUTABLE` and
`GEOCHEMISTRYPI_MCP_RUNS_ROOT` are temporary developer configuration. PR4 will
provide automated environment preparation, version handshake, doctor checks,
and client-neutral registration.

## Verification

The MCP unit and protocol suite runs in Python 3.11:

```text
python -m pytest tests/mcp_wrapper/interaction tests/mcp_wrapper/protocol
```

The real parity test requires a separate GeochemistryPi 0.8.0 CLI environment:

```text
python -m pytest -m mcp_cli_parity tests/mcp_wrapper/parity
```

The parity path starts the server over real stdio, queues the CLI run through
MCP, polls it, retrieves its result, confirms the protocol remains healthy,
and compares the complete 124-file output tree, metrics, parameters, split
membership, and predictions with a direct public CLI run.
