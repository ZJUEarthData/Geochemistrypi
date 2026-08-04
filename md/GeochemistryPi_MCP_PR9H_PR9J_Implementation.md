# GeochemistryPi MCP PR9H–PR9J Implementation

Status: implemented and locally verified in the working tree on Windows. Remote
CI, the complete scheduled parity matrix, other operating systems, and real
GUI-client runs remain external release evidence and must not be inferred from
source changes.

## PR9H: persistent experiments and managed MLflow UI

The installer now owns three independent persistent locations:

- `runs` for MCP run records and original CLI outputs;
- `tracking` for the shared local MLflow file store;
- `service-state` for managed MLflow UI identity and logs.

Repair and upgrade may replace the private Python environments, but they do not
replace these data roots. Uninstall removes the runtimes and owned client
registrations while preserving run and tracking data.

Every data-mining request receives the persistent tracking root. A request may
set `existing_experiment_id`; the wrapper resolves that stable ID in the CLI's
Python 3.9 environment and requires `experiment_name` to match exactly before it
creates a run directory or starts the analysis CLI. This avoids retry loops and
ambiguous name selection. The public lifecycle tools are:

- `list_experiments`;
- `get_experiment`;
- `start_mlflow_ui`;
- `mlflow_ui_status`;
- `stop_mlflow_ui`.

The UI binds to `127.0.0.1`, never starts automatically, detects port conflicts,
and persists PID, process creation time, exact command, tracking root, port, and
start time. Stop verifies every identity field before terminating the recorded
process tree. A reused PID or changed command produces `ownership_mismatch` and
is not killed. A dead recorded process is recovered as stopped after an unclean
server shutdown.

## PR9I: executable parity matrix

`tests/mcp_wrapper/parity/fixtures/full_parity_matrix_v1.json` is the objective
matrix. It enumerates:

- 36 manual single-model scenarios;
- all 11 classification AutoML branches;
- all 13 regression AutoML branches;
- five aggregate task scenarios;
- inference, label, missing-value, scaling, selector, feature-engineering, map,
  experiment, data-source, and Time Series dimensions.

The executable comparison contract requires unchanged input SHA-256 values,
identical complete recursive file inventories, preserved table values and
ordering, `1e-9` absolute and `1e-7` relative floating-point tolerances, and
structural image checks rather than cross-platform binary image hashes. Golden
updates require scientific review.

The normal PR gate retains representative real direct-CLI-versus-stdio-MCP
tests. The expensive full matrix has separate manual/scheduled shards for
classification manual, classification AutoML, regression manual, regression
AutoML, unsupervised manual, aggregates, and branch/rendering scenarios. Source
code for a shard is not evidence that a remote shard passed; release records
must retain the actual CI result.

## PR9J: user-facing workflow

`validate_analysis` resolves the dataset reference, hashes and inspects the
input, validates existing experiment selection, compiles the same interaction
plan used by execution, and returns:

- resolved task and data source;
- dataset path, SHA-256, size, and columns;
- selected model names and estimated model count;
- tuning and experiment modes;
- optional application source;
- warnings for aggregate, AutoML, inference, maps, and existing experiments.

It does not allocate a run ID, create a run workspace, or start the analysis
CLI. `start_analysis` repeats integrity validation at execution time and returns
the same model names and count in its acknowledgement.

Durable status records expose stable stages: `queued`, `running_cli`,
`indexing_outputs`, and final `completed`, `failed`, or `cancelled`. Aggregate
results include expected, succeeded, and failed child counts as well as the
ordered bounded child records.

Public validation failures report the invalid field, a bounded representation
of the actual value, valid alternatives or schema source, and a concrete next
action. Dataset contents and internal exception traces are not returned.

## Client boundary

The supported target list contains standard JSON plus Codex, Claude Desktop,
Claude Code, Cursor, VS Code, Gemini CLI, Windsurf, Cline, Roo Code, Zed,
Continue, Kiro, and OpenCode. Registration remains atomic, creates one recovery
backup for an existing file, preserves unrelated settings, and unregisters
only the still-owned command. Every client launches the same zero-argument MCP
server; no user configures an internal Python interpreter.

The controlled standard protocol has automated tools/list, capability,
validation, start, polling, result, and unregister evidence. Configuration
adapters have automated round-trip evidence for all 14 targets. Actual
natural-language runs inside the 13 external client applications remain marked
`pending_external_client_run` in the versioned acceptance matrix until those
applications are exercised; they remain a public-release gate.

## Local verification evidence

The final local Windows verification completed the following checks:

- Python 3.9 core and CLI suite: 91 passed with database environment variables
  unset;
- Python 3.11 MCP non-expensive suite: 210 passed and 75 real-run cases were
  deliberately deselected by marker;
- final wheel inspection: 184 core files and 30 MCP files, with no packaged test
  directories and with the new tracking, experiment, UI, and capability files;
- fresh wheel imports resolved from isolated `site-packages`, not the working
  tree;
- fresh-installed core: 91 passed; fresh-installed MCP: 210 passed;
- nine representative real direct-CLI-versus-MCP scenarios passed in isolated
  Python 3.9 and Python 3.11 environments;
- one additional real stdio scenario created an experiment in the CLI
  environment, listed and fetched it through MCP, attached a classification run
  by stable ID, and read the run back from the same experiment;
- the real installed MLflow UI completed explicit start, status, and verified
  stop on `127.0.0.1`, leaving no ownership state file;
- install, repeated install, repair, Doctor, and uninstall completed in an
  isolated local application root; Doctor passed 7/7 after install and repair;
- a real MLflow experiment survived repeated install, repair, and uninstall;
  uninstall removed both private runtimes while preserving runs, tracking data,
  and an unrelated MCP configuration entry;
- `uvx pre-commit run --all-files` passed every configured hook.

One intentionally cold-cache installer attempt reached the 15-minute network
and dependency-fetch timeout. Its verified orphaned installer child was stopped,
and the same lifecycle completed using the existing uv package cache. This is
not recorded as a cold-cache acceptance pass.

## Recommended user conversation

The geochemist needs only one sentence:

> 帮我分析桌面上的 `rocks.xlsx`，看看哪些元素最能区分不同岩性。

The client finds and inspects the file, asks one short scientific question only
when needed, explains the proposed work without model or schema terminology,
and waits for confirmation. It then starts, monitors, and summarizes the run
automatically. Experiment IDs, validation calls, polling, artifact references,
and MLflow process control are implementation details unless the user explicitly
asks to review history, outputs, or the local tracking interface.

If a scientist explicitly requests a task, column, model, tuning mode, or
parameter value, that choice is authoritative. The client maps it to the strict
request unchanged. Defaults are used only for omitted choices; invalid or
unsupported choices are explained and sent back for a decision instead of being
silently replaced.

## Release boundary

`public_release_ready` remains false. Do not change it until the full PR9I
matrix passes on the required platforms, the external client matrix is run, the
remaining platform and rollback lifecycle checks pass, and signed release
artifacts and publication decisions exist. Local Windows wheels, fresh installs,
and the install/repair/uninstall lifecycle now have evidence above; that evidence
does not substitute for Linux, macOS, remote CI, or real-client acceptance.
