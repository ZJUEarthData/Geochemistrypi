# GeochemistryPi MCP

GeochemistryPi MCP is a lightweight local wrapper around the existing
`geochemistrypi data-mining` command. It does not implement models, metrics,
plots, or preprocessing. The installed GeochemistryPi CLI remains the only
scientific execution path and keeps producing its original output files.

This package exposes all 36 single-model families, the supported supervised
AutoML branches, five real CLI aggregate branches, application inference,
semantic world maps, seeded Time Series analysis, persistent MLflow
experiments, and an explicitly managed local MLflow UI through stdio MCP.
It supports verified release-bundle installation as well as repository
development setup. The protected release workflow publishes the CLI to PyPI
and the signed MCP bundle to GitHub Releases without rebuilding either wheel.
MCP Registry publication is a separate future distribution target.

The wrapper supports explicit feature and supervised target columns,
classification label customization, numeric-target regression, target-free
clustering, decomposition, and anomaly detection, missing-value handling, all
three CLI scaling methods, both supervised CLI feature selectors, manual
hyperparameters, supported AutoML branches, and supervised application data.
Every accepted request is translated into the existing interactive CLI; the
wrapper never trains a model or recreates a metric, plot, prediction, or model
file.

## Environment boundary

The MCP process requires Python 3.10 or newer and the official MCP Python SDK.
GeochemistryPi 0.8.1 keeps its existing Python 3.9 and Pydantic 1 environment.
The wrapper starts the public CLI command as a subprocess, so neither process
has to share incompatible dependencies.

The versioned development compatibility policy is:

| Boundary | Accepted version |
| --- | --- |
| Private MCP runtime | Python `3.11` exactly; package metadata `>=3.10,<4` |
| MCP SDK | `==2.0.0` |
| Private GeochemistryPi CLI runtime | Python `3.9` exactly; package metadata `>=3.9,<3.10` |
| GeochemistryPi CLI | `0.8.1` |
| Interaction plan | schema 1 |
| CLI automation input/events | schema 1 |
| CLI capability manifest | schema 1 |
| Artifact index | schema 1 |

`get_capabilities` returns this policy and the current resource limits. The
0.2.1 package is the stable bundle channel and reports
`public_release_ready` only for artifacts published by the protected release
workflow. MCP Registry publication remains a separate future distribution
target and does not change the signed GitHub release-bundle installation.

Before execution, the wrapper verifies GeochemistryPi's installed distribution
version with the Python interpreter that owns the CLI launcher. Windows virtual
environments (`venv`, pip, or uv) and Conda environments use different
interpreter locations; both layouts are supported. Wrapper-only Python and
database environment variables are removed from the CLI subprocess.

## Verified release-bundle setup

A production release bundle contains exactly two wheels,
`release-manifest.json`, and one `.sigstore.json` bundle for each of those
three signed files. Setup rejects an unexpected version pair, an unlisted
wheel, packaged repository tests, a size or SHA-256 mismatch, an untrusted
workflow identity, or a missing signature before changing the installed
runtime.

On macOS, GeochemistryPi's XGBoost runtime requires the system OpenMP library.
Install it once with `brew install libomp` before setup. The installer Doctor
checks the complete scientific import path and reports this prerequisite
directly if it is missing; Windows and Linux require no equivalent step.

The intended scientist-facing installation is natural language. Download and
extract the signed release bundle, then give its folder to any supported Agent:

```text
请从这个文件夹安装 GeochemistryPi MCP：<发布包文件夹的绝对路径>。
请先验证发布签名，自动配置我正在使用的 MCP 客户端，安装后运行健康检查。
如果失败，不要跳过签名校验，请用普通用户能理解的语言告诉我原因和解决办法。
```

This prompt is operating-system neutral. The Agent should locate the MCP wheel
inside the bundle and run the verified setup command below; scientists should
not need to construct a Python environment, file URI, or client JSON by hand.
An explicit model or analysis parameter supplied by the scientist always takes
precedence over automatic defaults.

For manual or diagnostic use, install [uv](https://docs.astral.sh/uv/), then
bootstrap setup from the MCP wheel in the downloaded bundle. Replace the
example URI and bundle directory with absolute paths:

```text
uvx --python 3.11 --from "geochemistrypi-mcp[release] @ file:///ABSOLUTE/PATH/geochemistrypi_mcp-0.2.1-py3-none-any.whl" geochemistrypi-mcp-setup install --bundle /ABSOLUTE/PATH/release-bundle
```

Windows file URIs use forward slashes, for example
`file:///D:/Downloads/release-bundle/geochemistrypi_mcp-0.2.1-py3-none-any.whl`.
Linux and macOS use a URI such as
`file:///home/user/Downloads/release-bundle/geochemistrypi_mcp-0.2.1-py3-none-any.whl`.
Unsigned bundles fail closed. `--allow-unsigned-bundle` exists only for local
release-candidate testing and is recorded permanently in the install
manifest; it is not a production verification mode.

See the
[MCP architecture and implementation overview](<../../docs/source/For Developer/MCP/index.md>)
for the shared architecture, safety, verification, and release boundaries.

## One-action development setup from a local clone

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run setup
from its root:

```text
git clone https://github.com/ZJUEarthData/Geochemistrypi.git
cd Geochemistrypi
uv run --isolated --no-project --python 3.11 --with-editable packages/geochemistrypi-mcp geochemistrypi-mcp-setup install
```

The development command creates an MCP Python 3.11 environment and a separate
GeochemistryPi Python 3.9 environment under the platform-native application
data directory. It installs both packages from the current clone, verifies the
0.2.1/0.8.1 version handshake, persists private paths, runs an end-to-end
doctor, and registers one stable zero-argument server command in detected MCP
clients. Users never configure either private Python environment or a CLI path.
The CLI environment accepts third-party dependencies only as prebuilt wheels,
so setup never falls back to a platform compiler or system headers. Development
setup permits a source build only for the current local `geochemistrypi`
project; verified release-bundle setup installs both projects from wheels.

Safe auto-detection supports 13 MCP hosts: Codex, Claude Desktop, Claude Code,
Cursor, VS Code, Gemini CLI, Windsurf, Cline, Roo Code, Zed, Continue, Kiro,
and OpenCode. Codex's shared configuration also makes the same server available
to OpenAI surfaces that consume that configuration. A standard `mcpServers`
JSON file is always generated as a client-neutral fallback. Choose clients
explicitly by repeating `--client`, for example:

```text
uv run --isolated --no-project --python 3.11 --with-editable packages/geochemistrypi-mcp geochemistrypi-mcp-setup install --client codex --client cursor
```

To configure every adapter available on the current operating system, use:

```text
uv run --isolated --no-project --python 3.11 --with-editable packages/geochemistrypi-mcp geochemistrypi-mcp-setup install --client all
```

| Client | Registration interface | Installed user configuration |
| --- | --- | --- |
| Codex | TOML | `~/.codex/config.toml` |
| Claude Desktop | JSON | Platform Claude application-data directory |
| Claude Code | Official CLI | `claude mcp ... --scope user` |
| Cursor | JSON | `~/.cursor/mcp.json` |
| VS Code | JSON | Platform `Code/User/mcp.json` |
| Gemini CLI | JSON | `~/.gemini/settings.json` |
| Windsurf | JSON | `~/.codeium/windsurf/mcp_config.json` |
| Cline | JSON | `~/.cline/data/settings/cline_mcp_settings.json` |
| Roo Code | JSON | Default VS Code Roo extension global storage |
| Zed | JSON | Platform Zed `settings.json` |
| Continue | Round-trip YAML | `~/.continue/config.yaml` |
| Kiro | JSON | `~/.kiro/settings/mcp.json` |
| OpenCode | Nested JSON | `$XDG_CONFIG_HOME/opencode/opencode.json` or `~/.config/opencode/opencode.json` |

The adapters follow each client's own schema; the MCP server and scientific
behavior are identical for every client. Roo Code custom storage locations
cannot be inferred from outside the extension, so automatic setup targets its
default VS Code extension storage. OpenCode JSONC is deliberately not rewritten
because doing so could destroy comments; use its MCP settings UI when only an
`opencode.jsonc` file exists.

Existing client settings are preserved. Before the first change to an existing
JSON, TOML, or YAML file, setup creates one adjacent
`.geochemistrypi.bak` recovery copy. A conflicting `geochemistrypi` entry is
never replaced by normal install; use `repair` when replacement is intentional.
Uninstall removes an entry only when it still matches the command installed by
GeochemistryPi MCP, so a later user replacement is not deleted.

## Doctor, repair, upgrade, rollback, and uninstall

On Windows, run the installed doctor directly:

```powershell
& "$env:LOCALAPPDATA\GeochemistryPi MCP\environments\mcp\Scripts\geochemistrypi-mcp-doctor.exe"
```

On macOS or Linux:

```text
~/.local/state/geochemistrypi-mcp/environments/mcp/bin/geochemistrypi-mcp-doctor
```

The doctor performs ten checks: persisted paths and all six resource limits;
the exact CLI/MCP compatibility manifest; release-manifest and wheel hashes;
both installed distribution-inventory hashes; writable managed storage; exact
private Python/package versions; the public CLI command; the complete
scientific CLI import path (including native dependencies); zero-argument MCP
startup; and the 13 expected tools. It also rejects a claimed rollback whose
private snapshot is missing. Add `--json` for machine-readable output.

On Linux and macOS, `repair`, `rollback`, or `uninstall` may use the installed
setup command. On Windows, always use the same external wheel bootstrap shown
for install; a process running inside the private environment cannot replace or
delete its loaded DLLs. The installed Windows command detects this condition
before writing anything and prints a copyable external command. Repair
recreates both private environments from the active hash-checked bundle
and replaces only the owned `geochemistrypi` client entry. Upgrade requires a
new verified bundle, runs Doctor before and after replacement, and retains one
pre-upgrade private-runtime snapshot. Rollback restores that snapshot and then
removes the replaced runtime. Uninstall removes active and rollback runtimes
and owned client entries while preserving managed runs, persistent MLflow
tracking data, service logs, and recovery backups. A verified managed MLflow UI
is stopped before repair, upgrade, rollback, or uninstall; an ambiguous process
is never killed.

```text
uv run --isolated --no-project --python 3.11 --with-editable packages/geochemistrypi-mcp geochemistrypi-mcp-setup repair
uvx --python 3.11 --from "geochemistrypi-mcp[release] @ file:///ABSOLUTE/PATH/geochemistrypi_mcp-0.2.1-py3-none-any.whl" geochemistrypi-mcp-setup upgrade --bundle /ABSOLUTE/PATH/new-release-bundle
uvx --python 3.11 --from "geochemistrypi-mcp[release] @ file:///ABSOLUTE/PATH/geochemistrypi_mcp-0.2.1-py3-none-any.whl" geochemistrypi-mcp-setup rollback
uv run --isolated --no-project --python 3.11 --with-editable packages/geochemistrypi-mcp geochemistrypi-mcp-setup uninstall
```

If setup reports an existing client entry with a different command, inspect
that client's recovery backup and run `repair` only when replacing the
`geochemistrypi` entry is intentional. To copy the client-neutral fallback into
an unsupported client, print it with:

```text
uv run --isolated --no-project --python 3.11 --with-editable packages/geochemistrypi-mcp geochemistrypi-mcp-setup print-config
```

Environment variables remain available only as explicit development/test
overrides:

```text
GEOCHEMISTRYPI_MCP_APP_ROOT=<absolute isolated application root>
GEOCHEMISTRYPI_CLI_EXECUTABLE=<absolute path to geochemistrypi executable>
GEOCHEMISTRYPI_MCP_RUNS_ROOT=<absolute path for managed runs>
GEOCHEMISTRYPI_MCP_TRACKING_ROOT=<absolute path for persistent MLflow data>
GEOCHEMISTRYPI_MCP_SERVICE_STATE_ROOT=<absolute path for managed UI state>
GEOCHEMISTRYPI_MCP_SETTINGS_FILE=<absolute path to a test settings file>
GEOCHEMISTRYPI_MCP_MAX_DATASET_BYTES=<positive byte limit>
GEOCHEMISTRYPI_MCP_MAX_PENDING_RUNS=<positive active-and-queued run limit>
GEOCHEMISTRYPI_MCP_MAX_PROCESS_SECONDS=<positive total CLI timeout>
```

Normal setup persists a 512 MiB dataset limit, at most 256 columns and 200
model-facing artifact references, one concurrent run, at most eight active or
queued runs, and a 900-second total CLI timeout. Queue
admission is atomic: a rejected request does not receive a run ID or create a
partial run workspace. Development environment variables can tighten these
limits without exposing them as analysis-tool arguments.

On Windows, the wrapper also checks the longest projected CLI plot path before
starting a run. If a configured runs root would exceed the legacy plotting
library path limit, the request fails before training with an instruction to
choose a shorter runs root.

## Server behavior

After setup, every registered client launches the installed command with no
arguments:

```text
geochemistrypi-mcp
```

The command starts a stdio server and intentionally prints nothing while it
waits for an MCP client. Standard output is reserved for MCP messages.

## Classification, regression, clustering, decomposition, and anomaly-detection requests

`start_analysis` uses the `task` field to select a strict task-specific schema.
Requests created before regression support remain classification requests when
`task` is omitted. New clients should always send it explicitly.

A minimal regression request is:

```json
{
  "task": "regression",
  "training_dataset_path": "D:\\data\\rocks.csv",
  "experiment_name": "Geochemistry Regression",
  "run_name": "Ridge Reference",
  "identifier_column": "SampleID",
  "feature_columns": ["SIO2", "TIO2"],
  "target_column": "MeasuredValue",
  "model": {"type": "ridge_regression"}
}
```

Regression exposes all 15 single-model CLI families. The target must be one
finite numeric column. The wrapper validates the CLI's fixed 10-fold
cross-validation minimum, XGBoost-only unprocessed-missing-value branch,
feature-dependent regression plot prompts, and optional application-data
preprocessing before execution. Linear Regression and Polynomial Regression do
not offer AutoML in the public CLI; unsupported requests are rejected before a
process starts.

A minimal clustering request is:

```json
{
  "task": "clustering",
  "training_dataset_path": "D:\\data\\rocks.csv",
  "experiment_name": "Geochemistry Clustering",
  "run_name": "KMeans Reference",
  "identifier_column": "SampleID",
  "feature_columns": ["SIO2", "TIO2", "AL2O3"],
  "model": {"type": "kmeans", "number_of_clusters": 3}
}
```

Clustering exposes the five models in the public menu: KMeans, DBSCAN,
Agglomerative, AffinityPropagation, and MeanShift. It intentionally has no
target, train/test split, feature selection, AutoML, or application-data
inference. OPTICS exists internally in GeochemistryPi 0.8.1 but is not in the
public CLI menu and is therefore not advertised or accepted by MCP. The wrapper
validates numeric finite features, unique identifiers, model/data-size
constraints, missing-value resolution, and plot dimensions before execution.

A minimal decomposition request is:

```json
{
  "task": "decomposition",
  "training_dataset_path": "D:\\data\\rocks.csv",
  "experiment_name": "Geochemistry Decomposition",
  "run_name": "PCA Reference",
  "identifier_column": "SampleID",
  "feature_columns": ["SIO2", "TIO2", "AL2O3"],
  "model": {"type": "pca", "number_of_components": 2, "svd_solver": "auto"}
}
```

Decomposition exposes PCA, T-SNE, and MDS. Each request has no target,
train/test split, feature selection, AutoML, or application-data inference.
The wrapper validates finite numeric features, retained row counts, PCA
component/solver limits, T-SNE perplexity, missing-value resolution, and the
CLI's plot prerequisites. The existing CLI remains responsible for
`X Reduced.xlsx`, common visualizations, PCA bi/tri-plots, model files, and the
transform pipeline.

A minimal anomaly-detection request is:

```json
{
  "task": "anomaly_detection",
  "training_dataset_path": "D:\\data\\rocks.csv",
  "experiment_name": "Geochemistry Anomaly Detection",
  "run_name": "Isolation Forest Reference",
  "identifier_column": "SampleID",
  "feature_columns": ["SIO2", "TIO2", "AL2O3"],
  "model": {"type": "isolation_forest", "contamination": 0.1}
}
```

Anomaly detection exposes Isolation Forest and Local Outlier Factor. It is
target-free and intentionally has no supervised split, feature selection,
AutoML, or application-data inference. The wrapper validates row and feature
bounds before the CLI starts; the CLI remains responsible for the original
normal/anomalous tables, diagrams, model files, and transform pipeline.

## Development verification

From the repository root, the cross-platform wrapper suite is:

```text
uv run --isolated --project packages/geochemistrypi-mcp --extra test python -m pytest tests/mcp_wrapper/installation tests/mcp_wrapper/interaction tests/mcp_wrapper/protocol
```

The real parity suite additionally requires a supported GeochemistryPi 0.8.1
CLI command in `GEOCHEMISTRYPI_CLI_EXECUTABLE`.

## Tools

- `get_capabilities`
- `list_datasets`
- `inspect_dataset`
- `list_experiments`
- `get_experiment`
- `start_mlflow_ui`
- `mlflow_ui_status`
- `stop_mlflow_ui`
- `validate_analysis`
- `start_analysis`
- `get_run_status`
- `get_run_result`
- `cancel_run`

The request schema contains scientific choices only. It never accepts raw CLI
answers, shell commands, environment variables, or output directories.

`list_datasets` discovers the eight datasets shipped with the installed CLI and
supported `.csv`/`.xlsx` files directly inside `Desktop/geopi_input`. Discovery
is read-only: it does not create the Desktop directory, copy files, recurse into
subdirectories, or accept a resolved path outside that directory. Each entry
contains a stable ID, role, task, size, SHA-256, and any capability that blocks
analysis. Use the returned ID without copying its absolute installation path:

```json
{
  "dataset": {
    "source": "builtin",
    "dataset_id": "builtin:classification",
    "expected_sha256": "<hash returned by list_datasets>"
  },
  "sample_rows": 5
}
```

`training_dataset`, `application_dataset`, and the inspection `dataset` field
accept the same reference shape. It can also be
`{"source":"desktop","file_name":"rocks.xlsx"}` or
`{"source":"path","path":"D:\\data\\rocks.csv"}`. The legacy
`training_dataset_path` and `application_dataset_path` fields remain supported,
but a request must provide exactly one representation for each input. Dataset
inspection applies pandas-compatible `Unnamed: N` names to empty Excel headers.
Explicit datasets with duplicate or unsafe headers fail deterministically;
trusted bundled data can use pandas-compatible duplicate suffixes and returns a
visible `header_warnings` entry. GeochemistryPi deliberately supports `.csv`
and `.xlsx`; `.xls` is not advertised because the CLI reader has no reliable
legacy Excel dependency.

Call `get_capabilities` before planning a run. It returns separate versioned
classification, regression, clustering, decomposition, and anomaly-detection
model matrices, the runtime compatibility policy, resource limits, and
unsupported combinations. `model_selection.mode = "all"` executes the real CLI
aggregate branch and returns a parent summary plus ordered child results;
regression multiple-target behavior remains
outside the validated contract; classification sample balancing is not offered
because the GeochemistryPi 0.8.1 public CLI does not call its internal balancing
helper; and clustering, decomposition, or anomaly detection does not silently
inherit supervised controls.
Application-data inference supports the same validated feature-engineering
formulas used for supervised training; the CLI stores and replays those formulas
by source column name.

The response also includes every stable capability ID and status from
`cli_capability_manifest_v1.json`. Implemented PR9 capabilities link to precise
tests or full-matrix scenario IDs. Remaining gaps are returned explicitly and
are never presented as successful analysis support.

MCP-started analyses use CLI automation schema 1. The CLI consumes stable,
ordered input IDs inside its own process and writes a strict event document;
unknown, duplicate, missing, or unused inputs fail closed. Human CLI runs keep
the original prompts and scientific workflow. The legacy prompt-synchronized
driver remains covered only as a migration path and is no longer the default
for managed runs.

## Natural-language workflow

The same conversation works in Codex, Claude Desktop/Code, Cursor, VS Code,
Gemini CLI, Windsurf, Cline, Continue, Kiro, OpenCode, Roo Code, Zed, or any
client using the generated standard JSON configuration. A geochemist can begin
with one sentence:

> 帮我分析桌面上的 `rocks.xlsx`，看看哪些元素最能区分不同岩性。

In English, the same request is:

> Analyze `rocks.xlsx` on my Desktop and tell me which elements best distinguish
> the rock types.

The user does not need to name a model, MCP tool, schema field, identifier,
feature, target, run stage, or output path. The client should find and inspect
the named file, explain the proposed analysis in ordinary scientific language,
and ask only for information that cannot be determined safely. Useful follow-up
questions sound like “Which column represents the rock type?” and “Shall I start
the analysis now?” rather than requests for parameters or JSON.

A scientist may also specify as much detail as desired, still in natural
language:

> 请使用随机森林，把 `Lithology` 作为要判断的岩性，使用 `SiO2` 和 `TiO2`，
> 树的数量设为 500。

Explicit scientific choices always take priority over inferred or default
choices. The client translates them to the validated request without changing
them. If a requested model or value is unsupported or unsafe, it explains the
conflict and asks what to do; it never silently selects a replacement.

After confirmation, the client starts the work, watches it until completion,
and returns a concise scientific summary with links or references to the
original GeochemistryPi outputs. These tool calls are automatic and should not
be turned into extra instructions for the user.

`validate_analysis` never creates a run or starts the analysis CLI. A run status
uses the stable stages `queued`, `running_cli`, `indexing_outputs`, and a final
`completed`, `failed`, or `cancelled` stage. Large aggregate and AutoML requests
report their model count before execution.

For lifecycle work, first call `list_experiments`, then reuse the returned
stable ID with `existing_experiment_id` and the exact returned experiment name.
Name/ID mismatches fail before execution. The MLflow UI is never auto-started;
use `start_mlflow_ui`, open its `127.0.0.1` URL, inspect status, and use
`stop_mlflow_ui` when finished.

## Full parity gate

The versioned PR9I matrix contains 36 manual single-model scenarios, all 11
classification and 13 regression AutoML branches, five aggregate scenarios,
and the required inference, preprocessing, mapping, experiment, source, and
Time Series dimensions. Comparisons require unchanged input hashes, identical
recursive file inventories and table ordering, explicit `1e-9` absolute and
`1e-7` relative numeric tolerances, and structural checks for platform-dependent
images. Full AutoML/model execution is sharded behind the scheduled or manual
release-candidate workflow; the normal PR gate keeps representative real parity.
