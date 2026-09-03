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
private Python/package versions; the required public CLI commands and
`--scientific-config` option; the complete
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
GEOCHEMISTRYPI_MCP_ANALYSIS_SCHEMA_TASK=<classification|regression|clustering|decomposition|anomaly_detection|time_series>
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

## Classification, regression, clustering, decomposition, anomaly-detection, and Time Series requests

`validate_analysis` uses the `task` field to select a strict task-specific
schema. Every request also accepts additive `evaluation`, `reproducibility`,
and `artifact_requirements` contracts. Validation normalizes the request to a
paper-agnostic scientific family/mode/method and returns a stable
`validation_id`, `request_hash`, `canonical_contract_hash`,
`compiled_plan_hash`, readiness dimensions, and an expiry.
The preferred `start_analysis` request contains only that ID and hash. Before a
run is created, the wrapper verifies the HMAC-protected validation receipt and
checks that the exact request, input files, CLI executable/version, and compiled
interaction plan are unchanged. A changed or expired validation fails closed
and must be validated again. Strict full-request calls to `start_analysis`
remain accepted for backward compatibility.

For a development or experiment session already restricted to one analysis
family, `GEOCHEMISTRYPI_MCP_ANALYSIS_SCHEMA_TASK` limits only the advertised
`validate_analysis` schema to one of the six exact task names shown above. All
13 tool names and all six capabilities remain available. An unknown scope
prevents server construction. Requests created before regression support remain
classification requests when `task` is omitted in an unscoped session; new
clients should always send it explicitly.

Advertised input schemas omit only generated `title` annotations. In an
unscoped session, `validate_analysis` advertises a small task-routing envelope;
call `get_capabilities` with that task to receive its exact hash-bound schema,
including every scientific description, default, example, enum, constraint,
discriminator, closed-object rule, and semantic definition name. A task-scoped
session advertises that family's exact closed-object schema directly. The same
strict Pydantic models perform runtime validation in either mode.

### Scientific reproduction contract

Dataset references can pin the original file and describe a deterministic CLI
input view. Excel files with more than one worksheet require an explicit
`worksheet`; MCP never selects the active sheet silently. `header_row_index` is
zero-based, while `header_row_indices` can compose a multi-row scientific
header using explicit separator, empty-cell, and duplicate-name policies.
`header_whitespace_policy: strip` and `header_bom_policy: strip` are explicit,
hash-bound normalization choices for source workbooks whose published headers
contain padding or a leading Unicode byte-order mark. They are never applied by
default. When `selected_columns` is explicit, duplicate names in unrelated raw
columns do not block the prepared view; every selected, filtered, excluded, or
row-identity column must still resolve to exactly one normalized source column.
`selected_columns` and `excluded_columns` are mutually exclusive, and
column-value row identity can bind a separate source-mapping file by path and
SHA-256. Preparation `operations` record whether missing-value handling,
filtering, transformation, or feature selection occurred; they are provenance,
not new scientific calculations performed by MCP. The executable preparation
DSL supports `not_null`, equality/inequality, ordered comparisons, inclusive or
exclusive `between`, and membership predicates with typed operands. Rules are
applied before column projection, and the retained source-row sequence,
input/retained/dropped counts, prepared file, and contract are all hashed.

```json
{
  "source": "path",
  "path": "D:\\data\\paper-source.xlsx",
  "expected_sha256": "<original-file-sha256>",
  "preparation": {
    "worksheet": "Analysis Data",
    "header_row_index": 2,
    "excluded_columns": ["Comment"],
    "filters": [{"column": "PublishedClass", "operator": "not_null"}],
    "row_identity": {
      "strategy": "column_values",
      "columns": ["SampleID"],
      "source_mapping_path": "D:\\data\\source-row-map.json",
      "source_mapping_sha256": "<mapping-sha256>"
    },
    "operations": ["filtering", "transformation"]
  }
}
```

Validation distinguishes `source_file` from `prepared_input` and returns the
complete preparation record plus its hash. The cached prepared CSV is derived
only for table selection; the original source is never modified. Both hashes,
the ordered row-identity hash, and the source/prepared mapping are repeated in
the validation receipt and scientific run manifest.

The backward-compatible `reproducibility.environment` field can require the
complete observed environment identity, exact
Python/GeochemistryPi/MCP/platform/runtime values, and exact dependency
versions. New reproduction requests can instead provide one named
`environment_profile` with an exact Python version, exact package versions,
and supported runtime constraints. Its content hash and profile ID are bound
into the interaction-plan identity. MCP validates the selected isolated CLI
runtime; it does not silently install or switch environments. Validation
reports `environment_status` as `READY`, `MISMATCH`, or `UNSPECIFIED`; a
mismatch makes `execution_ready=false` and no run is created. The plan records
requested and effective model, preprocessing, and split/model/tuning seed
values. Manual random and conditionally random models receive the effective
seed through the CLI scientific sidecar, preserve zero, and attest the fitted
estimator's `random_state`; deterministic models report model seed as not
applicable. PCA `solver="auto"`, all-model execution, and AutoML report their
model/tuning seed stage as unbound instead of claiming the CLI's internal 42.
They may use `adapter_default` only when no fixed model/tuning seed was
requested, and cannot support a model/tuning reproducibility claim; an explicit
seed or fixed-seed policy blocks validation. Time Series uses its public
top-level seed and can freeze the full runtime with `expected_identity_sha256`.

Evaluation metrics can be bound to artifact requirement IDs. Artifact
requirements match workflow-aware scientific types and roles, safe paths or
patterns, media types, cardinality, and required JSON keys. Produced artifact
records include SHA-256, producer, scientific role, and every satisfied
requirement ID. Adapter mappings bind each scientific role to an original CLI
relative path; unavailable roles are explicit capabilities with a reason. For
example, the classification adapter maps metrics, raw confusion matrices,
predictions, scores, and supported feature importance outputs, but declares a
normalized confusion-matrix table unavailable because MCP does not recalculate
it. Missing or malformed required evidence yields an incomplete
contract rather than a successful reproduction claim.

Configuration-only benchmark profiles are supported by
`geochemistrypi_mcp.planning.profiles`. A bounded YAML profile contains
`benchmark`, `profile_state`, `workflow`, `dataset`, `environment` or
`environment_profile`, `reproducibility`, `parameters`, `expected_artifacts`,
and `acceptance_rules`. A ready profile compiles into the same strict public
analysis request and can attest the resulting plan family, mode, method, and
readiness. An incomplete profile retains literal `UNKNOWN` values only as a
non-executable template and compiles to a diagnostic plan with no public
command. Workflow stages form a validated generic DAG and remain blocked until
one adapter can execute the complete chain. Benchmark names are metadata only
and never participate in execution dispatch. See
`benchmark_profiles/README.md` for the format.

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

For several numeric outcomes from the same samples, replace `target_column`
with `target_columns`, for example `"target_columns": ["MeasuredValue",
"MeasuredValueB"]`. Send exactly one of these two fields. `target_column`
remains the backward-compatible single-target form; `target_columns` accepts
one or more targets. The CLI resolves target order from the source dataset and
`validate_analysis.target_columns` reports that order before execution.

Classification metric semantics are explicit and label-safe. `metric_average`
defaults to `auto` (`binary` for two final classes and `weighted` otherwise);
an explicit `binary` request requires `positive_label`. Semantic label identity
is typed, so numeric `1` and string `"1"` are never conflated. Binary aggregate
metrics bind that requested class through holdout and cross-validation; micro,
macro, and weighted aggregate metrics do not accept a user positive class. For
two-class data, precision-recall, threshold, and ROC outputs independently bind
a curve-positive class to the matching `estimator.classes_` probability column
instead of assuming column 1. Both aggregate and curve semantics are recorded
and independently checked by the scientific execution attestation for all 11
manual classification models.

Regression exposes all 15 single-model CLI families and the exact all-model
branch. Every target must be a finite numeric column. Holdout results retain the
legacy uniformly averaged metrics and add a named `Per Target` section;
prediction artifacts use `Predicted_<target>` columns. Cross-validation remains
uniformly averaged across targets. Because the public CLI feature selectors are
univariate, multi-target requests with feature selection are rejected instead
of silently dropping the scientist's choice. The wrapper validates the CLI's fixed 10-fold
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
validates numeric finite features, internal source-row lineage, model/data-size
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

A Time Series request selects `mode="subaerial_proportion"` (the backward-
compatible default), `mode="continuous"`, `mode="element_mean"`, or
`mode="reference_anomaly_series"`. Subaerial-proportion requests
can reproduce the complete selected-data and missing-value preparation
performed by the interactive workflow. For example,
`selected_columns` may contain the nine consecutive scientific fields selected
by CLI range `[6,14]`; `missing_values={"method":"drop_rows","columns":[]}`
means drop rows missing any selected field. `identifier_column` records the
sample-name field, and `feature_engineering` currently accepts only `none`.
The public noninteractive CLI applies these operations before the seeded
bootstrap and records input, retained, and dropped row counts in `Time Series
Parameters.json`. Continuous requests bind central/minimum/maximum age,
arbitrary numeric value, latitude/longitude, optional inclusive numeric
filter, relative two-sigma analytical uncertainty, bootstrap iterations,
seed, minimum samples per bin, and plotting controls to the public
`time-series --analysis-mode continuous` producer. The producer uses generic
spatiotemporal density weights and writes a machine-readable mean/uncertainty
table plus PNG/PDF figures; no dataset, paper, or element name changes its
algorithm. Element-mean requests express age/value/filter roles, bin
width, arithmetic-mean aggregation, standard-error uncertainty, and minimum
sample counts. The current public CLI has no matching element-mean command, so
validation returns `valid=true`, `execution_ready=false`, and an unavailable
adapter; starting that receipt fails before process creation. It is never
substituted with the subaerial-proportion workflow.

`reference_anomaly_series` renders externally supplied anomaly labels and
optional event records without fitting or changing a detector. Its CLI command
is probed before validation can report execution readiness. Decomposition also
offers `mode="embedding_label_overlay"` as a generic artifact-composition
producer: it joins an existing two-coordinate table and a label table by exact
one-to-one identifier sets, then writes the joined CSV, anomaly counts,
PNG/PDF figures, parameters, artifact index, and scientific manifest. It does
not rerun PCA or anomaly detection and uses the existing analysis tools rather
than adding another MCP tool.

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

The `validate_analysis` request schema contains scientific choices only. It
never accepts raw CLI answers, shell commands, environment variables, or output
directories. The preferred start request contains only the immutable validation
reference. Every tool-list entry publishes a small hash-addressed output
envelope. Resolve its full strict success/public-error JSON Schema through
`get_capabilities(output_contract_sha256=...)` when client-side validation is
needed; response fields and server-side validation are unchanged.
Public validation errors group repeated evidence and return bounded prefixes,
complete counts, truncation flags, and SHA-256 identities. Both structured
errors and their model-facing text have hard size budgets.

Scientific identifier columns such as sample or rock names retain their
original user meaning. Their values may be duplicated or missing when the
public CLI workflow does not require uniqueness; they are never promoted to ML
features or targets. MCP separately derives a deterministic, non-null internal
row identity from the input SHA-256 and the one-based effective source-row
position. Effective Excel/CSV row boundaries match the public CLI readers, so
all-empty worksheet tail rows are not misclassified as samples. For non-Time
Series runs, the wrapper verifies the complete ordered `Data Original` table
against the source before publishing the result. Identity collisions, schema or
row-count mismatches, changed/reordered rows, and indeterminate pairing remain
fail-closed. Numeric CSV-to-XLSX round trips use the explicit policy
`max(1e-14 * magnitude, 1e-15)` so spreadsheet serialization noise does not
invalidate an otherwise identical row; text, identifiers, dates, missingness,
row order, and numeric changes above that bound are still checked strictly.

`list_datasets` discovers the eight datasets shipped with the installed CLI and
supported `.csv`/`.xlsx` files directly inside `Desktop/geopi_input`. Discovery
is read-only: it does not create the Desktop directory, copy files, recurse into
subdirectories, or accept a resolved path outside that directory. Each entry
contains a stable ID, role, task, size, SHA-256, and any capability that blocks
analysis. Use the returned ID without copying its absolute installation path:

The public directory defaults to `detail: "compact"`, a deterministic page of
16 entries with `total_count`, `returned_count`, `next_offset`, and an exact
`view_sha256`; compact entries keep identity, routing, filename, size, hash,
shape, and blocker receipts but omit absolute installation paths. Request
`detail: "full"` only when every legacy path and field is needed. Full pages are
lossless and limited to 2,100,000 UTF-8 JSON bytes, while compact pages are
limited to 64 KiB. Reuse `if_view_sha256` only with the exact same source,
detail, offset, and limit; another projection cannot produce an unchanged
receipt.

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
but a request must provide exactly one representation for each input. Paths may
be absolute or relative to the MCP server's fixed startup working directory.
Relative paths must remain inside that directory after real-path resolution;
parent-directory and symbolic-link escapes are rejected before file content is
read. Accepted paths still receive the regular-file, format, size, hash,
row-lineage, and change-during-read checks. Dataset
inspection applies pandas-compatible `Unnamed: N` names to empty Excel headers.
Explicit datasets with duplicate or unsafe headers fail deterministically;
trusted bundled data can use pandas-compatible duplicate suffixes and returns a
visible `header_warnings` entry. GeochemistryPi deliberately supports `.csv`
and `.xlsx`; `.xls` is not advertised because the CLI reader has no reliable
legacy Excel dependency. The default `detail: "names"` response returns column
names, exact row and column counts, header warnings, and separately labelled
source-file and prepared-view SHA-256 identities without replaying the complete
preparation record or inferred-type summaries. Sample values are included only
when a positive `sample_rows` is explicitly requested; use `detail: "full"`
when inferred types and the complete bounded inspection record are also needed.

Call `get_capabilities` before planning a run. Its default compact response
retains all six task names, per-task model indexes, compatibility and resource
limits, unsupported combinations, capability-boundary identities, and a stable
SHA-256 of the complete capability snapshot. Supplying `task` narrows the
planning details without hiding any tool or task. The task-filtered view also
returns a hash-bound `validation_request_contract` generated from the same
strict Pydantic model used by `validate_analysis`. It includes the exact
task-level JSON Schema and machine-readable locations for dataset references,
model discriminators, and reproducibility seeds, so clients construct one
request instead of receiving the complete six-task union on every tool-list
replay. Accordingly, an unscoped `validate_analysis` tool advertises a small
task-routing envelope; the selected task's hash-bound contract is the complete
request contract, and the server continues to validate that strict model,
including unknown-field and cross-field rejection, before any process starts.
An explicitly task-scoped server still advertises that task's complete schema
directly. `detail: "full"` returns the
complete evidence-bearing inventory. Each response includes the global snapshot
hash plus a view hash computed from the fields in that exact projection. A
full-only evidence change therefore does not force an unchanged compact task
view to be replayed. Reuse
`if_capability_view_sha256` for compact or task-filtered unchanged receipts;
legacy `if_capabilities_sha256` is accepted only for the unfiltered full view,
so a hash from one projection cannot suppress delivery of another.

All 13 tools remain visible and callable. Their complete serialized success and
public-error response contracts are still generated from the same Pydantic
models and enforced at the server boundary. The tool listing carries each full
output contract's canonical SHA-256 and UTF-8 byte count instead of replaying
the same large response unions in every model continuation. Its resolver
metadata names the existing `get_capabilities` tool, the
`output_contract_sha256` argument, and the `output_contract_schema` response
field, so any MCP client can fetch and independently verify the exact contract
without a private package import. This changes only contract transport: no
response field, scientific option, validation rule, or tool is removed.

Both full and compact capability views expose the same machine-readable
`scientific_attestation` boundary, derived from the public model registries
rather than a separate documentation list. The public manual surface contains
36 task/method identities. Scientific-config v4 binds and requires a verified
execution attestation for 27 of them: all 11 classification methods, nine
regression methods, two clustering methods, all three decomposition methods,
and both anomaly-detection methods. The remaining nine are executable only
through the legacy adapter: regression `linear_regression`,
`polynomial_regression`, `k_nearest_neighbors`, `support_vector_machine`,
`bayesian_ridge`, and `ridge_regression`; and clustering `dbscan`,
`agglomerative`, and `mean_shift`. AutoML and all-model selection do not claim
a per-child v4 scientific-config sidecar. Time Series uses a separate
route-native contract and is not included in the 36/27/9 counts. Registry
invariants require the 27 and nine identity sets to be disjoint and to partition
all 36 public manual identities.

`model_selection.mode = "all"` executes the real CLI
aggregate branch and returns a parent summary plus ordered child results;
regression supports one or more finite numeric targets, including named
per-target metrics and application predictions, while explicitly rejecting
multi-target use of the CLI's univariate feature selectors; classification sample balancing is not offered
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

`validate_analysis` returns a compact start/stop receipt by default. It retains
the validation, request, canonical-contract and compiled-plan hashes; every
readiness dimension and blocking issue; resolved data identities, row counts,
column roles, seeds and model parameters; requested and effective evaluation,
split, CV, metric, typed-positive-label, preprocessing, application, and binding
decisions; exact artifact paths, roles and cardinalities; and environment and
experiment identities. Optional Time Series event data has its own path, size,
and SHA-256 identity. The immutable validation record retains the complete
preparations, mappings, environment record, and compiled interaction plan so
compact delivery never weakens execution checks.
The compact validation JSON has a 64 KiB hard limit. Large column-role lists,
artifact requirements, warnings, blockers, and nested decision collections use
typed prefixes with complete counts, truncation flags, and canonical SHA-256
identities; long individual diagnostics also retain their full-text hash. The
immutable full validation receipt remains byte-complete. The preregistered S06
contract therefore retains all selected columns, mode parameters, and native
output requirements while only the redundant observed-column inventory is
projected. `truncated_sections` identifies every affected receipt and
`start_relevant_content_complete` distinguishes that supplemental unselected
column inventory from truncated blockers, selected roles, scientific decisions,
or artifact requirements. A complete start-relevant compact receipt can be
started directly; full detail is read only when a start-relevant section is
incomplete or when the caller explicitly needs the supplemental inventory.
Every compact response still returns an exact `full_detail_request`. Calling
`validate_analysis` once with that
`validation_id`, `request_hash`, and `detail="full"` reads the HMAC-protected
stored record without inspecting data or compiling the same request again. The
full decision view returns every blocker, warning, and artifact requirement plus
hashes that match the compact receipts.

An explicit instruction to run or execute is already confirmation; otherwise
the client asks once before starting. It inspects a dataset only when exact
columns, shape, hashes, values, or types are still needed, rather than as a
ritual step before every validation. The client then starts the exact validated
request and normally calls `get_run_result` once. Its default `wait_seconds` is
300 and the call returns immediately when the run becomes terminal, so the
ordinary path does not need a preliminary status poll or a zero-second pending
read.
If that bounded wait ends while the run is still queued or running,
`get_run_result` returns a successful `response_detail="pending"` receipt with
the durable stage and retry guidance. It is continuing work, not a tool or
scientific failure; the client may make one later bounded result call without
starting an error-recovery flow.
It returns a concise
scientific summary with links or references to the original GeochemistryPi
outputs. The default `detail="compact"` and `artifact_view="canonical"` return
token-bounded artifact receipts and suppress only proven flat
`summary/<basename>` copies that have one unambiguous same-scope, same-name,
same-size, same-hash original. Nested, unique, or requirement-bound summary
outputs remain visible. `artifact_offset` and `artifact_limit` retrieve a
bounded page; compact delivery caps a page at 32 receipts even if a larger limit
is requested. `detail="full"` restores full artifact metadata and
`artifact_view="all"` explicitly includes mirrors. The complete wrapper-owned
artifact index is never rewritten by these views and is returned with its path
and SHA-256. Each artifact includes SHA-256, requirement binding when planned,
scientific type, and producer metadata. The first compact terminal page retains
the complete compact scientific core. A compact request with
`artifact_offset > 0` returns an additive
`response_detail="artifact_page"` receipt containing only immutable run,
result-record, and artifact-index identities plus view counts, page number,
offset, effective limit, next offset, compact artifact records, and the page
SHA-256. It does not replay dataset preparation, reported metrics, artifact
contract details, child summaries, or limitations. Explicit `detail="full"`
delivery remains unchanged.

The first successful terminal response also includes
`required_tabular_observations`, a read-only view over requirement-bound
canonical CSV, XLSX, JSON, and JSON-in-TXT outputs from that run's immutable
artifact index. Every observation carries the artifact identity, relative path,
requirement IDs, file SHA-256, format, worksheet when applicable, and the
observed output row count, column count, and column names. The wrapper verifies
the index and file identities before and after reading and never substitutes a
validation/input row count for an output row count. Complete native-order rows
are included only when the whole table fits the per-cell limit, the 512-cell
global budget, and the 16 KiB observation JSON budget; large prediction,
assignment, and coordinate tables return metadata only. Delivery is capped at
32 observations and 64 returned column names, with total counts, truncation
flags, and hashes binding complete identities. Proven summary mirrors are not
observed twice. A changed hash, unsafe path, or symlink remains an integrity
failure. A format that is safely too large or cannot be parsed within the
observation limits is omitted from this convenience view with bounded reason
counts and an omission hash; that additive omission does not redefine an
otherwise successful scientific/artifact contract.

Every completed run writes a wrapper-owned
`scientific-run-manifest.json` binding request, validation, run, plan,
training/application/event input identities, runtime, artifact hashes, and
missing required artifacts. Use
`get_run_status` only when progress detail is needed; it supports the same
bounded wait and should not be polled in a tight loop. These tool calls are
automatic and should not be turned into extra instructions for the user.

Compact delivery replaces the complete preparation provenance with its
canonical hash, source/prepared identities, row counts, projection/filter
hashes, and executed-operation summary. It also applies an 8 KiB JSON budget to
`reported_metrics`. Normal results remain unchanged; an oversized response sets
`reported_metrics_truncated=true` and reports the omitted top-level group count.
The complete compact result JSON has a 64 KiB hard limit. Artifact requirement
bindings, missing-requirement IDs, aggregate child summaries, and limitations
return bounded prefixes plus total counts, truncation flags, and SHA-256 values
of their complete sequences. Required tabular observations use their own 16 KiB
and 512-cell ceilings inside that response budget. Artifact pagination is
reduced dynamically if needed to stay within the limit.
The immutable result record and explicit `detail="full"` response retain the
complete preparation and CLI-reported metric structures. Missing required
artifacts produce `state="partial_failure"`, `contract_status="incomplete"`, and
the bounded missing requirement IDs; they are never presented as a complete
scientific result.

A successful terminal response also returns the SHA-256 of immutable
`result.json`. It is complete and should not be fetched a second time. A client
that must confirm the same terminal state passes that value as
`if_result_sha256`; matching successful, partial-failure, failed, and cancelled
receipts return a small `not_modified` response without replaying metrics,
artifacts, or diagnostics. Conditional identity checks cannot be combined with
full, all-artifact, or paginated delivery. Omitting the conditional hash remains
an explicit replay.

Successful full and compact terminal results also expose
`cli_started_at`, `cli_finished_at`, and
`cli_execution_duration_seconds`. These values are copied from the immutable
interaction trace for the actual GeochemistryPi CLI child process. The elapsed
value uses a monotonic clock; the timestamps and duration are fixed once and
remain identical across compact/full views and later reads. They never fall
back to the broader managed-run interval.

New successful `result.json` records are published once without replacement;
terminal `status.json` binds their exact path and SHA-256, and every later read
verifies that identity. This keeps conditional receipts anchored to the result
that actually reached terminal status.

Failed and cancelled runs publish a separate immutable `terminal-result.json`
receipt before their terminal status. It contains the bounded error,
process/exit status, the same route-native CLI child timing fields when a child
was created, and allowlisted wrapper-log identities only. A queued cancellation
with no CLI child reports all three timing fields as null. It explicitly
records that scientific
validity was not established, the artifact contract was not evaluated, and the
verified artifact count is zero; unvalidated workspace files are never exposed
as scientific artifacts.

`validate_analysis` never creates a run or starts the analysis CLI. Its
30-minute validation receipt is wrapper state, not a scientific result. A run
status uses the stable stages `queued`, `running_cli`, `indexing_outputs`, and a
final `completed`, `failed`, or `cancelled` stage. Large aggregate and AutoML
requests report their model count before execution.

For lifecycle work, first call `list_experiments`, then reuse the returned
stable ID with `existing_experiment_id` and the exact returned experiment name.
Name/ID mismatches fail before execution. The MLflow UI is never auto-started;
use `start_mlflow_ui`, open its `127.0.0.1` URL, inspect status, and use
`stop_mlflow_ui` when finished.

`list_experiments` and `get_experiment` use the same compact/full, offset/limit,
count, continuation, and projection-specific `view_sha256` contract. Compact
experiment pages retain ID, name, and lifecycle stage; compact run pages retain
run ID, name, status, and start/end times. Full detail preserves tracking roots,
artifact locations, tags, metrics, parameters, and artifact URIs. The direct
`get_experiment` lookup never expands an experiment listing first.

## Full parity gate

The versioned PR9I matrix contains 36 manual single-model scenarios, all 11
classification and 13 regression AutoML branches, five aggregate scenarios,
and the required inference, preprocessing, mapping, experiment, source, and
Time Series dimensions. Comparisons require unchanged input hashes, identical
recursive file inventories and table ordering, explicit `1e-9` absolute and
`1e-7` relative numeric tolerances, and structural checks for platform-dependent
images. Full AutoML/model execution is sharded behind the scheduled or manual
release-candidate workflow; the normal PR gate keeps representative real parity.
