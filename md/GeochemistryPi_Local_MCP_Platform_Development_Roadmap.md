# GeochemistryPi MCP CLI Wrapper Development Roadmap

> Status: Replacement roadmap
> Updated: 2026-08-03
> Core project baseline: GeochemistryPi 0.8.0
> Audience: maintainers, contributors, reviewers, and release engineers

## 1. Purpose

This roadmap defines how to expose the existing GeochemistryPi command-line
application through the Model Context Protocol (MCP) without creating a second
machine-learning implementation.

The product goal is simple:

1. A user installs and configures GeochemistryPi MCP once.
2. The user describes a geoscience task in natural language to any compatible
   MCP client.
3. The client converts that request into structured MCP arguments.
4. GeochemistryPi MCP drives the existing GeochemistryPi CLI.
5. The existing CLI performs the analysis and creates its original outputs.
6. MCP returns a concise result and references to those original outputs.

This document supersedes the previous roadmap that proposed independent
application services, model adapters, machine-learning workers, and separately
generated artifacts. That design duplicated validated CLI behavior and is no
longer an accepted implementation direction.

## 2. Product Contract

### 2.1 User experience

After setup, the expected user interaction is:

```text
帮我分析桌面上的 rocks.xlsx，看看哪些元素最能区分不同岩性。
```

The user may provide only this scientific goal. The client locates and inspects
the named file, proposes a safe analysis, and asks a short domain-language
question only when a choice cannot be inferred safely. Tool names, JSON fields,
model identifiers, and run-control steps remain invisible.

Natural language may still be precise. When the user explicitly names columns,
a model, a tuning approach, or parameter values, the client must preserve those
choices exactly. Defaults apply only to omitted choices. Unsupported or invalid
requests produce a plain-language explanation and a follow-up question, never a
silent replacement.

The user must not need to:

- open the interactive CLI manually;
- write Python or machine-learning code;
- choose MCP tool names;
- locate a Python interpreter;
- configure an Engine path or runs directory;
- select an internal worker implementation;
- copy data into a special trusted directory;
- understand the CLI menu numbering.

The MCP client may ask the user for scientific information that cannot be
inferred safely, such as the target column or whether an application dataset
should be used. It must not ask for internal implementation details.

### 2.2 CLI compatibility

The existing GeochemistryPi CLI is the only machine-learning implementation.

MCP development must not change:

- the `geochemistrypi` command;
- the `data-mining` command and its options;
- the interactive CLI menu flow;
- default model behavior;
- preprocessing behavior;
- metric calculations;
- plot generation;
- MLflow behavior;
- the existing `artifacts`, `metrics`, `parameters`, and `summary` output
  directories;
- the way existing CLI users launch or operate GeochemistryPi.

If a defect is discovered in the CLI, it must be handled in a separate
GeochemistryPi core change with explicit scientific review. MCP code must never
silently correct, replace, or compensate for a CLI result.

### 2.3 Token and privacy goals

The MCP integration exists to reduce unnecessary model-generated code and
context usage.

- Machine-learning execution stays local.
- Complete datasets are not returned to the language model.
- Large logs and tables are summarized locally.
- Binary models and images are returned as artifact references, not embedded
  in tool responses.
- The client receives only bounded dataset summaries, run state, scientific
  summaries, and artifact metadata.

## 3. Non-Goals

The first local release will not:

- reimplement classification, regression, clustering, decomposition, anomaly
  detection, feature engineering, metrics, or plots;
- introduce a new `ClassificationService` or equivalent training service;
- introduce model-specific legacy adapters;
- introduce a second ML pipeline;
- expose arbitrary Python, shell, module, or command execution;
- accept a raw list of CLI answers from the MCP client;
- require a database, remote queue, cloud service, authentication service, or
  object store;
- publish an incomplete package to PyPI or an MCP registry;
- change the existing Dash or FastAPI product;
- claim that behavioral equivalence alone proves scientific correctness.

Remote MCP, multi-user execution, cloud scheduling, billing, and Chemical
Modeling remain separate future tracks.

## 4. Verified Repository Baseline

The current CLI entry point is:

```text
geochemistrypi = geochemistrypi.cli:app
```

The `data-mining` command already accepts:

- a training-only data path;
- separate training and application data paths;
- desktop data discovery;
- built-in datasets.

For an explicit data path, the CLI uses the process working directory as its
working path. This allows MCP to isolate each run without changing the CLI: MCP
starts the command in a dedicated run workspace, and the CLI creates
`geopi_output` and `geopi_tracking` there.

The complete interactive workflow is implemented in
`geochemistrypi/data_mining/cli_pipeline.py`. It uses both Python `input()` and
Rich `Prompt`/`Confirm` interactions. The same workflow creates the experiment,
selects data, preprocesses features, selects a task and model, trains the
model, evaluates it, performs optional inference, and writes the original
outputs.

The repository also contains a FastAPI classification route. It calls selected
classification internals directly and does not execute the complete CLI
workflow. It is therefore not the foundation for CLI-equivalent MCP execution.

GeochemistryPi currently targets Python 3.9 and Pydantic 1. The current official
MCP Python SDK targets Python 3.10 or newer. The release design must therefore
keep the MCP process and CLI process in compatible environments while hiding
that detail from users.

## 5. Architecture

### 5.1 Runtime flow

```text
Natural-language request
        |
        v
Any MCP-compatible client
        |
        | stdio MCP
        v
geochemistrypi-mcp (lightweight MCP environment)
        |
        | validated semantic request
        v
Interaction-plan compiler
        |
        | expected prompts and controlled responses
        v
CLI process driver
        |
        | subprocess, isolated working directory
        v
Existing `geochemistrypi data-mining` command
        |
        v
Original GeochemistryPi outputs and MLflow tracking
```

The existing CLI process is the execution worker. No second machine-learning
worker is required.

### 5.2 Process environments

The implementation uses two internal process environments only when required
by dependency compatibility:

```text
MCP environment
  - Python 3.10+
  - official MCP SDK
  - request validation
  - run control
  - bounded dataset inspection

CLI environment
  - Python 3.9
  - GeochemistryPi 0.8.x
  - existing scientific dependencies
  - existing CLI and output behavior
```

This separation must be invisible after setup. Users interact with one MCP
server command and never provide interpreter paths.

### 5.3 Dependency rules

The following dependency rules are mandatory:

```text
MCP server -> CLI driver -> installed GeochemistryPi command
```

The following dependencies are forbidden:

```text
MCP server -> scikit-learn / XGBoost / FLAML / MLflow training APIs
MCP server -> GeochemistryPi model classes
MCP server -> FastAPI classification route
CLI production code -> MCP package
```

Dataset inspection may use lightweight readers, but it must not perform learned
preprocessing or model execution.

## 6. Target Repository Structure

Only one new distributable package is used. The source modules remain in one
package because `server`, `setup`, `doctor`, and `release` are stable installed
entry points, while `tools`, `schemas`, and `interaction_plan` are shared by
multiple test and runtime paths. The names and dependency direction provide the
internal layers without adding forwarding modules or fragile import aliases:

```text
packages/
└── geochemistrypi-mcp/
    ├── pyproject.toml
    ├── README.md
    └── src/
        └── geochemistrypi_mcp/
            ├── __init__.py
            ├── __main__.py
            │
            │   # MCP protocol boundary
            ├── server.py
            ├── tools.py
            ├── schemas.py
            │
            │   # CLI execution boundary
            ├── interaction_plan.py
            ├── cli_driver.py
            ├── runs.py
            ├── artifacts.py
            │
            │   # Scientific capability contracts and manifest
            ├── capability_manifest.py
            ├── cli_capability_manifest_v1.json
            ├── classification_contract.py
            ├── regression_contract.py
            ├── clustering_contract.py
            ├── decomposition_contract.py
            ├── anomaly_detection_contract.py
            │
            │   # Bounded data and experiment metadata
            ├── dataset_catalog.py
            ├── dataset_headers.py
            ├── dataset_inspector.py
            ├── experiments.py
            ├── tracking_ui.py
            │
            │   # Installation and release operations
            ├── constants.py
            ├── settings.py
            ├── client_config.py
            ├── setup.py
            ├── doctor.py
            └── release.py
```

Tests will be grouped by responsibility:

```text
tests/
├── cli_contract/
│   ├── fixtures/
│   └── test_classification_cli_contract.py
└── mcp_wrapper/
    ├── interaction/
    ├── parity/
    ├── protocol/
    └── installation/
```

Developer records and current handoff documents follow the repository's
existing documentation boundary:

```text
docs/source/For Developer/MCP/   # chronological PR implementation records
md/                              # roadmap, parity plan, and release handoff
```

The roadmap does not introduce:

- `geochemistrypi/application/`;
- `geochemistrypi/legacy_adapters/`;
- `geochemistrypi/worker/`;
- a separate contracts distribution;
- a separate runtime distribution.

Small request, status, and result schemas stay inside the MCP package until a
real independent consumer requires a separate package.

New subpackages are not introduced merely to shorten the root file listing.
They require a stable independent boundary and must preserve the installed
console entry points and the one-way `MCP -> CLI subprocess` dependency rule.

## 7. Semantic Request Design

### 7.1 Client-facing request

The MCP client sends scientific intent, not raw CLI menu answers.

An analysis request will contain fields such as:

```text
task
training_dataset_path
application_dataset_path
identifier_column
feature_columns
target_column
model_preference
use_automl
experiment_name
run_name
```

Task-specific fields use a discriminated schema. Unsupported fields are
rejected. Defaults must match documented CLI defaults or an explicitly tested
MCP policy; they must never be guessed from an unrelated model.

### 7.2 Interaction plan

The interaction-plan compiler converts semantic values into an ordered,
versioned plan for the CLI.

Each expected interaction includes:

- a stable prompt identifier;
- an expected prompt pattern;
- the response derived from validated scientific input;
- the reason for the response;
- whether the response may contain sensitive data;
- an occurrence index when a prompt repeats.

The client cannot submit raw prompt identifiers, raw answer sequences, shell
arguments, environment variables, or executable code.

### 7.3 Fail-closed behavior

The driver must stop safely when:

- an unexpected prompt appears;
- an expected prompt does not appear;
- a response is unused;
- the installed CLI version is unsupported;
- a column name cannot be mapped unambiguously;
- the CLI exits before completion;
- output appears outside the managed workspace;
- the input file changes during execution.

It must never continue with a shifted or guessed answer sequence.

## 8. CLI Interaction Strategy

### 8.1 Feasibility gate

Before implementing MCP tools, PR1 must prove a reliable cross-platform way to
drive the existing interactive CLI.

The preferred order is:

1. Launch the public CLI command with a subprocess and controlled stdin.
2. Synchronize responses with observed prompts instead of blindly writing all
   answers.
3. Normalize ANSI control sequences only for prompt matching and logs; do not
   alter scientific output files.
4. If Windows pipe behavior prevents reliable synchronization, use a minimal
   engine-side input bridge loaded only into the MCP-started subprocess.

An engine-side input bridge may intercept Python `input()` and Rich
`Prompt`/`Confirm` calls. It may not import, replace, or patch model,
preprocessing, metric, plotting, output, or MLflow functions.

No later PR may proceed until the chosen strategy passes Windows and Linux
integration tests. macOS validation is required before public release.

### 8.2 Interaction trace

Every MCP-started run records a wrapper-owned interaction trace containing:

- CLI and MCP versions;
- normalized prompt identifiers;
- non-sensitive responses;
- timestamps;
- subprocess exit code;
- whether every expected interaction was consumed.

The trace is metadata. It does not replace the CLI log and must not contain
credentials or complete dataset values.

## 9. Run Lifecycle

Long-running analyses must not block the MCP protocol loop.

The MCP package owns a small local run lifecycle:

```text
queued -> running -> succeeded
                  -> failed
                  -> cancelled
```

The existing CLI subprocess performs the computation. The MCP package only
owns process state, timestamps, cancellation, logs, and artifact discovery.

Each run uses this layout:

```text
runs/<run-id>/
├── wrapper/
│   ├── request.json
│   ├── status.json
│   ├── interaction-trace.json
│   ├── stdout.log
│   ├── stderr.log
│   └── artifact-index.json
└── workspace/
    ├── geopi_output/
    │   └── <experiment>/<run>/
    │       ├── artifacts/
    │       ├── metrics/
    │       ├── parameters/
    │       └── summary/
    └── geopi_tracking/
```

Rules:

- wrapper metadata and CLI outputs remain separate;
- MCP never regenerates or renames CLI artifacts;
- status files use atomic replacement;
- only one process owns a run state transition;
- cancellation targets only the recorded CLI process tree;
- process IDs are validated before termination;
- stale `running` states are repaired after an unclean shutdown;
- the default local concurrency is one until resource tests justify more.

## 10. Data and File Safety

### 10.1 Input paths

The local server may read any existing file that the user explicitly identifies
and that the current operating-system account can read. A global trusted-data
root is not required.

Before execution, MCP must:

- require an absolute path;
- resolve links and record the resolved path;
- verify that the path is a supported regular file;
- reject directories, devices, sockets, and broken links;
- enforce a configurable file-size limit;
- compute a SHA-256 hash;
- open the file read-only during inspection.

After execution, MCP recomputes the hash and fails the integrity check if the
input changed.

### 10.2 Output paths

All MCP-started outputs are written inside the managed runs root by launching
the CLI with the run workspace as its current directory.

The client cannot choose an arbitrary output path. The MCP package must not
write next to the source dataset.

### 10.3 Artifact access

Tool responses use run-scoped artifact identifiers. They do not accept an
arbitrary read path.

- Text and JSON artifacts are read with byte and line limits.
- CSV/XLSX results are summarized or paginated.
- Images and models are returned as metadata and local artifact references.
- Joblib, pickle, and model files are never automatically loaded by the MCP
  process.
- Logs are bounded and sanitized before being returned.

## 11. MCP Interface

### 11.1 Initial tools

The initial public tool set is intentionally small:

```text
get_capabilities
inspect_dataset
start_analysis
get_run_status
get_run_result
cancel_run
```

Later, if needed:

```text
list_artifacts
read_artifact_text
list_runs
```

The user is not expected to select tools. Server instructions and tool
descriptions must allow the MCP client to perform discovery, inspection,
validation, execution, polling, and result retrieval automatically.

### 11.2 Response design

Responses are concise and structured. A successful result contains:

- run ID and final state;
- task and model reported by the CLI;
- a bounded scientific summary parsed from existing CLI output files;
- original output directory location;
- artifact count and categorized artifact references;
- input hash verification;
- CLI exit code and version;
- warnings and limitations.

MCP may parse existing output values. It may not recalculate metrics or create
replacement reports.

### 11.3 Protocol rules

- Use the official MCP SDK.
- Use stdio for the first release.
- Reserve stdout exclusively for MCP protocol messages.
- Send server diagnostics to stderr or protocol logging.
- Validate all tool arguments with strict schemas.
- Reject unknown fields.
- Never return an unrestricted traceback to the client.
- Bound every text response.
- Keep tools client-neutral; client names belong only in setup adapters.

## 12. Installation and Client Setup

### 12.1 Development installation

Until the feature set is complete, installation is performed from the local
repository or an approved GitHub revision. PyPI and MCP registry publication
are explicitly deferred.

The development setup must:

1. create or locate the lightweight MCP environment;
2. create a private, version-pinned GeochemistryPi CLI environment or locate an
   explicitly approved development environment;
3. verify both versions;
4. run a real CLI smoke test;
5. run an MCP protocol smoke test;
6. save local settings using platform-native application directories;
7. register the MCP command with a selected compatible client or generate a
   standard configuration fallback.

### 12.2 Public target experience

The future public experience is one setup action followed by natural-language
use. The final client entry contains one stable command and no Engine, data,
runs, or model arguments.

Client support is adapter-based:

- standard `mcpServers` JSON fallback;
- Codex-compatible configuration;
- Claude Desktop and Claude Code configuration;
- Cursor configuration;
- VS Code configuration;
- additional compatible clients without server-side business changes.

Setup must be idempotent, preserve unrelated client settings, validate the
resulting configuration, create one recoverable backup, and provide clear
repair and uninstall behavior.

## 13. Equivalence and Quality Strategy

### 13.1 Two different correctness questions

The test suite keeps two questions separate:

1. **Wrapper equivalence:** Did MCP execute the same CLI behavior and preserve
   the same outputs?
2. **Scientific correctness:** Is the underlying CLI behavior scientifically
   valid?

MCP parity tests answer the first question. Scientific tests for the core
project answer the second. A scientific defect must be fixed in the core CLI,
after which the wrapper baseline is intentionally updated.

### 13.2 Direct-versus-wrapper parity

For each supported workflow, tests run:

```text
same CLI version
+ same dataset
+ same semantic choices
+ same random settings
+ same working-directory structure

through:
  A. direct CLI execution
  B. MCP CLI driver execution
```

The comparison includes:

- exit state;
- normalized output tree;
- required artifact names;
- selected feature and target names;
- split membership when recorded by the CLI;
- metric values with explicit tolerances;
- prediction values and row identifiers;
- model behavior on a fixed inference fixture;
- image format, dimensions, and non-empty content;
- parameters and summary files;
- MLflow run metadata when deterministic;
- input-file hash before and after execution.

Binary hashes are used only for deterministic files. Platform or library
metadata that is known to vary is normalized explicitly and documented.

### 13.3 Architecture guards

CI must fail if an MCP change:

- modifies CLI production files without a separately approved core change;
- imports prohibited ML libraries in the MCP package;
- imports GeochemistryPi model or preprocessing classes in the MCP package;
- creates metrics, plots, or models outside the CLI output tree;
- exposes raw commands, raw answer scripts, or arbitrary output paths;
- packages repository tests in a production wheel;
- permits protocol output to be mixed with CLI stdout;
- claims support without a direct-versus-wrapper parity fixture.

### 13.4 Test layers

```text
Unit
  - schemas
  - path validation
  - prompt matching
  - interaction-plan compilation
  - state transitions

Characterization
  - direct CLI prompt sequences
  - direct CLI output manifests

Parity
  - direct CLI versus driver
  - direct CLI versus MCP end to end

Protocol
  - tool discovery
  - strict request validation
  - stdout isolation
  - cancellation and failure responses

Installation
  - clean Windows environment
  - clean Linux environment
  - clean macOS environment before release
  - repeated setup, repair, and uninstall
```

Every Python or CI PR must finish with:

- relevant targeted tests;
- the full supported Python 3.9 core suite with database configuration absent;
- the MCP environment suite;
- `pre-commit run --all-files` with no file modifications;
- wheel build and archive inspection when packaging changes;
- installed-wheel tests when packaging changes;
- final `git diff --check` and status review.

## 14. Revised Pull Request Sequence

### PR0: CLI Contract Baseline

**Product purpose:** Establish a trustworthy definition of the CLI behavior
that MCP must preserve.

Deliverables:

- one small deterministic classification fixture;
- a complete direct CLI classification run in an isolated temporary workspace;
- versioned prompt-sequence characterization;
- normalized output manifest;
- metric and prediction Golden values with tolerances;
- input integrity verification;
- a developer-facing baseline document;
- CI checks that MCP work has not changed CLI production files.

Acceptance:

- the direct run completes from the public CLI entry point;
- all four original output directories exist;
- expected model, data, image, metric, parameter, and summary artifacts exist;
- the input hash is unchanged;
- the baseline is reproducible on the supported Python 3.9 environment;
- no MCP server or duplicate ML implementation exists yet.

### PR1: Cross-Platform CLI Interaction Driver

**Product purpose:** Allow software to operate the existing interactive CLI
reliably without changing how human CLI users work.

Deliverables:

- semantic classification request schema used only by the driver;
- versioned interaction-plan compiler;
- prompt-synchronized subprocess driver;
- strict unexpected-prompt and unused-response failures;
- isolated run workspace;
- stdout, stderr, and interaction trace capture;
- Windows and Linux integration tests;
- direct-versus-driver parity test.

Acceptance:

- the driver starts the public CLI command;
- it does not import ML implementations;
- it fails closed when the prompt contract changes;
- direct and driven runs satisfy the PR0 parity contract;
- CLI production files remain unchanged.

### PR2: Minimal MCP Package and Local Run Control

**Product purpose:** Make the validated CLI driver callable by any compatible
MCP client.

Deliverables:

- `geochemistrypi-mcp` package using the official SDK;
- stdio server;
- strict tool schemas;
- bounded dataset inspection;
- local run state and atomic metadata files;
- classification-capable `start_analysis`;
- status, result, and cancellation tools;
- protocol and stdout-isolation tests;
- MCP-to-CLI parity test.

Acceptance:

- an MCP client starts a classification run and retrieves its result;
- long execution does not block protocol handling;
- cancellation terminates only the recorded CLI process tree;
- CLI output is never written to MCP stdout;
- results reference original CLI outputs.

### PR3: Complete Classification Coverage

**Product purpose:** Deliver one fully polished reference workflow before
expanding to other task families.

Deliverables:

- classification capability matrix;
- supported feature, target, label, split, balancing, scaling, selection,
  model, tuning, and inference interactions;
- application-data support;
- original artifact discovery and bounded summaries;
- parity fixtures covering every supported classification model family and
  every materially different prompt branch;
- explicit unsupported behavior with user-actionable errors.

Acceptance:

- no supported CLI classification branch is silently skipped;
- unsupported branches are reported before execution;
- direct CLI and MCP outputs meet the parity contract;
- no metric, plot, prediction, or model is recreated by MCP.

### PR4: Setup, Doctor, and Client-Neutral Registration

**Product purpose:** Hide environment and configuration complexity from users.

Deliverables:

- private environment preparation;
- version handshake;
- zero-argument server startup after setup;
- real doctor checks for both processes;
- standard MCP JSON fallback;
- client registration adapters;
- atomic configuration updates and recoverable backups;
- repeated setup, repair, and uninstall tests;
- local GitHub installation documentation.

Acceptance:

- a clean machine reaches a working classification MCP without manually
  selecting Python paths;
- client configuration contains one stable server command;
- setup preserves unrelated client configuration;
- no PyPI or registry release occurs yet.

### PR5: Regression Coverage

**Product purpose:** Expose the existing CLI regression workflow through the
same validated wrapper.

Deliverables and acceptance follow the classification pattern:

- prompt and capability matrix;
- direct CLI characterization;
- interaction-plan support;
- training and application-data parity;
- model and plot coverage;
- explicit unsupported branches.

**Local implementation checkpoint (2026-08-02):** Implemented in the current
uncommitted worktree. The wrapper now exposes all 15 single-model regression
families, strict task-discriminated requests, the 13 CLI-supported regression
AutoML branches, numeric-target and fixed 10-fold validation, XGBoost-only
unprocessed-missing-value handling, conditional regression plot-dimension
prompts, application-data inference, original-output indexing, and direct
CLI/stdio MCP parity coverage. `all_models`, multiple regression targets,
previous-experiment attachment, and AutoML for Linear/Polynomial Regression are
explicitly rejected. This is a local development checkpoint, not a release or
remote-CI claim. Local verification completed with 49 core/CLI-contract tests,
125 MCP tests including 3 real CLI/MCP parity scenarios, all pre-commit hooks,
and a clean-wheel install that passed 122 non-parity MCP tests. PR6 remains the
next stage at that historical checkpoint; its later result is recorded below.

### PR6: Clustering Coverage

**Product purpose:** Expose existing unsupervised clustering without inventing
supervised assumptions.

Required coverage includes task-specific interactions, cluster outputs,
visualizations, model behavior, and parity fixtures. Classification or
regression defaults must not be reused silently.

**Local implementation checkpoint (2026-08-02):** Implemented in the current
uncommitted worktree. The wrapper now exposes the five models in the public
clustering menu through a strict target-free request and exact task-specific
interactions. It validates numeric features, identifier uniqueness,
missing-value resolution, fixed k=2 through k=10 silhouette prerequisites,
conditional model parameters, plot dimensions, and precomputed affinity shape
before process startup. A pre-existing core failure in transform-pipeline
construction was repaired by accepting the unsupervised `y_train=None` path,
with focused and real-CLI regression evidence. KMeans direct-CLI versus stdio
MCP parity compares the complete output inventory, scores, labels, plots, input
hashes, and artifact count. `all_models`, application inference, targets,
supervised feature selection and splitting, AutoML, unresolved missing values,
previous-experiment attachment, and internal-only OPTICS are explicitly not
exposed. Local verification completed with 50 core/CLI-contract tests, 142 MCP
tests including 4 real parity scenarios, all pre-commit hooks, both production
wheels, and a clean MCP-wheel install that passed 138 non-parity tests. This is
a local development checkpoint, not a release or remote-CI claim. PR7 remains
the next stage at that historical checkpoint; its later result is recorded
below.

### PR7: Decomposition Coverage

**Product purpose:** Expose the existing decomposition and embedding workflows
while preserving original CLI plots and transformed-data outputs.

Required coverage includes component selection, method-specific parameters,
transformed data, visual outputs, and parity fixtures.

**Local implementation checkpoint (2026-08-02):** Implemented in the current
uncommitted worktree. The wrapper now exposes PCA, T-SNE, and MDS from the
public dimensional-reduction menu through a strict target-free request. It
validates finite numeric data, identifiers, missing-value resolution, final
feature count, PCA component and ARPACK limits, and T-SNE perplexity before
process startup. Exact method-specific parameters, preprocessing, conditional
PCA bi/tri-plot component selection, model completion, transformed data, and
transform-pipeline output are compiled into guarded CLI interactions. Real
characterization completed for all three models. PCA direct-CLI versus stdio
MCP parity compares the complete file inventory, transformed data, loading
table, hyperparameters, plots, hashes, result semantics, and artifact count.
Two pre-existing PCA failures were repaired by retaining the generated loading
table and selecting matching score/loading columns for plots, with focused and
real-CLI regression evidence. Aggregate models, targets, supervised feature
selection and splitting, AutoML, application inference, unresolved missing
values, and previous-experiment attachment are explicitly not exposed. Local
verification completed with 52 core/CLI-contract tests, 153 MCP tests including
5 real parity scenarios, all pre-commit hooks, both production wheels, and a
clean MCP-wheel install that passed 148 non-parity tests. This is a local
development checkpoint, not a release or remote-CI claim. PR8 was the next
stage at that historical checkpoint; its later result is recorded below.

### PR8: Anomaly Detection and Remaining Inference Paths

**Product purpose:** Complete the main CLI task families and application-data
behavior.

This PR may be divided into independently reviewable PR8A and PR8B if the
capability matrix is large. AutoML support is included only after its exact CLI
interaction and reproducibility limits are characterized.

**Local implementation checkpoint (2026-08-03):** Implemented in the current
uncommitted worktree. PR8A exposes Isolation Forest and Local Outlier Factor
through a strict target-free request, exact preprocessing and model prompts,
bounded parameter validation, original-output indexing, and real Isolation
Forest direct-CLI-versus-stdio-MCP parity. Characterization repaired three
connected pre-existing core failures: disabled bootstrap now preserves
Isolation Forest's `max_samples="auto"` default, downstream diagrams receive
the indexed `-1`/`1` prediction series, and density grouping uses the detector's
actual inlier/outlier polarity. Direct public-CLI characterization also
completed successfully for Local Outlier Factor. PR8B audited the remaining
application-data paths and added the missing real stdio comparison for
classification feature-engineering replay and prediction; regression's real
application parity remains in place. Unsupervised application inference,
aggregate models, targets, supervised splitting/selection, anomaly AutoML,
unresolved missing values, and previous-experiment attachment are not exposed.
The wrapper now covers all 36 public single-model menu families across the five
main task families. Local verification completed with 55 core/CLI-contract
tests and 163 MCP tests including 6 real parity scenarios. The final pre-commit
run made no changes; the 24-entry MCP wheel and 179-entry core wheel contained
no repository tests; the clean MCP-wheel environment passed 157 non-parity
tests; and the installed core wheel passed Python 3.9 anomaly smoke checks.
This is a local development checkpoint, not a release or remote-CI claim. PR9
release hardening remains the next stage.

### PR9: Release Hardening

**Product purpose:** Turn the complete local integration into a dependable
public product.

Deliverables:

- full capability and parity matrix;
- clean Windows, Linux, and macOS installation tests;
- real-client acceptance tests;
- resource limits and failure recovery;
- versioned compatibility policy;
- signed release manifest and hashes;
- upgrade and uninstall behavior;
- final user and developer documentation;
- release decision for PyPI and MCP registry publication.

**Local foundation checkpoint (2026-08-03):** PR9 is in progress in the
current uncommitted worktree. The wrapper now publishes a versioned
compatibility policy through `get_capabilities`, identifies the channel as
development, and reports the exact MCP Python, MCP SDK, CLI Python, CLI,
interaction-plan, artifact-index, target-OS, and remaining-release-gate
contract. Core package metadata now enforces the actual Python 3.9-only CLI
boundary. Setup persists the compatibility policy in its install manifest;
legacy manifests trigger a runtime refresh; and doctor rejects stale policy
metadata. Resource hardening adds an atomic eight-run active/queued limit and
a configurable 900-second total CLI timeout, both exposed in capabilities and
kept out of analysis arguments. The wrapper CI matrix now includes the explicit
Intel `macos-15-intel` runner in addition to Ubuntu and Windows, avoiding the
arm64 `macos-latest` label for the Python 3.9-only legacy scientific stack, and
exercises a real setup/install/repair/uninstall lifecycle with pinned uv 0.11.7
on every matrix system. Real remote
macOS/Linux results, real-client natural-language acceptance, signed release
hashes, published-bundle upgrade testing, and the explicit
PyPI/MCP-registry decision remain open. This is not a release-complete or
remote-CI claim.

Local PR9A verification completed with 55 core/CLI-contract tests and 172 MCP
tests including six real parity scenarios. A wheel-bootstrapped temporary
Windows installation passed repeated install, forced repair, doctor 7/7, and
uninstall while preserving managed run evidence. The 179-entry core wheel and
24-entry MCP wheel contained no repository tests; standard pip rejected the
core wheel on Python 3.11; Python 3.9 installed-core smoke checks passed; and a
clean MCP-wheel environment passed 166 non-parity tests. Real remote CI remains
unverified without an authorized push.

## 15. Capability Expansion Rules

Every new task or model branch must follow this order:

```text
1. Run and document the direct CLI branch.
2. Record prompts and original output artifacts.
3. Add semantic request fields only when required.
4. Add interaction-plan translation.
5. Run the real CLI through the driver.
6. Compare direct and wrapped results.
7. Add capability discovery and user-facing errors.
8. Mark the branch supported only after parity passes.
```

Do not create schemas for unimplemented future tasks. Do not use a universal
parameter dictionary to bypass task-specific validation.

## 16. Versioning and Compatibility

The MCP package has its own semantic version. Each MCP release declares:

- supported GeochemistryPi CLI versions;
- supported MCP protocol/SDK version;
- supported operating systems;
- supported task and model branches;
- prompt-contract version;
- artifact parser version.

Startup refuses an unsupported CLI version before running an analysis. A prompt
contract change requires new characterization and parity evidence.

Release dependencies use tested bounds and a lock or constraints file for
development and bundles. The public setup verifies downloaded wheels and the
release manifest before installation.

## 17. Definition of Done

The local MCP product is complete only when:

1. Existing CLI users observe no command or workflow change.
2. MCP runs the installed GeochemistryPi CLI rather than a second ML pipeline.
3. Classification, regression, clustering, decomposition, and anomaly
   detection have explicit capability and parity evidence.
4. Training and application-data behavior preserve original CLI outputs.
5. The four original output directories, CLI logs, and MLflow records remain
   available.
6. Every supported branch passes direct-versus-MCP parity tests.
7. MCP contains no model training, metric calculation, or plot generation.
8. User datasets remain unchanged and are never sent in full to the language
   model.
9. Users can reference data at an explicit local path without configuring a
   trusted-data root.
10. Outputs stay inside managed run workspaces.
11. Long tasks support status, cancellation, and failure recovery.
12. Setup hides internal environments and produces a client-neutral server
    command.
13. Windows, Linux, and macOS clean-install tests pass.
14. At least one real client from each supported configuration family passes a
    natural-language end-to-end test.
15. Installation, repair, upgrade, and uninstall preserve user data and
    unrelated client settings.
16. Release documentation distinguishes supported, unsupported, and deferred
    capabilities accurately.

## 18. Risks and Mitigations

### Interactive prompt drift

Risk: a CLI release changes prompt text or order.

Mitigation: versioned prompt contracts, strict prompt matching, fail-closed
execution, and direct CLI characterization for every supported version.

### Cross-platform terminal behavior

Risk: Windows, Linux, and macOS handle pipes, encoding, ANSI output, or process
trees differently.

Mitigation: prove the interaction transport in PR1, keep a minimal input bridge
fallback, normalize only protocol/log text, and test real operating systems.

### Silent behavioral divergence

Risk: MCP begins reproducing or correcting CLI results independently.

Mitigation: prohibited-import checks, CLI source protection, artifact-origin
checks, and direct-versus-wrapper parity gates.

### Excessive tool responses

Risk: large outputs consume unnecessary model context.

Mitigation: bounded local summaries, pagination, artifact references, and no
binary or full-dataset responses.

### Dependency incompatibility

Risk: the current GeochemistryPi Python 3.9 environment conflicts with the MCP
SDK environment.

Mitigation: two private processes with a version handshake, one user-facing
setup, and tested release bundles.

### Accidental input or client-configuration modification

Risk: local automation overwrites user files.

Mitigation: input hashing, managed output roots, atomic configuration updates,
recoverable backups, and explicit repair/uninstall tests.

## 19. Immediate Next Actions

The next implementation work is limited to PR0.

1. Create a clean feature branch from the latest `origin/main`.
2. Keep previous recovery branches and stash entries only as local recovery
   material; do not cherry-pick the old MCP implementation.
3. Inventory the exact classification prompt sequence for one small fixture.
4. Run the public CLI entry point in an isolated temporary workspace.
5. Record the original output tree and stable scientific values.
6. Add input-integrity and CLI-source-protection tests.
7. Run the full Python 3.9 baseline and pre-commit checks.
8. Review PR0 before creating the MCP package.

No `ClassificationService`, model adapter, runtime package, worker package, or
MCP server should be created during PR0.

## 20. Technical References

- Model Context Protocol documentation: <https://modelcontextprotocol.io/>
- Official MCP Python SDK: <https://github.com/modelcontextprotocol/python-sdk>
- GeochemistryPi project: <https://github.com/ZJUEarthData/geochemistrypi>
