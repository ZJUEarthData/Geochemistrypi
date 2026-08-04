# GeochemistryPi Complete CLI-to-MCP Parity Plan

## 1. Document status

This document defines the implementation and release plan for turning the
current GeochemistryPi MCP wrapper into a complete, convenient, and verifiable
encapsulation of the public GeochemistryPi command-line product.

It is a forward-looking plan, not a claim that the listed target capabilities
already exist.

Current local baseline:

- GeochemistryPi CLI version: `0.8.0`;
- GeochemistryPi MCP version: `0.2.0`;
- current release channel: `development`;
- current branch: `feat/geochemistrypi-mcp-wrapper-v2`;
- current work is local and uncommitted;
- the MCP already covers 36 public single-model families across
  classification, regression, clustering, decomposition, and anomaly
  detection;
- the latest recorded local checkpoint contains 55 core/CLI-contract tests
  and 172 MCP tests, including six real direct-CLI-versus-MCP scenarios;
- public-release readiness remains false.

Recommended complete-product target:

- GeochemistryPi CLI: `0.9.0`;
- GeochemistryPi MCP: `0.3.0`;
- stable CLI automation contract: schema 1;
- capability manifest: schema 1;
- all public CLI capabilities either implemented and parity-tested or removed
  from the public CLI contract with an explicit compatibility decision.

## 2. Product objective

The finished product must let a user operate the complete public
GeochemistryPi CLI through natural language without learning menu numbers,
prompt sequences, Python environment details, executable paths, or output
directory conventions.

The MCP package must remain an orchestration and safety layer. The installed
GeochemistryPi CLI must remain the sole owner of:

- scientific preprocessing;
- model training and prediction;
- AutoML;
- evaluation metrics;
- model inference;
- maps and scientific plots;
- time-series calculations;
- model and pipeline serialization;
- MLflow records;
- original result files.

The MCP package must not import or recreate these scientific implementations.

## 3. Definition of complete CLI coverage

### 3.1 Included public capabilities

The complete MCP product must cover:

1. CLI discovery and version compatibility.
2. Explicit training-data paths.
3. Separate training and application-data paths.
4. Desktop `geopi_input` data discovery.
5. Built-in GeochemistryPi datasets.
6. CSV, XLSX, and any deliberately retained public XLS format support.
7. Dataset inspection and column-role selection.
8. World-map projection, including repeated element selection.
9. Missing-value rejection, retention where supported, row dropping, and
   imputation.
10. Feature engineering.
11. Feature scaling.
12. Supervised feature selection.
13. Classification label customization and multiclass metric averaging.
14. Train/test splitting.
15. All 36 current public single-model families.
16. Every AutoML branch actually offered by the CLI.
17. The public `all_models` aggregate branches.
18. Classification and regression application-data inference.
19. Time Series.
20. New and existing MLflow experiments.
21. MLflow UI lifecycle management.
22. Original outputs, logs, model files, plots, and tracking records.
23. Asynchronous status, cancellation, timeout, and failure recovery.

### 3.2 Explicit non-goals

The following are not part of complete public CLI parity unless they are first
made intentional public CLI features:

- the disabled Web path guarded by `web = False`;
- commented-out `web_setup` code;
- the independent Dash and FastAPI products;
- internal-only OPTICS, which is not in the public CLI model menu;
- the sample-balancing helper, which the public `data-mining` workflow does
  not call;
- remote execution, multi-user scheduling, cloud storage, authentication,
  billing, or a hosted MCP service.

## 4. Current verified coverage

| Capability | Current state |
| --- | --- |
| Classification | 11 public single-model families exposed |
| Regression | 15 public single-model families exposed |
| Clustering | 5 public single-model families exposed |
| Decomposition | 3 public single-model families exposed |
| Anomaly detection | 2 public single-model families exposed |
| Total single-model families | 36 |
| Supervised application inference | Classification and regression exposed |
| Missing values | Main supported CLI branches exposed |
| Scaling | None plus all three public scaling methods |
| Feature selection | Both public supervised selectors |
| Feature engineering | Semantic formulas compiled into the existing CLI |
| Label customization | All four public classification strategies |
| AutoML | Current supported classification and regression branches |
| Original outputs | Indexed without recalculation |
| Run control | Queue, status, cancellation, timeout, and recovery |
| Setup | Private runtimes, doctor, repair, uninstall, and client registration |

This coverage is substantial, but it is not yet complete CLI parity.

## 5. Confirmed gaps and root causes

### 5.1 World-map prompt mismatch

The CLI enters an additional interactive branch when both latitude and
longitude columns are present. Current MCP plans assume the map branch was
skipped and wait for the post-skip continuation prompt. A coordinate-bearing
dataset can therefore make the CLI wait for a Yes/No answer while the wrapper
waits for Enter.

This is a real execution defect, not merely an unavailable optional feature.

### 5.2 Built-in workbook incompatibility

The bundled `Data_*.xlsx` training workbooks contain coordinate columns and an
empty header cell. pandas assigns a name such as `Unnamed: 6` when the CLI reads
the workbook, while the current MCP inspector rejects every empty header.

The current wrapper therefore cannot directly use the CLI's own bundled
training workbooks through its explicit-path contract.

### 5.3 Time Series is advertised but not routed

The interactive mode list includes Time Series as option 6, but the later
`Modes2Models` mapping contains only keys 1 through 5. Selecting Time Series
can reach `Modes2Models[6]` and fail.

The repository contains a separate `run_time_series.py` module, but it is not
registered as a production console script and is not represented by the MCP
analysis schema.

Time Series must first become a tested, supported CLI path. The MCP must then
invoke that path instead of copying its calculation code.

### 5.4 Aggregate model execution is not equivalent

The current MCP represents `all_models` as multiple reproducible single-model
requests. This covers the algorithms but does not preserve the exact CLI
aggregate output hierarchy or nested MLflow behavior.

Complete parity requires the real CLI aggregate branch.

### 5.5 Previous-experiment lifecycle is unavailable

Each managed MCP run currently has an isolated working directory and tracking
directory. It cannot intentionally attach a new run to a previously selected
MLflow experiment.

### 5.6 Data-source convenience is incomplete

The MCP accepts an explicit absolute path. It does not yet provide safe,
semantic built-in or Desktop discovery, even though those are public CLI
convenience paths.

### 5.7 MLflow UI is not managed

The active CLI `--mlflow` option starts a local MLflow UI. The MCP does not
currently expose start, status, or stop operations for this service.

### 5.8 End-to-end model evidence is incomplete

All 36 models have interaction-plan compilation coverage, but the six current
real parity scenarios exercise five unique model families. The remaining model
families require real default manual runs before the product can claim
full-model parity.

### 5.9 Capability discovery is not exhaustive

The current `get_capabilities` response describes the five implemented task
families and documented unsupported combinations, but it does not fully explain
the map, built-in-data, Time Series, previous-experiment, and MLflow-service
boundaries discovered in this audit.

## 6. Target architecture

The complete product should replace prompt-text-dependent runtime operation
with a stable, versioned CLI automation boundary while retaining the human CLI
and the same scientific execution engine.

```text
Human CLI
    |
    v
HumanInputAdapter -----------+
                             |
                             v
                      SharedWorkflowExecutor
                             |
                             +--> existing preprocessing
                             +--> existing model workflows
                             +--> existing inference
                             +--> existing plots and maps
                             +--> existing Time Series
                             +--> existing MLflow and outputs
                             ^
                             |
AutomationInputAdapter ------+
    ^
    |
GeochemistryPi MCP semantic request
```

### 6.1 Proposed CLI automation command

```text
geochemistrypi automation run \
  --request <managed-request.json> \
  --events <managed-events.jsonl>
```

The exact command name may change during implementation, but the following
properties are mandatory:

- additive: the human `data-mining` workflow remains available;
- versioned: request, event, and result schemas declare versions;
- local: request and event files remain inside the managed workspace;
- strict: unknown, missing, and unused fields fail closed;
- semantic: the request contains scientific intent, not prompt answers;
- observable: stable events report meaningful stages;
- shared: both input adapters call the same workflow executor;
- non-duplicative: no second scientific pipeline is introduced.

### 6.2 Stable events

Events should use stable identifiers such as:

- `dataset.loaded`;
- `map.completed`;
- `preprocessing.completed`;
- `model.started`;
- `model.completed`;
- `inference.completed`;
- `outputs.completed`;
- `run.failed`.

Do not invent a percentage when the CLI cannot calculate one reliably. Batch
runs may report `completed_models` and `total_models`.

### 6.3 Migration from the current prompt driver

1. Retain the current driver as characterization evidence during migration.
2. Add the automation contract to the CLI without changing scientific logic.
3. Run identical requests through the direct human-characterized path and the
   automation path.
4. Compare complete outputs.
5. Switch MCP `0.3.0` to the automation path only after parity passes.
6. Do not retain redundant production paths unless an explicit legacy-support
   decision requires them.

## 7. Target MCP tool surface

The public tool set should remain small and semantic.

| Tool | Purpose |
| --- | --- |
| `get_capabilities` | Report exact versions, tasks, models, limits, and readiness |
| `list_datasets` | Discover built-in and Desktop datasets without reading full data |
| `inspect_dataset` | Inspect bounded schema, types, and samples read-only |
| `validate_analysis` | Validate a complete request and preview work without execution |
| `start_analysis` | Start one single-model, aggregate-model, or Time Series request |
| `get_run_status` | Read durable stage and state |
| `get_run_result` | Return bounded summaries and original artifact references |
| `cancel_run` | Cancel only the recorded wrapper-owned process tree |
| `list_experiments` | Discover managed MLflow experiments |
| `get_experiment` | Inspect one experiment and its runs |
| `start_mlflow_ui` | Start a managed local-only MLflow UI |
| `get_mlflow_status` | Return URL, state, and ownership information |
| `stop_mlflow_ui` | Stop only the MCP-owned MLflow process |

The MCP client must never submit shell commands, executable paths, raw prompt
IDs, response sequences, arbitrary environment variables, process IDs, or
unrestricted output paths.

## 8. Proposed semantic schemas

These are design examples. Final schemas must be derived from characterized CLI
behavior and validated through tests.

### 8.1 Data source

```json
{
  "data_source": {
    "type": "path",
    "training_dataset_path": "D:/data/train.xlsx",
    "application_dataset_path": "D:/data/application.xlsx"
  }
}
```

```json
{
  "data_source": {
    "type": "builtin",
    "training_dataset_id": "classification",
    "include_builtin_application_data": true
  }
}
```

```json
{
  "data_source": {
    "type": "desktop",
    "training_file": "training.xlsx",
    "application_file": "application.xlsx"
  }
}
```

### 8.2 Map projection

```json
{
  "map_projection": {
    "enabled": true,
    "longitude_column": "LONGITUDE",
    "latitude_column": "LATITUDE",
    "value_columns": ["SIO2(WT%)", "MGO(WT%)"]
  }
}
```

### 8.3 Model selection

```json
{
  "model_selection": {
    "mode": "single",
    "model": {
      "type": "random_forest"
    }
  }
}
```

```json
{
  "model_selection": {
    "mode": "all"
  }
}
```

### 8.4 Experiment selection

```json
{
  "experiment": {
    "mode": "new",
    "name": "Regional Basalt Classification",
    "run_name": "All Models"
  }
}
```

```json
{
  "experiment": {
    "mode": "existing",
    "experiment_id": "<stable-managed-id>",
    "run_name": "Updated Dataset"
  }
}
```

Experiment reuse should use an unambiguous managed ID. A user-facing name may
be displayed but should not be the only identity when duplicate or renamed
experiments are possible.

## 9. Implementation sequence

The work should continue as independently reviewable PR9 sub-phases. No phase
may advertise support before its direct CLI and MCP parity gates pass.

### PR9B: Complete capability inventory and coverage guard

**Local implementation checkpoint (2026-08-03):** implemented in the current
uncommitted worktree. The packaged schema-1 manifest tracks Typer declarations,
six modes, 36 model families, bundled datasets, evidence, and known gaps.

Purpose: make hidden public branches impossible.

Deliverables:

- create `cli_capability_manifest_v1.json`;
- enumerate active Typer commands and options;
- enumerate interactive modes and conditional branches;
- enumerate public model constants and AutoML availability;
- enumerate data sources, map behavior, inference, aggregate execution,
  experiments, Time Series, and MLflow UI;
- record each capability as `implemented`, `verified`, `known_gap`, or
  `not_public`;
- add a CI guard comparing current CLI declarations with the manifest;
- update `get_capabilities` so every known gap is discoverable.

Acceptance:

- every active public CLI feature has one stable capability ID;
- no capability is marked supported without a parity-test reference;
- changing a CLI menu or model list without updating the manifest fails CI;
- no permanent non-strict `xfail` hides a missing capability.

### PR9C: Stable CLI automation boundary

**Local implementation checkpoint (2026-08-03):** implemented in the current
uncommitted worktree with CLI automation input/event schema 1. Human and managed
runs share the unchanged scientific pipeline; managed runs no longer use prompt
text as their default transport contract. Compatibility policy schema 2 records
this boundary under the current development package versions; the planned
public CLI `0.9.0` and MCP `0.3.0` version bump remains a release gate rather
than being silently applied to an uncommitted checkpoint.

Purpose: remove prompt-text matching as the full-product runtime contract.

Deliverables:

- introduce human and automation input adapters;
- introduce a shared workflow executor without duplicating science;
- add strict versioned request and event schemas;
- keep human prompts and outputs behaviorally compatible;
- validate that every supplied automation input is consumed;
- reject missing or unexpected automation inputs;
- preserve CLI stdout, stderr, and original files;
- add compatibility and migration tests from the current driver.

Acceptance:

- human CLI baseline output remains unchanged except for approved bug fixes;
- automation mode completes one reference scenario for each main task family;
- automation mode fails closed on unknown, missing, duplicate, or unused fields;
- the MCP package still imports no scientific implementation modules;
- CLI `0.9.0` and MCP `0.3.0` compatibility is explicit.

### PR9D: Complete data-source parity

**Local implementation checkpoint (2026-08-03):** the safe data-source boundary
is implemented in the current uncommitted worktree: installed built-ins, safe
Desktop discovery, explicit paths, hashes, header normalization, and the
CSV/XLSX policy. Bundled training execution deliberately reports
`branch.world_map` as a blocker until PR9E rather than entering an unconfigured
interactive branch.

The discovery, inspection, integrity, and explicit/Desktop non-coordinate
execution boundary is complete. The acceptance item requiring bundled training
workbooks to execute is intentionally not marked complete yet: those workbooks
contain coordinate columns and require PR9E's semantic map choice. They remain
listable and inspectable, and attempts to train with them fail before process
creation with `branch.world_map` instead of hanging or guessing for the user.

Purpose: make data selection convenient without weakening path or integrity
controls.

Deliverables:

- add `list_datasets`;
- add built-in dataset IDs and metadata;
- add safe Desktop `geopi_input` discovery;
- preserve explicit absolute paths;
- normalize Excel headers exactly like the direct CLI/pandas path;
- resolve the public `.xls` claim by either implementing and testing it or
  removing the unsupported claim from every public message and document;
- preserve source hashes before and after every run;
- validate training/application schema compatibility.

Acceptance:

- every bundled dataset can be listed, inspected, and used through MCP;
- Desktop discovery is read-only and cannot escape the expected directory;
- empty, duplicate, oversized, and unsafe headers receive deterministic errors;
- CSV, XLSX, and the final deliberate XLS policy match between CLI and MCP;
- no input file is modified.

### PR9E: Complete world-map parity

Implementation status: **implemented locally in PR9E**. Semantic map roles,
explicit disablement, numeric/range validation, conditional Basemap/Cartopy
dependencies, artifact preservation, and actionable renderer failures are in
place. Remote Linux/macOS smoke results remain release evidence rather than a
local claim.

Purpose: support coordinate-bearing geochemical datasets without prompt
deadlock or lost map functionality.

Deliverables:

- add semantic map configuration to all relevant task requests;
- allow explicit latitude and longitude columns;
- support zero, one, or multiple projected value columns;
- validate numeric and finite map values;
- validate coordinate ranges and missing values;
- preserve Basemap on Windows/Linux and Cartopy on macOS where that remains the
  direct CLI contract;
- index original map files and supporting tables;
- add actionable dependency and renderer errors.

Acceptance:

- datasets without coordinates skip maps cleanly;
- coordinate datasets with maps disabled do not block;
- single- and multi-map requests complete;
- direct CLI and MCP inventories match;
- Windows, Linux, and macOS map smoke tests pass;
- no map package is installed dynamically at analysis runtime.

### PR9F: Repair and expose Time Series

Implementation status: **implemented locally in PR9F**. Menu option 6 and the
standalone `time-series` command share one seeded numerical workflow; MCP uses a
discriminated zero-interaction plan and indexes the CLI's CSV, PDF, metrics, and
parameter files. Installed parity is part of the final PR9E–PR9G gate.

Purpose: convert the advertised but broken path into a supported scientific
workflow before wrapping it.

CLI deliverables:

- repair option 6 routing;
- register a supported production Time Series command;
- make interactive and argument-driven Time Series call the same functions;
- validate required columns, bin width, iterations, finite values, and empty
  results;
- document random seed and numerical behavior;
- produce stable CSV and PDF artifacts;
- add scientific tests separate from legacy characterization.

MCP deliverables:

- add a discriminated `time_series` request;
- validate column roles before execution;
- invoke the installed CLI command;
- index original numeric and plot outputs;
- add status, cancellation, timeout, and parity evidence.

Acceptance:

- the CLI menu and standalone command agree;
- direct CLI and MCP numeric results match within recorded tolerances;
- input data, seed, dependency versions, and tolerances are documented;
- Windows, Linux, and macOS runs pass;
- no Time Series calculation exists in the MCP package.

### PR9G: Exact aggregate-model parity

Implementation status: **implemented locally in PR9G**. All five task schemas
compile the real CLI aggregate branch, preserve nested children, index their
files recursively, and expose complete or partial-failure results. Exhaustive
real-model execution remains deliberately assigned to PR9I. A representative
installed-wheel anomaly aggregate now has direct-CLI-versus-MCP evidence for
both public child models and the complete recursive file inventory.

Purpose: expose the real CLI `all_models` branches rather than approximating
them with unrelated independent runs.

Deliverables:

- add `model_selection.mode = "all"`;
- characterize aggregate prompts for every applicable task;
- preserve nested MLflow runs;
- preserve the aggregate output hierarchy;
- extend results with parent and child model summaries;
- report per-model state and artifacts;
- reproduce the CLI's AutoML behavior for aggregate supervised runs;
- define failure semantics for a child model failure.

Acceptance:

- all five current task families complete their public aggregate branch;
- output inventory and nested tracking match the direct CLI;
- aggregate cancellation terminates the recorded process tree;
- bounded result summaries do not hide child failures;
- artifact indexing remains bounded without losing the complete durable index.

### PR9H: Previous experiments and managed MLflow UI

Implementation record: see
`md/GeochemistryPi_MCP_PR9H_PR9J_Implementation.md`. Source and local tests may
establish implementation evidence, but remote platform and real-client release
evidence remains separate.

Purpose: complete the CLI's machine-learning lifecycle functionality.

Deliverables:

- introduce a persistent installer-owned tracking root;
- add `list_experiments` and `get_experiment`;
- support explicit existing-experiment IDs;
- preserve run and experiment data across repair, upgrade, and uninstall;
- add managed MLflow UI start, status, and stop tools;
- bind the UI to a local interface by default;
- detect port conflicts;
- record PID and process creation time;
- stop only a verified MCP-owned process.

Acceptance:

- a run can intentionally attach to a previously created experiment;
- ambiguous names fail before execution;
- tracking data survives product lifecycle operations;
- an unrelated process is never terminated;
- MLflow UI state is recovered safely after an unclean server shutdown;
- no UI service starts without an explicit user or client request.

### PR9I: Full real-model and branch parity matrix

Purpose: replace compilation-only confidence with real execution evidence.

Required real runs:

- 36 manual single-model direct-CLI-versus-MCP scenarios;
- every CLI-supported classification AutoML branch;
- all 13 CLI-supported regression AutoML branches;
- five aggregate-model scenarios;
- classification and regression application inference;
- all label strategies;
- all missing-value strategies;
- all scaling methods;
- both supervised feature selectors;
- representative feature-engineering chains;
- map off, single map, and multiple maps;
- new and existing experiments;
- explicit, built-in, and Desktop data;
- Time Series.

Acceptance:

- every supported capability-manifest row links to direct and MCP evidence;
- every output comparison checks input hashes and complete file inventories;
- identifiers, row order, feature order, targets, and predictions are preserved;
- floating-point tolerances are explicit;
- platform-dependent images use structural and content checks instead of
  unjustified cross-platform binary hashes;
- no Golden file is changed only to make CI pass.

### PR9J: User experience and client acceptance

Purpose: make the complete product easy to install and use from mainstream MCP
clients.

User-experience deliverables:

- `validate_analysis` previews resolved task, models, columns, data source,
  estimated model count, and warnings without starting work;
- errors explain the invalid field, actual value, valid alternatives, and next
  action;
- status uses stable scientific stages;
- aggregate results summarize child models;
- large or AutoML workloads report expected model counts before execution;
- documentation includes short natural-language workflows;
- no user must configure either internal Python environment.

Client acceptance targets:

- standard JSON;
- Codex;
- Claude Desktop;
- Claude Code;
- Cursor;
- VS Code;
- Gemini CLI;
- Windsurf;
- Cline;
- Continue;
- Kiro;
- OpenCode;
- Roo Code;
- Zed.

Each target must cover:

1. configuration discovery;
2. atomic registration;
3. backup preservation;
4. `tools/list`;
5. capability discovery;
6. one natural-language dataset workflow;
7. status polling;
8. result retrieval;
9. safe unregister;
10. unrelated-setting preservation.

### PR9K: Packaging, upgrade, signing, and public release

Purpose: convert the complete local implementation into a dependable product.

Implementation and operator handoff:
`GeochemistryPi_MCP_PR9K_Release_Implementation.md`. The local implementation
keeps public readiness false until the remote, signed, previous-bundle, and
real-client evidence below exists for the exact candidate.

Deliverables:

- build CLI and MCP wheels from clean source;
- inspect wheel contents;
- install only exact supported runtime versions;
- test clean install, repeated install, repair, upgrade, and uninstall;
- preserve runs, experiments, and unrelated client settings;
- generate a versioned release manifest;
- record wheel hashes;
- sign the manifest and published artifacts;
- decide and document PyPI and MCP registry publication;
- publish final user, operator, and developer documentation;
- provide a rollback procedure.

Acceptance:

- clean Windows, Linux, and macOS lifecycle jobs pass remotely;
- upgrade from the last supported bundle passes;
- doctor validates versions, manifests, hashes, paths, and resource limits;
- neither production wheel contains repository tests;
- at least ten target clients pass real acceptance;
- no known critical, high, or medium parity defect remains;
- public-release readiness changes to true only after every gate is complete.

## 10. Verification strategy

### 10.1 Capability-manifest gate

Every public capability row should record:

| Field | Meaning |
| --- | --- |
| `capability_id` | Stable public identifier |
| `cli_version` | Owning CLI version |
| `direct_cli_test` | Characterization or scientific test reference |
| `mcp_schema` | Semantic request path |
| `automation_test` | CLI automation-contract test |
| `parity_test` | Direct-versus-MCP test reference |
| `platforms` | Verified operating systems |
| `status` | Planned, implemented, verified, or deliberately not public |

A capability can be advertised only when its status is `verified`.

### 10.2 Test layers

1. CLI characterization tests freeze current public behavior.
2. Scientific tests verify correctness invariants.
3. Schema tests reject invalid or unknown requests.
4. Automation-contract tests verify stable request consumption and events.
5. Direct CLI tests prove the real branch works.
6. MCP parity tests compare the direct and wrapped branch.
7. Protocol tests prove strict MCP behavior.
8. Run-control tests cover status, queue, cancellation, timeout, and recovery.
9. Installation tests cover client registration and product lifecycle.
10. Real-client tests prove natural-language usability.

### 10.3 CI organization

Use sharded jobs to keep the complete matrix practical:

- fast PR gate: schemas, unit tests, manifests, representative parity;
- task shards: classification, regression, clustering, decomposition,
  anomaly detection, and Time Series;
- AutoML shards;
- map and platform-specific rendering shards;
- installation and client-config shards;
- scheduled complete matrix;
- mandatory full release-candidate matrix.

No release may rely only on the fast PR gate.

### 10.4 Required platform matrix

- `windows-latest` or the exact supported Windows image;
- `ubuntu-latest` or the exact supported Ubuntu image;
- `macos-15-intel` while the Python 3.9-era scientific stack requires Intel.

Runner labels and architectures must be explicit in the compatibility policy.

## 11. Scientific safety rules

- Preserve train-only fitting for supervised learned preprocessing.
- Reuse training-fitted transforms for test, display, and application data.
- Keep supervised and unsupervised workflows separate.
- Preserve identifier, row, feature, split, and target alignment.
- Preserve original class-label mappings.
- Do not recalculate metrics in MCP.
- Do not recreate missing plots or result files in MCP.
- Do not replace a direct CLI failure with partial MCP success.
- Keep characterization evidence separate from scientific correctness claims.
- Record dataset provenance, hashes, seeds, versions, and tolerances for Golden
  data.

## 12. Security and resource rules

- Accept only semantic task fields.
- Reject arbitrary commands, code, arguments, executables, and environment
  variables.
- Keep explicit-path access local and read-only before execution.
- Hash every input before and after the run.
- Keep outputs inside managed workspaces.
- Keep the durable artifact index complete but bound model-facing references.
- Preserve the atomic pending-run limit.
- Preserve configurable process timeouts.
- Terminate only process trees whose PID and creation time were recorded.
- Bind managed MLflow UI to a local interface by default.
- Never send complete datasets to the language model.

## 13. User experience requirements

The complete product should support this natural-language workflow:

```text
帮我分析桌面上的 rocks.xlsx，看看哪些元素最能区分不同岩性。
```

That single sentence is the user workflow. The client performs dataset
discovery, read-only inspection, request construction, validation, status
polling, and result retrieval automatically. If a safe scientific choice is
missing, it asks one short question such as “Which column represents the rock
type?” It explains the proposed analysis in plain language and waits for the
user's confirmation before starting. Experiment reuse and the local MLflow UI
are offered only when the user asks for history or visual tracking.

Explicit scientific intent has higher priority than inference or defaults. A
named task, dataset, column role, model, tuning mode, or parameter value is
preserved in the semantic request and validated against the capability
contract. An unsupported or unsafe choice is explained and returned to the user
for a decision; the client must not silently substitute another choice.

The user should never need to know:

- CLI menu numbers;
- prompt text;
- internal interaction IDs;
- the CLI Python 3.9 environment path;
- the MCP Python environment path;
- the private CLI executable path;
- raw MCP configuration syntax;
- internal output directory rules.

## 14. Versioning and migration

Recommended policy:

| Product | Current | Complete target |
| --- | --- | --- |
| GeochemistryPi CLI | 0.8.0 | 0.9.0 |
| GeochemistryPi MCP | 0.2.0 | 0.3.0 |
| Prompt interaction plan | schema 1 | migration-only evidence |
| CLI automation contract | unavailable | schema 1 |
| Capability manifest | unavailable | schema 1 |

The version numbers are proposals until the release decision, but the final
release must not claim full CLI `0.8.0` parity while the known CLI and wrapper
gaps remain.

Migration requirements:

- setup installs an exact compatible CLI/MCP pair;
- doctor rejects a partial or mismatched pair;
- existing managed runs remain readable;
- existing client registrations are repaired atomically;
- existing tracking data is migrated or preserved without silent relocation;
- rollback restores the previous executable/configuration while retaining user
  data;
- compatibility policy and documentation explain legacy behavior explicitly.

## 15. Documentation deliverables

Documentation must reflect actual state throughout implementation.

Required final documents:

- installation and one-command setup;
- supported client matrix;
- complete capability matrix;
- semantic tool reference;
- task examples;
- built-in and Desktop data workflows;
- map workflow;
- Time Series workflow;
- all-model workflow;
- experiment and MLflow workflow;
- status, cancellation, and recovery;
- troubleshooting;
- compatibility and upgrade policy;
- security and privacy behavior;
- release notes and signed-manifest verification.

Planned features must remain labelled planned until their acceptance tests pass.

## 16. Final release gates

The product is complete only when all of the following are true:

1. No active public CLI capability is missing from the capability manifest.
2. No manifest row is advertised before direct and MCP parity pass.
3. All 36 single-model families pass real parity.
4. Every supported AutoML branch passes real parity.
5. Every public aggregate-model branch passes real parity.
6. Classification and regression inference preserve original outputs.
7. Maps pass disabled, single, and repeated-selection cases.
8. Time Series passes scientific and parity tests.
9. Explicit, built-in, and Desktop data sources pass.
10. New and previous experiments pass.
11. Managed MLflow UI start, status, recovery, and stop pass.
12. Inputs remain unchanged.
13. Outputs and tracking data remain recoverable.
14. Windows, Linux, and macOS remote CI passes.
15. At least ten mainstream clients pass real natural-language acceptance.
16. Install, repeat install, repair, upgrade, uninstall, and rollback pass.
17. Production wheels contain no repository tests.
18. Release hashes and manifest signatures verify.
19. No known critical, high, or medium parity defect remains.
20. Documentation, `get_capabilities`, and actual implementation agree.
21. `public_release_ready` changes to true only after gates 1 through 20 pass.

## 17. Recommended next action

Begin with PR9B. Do not start another feature-specific patch until the complete
capability manifest exists, because that manifest becomes the objective source
of truth for every later implementation and release gate.

PR9B itself should not change scientific behavior. Its first responsibility is
to make the remaining scope measurable and to prevent another public CLI branch
from becoming an undiscovered MCP gap.
