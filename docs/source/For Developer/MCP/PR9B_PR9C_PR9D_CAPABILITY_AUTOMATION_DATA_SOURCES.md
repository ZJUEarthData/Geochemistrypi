# PR9B–PR9D: Capability Inventory, CLI Automation, and Data Sources

## Outcome

This local development checkpoint implements PR9B, PR9C, and the safe
data-source boundary of PR9D without duplicating GeochemistryPi's scientific
code. Human CLI execution still calls the original `cli_pipeline`. Managed MCP
runs call the same function with a versioned input adapter active inside the
CLI process.

No release, commit, push, registry publication, or remote-CI result is claimed
by this document.

## PR9B: complete capability inventory

`cli_capability_manifest_v1.json` is packaged with GeochemistryPi MCP. It pins:

- active Typer commands and options;
- all six CLI menu modes;
- all 36 single-model menu families;
- all eight bundled dataset IDs;
- explicit CLI/MCP status for data sources, inference, AutoML, aggregate model
  execution, world maps, experiment reuse, Time Series, MLflow UI, and dataset
  formats;
- one stable capability ID for every inventoried item;
- parity or contract-test evidence for every item advertised as MCP-supported.

The manifest permits only `implemented`, `verified`, `known_gap`, and
`not_public`. An MCP-supported entry must be `verified` and have evidence. A
known gap cannot be marked supported. Duplicate IDs and malformed records fail
server capability discovery.

The CLI-side guard compares the manifest with Typer command metadata, public
mode/model constants, and bundled dataset declarations. Adding or changing a
command, option, mode, model, or bundled file without updating the inventory
fails the source test. The guard also rejects non-strict `xfail` escape hatches.

`get_capabilities` now publishes the manifest ID, every capability record,
known-gap IDs, supported data sources, and automation schema version. Important
known gaps remain visible rather than being implied as working:

- `branch.world_map`;
- `task.time_series`;
- `branch.previous_experiment`;
- `branch.all_models`;
- `branch.unsupervised_inference`;
- `cli.data_mining.mlflow_ui`.

## PR9C: stable CLI automation boundary

Managed runs no longer depend on waiting for ANSI terminal text and matching a
prompt before every response. The MCP driver writes this bounded plan inside
the managed wrapper directory:

```json
{
  "schema_version": 1,
  "plan_name": "classification-v1",
  "inputs": [
    {"id": "use_previous_experiment", "response": "n"},
    {"id": "experiment_name", "response": "Reference"}
  ]
}
```

The CLI receives the plan only when both `--automation-plan` and
`--automation-events` are present. It temporarily installs an input adapter,
runs the unchanged scientific workflow, restores the original `input`, and
atomically writes schema-1 events. Each event contains the stable input ID,
sequence number, prompt SHA-256, prompt length, and timestamp. Human-readable
prompts remain in stdout, but prompt text is diagnostic evidence rather than
the transport contract.

The boundary fails closed when:

- the plan is relative, oversized, malformed, or has unknown fields;
- the schema version is unsupported;
- IDs are invalid or duplicated;
- a response is not a bounded single line;
- the CLI asks for an unplanned input;
- the workflow exits with unused inputs;
- event fields, ordering, hashes, plan name, or completion state are invalid.

The MCP subprocess still preserves stdout, stderr, the original output tree,
process cancellation, total timeout, and input hashes. The legacy
prompt-synchronized driver remains available for migration characterization,
but `RunManager` uses CLI automation by default. Real direct-CLI-versus-MCP
parity covers classification, regression, clustering, decomposition, anomaly
detection, and supervised application-data output under the new transport.

The compatibility policy is schema 2 and explicitly records CLI automation
schema 1. Package versions remain the local development pair MCP `0.2.0` and
CLI `0.8.0`; public `0.3.0`/`0.9.0` version assignment remains a release
decision rather than being silently claimed in an unfinished PR9 branch.

## PR9D: complete safe data-source boundary

The seventh MCP tool, `list_datasets`, queries the installed CLI in its own
Python 3.9 environment. MCP never imports the scientific package. The CLI
returns schema-1 metadata for:

- eight bundled datasets with stable IDs, task, and training/application role;
- supported files directly inside `Desktop/geopi_input`;
- absolute path, format, size, row/column counts, and SHA-256;
- analysis blockers known at discovery time.

Desktop discovery is read-only. It does not create the directory, copy bundled
files, recurse, accept unsupported suffixes, or allow a resolved path outside
the expected root. MCP revalidates the returned path, immediate-parent rule,
size, and SHA-256 before returning it.

Analysis and inspection schemas accept one of these strict references:

```json
{"source":"path","path":"D:\\data\\rocks.csv"}
{"source":"builtin","dataset_id":"builtin:classification"}
{"source":"desktop","file_name":"rocks.xlsx"}
```

An optional `expected_sha256` gives callers optimistic integrity locking. The
legacy `training_dataset_path` and `application_dataset_path` remain compatible.
Supplying neither, supplying both representations, a Desktop path component,
an unsupported extension, a wrong built-in task/role, or a changed expected
hash fails before queuing work.

The run record preserves the semantic request plus resolved source, stable ID,
absolute path, size, format, and pre-run SHA-256. Existing pre-execution,
during-execution, and post-execution hash checks still protect training and
application inputs.

### Header behavior

Excel and CSV blank headers use pandas-compatible zero-based names such as
`Unnamed: 6`. Leading/trailing whitespace, control characters, names over 128
characters, too many columns, and explicit duplicate headers receive bounded,
deterministic errors.

One bundled Time Series workbook already contains a duplicate `FEOT` header.
Trusted bundled inspection uses pandas-compatible suffixing (`FEOT.1`) and
returns `header_warnings`; arbitrary external files do not receive this trust
exception. All eight installed workbooks can therefore be listed and inspected
without modifying a byte.

The CLI reader actually supports `.csv` and `.xlsx`. Public `.xls` claims were
removed from the interactive message, Desktop discovery, upload route, Time
Series entry points, and file dialog because no reliable legacy Excel reader is
installed. MCP advertises the same deliberate policy.

### World-map dependency between PR9D and PR9E

All bundled training workbooks contain recognized coordinate columns. The
human CLI enters the interactive map branch before feature selection. PR9D does
not invent a hidden answer for that scientific/user choice. Dataset entries
therefore report `branch.world_map` as an analysis blocker, while remaining
fully listable and inspectable. A coordinate-bearing explicit or Desktop
training file is rejected during plan compilation before a process starts with
the same actionable capability ID.

PR9E will replace this blocker with semantic map configuration. Until then,
users who do not need maps can deliberately provide an analysis copy without
coordinate columns; MCP never edits the original file for them.

## Local verification evidence

Completed on Windows in isolated Python 3.9 and Python 3.11 environments:

- 65 core/CLI tests passed;
- 182 MCP installation, interaction, and protocol tests passed both against
  current source and against the installed final MCP wheel;
- seven installed-wheel parity scenarios passed: six real
  direct-CLI-versus-MCP analyses under CLI automation schema 1, plus one
  scenario that listed and inspected all eight bundled datasets, including the
  duplicate-header warning;
- the core wheel contained 181 entries and the MCP wheel contained 28 entries;
  neither wheel contained repository tests, and both included their new PR9B-D
  runtime data/modules;
- an isolated cold setup completed after the local 15-minute observation
  harness expired; the resulting runtime independently passed doctor at 7/7.
  A forced repair then completed in 33.1 seconds and also passed doctor at 7/7;
  uninstall removed only the private runtimes/settings/manifest and preserved
  the runs directory;
- pre-commit formatting, import ordering, whitespace, EOF, YAML, large-file,
  and Flake8 checks passed.

The cold-install duration is a release-user-experience observation, not hidden
as a successful fast install. Publication-oriented wheel bootstrapping and
install-progress improvements remain later release-hardening work. Remote CI,
signing, publication, registry mutation, and real user-client configuration
were not performed by this local checkpoint.
