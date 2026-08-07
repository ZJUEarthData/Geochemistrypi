# GeochemistryPi MCP Architecture and Implementation Overview

This is the single developer overview for the GeochemistryPi MCP wrapper. It
describes the current product architecture and maintenance boundaries without
retaining a separate document for every implementation phase.

For installation and daily operation, use the
[MCP package README](../../../../packages/geochemistrypi-mcp/README.md). This
overview contains the shared architecture, maintenance, verification, and
release boundaries that previously appeared in separate phase documents.

## Product purpose

The wrapper lets an MCP-compatible AI client operate the existing
GeochemistryPi command-line workflow through validated scientific requests.
Scientists can describe the analysis in natural language. When they explicitly
choose a model or parameter, the client must preserve that choice in the MCP
request instead of replacing it with a default.

The MCP layer does not reimplement model training. It translates a validated
request into a deterministic interaction plan, runs the public CLI in an
isolated subprocess, and returns bounded status, result, artifact, dataset, and
experiment metadata.

## Runtime architecture

```text
MCP-compatible client
        |
        | stdio MCP
        v
geochemistrypi-mcp
        |
        | validated request and interaction plan
        v
isolated GeochemistryPi CLI subprocess
        |
        v
original outputs and local MLflow tracking
```

The two-process design keeps incompatible dependencies isolated:

- the MCP runtime uses Python 3.10 or newer and the official MCP SDK;
- the scientific CLI runtime uses the supported Python 3.9 GeochemistryPi
  environment;
- setup stores both interpreter locations so users do not provide them for
  every analysis;
- protocol output remains separate from CLI stdout and stderr.

## Repository boundaries

- `geochemistrypi/` owns the public CLI, scientific preprocessing, model
  execution, plots, maps, time-series analysis, aggregation, and MLflow writes.
- `packages/geochemistrypi-mcp/` owns request schemas, capability metadata,
  interaction-plan compilation, subprocess control, managed runs, artifact
  discovery, client configuration, diagnostics, and release tooling.
- `tests/cli_contract/` freezes observable CLI behavior and stable fixtures.
- `tests/mcp_wrapper/interaction/` verifies request-to-prompt translation.
- `tests/mcp_wrapper/protocol/` verifies MCP tools and managed run behavior.
- `tests/mcp_wrapper/parity/` compares direct CLI and MCP results.
- `tests/mcp_wrapper/installation/` verifies setup, repair, rollback, doctor,
  release, and client registration behavior.

The MCP source package is layered by stable responsibility while keeping the
installed entry points at the package root:

```text
geochemistrypi_mcp/
├── __main__.py, server.py          # stdio application boundary
├── setup.py, doctor.py, release.py # stable console entry points
├── api/                            # MCP schemas and tool dispatch
├── config/                         # settings, constants, client adapters
├── contracts/                      # scientific capability declarations
├── data/                           # bounded dataset discovery and inspection
├── lifecycle/                      # install, diagnose, upgrade, and release
├── planning/                       # semantic request to CLI interaction plan
├── runtime/                        # subprocess, managed runs, and artifacts
└── tracking/                       # experiment metadata and managed MLflow UI
```

Dependencies point inward toward contracts and configuration, then outward
through planning and runtime to the CLI subprocess. The root console modules
are intentionally thin so installed command names remain stable without
pulling lifecycle implementation into the protocol layer.

## Supported workflow surface

The wrapper exposes capability discovery, safe dataset discovery and
inspection, analysis start/status/result/cancel operations, experiment lookup,
and managed MLflow UI control. Its analysis schemas cover:

- classification;
- regression;
- clustering;
- decomposition;
- anomaly detection;
- world-map configuration;
- time-series workflows;
- exact all-model execution;
- training-only and training-plus-application data paths;
- built-in, local-path, and supported Desktop dataset sources.

The versioned capability manifest is the machine-readable source of truth for
supported tasks, models, modes, and known restrictions. Request schemas reject
unknown fields and invalid combinations before a CLI process starts.

## Installation and client configuration

The package provides four stable console commands:

```text
geochemistrypi-mcp
geochemistrypi-mcp-setup
geochemistrypi-mcp-doctor
geochemistrypi-mcp-release
```

Setup supports the registered client adapters documented in the package
README, plus a standard JSON fallback. Configuration updates are atomic and
retain backups. Repeated setup repairs managed state, upgrade and rollback
preserve user runs and tracking data, and uninstall removes managed runtime
configuration without deleting scientific results.

## Safety and integrity rules

- MCP code must not import GeochemistryPi model classes or heavy scientific
  training libraries directly.
- The CLI is invoked only through validated plans; raw commands and arbitrary
  answer scripts are not public inputs.
- Dataset paths, output paths, resource limits, process timeouts, pending-run
  limits, and artifact counts are bounded before use.
- Managed state uses private application directories and atomic writes.
- Cancellation terminates the CLI process tree and records a durable terminal
  state.
- Result metadata refers to artifacts inside the managed run directory.
- The production wheels must not contain repository tests.
- A capability is not considered complete without direct-CLI versus wrapper
  parity evidence.

## Verification and release boundary

Local verification covers Python 3.9 CLI tests, MCP interaction and protocol
tests, installed-wheel tests, parity scenarios, formatting, linting, and wheel
content inspection. The release workflows add Windows, Linux, and macOS gates,
sharded real-model parity, artifact signing, and attestations. Before opening a
release PR, run the cross-platform preflight from the repository root:

```text
uv run --isolated --no-project --python 3.11 python packages/geochemistrypi-mcp/tools/release_preflight.py
```

The default command includes all seven slow full-model parity shards. During
iteration, `--quick` skips only those shards; it is not release evidence. The
preflight writes wheels, private environments, and lifecycle state to a system
temporary directory, never to the repository or the user's real Agent
configuration. On failure it retains that directory for diagnosis.

The Tag workflow downloads the final signed artifact into clean ordinary-user
jobs on Windows, Linux, and Intel macOS. Each job verifies the Sigstore bundles
offline against the pinned GitHub workflow identity, installs the exact wheels
under a path containing spaces and non-ASCII characters, runs Doctor, repairs
the installation, uninstalls it, and confirms scientific run/tracking data was
preserved. Native macOS arm64 is not claimed until that same final-artifact
gate is available and green on arm64.

Local success is not a public-release claim. A release is ready only after all
required remote jobs finish successfully and the generated artifacts satisfy
the release manifest and signature policy.

## Maintainer source of truth

- Current user commands and supported clients:
  [MCP package README](../../../../packages/geochemistrypi-mcp/README.md)
- Machine-readable supported capabilities:
  [CLI capability manifest](../../../../packages/geochemistrypi-mcp/src/geochemistrypi_mcp/contracts/cli_capability_manifest_v1.json)
- Cross-platform validation and release gates:
  [CI workflow](../../../../.github/workflows/geochemistrypi.yml) and
  [release workflow](../../../../.github/workflows/release.yml)
- Executable behavior and parity evidence: `tests/cli_contract/` and
  `tests/mcp_wrapper/`
