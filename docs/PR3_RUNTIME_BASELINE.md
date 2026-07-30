# PR3 runtime baseline

## Purpose

PR3 creates the durable local storage layer used by future GeochemistryPi
workers and MCP services. It turns the versioned PR2 request into a complete run
directory and protects every later update against partial writes, stale writers,
path escape, and silent artifact changes.

This PR does not execute machine-learning code, start worker processes, manage a
queue, or expose MCP tools. Those components will use this storage boundary in
later PRs.

## Package boundary

The independent package is located at
`packages/geochemistrypi-runtime/` and installs as
`geochemistrypi-runtime`.

It:

- supports Python 3.9 and later;
- depends only on `geochemistrypi-contracts` and `filelock`;
- does not import the GeochemistryPi Engine, pandas, scikit-learn, FastAPI, or
  MCP;
- stores JSON records with deterministic serialization;
- uses atomic file replacement and per-record inter-process locks;
- ships no tests or Engine modules in its wheel.

Architecture tests enforce this dependency direction instead of relying only on
documentation.

## Run directory

`RunContext.create()` prepares the complete directory under a private staging
name and publishes it with a single rename:

```text
runs/<run_id>/
├── request.json
├── request.sha256
├── status.json
├── control.json
├── manifest.json
├── provenance.json
├── inputs/
├── artifacts/
├── errors/
└── .locks/
```

`result.json` is created only when a terminal `ExperimentResult` is available.
Worker identity and log files are introduced with the worker process in a later
PR.

The request is immutable. `request.sha256` hashes the exact canonical bytes of
`request.json`; reading a request verifies this digest before deserialization.

## State ownership

Atomic replacement prevents half-written JSON, but it cannot by itself prevent
two processes from overwriting each other. Runtime therefore combines file
locks, monotonically increasing revisions, and explicit owners.

| Record or transition | Writer |
| --- | --- |
| Initial `queued` status | Run manager |
| `queued` to `validating` | Worker claiming the run |
| Active status after claim | The same worker identity |
| Cancellation request | Run manager or MCP layer through `control.json` |
| `orphaned` or `corrupted` repair | Recovery logic after the worker is confirmed stopped |

Every mutable update is compare-and-swap: a caller supplies the revision it
read, and a stale update fails with `RevisionConflictError`. A worker identity
cannot be replaced by another worker or by the run manager.

Normal transitions are:

```text
queued -> validating -> running -> completed
                    \-> cancel_requested -> cancelled
                    \-> failed
```

`orphaned` means a worker was lost; it does not imply checkpoint resume.
`corrupted` means the previous status cannot be trusted. Corruption repair first
archives bounded raw evidence under `errors/`, then writes recovery-owned
status. Both repair operations require explicit confirmation that the worker
has stopped.

## Manifest, provenance, and artifacts

`manifest.json` records the request digest, PR2 contract version, request-schema
ID and schema digest, status and provenance paths, result path, warnings, and
artifact references.

`provenance.json` starts with Runtime, contract, Python, platform, dependency,
and optional Git version facts. Later workers can update named, revisioned
sections for the dataset, input, split, preprocessing, feature engineering,
resampling, model, evaluation, resources, timing, determinism, and failure.

An artifact:

- must be a regular file below the run's `artifacts/` directory;
- cannot use an absolute path, parent traversal, backslashes, or a symbolic-link
  escape;
- is registered only after its size and SHA-256 are calculated;
- is verified again before access or result publication.

`result.json` must match the run ID, immutable request digest, manifest paths,
and registered artifact records. A retry may complete an interrupted matching
result write, but a different existing result is rejected.

## Validation

Run the Runtime suite:

```text
python -m pytest tests/runtime
```

Run the complete repository suite without database configuration:

```text
SQLALCHEMY_DATABASE_URL="" python -m pytest
```

The Runtime suite covers concurrent revision conflicts, status ownership,
separate cancellation control, simulated creation failure, corrupt-status
evidence, request and artifact tampering, unsafe paths, provenance revisions,
result consistency, import boundaries, wheel contents, and installed-wheel
imports. CI builds and inspects the Engine, Contracts, and Runtime wheels before
running the complete suite.
