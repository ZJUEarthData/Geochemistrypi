# GeochemistryPi Runtime

`geochemistrypi-runtime` owns the durable, local representation of an
experiment run. It creates a complete run directory, writes JSON records
atomically, enforces status ownership and revisions, and verifies artifact
integrity.

The package deliberately does not import the GeochemistryPi machine-learning
engine. A future worker package will execute experiments and use this package
to persist requests, status, results, manifests, and provenance.

## Run directory

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

`result.json` is added only after a worker has produced a validated terminal
result.

## Safety model

- A new run is first prepared in a private staging directory and is published
  with one atomic rename.
- Mutable records use atomic file replacement plus per-record file locks.
- Every mutable record has a revision so callers can detect stale writes.
- Worker-owned status cannot be overwritten by the run manager.
- Cancellation is requested through `control.json`; it is not a status
  overwrite.
- Recovery requires an explicit statement that the worker has stopped.
- Artifacts must stay inside `artifacts/` and are recorded with size and SHA-256.

This package stores facts. It does not schedule processes, run models, or
provide an MCP server.
