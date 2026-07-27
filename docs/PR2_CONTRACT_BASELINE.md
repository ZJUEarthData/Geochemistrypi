# PR2 contract baseline

## Purpose

PR2 defines the first versioned wire contract shared by the GeochemistryPi
engine and future runtime and MCP packages. It does not implement experiment
execution, workers, run directories, or MCP tools.

The contract prevents the two environments from exchanging pandas objects,
scikit-learn objects, Python class paths, callables, or other process-specific
values. Requests and results cross the boundary as validated JSON data.

## Package boundary

The independent package is located at
`packages/geochemistrypi-contracts/` and installs as
`geochemistrypi-contracts`.

It:

- supports Python 3.9 and later;
- has no runtime dependencies;
- does not import the main engine, pandas, scikit-learn, Pydantic, or MCP;
- includes its public JSON Schema files in the wheel;
- provides standard-library dataclasses for engine-side use.

The JSON Schema files are the normative wire format. Dataclass round-trip tests
prove that the engine representation serializes to and restores from the same
JSON shape.

## Public v1 schemas

| Schema | Stable ID | Responsibility |
| --- | --- | --- |
| `dataset-ref.schema.json` | `https://schemas.geochemistrypi.org/contracts/v1/dataset-ref.schema.json` | Local CSV/XLSX reference, deterministic read options, optional input digest, and snapshot policy |
| `classification-experiment-spec.schema.json` | `https://schemas.geochemistrypi.org/contracts/v1/classification-experiment-spec.schema.json` | Classification request, preprocessing, split, model, and evaluation configuration |
| `experiment-result.schema.json` | `https://schemas.geochemistrypi.org/contracts/v1/experiment-result.schema.json` | Terminal metrics, warnings, and safe artifact references |
| `error-response.schema.json` | `https://schemas.geochemistrypi.org/contracts/v1/error-response.schema.json` | Stable error code, stage, retry guidance, and bounded details |

Every schema uses JSON Schema Draft 2020-12, declares
`x-contract-version: "1.0"`, and rejects unknown fields in its fixed public
objects. Model parameters, metric names, and error details remain bounded
dictionaries because their keys are selected by later registries or runtime
logic.

## Version and provenance rules

- Contract version `1.0` identifies the compatible wire shape.
- Package version `0.1.0` identifies this package release; it is not a
  replacement for the wire version.
- `$id` values remain stable for the complete lifetime of v1.
- `schema_sha256()` hashes the exact schema bytes installed from the wheel.
  Later run manifests must record the contract version and relevant schema
  digest.
- An incompatible request fails explicitly instead of being upgraded or
  guessed silently.
- JSON Schema `default` values are annotations; validation does not mutate an
  incoming payload. Dataclass deserialization applies those defaults and
  serialization emits a canonical complete payload.

## Scientific and safety decisions

- The classification contract supports `stratified_random` and `group`
  splitting in v1.
- Group splitting requires consistent group-column declarations.
- Learned preprocessing remains a configuration only; PR1's train-only fitting
  rule continues to control execution.
- Dataset format must match the file extension.
- Input digests use lowercase SHA-256.
- Artifact paths are portable relative paths and cannot contain absolute paths
  or parent traversal.
- Error response fields are bounded. Runtime code must sanitize traceback,
  secrets, and environment content before constructing a response.
- Model parameters are JSON-only and require a second strict model-specific
  validation when the model registry is introduced.

## Validation

Run the contract tests:

```text
python -m pytest tests/contracts
```

Run the complete engine suite without database configuration:

```text
SQLALCHEMY_DATABASE_URL="" python -m pytest
```

The wheel test builds `geochemistrypi-contracts`, verifies all four schema files
are packaged, installs the wheel into an isolated target directory, and loads
the schemas from that installed copy. CI repeats the wheel-content assertion
before running the full suite.
