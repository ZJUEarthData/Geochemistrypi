# Geochemistryπ test suites

The local MCP platform work keeps compatibility evidence separate from scientific
correctness evidence.

- `characterization/`: records observable behavior of the existing CLI and
  legacy machine-learning implementation. These tests are not scientific truth.
- `scientific/`: verifies leakage-free preprocessing, metrics, invariants, and
  reproducibility against independent expectations.
- `integration/`: verifies boundaries between configuration, API modules,
  workers, runtime storage, and later package boundaries.
- `contracts/`: verifies v1 JSON Schema validity, dependency-free engine
  dataclass round trips, strict rejection behavior, and installed wheel
  resources.
- `runtime/`: verifies atomic run creation, request integrity, status ownership,
  cancellation control, crash recovery evidence, artifact hashes, provenance,
  package boundaries, and the installed Runtime wheel.
- `mcp/`: verifies MCP protocol behavior. This directory is intentionally empty
  until the MCP package is introduced.

Existing legacy tests remain under `geochemistrypi/data_mining/tests/` until a
later focused migration. Moving them is not required for PR0.
