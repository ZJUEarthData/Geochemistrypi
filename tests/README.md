# Geochemistryπ test suites

The local MCP platform work keeps compatibility evidence separate from scientific
correctness evidence.

- `characterization/`: records observable behavior of the existing CLI and
  legacy machine-learning implementation. These tests are not scientific truth.
- `scientific/`: verifies leakage-free preprocessing, metrics, invariants, and
  reproducibility against independent expectations.
- `integration/`: verifies boundaries between configuration, API modules,
  workers, runtime storage, and later package boundaries.
- `mcp/`: verifies MCP protocol behavior. This directory is intentionally empty
  until the MCP package is introduced.

Existing legacy tests remain under `geochemistrypi/data_mining/tests/` until a
later focused migration. Moving them is not required for PR0.
