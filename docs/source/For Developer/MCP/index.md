# GeochemistryPi MCP Developer Notes

This directory contains the chronological implementation baselines for the
GeochemistryPi MCP wrapper. These documents explain why each capability was
added and record the verification boundary at that point in development.

For current user installation and operation, use the
[MCP package README](../../../../packages/geochemistrypi-mcp/README.md). For the
current release checklist and operator handoff, use the
[PR9K implementation guide](../../../../md/GeochemistryPi_MCP_PR9K_Release_Implementation.md).
The phase documents below are historical engineering records, not a substitute
for those current instructions.

## Repository boundaries

- `geochemistrypi/` owns the public CLI and all scientific computation.
- `packages/geochemistrypi-mcp/` owns protocol validation, interaction-plan
  compilation, local run control, client registration, and release tooling.
- `tests/cli_contract/` freezes observable CLI behavior.
- `tests/mcp_wrapper/` verifies interaction, protocol, parity, and installation
  behavior without duplicating the scientific implementation.
- `md/` contains the user-requested roadmap, parity plan, requirements, and
  current release handoff documents.

The MCP package deliberately invokes the installed CLI in a separate process.
It must not import GeochemistryPi model classes or heavy machine-learning
libraries directly.

## Implementation history

1. [PR0: CLI contract baseline](PR0_CLI_CONTRACT_BASELINE.md)
2. [PR1: CLI interaction driver](PR1_CLI_INTERACTION_DRIVER.md)
3. [PR2: MCP package and local run control](PR2_MCP_LOCAL_RUN_CONTROL.md)
4. [PR3: complete classification coverage](PR3_COMPLETE_CLASSIFICATION_COVERAGE.md)
5. [PR4: setup, doctor, and client registration](PR4_SETUP_DOCTOR_AND_CLIENT_REGISTRATION.md)
6. [PR5: complete regression coverage](PR5_COMPLETE_REGRESSION_COVERAGE.md)
7. [PR6: complete clustering coverage](PR6_COMPLETE_CLUSTERING_COVERAGE.md)
8. [PR7: decomposition coverage](PR7_DECOMPOSITION_COVERAGE.md)
9. [PR8: anomaly detection and inference](PR8_ANOMALY_DETECTION_AND_INFERENCE.md)
10. [PR9 release-hardening foundation](PR9_RELEASE_HARDENING_FOUNDATION.md)
11. [PR9B–PR9D: capability inventory, automation, and data sources](PR9B_PR9C_PR9D_CAPABILITY_AUTOMATION_DATA_SOURCES.md)
12. [PR9E–PR9G: maps, time series, and all-model execution](PR9E_PR9F_PR9G_MAP_TIME_SERIES_ALL_MODELS.md)
