# PR9 Release Hardening Foundation

## Outcome

This PR9 checkpoint converts implicit development assumptions into an explicit,
machine-readable release contract. It does not publish GeochemistryPi MCP or
claim that every release gate has passed. The server continues to run the
existing GeochemistryPi CLI as its only scientific engine.

`get_capabilities` now reports both the compatibility policy and the installed
resource limits. The policy deliberately identifies the current channel as
`development`, sets `public_release_ready` to false, and lists the gates still
required before a public release.

## Compatibility policy

The first compatibility-policy schema declares:

| Boundary | Contract |
| --- | --- |
| GeochemistryPi MCP | 0.2.0 |
| MCP Python | `>=3.10,<4` |
| MCP SDK | `==2.0.0` |
| GeochemistryPi CLI Python | `>=3.9,<3.10` |
| GeochemistryPi CLI | 0.8.0 |
| Interaction plan | schema 1 |
| Artifact index | schema 1 |
| Target operating systems | Windows, Linux, macOS |

The core package metadata now expresses the real Python 3.9-only CLI boundary
as `>=3.9,<3.10`. The previous `~=3.9` declaration was interpreted by packaging
tools as allowing later Python 3.x versions and therefore did not match the
private CLI runtime validated by setup and doctor. The MCP package now states
its complementary `>=3.10,<4` boundary explicitly.

Setup writes the compatibility-policy version and both runtime requirements to
its install manifest. A manifest created before this policy triggers a private
runtime refresh during the next install. Doctor adds an install-manifest check
and rejects missing, malformed, or stale compatibility metadata.

A real wheel-bootstrapped setup exercise exposed an additional environment
boundary defect. uv-generated Python launchers can inject `PYTHONHOME` and
`UV_INTERNAL__PYTHONHOME` into the setup process. The installer previously
forwarded them into uv's build subprocess, allowing the build backend to mix
incompatible standard libraries. Setup now removes the same isolated-runtime
variables as the CLI driver and doctor before every environment command. A
focused test protects this boundary, and the real install/repair flow now
passes.

## Resource limits

Normal setup persists these installer-owned limits:

- maximum dataset size: 512 MiB;
- maximum columns inspected: 256;
- maximum artifact references returned: 200;
- maximum concurrent CLI runs: 1;
- maximum active or queued runs: 8;
- maximum total CLI process time: 900 seconds.

The queue limit closes an unbounded-work-submission path. Admission is checked
atomically under the run-manager lock before a run directory, durable request,
or run ID is exposed. A full queue returns an actionable error and accepts new
work again after a run finishes or is cancelled. The configured total process
limit is now passed to the CLI driver instead of remaining an implicit driver
default.

The pending-run and process-time limits can be tightened through explicit
development environment variables or persisted setup settings. They remain
server configuration and are never accepted as analysis-tool arguments.

## Cross-platform release gate

The MCP wrapper CI matrix now targets Ubuntu, Windows, and the GitHub-hosted
`macos-15-intel` runner for installation, interaction, protocol, wheel-archive,
and installed-wheel tests. The explicit Intel label keeps the Python 3.9-only
CLI and its legacy scientific dependencies off the arm64 `macos-latest` runner.
Each job also installs the verified uv 0.11.7 setup runtime and exercises real
repeated install, forced repair, doctor, and uninstall commands. Local Windows
verification can validate the workflow definition and all shared tests, but the
new macOS and existing Linux jobs remain unverified until an authorized commit
and push produce successful remote terminal runs.

Real-client natural-language acceptance, signed release hashes, upgrade tests
against an actual previously published bundle, and the PyPI/MCP-registry release
decision remain open PR9 gates. No signing key, external client configuration,
package publication, or registry mutation is performed by this checkpoint.

## Regression evidence

Focused tests cover persisted and legacy settings, invalid limit overrides,
queue saturation and reopening, configured driver timeout, compatibility-aware
install refresh, doctor manifest rejection, and protocol discovery of the new
policy. The completed local checkpoint includes:

- 55 GeochemistryPi core and CLI-contract tests;
- 172 MCP tests, including six real direct-CLI-versus-stdio-MCP scenarios;
- a real isolated `install -> install -> repair -> doctor -> uninstall` cycle,
  with doctor healthy at 7/7 checks and managed run evidence preserved;
- a 179-entry core wheel and 24-entry MCP wheel with zero repository-test
  paths;
- `Requires-Python: >=3.9,<3.10` for the core and `>=3.10,<4` for MCP;
- standard pip rejection of the core wheel under Python 3.11 and successful
  installed-core smoke checks under Python 3.9;
- 166 non-parity tests from a clean MCP-wheel installation, with the six real
  CLI scenarios deliberately deselected.

Direct local-wheel installation with `uv pip` does not by itself provide the
same negative-version rejection as standard pip. Product setup therefore does
not rely on installer metadata alone: it explicitly creates Python 3.9 for the
CLI and doctor independently rejects any other CLI runtime.

This document describes local uncommitted work. It does not claim remote CI, a
commit, a push, publication, signing, or a public release.

## PR9B–PR9D continuation

The later local PR9B–PR9D checkpoint adds a machine-validated CLI capability
manifest, CLI-owned automation input/event schema 1, the `list_datasets` tool,
stable built-in and safe Desktop data references, pandas-compatible blank Excel
headers, and a deliberate CSV/XLSX-only format policy. Compatibility-policy
schema 2 now includes the automation boundary. Managed runs use the new CLI
input adapter by default while legacy prompt synchronization remains under
migration tests.

Known gaps are returned by `get_capabilities`. In particular, coordinate-bearing
training files fail before process creation until PR9E supplies semantic map
configuration; bundled training entries expose this blocker while remaining
listable and inspectable. See
`PR9B_PR9C_PR9D_CAPABILITY_AUTOMATION_DATA_SOURCES.md` for the complete contract
and verification evidence.

The final local PR9B–PR9D evidence is 65 core tests, 182 MCP non-parity tests
against both source and the installed wheel, and seven installed-wheel parity
scenarios. The rebuilt core/MCP wheels contain 181/28 entries and no repository
tests. An isolated runtime passed doctor at 7/7 after cold install and again
after a 33.1-second forced repair; uninstall preserved the runs directory. The
cold install exceeded the local 15-minute observation harness before completing,
so installation progress and publication-oriented bootstrapping remain explicit
later release-hardening concerns.
