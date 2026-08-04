# GeochemistryPi MCP PR9K Release Implementation

Status: implemented locally as a release candidate; public publication is
deliberately blocked.

This document is the user, operator, and developer handoff for PR9K in
`GeochemistryPi_Complete_CLI_MCP_Parity_Plan.md`. It describes the code that is
present now, the evidence that can be produced locally, and the external gates
that must remain open until they run against the exact signed bundle.

## 1. Product boundary

The release is one exact pair:

| Component | Version | Private interpreter |
| --- | --- | --- |
| GeochemistryPi CLI | `0.8.0` | Python `3.9` |
| GeochemistryPi MCP | `0.2.0` | Python `3.11` |
| MCP Python SDK | `2.0.0` | MCP environment |
| Setup runtime | `uv 0.11.7` | Bootstrap host |

Package metadata remains `>=3.9,<3.10` for the CLI and `>=3.10,<4` for MCP,
but the product installer intentionally creates Python 3.9 and 3.11 exactly.
It rejects any other `uv` version. The installation manifest also stores a
canonical count and SHA-256 digest of every installed distribution in each
private environment, allowing Doctor to detect dependency drift after setup.

The MCP wrapper does not reimplement scientific calculations. The CLI wheel is
the only training, inference, metric, plot, and artifact-producing engine.

## 2. Release-bundle contract

A complete bundle has these files and no additional wheel:

```text
release-bundle/
  geochemistrypi-0.8.0-*.whl
  geochemistrypi_mcp-0.2.0-*.whl
  release-manifest.json
  geochemistrypi-0.8.0-*.whl.sigstore.json
  geochemistrypi_mcp-0.2.0-*.whl.sigstore.json
  release-manifest.json.sigstore.json
```

The manifest records:

- exact distribution names, versions, `Requires-Python`, sizes, and SHA-256
  hashes;
- the source Git commit and exact tag
  `mcp-v0.2.0-cli-v0.8.0`;
- the compatibility policy and private interpreter versions;
- the trusted Sigstore issuer and exact tag-workflow certificate identity;
- the deferred PyPI/MCP Registry decision;
- `public_release_ready: false` and every remaining external gate.

The verifier rejects malformed or extra fields, unsafe filenames, symlinks,
duplicate or unlisted wheels, missing metadata, wrong versions, packaged test
modules, size/hash changes, missing signatures, and an untrusted signing
identity. `--allow-unsigned` and `--allow-unsigned-bundle` are explicit local
release-candidate overrides. Their use is recorded as
`explicit-development-override`; neither command is a public-release path.

## 3. User installation

Install `uv`, download the whole signed bundle without renaming its files, and
bootstrap setup from the MCP wheel. Use an absolute file URI for the wheel and
an absolute filesystem path for the bundle:

```text
uvx --python 3.11 --from "geochemistrypi-mcp[release] @ file:///ABSOLUTE/PATH/geochemistrypi_mcp-0.2.0-py3-none-any.whl" geochemistrypi-mcp-setup install --bundle /ABSOLUTE/PATH/release-bundle
```

For Windows, a valid URI looks like:

```text
file:///D:/Downloads/release-bundle/geochemistrypi_mcp-0.2.0-py3-none-any.whl
```

Setup verifies all three Sigstore bundles before it stops a managed process,
creates an environment, or changes a client configuration. It then creates the
two private environments, validates the exact package pair, records their
dependency inventories, writes settings and the install manifest atomically,
runs Doctor, and registers detected clients. A standard `mcpServers` JSON file
is always created even when no named client is detected.

Repository installation without `--bundle` remains supported only for
developers working from a complete clone.

## 4. Repeat, repair, upgrade, and rollback

Repeat install is idempotent. When the active source fingerprint and runtime
are current, environments are not recreated and matching client entries are
left unchanged.

Repair creates fresh environments at the stable installed paths. The current
runtime is first moved to a private recovery directory. If preparation,
validation, Doctor, or finalization fails, the prior environment, settings,
manifest, and active bundle are restored. Client files are snapshotted before
multi-client registration and restored if a later client fails. Claude Code is
updated only through its official CLI; setup never removes a conflicting entry
before a replacement has succeeded.

Upgrade must receive a newly verified bundle. It:

1. requires a current-schema installation and a passing preflight Doctor;
2. stops only a process whose managed ownership can be proved;
3. moves the active environments and release files into a one-level private
   rollback transaction;
4. installs and validates the new exact pair;
5. runs Doctor against the new runtime and bundle hashes;
6. atomically promotes the rollback snapshot and runs the rollback-aware Doctor
   check;
7. repairs the same registered clients at their stable server path.

Any failure before completion restores the prior runtime. A successful upgrade
sets `rollback_available: true`. A second upgrade replaces the single retained
snapshot only after the new runtime has passed validation.

Use the same signed-wheel bootstrap to upgrade:

```text
uvx --python 3.11 --from "geochemistrypi-mcp[release] @ file:///ABSOLUTE/PATH/NEW/geochemistrypi_mcp-0.2.0-py3-none-any.whl" geochemistrypi-mcp-setup upgrade --bundle /ABSOLUTE/PATH/NEW/release-bundle
```

Rollback needs no network access or new download. On Linux and macOS, run the
setup executable next to the installed Doctor executable:

```text
~/.local/state/geochemistrypi-mcp/environments/mcp/bin/geochemistrypi-mcp-setup rollback
```

On Windows, use the downloaded or retained MCP wheel as an external bootstrap.
Do not run a destructive lifecycle action from inside the private environment,
because Windows locks loaded DLLs. The installed command refuses before any
write and prints a path-specific command when this is attempted:

```text
uvx --python 3.11 --from "geochemistrypi-mcp[release] @ file:///ABSOLUTE/PATH/geochemistrypi_mcp-0.2.0-py3-none-any.whl" geochemistrypi-mcp-setup rollback
```

Rollback validates the restored runtime and runs Doctor before deleting the
replaced version. If rollback validation fails, the upgraded runtime and the
rollback snapshot are put back. Successful rollback is intentionally one-way:
the replaced version is deleted and `rollback_available` becomes false.

## 5. Data and configuration preservation

Install, repair, upgrade, rollback, and uninstall never move or delete:

- managed run directories and original CLI artifacts;
- MLflow experiments and tracking metadata;
- managed-service logs and state records;
- unrelated keys or servers in client configurations;
- adjacent `.geochemistrypi.bak` client recovery copies.

Only these installer-owned runtime roots can be recursively removed:

- `environments/`;
- `release/`;
- `rollback/`.

Every removal target is resolved and checked to be inside the absolute
GeochemistryPi MCP application root. Uninstall removes the active runtime, the
one retained rollback runtime, settings, the install manifest, and only client
entries that still point to the installed stable command. A user-modified
replacement is not removed.

## 6. Doctor contract

Doctor reports nine named checks and supports `--json`:

1. settings schema, exact private paths, and all six positive resource limits;
2. CLI/MCP versions, compatibility policy, source fingerprint, client list,
   and rollback declaration;
3. active release-manifest identity and both wheel SHA-256 hashes;
4. current CLI and MCP distribution-inventory hashes;
5. writable run, tracking, and service-state directories;
6. Python 3.11 and MCP `0.2.0`;
7. Python 3.9 and CLI `0.8.0`;
8. the public `geochemistrypi --version` command;
9. zero-argument stdio startup and the exact 13-tool set.

For a signed install, Doctor confirms that all retained Sigstore bundles are
present and that the active artifact hashes still match the signed manifest.
Cryptographic identity verification happens before activation; a future
upgrade re-verifies the new bundle before any mutation.

## 7. Developer build and local verification

Build both wheels into one clean directory, create the canonical manifest, and
perform an explicit unsigned local inspection:

```text
python -m build --wheel --outdir release-bundle .
python -m build --wheel --outdir release-bundle packages/geochemistrypi-mcp
geochemistrypi-mcp-release build-manifest --dist release-bundle --release-tag mcp-v0.2.0-cli-v0.8.0 --source-commit uncommitted
geochemistrypi-mcp-release verify --bundle release-bundle --allow-unsigned
```

The normal cross-platform workflow builds the same bundle on Windows, Ubuntu,
and Intel macOS, installs through the wheel bootstrap, and exercises source
install, repeat install, bundle upgrade, rollback, uninstall, clean bundle
install, bundle repeat, repair, Doctor, and final uninstall. Sentinel run and
tracking artifacts must survive the entire lifecycle.

The tag-only `.github/workflows/release.yml` workflow adds keyless Sigstore
signatures using the exact tag workflow identity, verifies the resulting bundle
with the product verifier, generates GitHub artifact attestations, and uploads
the signed candidate as a temporary workflow artifact. It has no PyPI, MCP
Registry, or GitHub Release publication step.

## 8. Publication decision and remaining gates

The current decision is **deferred**. No publication, tag creation, commit,
push, or external client mutation is part of this implementation document.
`public_release_ready` must remain false until evidence for this exact versioned
bundle proves all of the following:

- the complete PR9I matrix on every required platform;
- clean Linux and macOS lifecycle acceptance;
- real natural-language acceptance in at least ten target clients;
- upgrade from the last actually published supported bundle;
- no known critical, high, or medium parity defect;
- successful tag-workflow signatures and release hashes;
- explicit authorization for PyPI and MCP Registry publication.

Local green tests or a locally generated unsigned bundle cannot close those
external gates. Once they are complete, readiness must change in code,
capability output, manifest, documentation, and release notes together.

## 9. Operator failure guide

| Symptom | Safe action |
| --- | --- |
| Signature, identity, size, or hash failure | Do not install; reacquire the complete bundle from the authorized release channel. |
| Setup says `uv` is not `0.11.7` | Install the pinned version, then rerun; do not bypass the check. |
| Upgrade preflight Doctor fails | Repair the current release first; the upgrade has not changed it. |
| Upgrade fails after preparation | Read the reported cause; setup restores the prior runtime automatically. |
| Client entry conflicts | Inspect that client and its `.geochemistrypi.bak`; use repair only when replacing the named entry is intentional. |
| Client registration fails | The healthy private runtime remains and file-backed client configs are restored; fix the named client and rerun setup. |
| Windows reports that the private environment is running | Use the printed external `uv`/wheel bootstrap; setup refused before changing files. |
| Rollback is unavailable | No complete successful-upgrade snapshot exists; repair from a trusted bundle instead. |
| Rollback Doctor fails | The upgraded runtime and rollback snapshot are restored automatically; do not delete the rollback directory manually. |
| Doctor reports inventory drift | Repair from the original verified bundle; do not edit either private environment. |
| Managed UI ownership is ambiguous | Stop or identify the process manually; setup deliberately refuses to kill it. |

This document does not claim remote or real-client gates have passed until their
external results exist. That distinction is part of the release safety model,
not an unfinished documentation detail.

## 10. Local verification evidence (2026-08-04)

The final local implementation was verified on Windows with the database URL
unset and non-interactive plotting enabled:

| Verification | Result |
| --- | --- |
| MCP installation, interaction, and protocol suite from source | 227 passed |
| GeochemistryPi CLI suite from source on Python 3.9.20 | 91 passed |
| Same MCP suite imported from the built MCP wheel outside the repository | 227 passed |
| Same CLI tests imported from the built CLI wheel outside the repository | 91 passed (85 package tests plus 6 CLI-contract/database tests) |
| Formatter, import ordering, whitespace, YAML, file-size, and Flake8 hooks | all passed |
| Product release-manifest verifier | exact two-wheel pair accepted; no packaged tests; sizes and SHA-256 matched |
| Real isolated Windows lifecycle | source install, repeat, bundle upgrade, rollback, uninstall, clean bundle install, repeat, active-bundle repair, and uninstall passed |
| Data-preservation sentinel | run JSON and experiment YAML remained byte-for-byte present after every lifecycle action |

The local final-candidate artifact hashes were:

| Artifact | SHA-256 |
| --- | --- |
| `geochemistrypi-0.8.0-py3-none-any.whl` | `183bcd56a2ee40865992343b4f0d7cad1e12c7c472dc7b1571e9f8c7275eff8f` |
| `geochemistrypi_mcp-0.2.0-py3-none-any.whl` | `78c991c365c9741145a2f5a97780e7ff0f9da4a4221304b562c47fd992005c89` |

Those hashes identify an **unsigned, uncommitted local candidate** only. They
are not public release hashes and must not be copied into release notes as if
the tag workflow had signed them. The authoritative public hashes, if release
is later authorized, must come from the exact tag workflow artifact after
remote platform and acceptance gates pass.

The real lifecycle exposed and closed three Windows-only failure classes that
mocked tests did not reveal: loaded private-environment DLL locks, recovery of
paths that had not actually moved, and repair wheels moving with the active
release directory. Each now has a fail-closed code path plus a regression test.
Windows destructive lifecycle actions require the external wheel bootstrap;
the installed private command refuses before mutation and prints the command.
