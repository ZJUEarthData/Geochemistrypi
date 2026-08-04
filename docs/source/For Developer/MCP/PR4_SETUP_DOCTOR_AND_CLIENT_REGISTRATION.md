# PR4 Setup, Doctor, and Client Registration

## Outcome

PR4 turns the local MCP wrapper into a source-installable product slice. From a
GeochemistryPi repository clone, one setup command prepares both required
Python environments, installs the local sources, persists their relationship,
registers a stable stdio command, and proves the installation through a real
MCP handshake. The package remains local-only and is not released to PyPI.

The scientific boundary is unchanged: GeochemistryPi 0.8.0 remains the only
classification engine, while `geochemistrypi-mcp` 0.2.0 validates requests,
drives the public interactive CLI, manages run state, and exposes original
artifacts.

## One-action local setup

Clone the repository and run setup from its root:

```text
git clone https://github.com/ZJUEarthData/Geochemistrypi.git
cd Geochemistrypi
uv run --isolated --no-project --python 3.11 --with-editable packages/geochemistrypi-mcp geochemistrypi-mcp-setup install
```

The administrative command uses uv to create two installer-owned environments:

| Runtime | Python | Responsibility |
| --- | --- | --- |
| MCP | 3.11 | MCP SDK, wrapper, setup, doctor, and stdio server |
| CLI | 3.9 | Unmodified GeochemistryPi 0.8.0 public CLI |

On Windows, the default root is
`%LOCALAPPDATA%\GeochemistryPi MCP`. On Unix-like systems it is
`$XDG_STATE_HOME/geochemistrypi-mcp` or
`~/.local/state/geochemistrypi-mcp`. The stable client command is the
`geochemistrypi-mcp` launcher inside the private MCP environment and takes no
arguments.

The installer records a source fingerprint. Repeating `install` does not
recreate healthy, unchanged environments. A source change refreshes them, and
`repair` always rebuilds both environments before revalidating the installation.

## Persisted runtime contract

`config/settings.json` contains only installer-owned runtime values:

- settings schema version;
- absolute private GeochemistryPi CLI command;
- absolute managed-runs directory;
- maximum dataset size.

The zero-argument MCP server reads this file at process startup. Explicit
environment variables continue to override it for development and tests, but
normal client configuration contains no Python path, CLI path, run path, shell
wrapper, or environment-variable block.

`config/install-manifest.json` records wrapper and CLI versions, source
fingerprint, stable command, run root, timestamps, and clients registered by
setup. `config/mcp.json` is the client-neutral `mcpServers` fallback.

## Client adapters and recovery

Setup supports 13 named MCP clients in addition to the standard JSON fallback.
They all launch the same zero-argument stdio command; only the host-side
registration differs.

| Client | Adapter schema or interface | Default user target |
| --- | --- | --- |
| Codex | `[mcp_servers.geochemistrypi]` TOML | `~/.codex/config.toml` |
| Claude Desktop | `mcpServers` JSON | Platform Claude application-data directory |
| Claude Code | Supported user-scope CLI | `claude mcp add-json --scope user` |
| Cursor | `mcpServers` JSON | `~/.cursor/mcp.json` |
| VS Code | `servers` JSON with `type: stdio` | Platform `Code/User/mcp.json` |
| Gemini CLI | `mcpServers` JSON | `~/.gemini/settings.json` |
| Windsurf | `mcpServers` JSON | `~/.codeium/windsurf/mcp_config.json` |
| Cline | `mcpServers` JSON | `~/.cline/data/settings/cline_mcp_settings.json` |
| Roo Code | `mcpServers` JSON | Default VS Code Roo extension global storage |
| Zed | `context_servers` JSON | Platform Zed `settings.json` |
| Continue | `mcpServers` YAML sequence | `~/.continue/config.yaml` |
| Kiro | `mcpServers` JSON | `~/.kiro/settings/mcp.json` |
| OpenCode | `mcp.servers` nested JSON | `$XDG_CONFIG_HOME/opencode/opencode.json` |

These formats follow the clients' documented configuration contracts:
[Gemini CLI](https://google-gemini.github.io/gemini-cli/docs/tools/mcp-server.html),
[Windsurf](https://docs.windsurf.com/windsurf/cascade/mcp),
[Cline](https://docs.cline.bot/getting-started/config),
[Roo Code](https://roocodeinc.github.io/Roo-Code/features/mcp/using-mcp-in-roo/),
[Zed](https://zed.dev/docs/ai/mcp),
[Continue](https://docs.continue.dev/reference),
[Kiro](https://kiro.dev/docs/mcp/configuration/), and
[OpenCode](https://opencode.ai/v2/docs/config).

`install` always writes the standard fallback and safely auto-detects installed
clients. `--client` can select one or more clients explicitly; `--client all`
selects every adapter available on the current platform.

Direct JSON, TOML, and YAML changes are atomic. Unrelated keys, comments where
the format permits them, and other MCP servers are preserved. An existing file receives one adjacent
`.geochemistrypi.bak` copy before its first mutation. Normal install refuses to
overwrite a different `geochemistrypi` entry; `repair` is the explicit recovery
path and replaces only that entry. Uninstall removes a client entry only while
it still matches the installed command.

Roo Code can use a custom extension storage path that an external installer
cannot reliably discover; setup therefore targets its default VS Code extension
storage. OpenCode accepts JSONC, but this installer refuses to rewrite a sole
`opencode.jsonc` because a JSON round trip would lose user comments. In that
case it reports a manual MCP-settings action instead of making an unsafe edit.

## Doctor contract

`geochemistrypi-mcp-doctor` returns nonzero when any check fails and supports a
machine-readable `--json` report. It verifies:

1. persisted settings match the installed private paths;
2. the managed-runs directory can be written;
3. the MCP interpreter and installed wrapper version are compatible;
4. the CLI interpreter is exactly Python 3.9 and GeochemistryPi is 0.8.0;
5. the public `geochemistrypi --version` command starts successfully;
6. the zero-argument stdio server initializes and lists exactly the six PR2/PR3 tools.

Setup runs this doctor before reporting success. This checks both actual
process boundaries instead of treating file existence as installation health.

## Repair and uninstall semantics

`repair` rebuilds both private environments, rewrites installer settings,
repairs selected owned client entries, refreshes the manifest, and reruns the
doctor. It is safe to repeat.

`uninstall` removes the two private environments, persisted runtime settings,
the installer manifest, and registered `geochemistrypi` client entries. It
deliberately preserves the runs directory, original scientific outputs,
unrelated client settings, other MCP servers, and recovery backups.
On Windows, private-environment removal uses an extended-length path after the
target has been verified beneath the application root, so dependencies whose
installed paths exceed the legacy 260-character limit can still be removed.

The maintenance commands remain source-based until a later release PR:

```text
uv run --isolated --no-project --python 3.11 --with-editable packages/geochemistrypi-mcp geochemistrypi-mcp-setup repair
uv run --isolated --no-project --python 3.11 --with-editable packages/geochemistrypi-mcp geochemistrypi-mcp-setup uninstall
uv run --isolated --no-project --python 3.11 --with-editable packages/geochemistrypi-mcp geochemistrypi-mcp-setup print-config
```

## Verification added in PR4

Installation tests cover JSON/TOML/YAML preservation, nested schemas, one-time
backups, Windows/macOS/Linux paths, all 13 client adapters, Claude Code CLI
calls, JSONC refusal, collision handling, auto-detection, repeated install,
source refresh, repair, doctor failure recovery, and uninstall data
preservation. Doctor tests cover healthy and failing version/protocol reports.
The cross-platform CI job runs these tests both from an editable installation
and again from the built wheel, alongside all earlier MCP interaction and
protocol tests.

This PR does not add regression, clustering, anomaly detection, decomposition,
time-series, or broader scientific capability. Those remain later roadmap
work; PR4 is limited to making the completed classification reference workflow
locally installable, diagnosable, and recoverable.
