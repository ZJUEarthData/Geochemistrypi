# GeochemistryPi MCP Implementation Requirements and Boundaries

> Status: Mandatory companion to the replacement development roadmap
> Created: 2026-08-02
> Audience: maintainers, contributors, reviewers, release engineers, and
> AI-assisted development sessions

## 1. How to Use This Document

This document records the product requirements, engineering boundaries, quality
expectations, and collaboration rules that must remain stable while implementing
GeochemistryPi MCP.

At the start of every new development session, read these two files completely:

1. `md/GeochemistryPi_Local_MCP_Platform_Development_Roadmap.md`
2. `md/GeochemistryPi_MCP_Implementation_Requirements.md`

The roadmap defines implementation order and PR deliverables. This document
defines the non-negotiable product direction and working rules.

The replacement roadmap supersedes every older MCP design that proposed a new
machine-learning service, model adapters, worker packages, a contracts package,
or a runtime package. Older PR numbering or handoff notes must not override the
replacement roadmap. Their safety and scientific lessons may still be valid,
but their obsolete implementation direction is not.

The current user's explicit request defines the immediate task. It does not
silently authorize commits, pushes, history rewrites, releases, unrelated
refactors, or changes to the CLI scientific workflow.

## 2. Product Goal

GeochemistryPi MCP exists to make the validated GeochemistryPi CLI usable from
natural-language conversations with any compatible MCP client.

The intended user experience is:

```text
1. Perform one setup action.
2. Open any supported MCP client.
3. Describe a geoscience task in natural language.
4. Let the client call GeochemistryPi MCP automatically.
5. Receive a concise scientific result and references to the complete original
   GeochemistryPi outputs.
```

The product has two primary benefits:

- users do not need to open the interactive CLI or write machine-learning code;
- clients call a validated and reproducible local tool instead of generating a
  new machine-learning implementation, reducing token use and execution risk.

Convenience is valid only when correctness remains owned by the existing CLI.

## 3. Fundamental Architecture Rule

The existing `geochemistrypi data-mining` CLI is the only machine-learning
implementation.

The required execution path is:

```text
MCP client
  -> geochemistrypi-mcp
  -> validated semantic request
  -> prompt-synchronized CLI driver
  -> installed public geochemistrypi command
  -> original CLI outputs
```

GeochemistryPi MCP may validate requests, inspect bounded dataset metadata,
translate semantic choices into CLI interactions, manage subprocesses, record
wrapper state, parse bounded values from existing files, and return artifact
references.

GeochemistryPi MCP must never:

- implement or import an alternative training pipeline;
- call GeochemistryPi model or preprocessing classes directly;
- call the FastAPI classification route as a substitute for the CLI;
- import scikit-learn, XGBoost, FLAML, Ray, or MLflow training APIs;
- calculate replacement metrics;
- create replacement plots, predictions, models, reports, or processed data;
- silently correct or compensate for a result produced by the CLI;
- hard-code expected scientific results into production code;
- add test-only behavior to production execution.

When an MCP result differs from a direct CLI result, first determine whether the
cause is the wrapper, the test fixture, the environment, or the CLI itself. Fix
wrapper defects in the wrapper. A genuine CLI scientific defect requires a
separate core change with explicit user authorization and scientific review.

## 4. Preserve the Existing CLI Product

MCP work must not change how existing CLI users install, launch, or operate
GeochemistryPi.

Do not change, as part of MCP implementation:

- the `geochemistrypi` command;
- the `data-mining` command or its options;
- the interactive menu sequence;
- existing preprocessing and model defaults;
- metric calculations or plot generation;
- MLflow behavior;
- the existing CLI output structure;
- the normal human CLI workflow.

The MCP driver must operate the public installed command as a black box. Any
necessary compatibility check must happen around the process, not by replacing
its scientific internals.

## 5. Output Ownership and Parity

The CLI remains the sole owner of scientific outputs. Successful MCP-started
runs must preserve the original directories:

```text
artifacts/
metrics/
parameters/
summary/
```

The CLI's original plots, processed datasets, split records, predictions,
metrics, parameters, models, transform pipeline, summaries, and MLflow records
must remain available exactly as the CLI writes them.

Wrapper metadata stays separate and may contain request state, status, bounded
logs, an interaction trace, input hashes, and an artifact index. Wrapper files
must not replace, rename, move, or masquerade as CLI scientific files.

For every supported workflow:

- characterize the direct public CLI branch first;
- drive the same installed CLI through MCP;
- compare the output tree and scientifically meaningful values;
- verify input files remain unchanged;
- mark a capability supported only after its required parity evidence passes.

Plan-compilation tests are necessary but are not a substitute for representative
real CLI and MCP end-to-end tests.

## 6. Installation and Environment Contract

The internal architecture contains two components because their dependency
requirements differ:

```text
MCP environment
  - Python 3.10+
  - official MCP SDK
  - Pydantic 2 request validation
  - no scientific training dependencies

CLI environment
  - supported Python for GeochemistryPi 0.8.x
  - installed GeochemistryPi distribution
  - existing scientific dependencies
```

During development before PR4, developers may need to prepare and connect both
environments manually.

The final user experience must not expose this separation. A user performs one
setup action. Setup then prepares or locates the lightweight MCP environment,
prepares a private version-pinned CLI environment, verifies both components,
runs real smoke tests, and registers one stable MCP server command.

Therefore:

- under the hood, both `geochemistrypi-mcp` and `geochemistrypi` are present;
- the public user does not install or connect them manually;
- the user never supplies interpreter, executable, Engine, or runs-root paths;
- normal server startup takes no scientific or environment arguments;
- setup and runtime details remain invisible after successful setup.

Until the implementation is complete, installation must use the local
repository or an approved GitHub revision. Do not publish to PyPI or an MCP
registry without an explicit release decision.

## 7. Client-Neutral Integration

The MCP server is not a Codex-only, Claude-only, Cursor-only, or VS Code-only
product.

Core tools, schemas, instructions, run control, and scientific behavior must be
client-neutral. Client names belong only in setup adapters.

Setup should support:

- a standard `mcpServers` JSON fallback;
- tested adapters for specific compatible clients;
- a clear configuration snippet when automatic registration is unavailable.

Adding a new client adapter must not change server-side scientific behavior.
The setup process must preserve unrelated client configuration, make an atomic
update, create a recoverable backup, validate the result, and support repair and
uninstall.

## 8. User Data and Local File Rules

Users may explicitly reference any supported absolute local file path that the
current operating-system account can read.

Do not require a global trusted-data root. Do not require users to copy files
into the repository or a special workspace. Do not silently copy a dataset to
bypass a path restriction.

Before execution:

- require and resolve an absolute path;
- accept only supported regular files;
- reject directories, devices, sockets, broken links, and unsupported formats;
- enforce configured size and column limits;
- inspect input read-only;
- record a SHA-256 hash.

Verify the hash again immediately before execution and after execution. Treat a
changed input as an integrity failure.

MCP-started outputs belong inside the managed run workspace. The user cannot
select an arbitrary output directory through an analysis tool. Artifact reads
are run-scoped, bounded, and must never automatically load pickle or Joblib
files.

Complete datasets, binary models, and images are not returned inline to the
language model. Return bounded summaries and local artifact references to
protect privacy and reduce token usage.

## 9. Semantic Tool Contract

The public MCP interface accepts scientific intent, not implementation details.

Allowed inputs include task-specific choices such as dataset paths, identifier,
features, target, label handling, preprocessing, model, tuning, and inference
requirements.

Public analysis tools must not accept:

- raw CLI prompt identifiers or response sequences;
- shell commands or Python code;
- arbitrary command-line arguments;
- executable paths;
- environment variables;
- process IDs;
- arbitrary output paths;
- unrestricted file-read paths.

Schemas must reject unknown fields. Defaults must match the documented CLI
behavior or a separately documented and tested MCP policy. Unsupported branches
must fail before execution with a specific, user-actionable explanation; they
must never be silently ignored.

The driver must fail closed on prompt drift, reordered prompts, missing prompts,
unused responses, unsupported CLI versions, input changes, premature process
exit, or output-path violations. It must never guess the next response.

## 10. User-Centered Engineering Standard

Every discovered problem must be resolved from the user's perspective and at
its real boundary. Do not add a narrow patch that only makes one test pass.

For each defect:

1. reproduce the user-visible failure;
2. identify the actual owning layer;
3. inspect the direct CLI behavior before changing the wrapper;
4. implement the smallest complete root-cause fix;
5. add a regression test that fails if the problem returns;
6. test adjacent branches and installation behavior when relevant;
7. explain the purpose and effect in simple language.

Examples of unacceptable shortcuts include:

- recreating missing CLI outputs in MCP;
- suppressing an error and returning partial success;
- accepting an unsupported parameter but ignoring it;
- using a perfectly separable toy dataset to hide evaluation problems;
- changing a Golden file only to make CI green;
- weakening validation because one client sends an inconvenient request;
- hard-coding one client, one machine, one username, or one local path;
- leaving redundant implementations after the accepted route changes.

Remove obsolete or redundant code only after proving it is not part of the
current accepted implementation or user-owned work. Destructive cleanup must be
precisely scoped and recoverable.

## 11. Scientific and Baseline Safety

Keep wrapper equivalence and scientific correctness as two separate questions.

- Characterization and parity tests prove that MCP operates the existing CLI.
- Scientific tests prove invariants of the GeochemistryPi core.

Do not claim that matching a legacy result proves scientific correctness. Do
not change a characterized CLI result inside MCP.

Preserve these existing safeguards:

- supervised learned preprocessing is fitted on training data only;
- test, display, and application data reuse training-fitted transforms;
- identifiers, row alignment, feature order, split membership, and target
  traceability remain intact;
- Golden data records provenance, hashes, seeds, versions, and tolerances;
- missing database configuration does not break pure imports or unrelated
  tests;
- production wheels do not package repository test modules.

## 12. Git and Workspace Control

Implementation permission means edit and test only unless the user explicitly
authorizes another Git operation.

Never perform any of the following without explicit authorization for that
specific operation:

- `git add` or staging files;
- creating a commit;
- pushing or force-pushing;
- rebasing or rewriting history;
- reverting a commit;
- deleting a branch, recovery branch, stash, or material file;
- resolving a destructive conflict by assumption;
- publishing a package or creating a release.

Starting or completing a roadmap PR does not authorize a commit. Leave changes
uncommitted when requested.

Continue to implement and review the roadmap one PR scope at a time even when
the user chooses to combine several completed PR scopes into one later commit.
The absence of an intermediate commit is not permission to skip acceptance
criteria, repeat completed work, or merge unrelated future scope into the
current task.

Always:

- inspect the branch, upstream, status, and recent history first;
- preserve unrelated tracked and untracked user changes;
- treat `.gitignore` and local memory/configuration files as user-owned unless
  the current task explicitly includes them;
- keep `pull.ff=only` enabled for this clone;
- use system temporary directories for build and test environments;
- keep `.venv`, build output, lock files created only for local verification,
  logs, and generated run artifacts out of the repository;
- use `apply_patch` for deliberate file edits;
- explain any operation that could open a GitHub authentication prompt before
  running it.

Never use a successful local history rewrite as proof that GitHub history was
updated. Remote verification is mandatory after any explicitly approved push.

## 13. Documentation Rules

Shared project documentation must be written in clear English for developers,
maintainers, reviewers, or users. Use human-facing titles rather than internal
handoff labels.

Tests, fixtures, Golden files, schemas, architecture documents, PR baselines,
and this requirements document are developer-shareable materials. Private
reasoning, temporary AI checklists, local codebase-memory indexes, credentials,
and machine-specific notes must stay outside commits or in local excludes.

Documentation must describe the actual implemented state. Clearly distinguish:

- implemented and verified;
- implemented but awaiting remote CI;
- deliberately unsupported;
- planned for a later PR;
- deferred until public release.

Do not describe the final one-action setup as already available before PR4 has
implemented and tested it.

## 14. Required Start-of-Task Procedure

Every new session must perform this sequence before editing:

1. Read the complete replacement roadmap and this document.
2. Read repository `AGENTS.md` and the mandatory GeochemistryPi task-safety
   skill.
3. Inspect `git status --short --branch`, branch name, upstream, `pull.ff`, and
   recent history.
4. Preserve and classify every existing tracked and untracked change.
5. Read the completed PR baseline documents relevant to the requested stage.
6. Use codebase-memory to inspect or refresh the code graph before code
   discovery; fall back to exact local inspection only when needed.
7. Determine the real current checkpoint from code, tests, documents, and Git;
   do not rely only on a roadmap's historical “Immediate Next Actions” section.
8. State the requested PR's product purpose and smallest acceptance criteria.
9. Implement only the requested scope and its necessary regression protection.
10. Run the required verification ladder and report the final Git state.

If the current checkpoint is unclear, investigate before changing files. Do not
restart completed PRs or copy code from an obsolete branch.

## 15. Verification Requirements

Choose checks proportional to the change, but Python, packaging, CLI
interaction, scientific, database, setup, CI, and history work require the full
relevant ladder:

1. Review the complete relevant diff, `git diff --check`, and `git diff --stat`.
2. Run focused regression tests for the changed behavior.
3. Run the complete supported GeochemistryPi core suite with database
   configuration absent.
4. Run the MCP interaction and protocol suites in the MCP environment.
5. Run real direct-CLI-versus-MCP parity for affected workflows.
6. Run `pre-commit run --all-files` until it exits zero without modifying files.
7. Build each affected production wheel in a temporary directory.
8. Inspect wheel contents and prove repository tests are excluded.
9. Install the built wheel into a clean temporary environment and run tests
   against the installed artifact.
10. For setup work, test clean install, repeated setup, doctor, repair,
    configuration preservation, uninstall, and failure recovery.
11. For CI work, verify the remote workflow reaches a successful terminal state
    after an authorized push; local success is not enough.
12. Recheck status, staged files, temporary artifacts, CLI source protection,
    and the exact HEAD before handoff.

Never report a command as successful before its final exit code is known. Record
meaningful counts such as tests passed, output files compared, and packaged test
paths found.

## 16. Required End-of-Task Report

Every implementation handoff must explain:

- the outcome in simple language;
- the purpose from the final user's perspective;
- every changed file and why it changed;
- exact validation commands and final results;
- unsupported or deferred boundaries;
- remaining risks, including remote CI not yet run;
- whether any files were staged;
- whether anything was committed, pushed, published, deleted, or rewritten.

Do not claim that all bugs are impossible. Claim only the behavior established
by completed evidence. Do not call a problem root-cause-fixed without a
regression test that would fail if it returned.

## 17. Current Local Checkpoint

This section is a handoff snapshot, not a substitute for live inspection.

As of 2026-08-03:

- branch: `feat/geochemistrypi-mcp-wrapper-v2`;
- HEAD: `b4d49ac74580818fa92cacbcd5184bf8601ce2fd`;
- the branch had no configured upstream;
- PR0 through PR8 and the first PR9 release-hardening foundation were
  implemented in the local uncommitted worktree;
- no files were staged;
- no MCP changes had been committed or pushed;
- the existing CLI entry point and complete `cli_pipeline.py` workflow remained
  unchanged by the wrapper work;
- classification, regression, clustering, decomposition, and anomaly detection
  were implemented through the original CLI rather than reimplemented in the
  MCP package;
- PR4's setup, doctor, configuration backup, atomic write, and client-neutral
  registration work was implemented locally;
- PR5 added strict regression requests, exact interaction plans for all 15 CLI
  regression models, application-data prediction, original output discovery,
  and direct-CLI-versus-MCP parity coverage;
- regression intentionally remains one numeric target and one model per run;
  `all_models`, multi-target regression, and previous-experiment loading are
  not silently approximated;
- AutoML is supported for the 13 CLI regression models that expose it; Linear
  Regression and Polynomial Regression remain manual-only because that is the
  real CLI contract;
- PR6 added a target-free clustering request, exact interactions for all five
  models in the public clustering menu, original-output discovery, and real
  KMeans direct-CLI-versus-MCP parity coverage;
- a pre-existing unsupervised core failure was repaired: transform-pipeline
  construction now accepts `y_train=None`, and the behavior is protected by a
  focused unit test and a successful direct public-CLI KMeans run;
- clustering intentionally remains one public model per run and does not expose
  targets, application inference, supervised feature selection or splitting,
  AutoML, unresolved missing values, previous experiments, or internal-only
  OPTICS;
- PR7 added a strict target-free decomposition request, exact public-CLI
  interactions for PCA, T-SNE, and MDS, original transformed-data and plot
  discovery, and real PCA direct-CLI-versus-MCP parity coverage;
- real CLI characterization repaired two pre-existing PCA failures: generated
  principal-component loadings are now retained for plotting, and selected
  component indices are applied correctly to both scores and loadings;
- decomposition intentionally remains one public model per run and does not
  expose targets, supervised feature selection or splitting, AutoML,
  application inference, unresolved missing values, or previous experiments;
- PR8 added a strict target-free anomaly-detection request, exact public-CLI
  interactions for Isolation Forest and Local Outlier Factor, original output
  discovery, and real Isolation Forest direct-CLI-versus-MCP parity coverage;
- real CLI characterization repaired three connected pre-existing anomaly
  failures: Isolation Forest now receives `max_samples="auto"` when bootstrap
  is disabled, downstream plots receive the one-dimensional prediction labels,
  and density grouping uses scikit-learn's `1` inlier and `-1` outlier contract;
- anomaly detection intentionally remains one public model per run and does not
  expose targets, supervised splitting or feature selection, AutoML,
  application inference, unresolved missing values, or previous experiments;
- PR8's remaining-inference audit confirmed that training-fitted transforms are
  replayed for both supervised application-data paths. Regression retained its
  real stdio application parity test, and classification now has equivalent
  direct-CLI-versus-stdio-MCP application parity evidence;
- the wrapper now covers all 36 public single-model menu families across the
  five main CLI task families. Release hardening and real-client/cross-platform
  acceptance remain PR9 work.
- PR9 now advertises a versioned development compatibility policy and exact
  resource limits through `get_capabilities`; public-release readiness remains
  false while the listed release gates are open;
- core package metadata now matches the real Python 3.9-only CLI runtime,
  setup refreshes manifests that predate the compatibility policy, and doctor
  validates the installed manifest before reporting health;
- local work submission is bounded to eight active or queued runs with one
  concurrent CLI process and a configurable 900-second total process timeout;
  queue admission is atomic and rejects saturation before creating a run;
- the CI definition now includes the Intel `macos-15-intel` runner beside
  Ubuntu and Windows, avoiding the arm64 `macos-latest` label for the Python
  3.9-only CLI stack, and runs a pinned-uv real setup/install/repair/uninstall
  lifecycle on every matrix system, but no new remote result is claimed without
  an authorized commit and push.

The local verification checkpoint was:

- 55 GeochemistryPi core tests passed with database configuration absent;
- 172 MCP interaction, installation, protocol, driver, and parity tests passed;
- the MCP total includes 6 real direct-CLI-versus-MCP parity tests, including
  classification and regression application-data prediction, KMeans
  clustering, PCA decomposition, and Isolation Forest anomaly detection;
- `pre-commit run --all-files` passed without modifying files on the final run;
- the MCP wheel contained 24 entries, included the anomaly-detection contract,
  and contained zero repository test paths;
- the MCP wheel imported from a clean Python 3.11 environment and passed 166
  non-parity tests with the 6 real CLI parity tests deliberately deselected;
- the GeochemistryPi 0.8.0 wheel contained 179 entries, included the repaired
  anomaly-detection modules, contained zero source-test paths, and passed
  installed anomaly-default and label-semantics smoke checks under Python 3.9;
- wheel metadata now restricts the core to Python `>=3.9,<3.10` and MCP to
  Python `>=3.10,<4`; standard pip rejected the core wheel on Python 3.11;
- a real wheel-bootstrapped temporary Windows setup passed repeated install,
  forced repair, doctor 7/7, and uninstall while preserving managed run data.

Re-run live Git and test checks because this snapshot may become stale after a
commit, merge, new edit, environment change, or new conversation.

Historical "Immediate Next Actions" text describes the starting point of the
replacement plan. At this checkpoint, do not restart completed stages. Determine
the next stage from the actual worktree, completed-stage documents, live tests,
and explicit user request.

## 18. Final Product Definition

The local MCP product is successful only when a user can perform one safe setup
action, ask a compatible client for a geoscience analysis in natural language,
and receive a concise result linked to the complete original GeochemistryPi CLI
outputs.

Internally, the product may maintain two isolated environments. Externally, the
user sees one dependable MCP integration and never needs to operate the CLI,
select menu numbers, write machine-learning code, choose tools, or configure
internal paths.

That convenience must never be achieved by duplicating, weakening, or changing
the validated CLI scientific workflow.
