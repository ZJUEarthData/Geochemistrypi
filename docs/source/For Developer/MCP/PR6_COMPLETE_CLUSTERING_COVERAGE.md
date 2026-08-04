# PR6 Complete Clustering Coverage

## Outcome

PR6 exposes all five single-model clustering families in the GeochemistryPi
0.8.0 public CLI through the local stdio MCP wrapper. GeochemistryPi remains the
only scientific execution engine: MCP validates semantic inputs, compiles the
interactive answers, runs `geochemistrypi data-mining`, and indexes the CLI's
original files.

`start_analysis` now accepts strict task-discriminated `classification`,
`regression`, and `clustering` requests. Clustering has its own target-free
schema and does not silently inherit supervised controls.

## Root CLI repair

Real KMeans characterization found a pre-existing failure after successful
model training. The unsupervised workflow calls `build_transform_pipeline`
with `y_train=None`, while its multi-output regression guard unconditionally
read `y_train.shape`. This raised `AttributeError` after the CLI had already
written most scientific outputs.

The core repair makes `y_train` explicitly optional and runs the multi-output
branch only when a target exists. A focused regression test proves that the
pipeline can fit with `None` as the target. A direct public KMeans CLI run then
completed successfully and produced its model, labels, scores, plots,
configuration, and transform pipeline.

## Versioned model contract

The clustering capability matrix is stored in
`tests/mcp_wrapper/parity/fixtures/clustering_capability_matrix_v1.json` and is
checked against `CLUSTERING_MODELS` in the public CLI source.

| MCP model ID | Existing public CLI model |
| --- | --- |
| `kmeans` | KMeans |
| `dbscan` | DBSCAN |
| `agglomerative` | Agglomerative |
| `affinity_propagation` | AffinityPropagation |
| `mean_shift` | MeanShift |

Every model has a strict settings schema matching its public CLI prompts,
including conditional DBSCAN metric choices and Minkowski power, KMeans
initialization and algorithm, Agglomerative linkage, AffinityPropagation
affinity, and MeanShift boolean and bandwidth branches.

An OPTICS implementation exists in the core source, but OPTICS is absent from
the GeochemistryPi 0.8.0 public model menu. MCP therefore reports it as an
explicitly unsupported internal-only branch instead of advertising a path the
public CLI cannot select.

## Clustering request contract

A clustering request supplies:

- one explicit CSV or XLSX training path;
- safe experiment and run names;
- one unique, non-missing identifier column;
- at least two final numeric, finite features;
- explicit missing-value, feature-engineering, scaling, and model choices.

The wrapper rejects conditions known to fail or diverge in the existing CLI,
including:

- missing, duplicate, or conflicting column roles;
- non-numeric or non-finite features;
- duplicate or missing identifiers;
- unresolved missing values, because the public clustering menu becomes empty;
- empty or undersized data after row dropping;
- KMeans or Agglomerative data with fewer than 11 rows, because the CLI always
  evaluates silhouette scores for k=2 through k=10;
- cluster counts that do not leave silhouette scoring defined;
- DBSCAN or MeanShift sample settings larger than the retained dataset;
- non-square input for AffinityPropagation with `affinity="precomputed"`;
- unsafe or unavailable engineered-feature references.

These checks do not calculate clusters or replace scientific results. They
only stop known-invalid interactions before the public CLI starts.

## Interaction-plan coverage

`ClusteringPlanCompiler` reuses the validated common flow for identifier and
data selection, missing-value handling, and feature engineering. Its
unsupervised branch then compiles:

- clustering mode number 3;
- optional feature scaling without supervised feature selection or splitting;
- one of the five public model numbers;
- every material manual hyperparameter prompt;
- two distinct feature selections for the 2D plot and three distinct feature
  selections for the 3D plot whenever at least three final features exist;
- completion guards for the model and transform-pipeline construction.

With exactly two final features, the CLI creates the 2D diagram directly and
does not ask dimension questions. With three or more, the compiler answers all
five real plot prompts. No target, AutoML, train/test split, application-data
inference, or supervised feature-selection response is synthesized.

## Original outputs and parity

MCP does not create clustering models, labels, metrics, or plots. The existing
CLI continues to create and own:

- cluster labels and supported cluster centers;
- model score and model-specific metrics;
- 2D and optional 3D cluster diagrams;
- silhouette and model-specific diagrams;
- hyperparameter files;
- the trained model joblib;
- transform-pipeline configuration and, when transformations exist, the
  transform-pipeline joblib;
- copied `summary` files and MLflow tracking data.

The PR6 parity test runs the same three-feature KMeans request directly through
the public CLI and through a real stdio MCP server. It compares the complete
output file inventory, model scores, k=2 through k=10 silhouette scores, cluster
labels, plot presence, input hashes, result semantics, and artifact count. The
test also verifies that the MCP protocol remains healthy after the run.

## Explicit boundaries

PR6 intentionally does not expose:

- the aggregate `all_models` branch, whose nested output layout differs from a
  reproducible single-model run;
- application-data inference, which the public CLI enables only for supervised
  tasks;
- target columns, train/test splitting, supervised feature selection, or
  AutoML;
- unresolved missing values;
- internal-only OPTICS;
- previous-experiment attachment, because MCP runs use explicit new experiment
  and run names;
- decomposition or anomaly-detection families planned for later stages.

The package remains source-installable and local-only. Nothing in PR6 publishes
it to PyPI or an MCP registry.

## Verification

PR6 adds a versioned five-model capability fixture, plan coverage for every
public model, conditional hyperparameter and plot tests, invalid-input tests,
protocol discovery and dispatch tests, run-manager regression protection, a
core unsupervised pipeline regression test, and a real direct-CLI versus
stdio-MCP KMeans parity test.

The completed local verification checkpoint on 2026-08-02 is:

- 50 GeochemistryPi core and CLI-contract tests passed under Python 3.9 with
  database configuration absent;
- 142 MCP tests passed, including all 4 real direct-CLI-versus-MCP parity
  scenarios;
- the new KMeans parity scenario completed both executions and compared the
  complete file inventory and selected scientific outputs;
- `pre-commit run --all-files` passed;
- the production MCP wheel built with 22 entries, included the clustering
  contract, and contained zero repository test paths;
- that wheel imported from a clean Python 3.11 `site-packages` environment and
  passed all 138 non-parity MCP tests, with only the 4 external real-CLI parity
  tests deliberately deselected;
- the GeochemistryPi 0.8.0 wheel built with 179 entries, included the repaired
  inference module, contained zero source-test paths, and passed an installed
  unsupervised transform-pipeline smoke test under Python 3.9;
- `git diff --check` passed apart from Git's informational line-ending
  warnings.

Remote CI is not claimed because this local worktree has not been committed or
pushed. Re-run the live verification ladder after later changes rather than
treating these recorded counts as permanent execution evidence.
