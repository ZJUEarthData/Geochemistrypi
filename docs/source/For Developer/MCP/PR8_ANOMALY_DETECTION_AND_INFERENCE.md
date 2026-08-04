# PR8 Anomaly Detection and Remaining Inference Paths

## Outcome

PR8 completes the five main task families in the GeochemistryPi 0.8.0 public
CLI. The local stdio MCP wrapper now exposes Isolation Forest and Local Outlier
Factor through a strict, target-free anomaly-detection request while preserving
the existing CLI as the only scientific execution engine.

Across classification, regression, clustering, decomposition, and anomaly
detection, the wrapper now covers all 36 public single-model menu families.
Classification and regression application-data inference are also protected by
real direct-CLI-versus-MCP evidence. Unsupervised application inference is not
invented because the public CLI does not expose that workflow.

## Root CLI repairs

Real Isolation Forest characterization exposed two pre-existing core failures.
When bootstrap sampling was disabled, the CLI skipped the maximum-sample prompt
but passed `None` to scikit-learn; Isolation Forest requires its public default
`"auto"` instead. After fitting, the workflow stored the complete anomaly
result table where downstream diagrams expected the one-dimensional `-1`/`1`
prediction labels. The density plot also interpreted the label polarity as
`0`/`1`, contrary to the detector contract.

The core repair preserves `"auto"`, stores the prediction labels as an indexed
`is_abnormal` series, and treats `1` as normal and `-1` as anomalous. Focused
regression tests protect all three conditions. Direct public-CLI Isolation
Forest and Local Outlier Factor runs now complete and create their original
outputs.

## Versioned model contract

The capability matrix is stored in
`tests/mcp_wrapper/parity/fixtures/anomaly_detection_capability_matrix_v1.json`
and checked against the public CLI source.

| MCP model ID | Existing public CLI model | Exposed parameters |
| --- | --- | --- |
| `isolation_forest` | Isolation Forest | estimators, contamination, maximum features, bootstrap, conditional maximum samples |
| `local_outlier_factor` | Local Outlier Factor | neighbors, leaf size, Minkowski power, contamination, parallel jobs |

Every request names exactly one model. Targets, aggregate-model execution,
supervised feature selection and splitting, AutoML, application-data inference,
unresolved missing values, and previous-experiment attachment are rejected
before process startup.

## Validation and interaction coverage

Before execution, the wrapper checks that identifiers are unique and
non-missing; selected features are numeric and finite; missing-value handling
retains usable rows; and the model parameters are compatible with the retained
dataset. Isolation Forest maximum features cannot exceed the final feature
count, bootstrap maximum samples cannot exceed the retained row count, and
Local Outlier Factor neighbors must be smaller than the retained row count.

The interaction compiler covers the real anomaly-detection mode number, both
model numbers, scaling, exact method-specific prompts, optional feature
engineering, conditional two- and three-dimensional diagrams, Local Outlier
Factor's score diagram, model saving, and transform-pipeline completion. The
driver continues to fail closed if the CLI prompt contract drifts.

## Original outputs and parity

The CLI remains the sole owner of:

- `X Abnormal Detection.xlsx`, `X Normal.xlsx`, and `X Abnormal.xlsx`;
- density-estimation and two-/three-dimensional plots and tables;
- the Local Outlier Factor score plot and table;
- trained model and hyperparameter files;
- transform-pipeline configuration and optional fitted pipeline;
- summaries and MLflow tracking data.

The PR8 parity scenario runs the same Isolation Forest request once by feeding
the compiled answers directly to the installed public CLI and once through a
real stdio MCP server. It compares the complete output inventory, input hash,
transformed feature data, normal and anomalous rows, prediction labels,
hyperparameters, plots, task/result semantics, and artifact count.

## Remaining inference audit

PR8B audited the existing training-fitted transform replay and application-data
validation paths. Regression application-data prediction already had a real
stdio parity scenario. Classification had the real driver path but lacked the
same end-to-end protocol comparison, so PR8 adds it. The test compares the
training and application input hashes, engineered selected-feature workbook,
application prediction workbook, output inventory, and continued MCP protocol
health.

This evidence preserves the public CLI's classification output representation;
the wrapper does not decode, recalculate, or replace predictions. Clustering,
decomposition, and anomaly detection remain training-only because their public
CLI menus do not offer application-data inference.

## Verification

The completed local verification checkpoint on 2026-08-03 includes 55
GeochemistryPi core and CLI-contract tests and 163 MCP tests, of which six run
the real direct-CLI-versus-stdio-MCP comparison. The final pre-commit run made
no changes. The 24-entry MCP wheel and 179-entry core wheel contained no
repository tests; a clean MCP-wheel installation passed 157 non-parity tests,
and the installed core wheel passed Python 3.9 anomaly smoke checks.

This document describes local uncommitted work. It does not claim remote CI, a
commit, a push, publication, or a public release.
