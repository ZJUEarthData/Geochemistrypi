# PR5 Complete Regression Coverage

## Outcome

PR5 exposes all 15 single-model regression families from the existing
GeochemistryPi 0.8.0 public CLI through the same local stdio MCP wrapper used by
classification. GeochemistryPi remains the only scientific execution engine.
The wrapper validates semantic inputs, compiles controlled prompt responses,
starts the public CLI in an isolated workspace, and returns references to the
CLI's original outputs.

The public MCP surface still has six tools. `start_analysis` now accepts a
strict task-discriminated request for either `classification` or `regression`.
Classification requests that predate PR5 remain compatible when `task` is
omitted; new clients should send the task explicitly.

## Versioned model contract

The regression capability matrix is stored in
`tests/mcp_wrapper/parity/fixtures/regression_capability_matrix_v1.json` and is
checked against `REGRESSION_MODELS` in the public CLI source.

| MCP model ID | Existing CLI model | AutoML offered by CLI |
| --- | --- | --- |
| `linear_regression` | Linear Regression | No |
| `polynomial_regression` | Polynomial Regression | No |
| `k_nearest_neighbors` | K-Nearest Neighbors | Yes |
| `support_vector_machine` | Support Vector Machine | Yes |
| `decision_tree` | Decision Tree | Yes |
| `random_forest` | Random Forest | Yes |
| `extra_trees` | Extra-Trees | Yes |
| `gradient_boosting` | Gradient Boosting | Yes |
| `xgboost` | XGBoost | Yes |
| `multi_layer_perceptron` | Multi-layer Perceptron | Yes |
| `lasso_regression` | Lasso Regression | Yes |
| `elastic_net` | Elastic Net | Yes |
| `stochastic_gradient_descent` | SGD Regression | Yes |
| `bayesian_ridge` | BayesianRidge Regression | Yes |
| `ridge_regression` | Ridge Regression | Yes |

Every model has a strict settings schema matching the choices and conditional
prompts offered by the CLI. AutoML requests reject manual hyperparameters.
Linear Regression and Polynomial Regression reject AutoML before process
startup because the public CLI deliberately skips the AutoML prompt for those
two models.

## Regression request contract

A regression request supplies:

- one explicit CSV or XLSX training path;
- safe experiment and run names;
- one unique identifier column;
- one or more numeric feature columns;
- exactly one finite numeric target column;
- optional application data;
- explicit missing-value, feature-engineering, scaling, feature-selection,
  split, tuning, and model choices.

The wrapper rejects conditions known to fail or diverge in the existing CLI,
including:

- missing, duplicate, or conflicting column roles;
- non-numeric or non-finite features and targets;
- duplicate or missing identifiers;
- a split with fewer than 10 training rows, because the existing regression
  workflow always performs 10-fold cross-validation;
- KNN neighbor counts larger than the resulting training split;
- a Poisson decision tree with negative targets;
- feature-selection counts or tree feature counts outside the available
  processed feature set;
- unprocessed missing values for any model other than XGBoost;
- training/application preprocessing mismatches;
- unsafe or target-leaking engineered-feature formulas.

These checks do not recalculate or replace scientific results. They only stop a
known-invalid CLI interaction before expensive training starts.

## Interaction-plan coverage

`RegressionPlanCompiler` reuses the validated common CLI flow for dataset
selection, missing values, feature engineering, feature scaling, feature
selection, data splitting, and optional application data. It then compiles the
regression task number, one of 15 model numbers, conditional AutoML selection,
and every material manual hyperparameter branch.

Five linear-family models ask additional plot-dimension questions when more
than one processed feature remains:

- Linear Regression;
- Lasso Regression;
- Elastic Net;
- SGD Regression;
- Ridge Regression.

The plan derives these responses from the final feature count after feature
engineering and feature selection. With two features it selects one feature
for the 2D plot. With more than two it also selects two distinct features for
the 3D plot. This dynamic branch was confirmed against a real CLI run.

## Original outputs and parity

MCP does not create regression models, predictions, metrics, or plots. The
existing CLI continues to create and own:

- the selected, transformed, train, test, and application datasets;
- `Y Train Predict.xlsx`, `Y Test Predict.xlsx`, and optional
  `Application Data Predicted.xlsx`;
- model score and 10-fold cross-validation files;
- hyperparameter files;
- Predicted-vs-Actual, Residuals, and Permutation Importance plots;
- model-specific plots or formula files;
- the trained model joblib;
- `Transform Pipeline.joblib`;
- the copied `summary` directory and MLflow tracking data.

The PR5 parity test runs the same Linear Regression request directly through
the public CLI and through a real stdio MCP server. It compares the complete
output file list, score, cross-validation, hyperparameters, test predictions,
application predictions, common plots, input hashes, and artifact count.

## Explicit boundaries

PR5 intentionally does not expose:

- the aggregate `all_models` branch, whose nested output layout differs from a
  reproducible single-model run;
- multiple regression target columns, because the current CLI contains only
  partial multi-target handling and the wrapper contract is one numeric target;
- previous-experiment attachment, because an MCP run always uses explicit new
  experiment and run names;
- AutoML for Linear Regression or Polynomial Regression;
- clustering or later task families from PR6 onward.

The package remains source-installable and local-only. Nothing in PR5 publishes
the package to PyPI or an MCP registry.

## Verification

PR5 adds a versioned 15-model capability fixture, manual-plan coverage for all
15 models, AutoML-plan coverage for the 13 supported models, conditional model
and plot prompt tests, invalid scientific-input tests, protocol discovery and
dispatch tests, training/application-data parity, and a real direct-CLI versus
stdio-MCP parity test.

The completed local verification checkpoint on 2026-08-02 is:

- 49 GeochemistryPi core and CLI-contract tests passed under Python 3.9 with
  database configuration absent;
- 125 MCP tests passed, including all 3 real direct-CLI-versus-MCP parity
  scenarios;
- the new regression parity scenario completed both executions, compared the
  full file inventory and selected scientific artifacts, and preserved the
  original CLI model and plot outputs;
- `pre-commit run --all-files` passed;
- the production MCP wheel built successfully with 21 entries, including the
  regression contract and zero repository test paths;
- that wheel imported from a clean Python 3.11 `site-packages` environment and
  passed all 122 non-parity MCP tests, with only the 3 external real-CLI parity
  tests deliberately deselected;
- `git diff --check` passed apart from Git's informational line-ending
  warnings.

Remote CI was not run because this local worktree has not been committed or
pushed. Re-run the live verification ladder after later changes rather than
treating these recorded counts as permanent execution evidence.
