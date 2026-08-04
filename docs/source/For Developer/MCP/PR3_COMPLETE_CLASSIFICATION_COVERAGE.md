# PR3 Complete Classification Coverage

## Product purpose

PR3 turns classification into the first complete MCP reference workflow. A
client can describe a supported classification job with scientific parameters,
and the wrapper converts that request into the public
`geochemistrypi data-mining` interaction. GeochemistryPi 0.8.0 remains the only
owner of preprocessing, training, tuning, inference, metrics, plots,
predictions, models, and the four original output directories.

The wrapper does not import the GeochemistryPi Python package and does not
contain an alternative machine-learning implementation.

## Supported model families

The order is part of the public CLI contract:

1. Logistic Regression
2. Support Vector Machine
3. Decision Tree
4. Random Forest
5. Extra-Trees
6. XGBoost
7. Multi-layer Perceptron
8. Gradient Boosting
9. K-Nearest Neighbors
10. Stochastic Gradient Descent
11. AdaBoost

Every model supports the CLI manual-hyperparameter path and the CLI AutoML
path. Conditional prompts, such as kernel-specific SVM settings, bootstrap
forest settings, and distance-specific KNN settings, are compiled only when
the CLI will ask for them.

## Supported classification interactions

| Area | Supported values |
| --- | --- |
| Labels | encode original, explicit mapping, numeric intervals, quantiles |
| Missing values | reject, keep, drop rows, impute |
| Scaling | none, min-max, standardization, mean normalization |
| Feature selection | none, generic univariate, Select K Best |
| Tuning | manual, AutoML |
| Inference | optional application CSV/XLSX |
| Engineered features | validated arithmetic formulas for training and application inference |

Before starting a subprocess, the compiler checks file and column existence,
numeric feature values, missing targets, label coverage, final class sizes,
stratified split feasibility, feature counts, KNN neighbor counts, model
constraints, and conditional CLI compatibility. Unsupported behavior is an
actionable request error rather than a silently ignored option.

## Deliberate boundaries

- `all_models` is not exposed as one run because the CLI changes to a nested
  output layout. Clients should start one reproducible run per model.
- Sample balancing is not exposed. GeochemistryPi 0.8.0 contains a helper, but
  its public `data-mining` workflow never invokes it; adding it in MCP would
  create a second scientific path.
- A previous MLflow experiment cannot be selected. Explicit experiment and run
  names keep MCP results deterministic and locally discoverable.
- Feature-engineering formulas use readable source-column placeholders in the
  MCP request. The wrapper maps them to the CLI interaction, and the CLI stores
  the resulting name-based expressions for safe reuse on application data.

## Output and integrity contract

Successful runs reference only files originally written under `artifacts`,
`metrics`, `parameters`, and `summary`. Reported metrics are bounded readings
of original CLI text files. Training and application inputs are independently
hashed before execution and verified again before a result is published.

The wrapper resolves the Python interpreter that owns the configured CLI for
both Windows virtual-environment and Conda layouts before checking the installed
GeochemistryPi version. It also projects the longest model-output plot path on
Windows and rejects an unsafe runs root before training starts, rather than
allowing the CLI to fail after producing a partial result tree.

The versioned capability fixture is
`tests/mcp_wrapper/parity/fixtures/classification_capability_matrix_v1.json`.
Tests compare its model order with the CLI constants, compile every model in
manual and AutoML modes, cover materially different prompt branches, verify
strict MCP schemas, exercise application-input integrity, and retain the real
installed CLI/MCP parity test for the reference classification workflow.
