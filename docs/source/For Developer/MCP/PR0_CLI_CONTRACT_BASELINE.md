# PR0 CLI Contract Baseline

## Purpose

This baseline defines the existing GeochemistryPi classification behavior that
the future MCP wrapper must preserve. It runs the installed public
`geochemistrypi data-mining` command as a black box. It does not implement an
MCP server, import a second machine-learning pipeline, or change the human CLI
workflow.

The baseline answers one product question: after an Agent invokes the future
wrapper, did the real CLI still produce the same scientifically meaningful
result set that a CLI user receives today?

## Reference run

The reference run uses:

- GeochemistryPi 0.8.0 on supported Python 3.9;
- a deterministic 80-row binary classification fixture;
- seven geochemical oxide features;
- the existing CLI standardization option;
- the existing CLI logistic-regression implementation without AutoML;
- a 20% stratified test split with the CLI's seed of 42;
- the existing label-customization path, keeping and encoding the original
  labels.

The run is started through the installed console entry point in a temporary
working directory. Its input responses are recorded in
`tests/cli_contract/fixtures/classification_interaction_v1.json` as a
versioned, ordered sequence of 41 prompts.

## Fixture provenance

`classification_baseline.csv` is derived from the repository's built-in
`Data_Classification.xlsx`, not from a synthetic or tutorial dataset. The
generation rule is:

1. group by `Label`;
2. sample 40 rows per label with seed `20260801`;
3. sort by the original workbook row index;
4. retain `Label` and seven named oxide columns;
5. create `SampleID` from the original one-based row position.

The Golden metadata records SHA-256 hashes for both the source workbook and the
derived CSV. The contract test hashes the input before and after the CLI run to
prove that the CLI did not modify the user's data.

## Expected product outputs

The run must create the original four top-level directories:

- `artifacts`
- `metrics`
- `parameters`
- `summary`

The normalized manifest contains 124 files. It covers the saved source and
processed data, train/test split, encoded and decoded predictions, target
traceability, model files, transform pipeline, statistical plots, model plots,
metrics, parameters, and the flattened summary copies. The expected files are
defined in `classification_output_manifest_v1.json`.

The reference test-set Accuracy, Precision, Recall, and F1 score are all
`0.875`. This intentionally non-perfect result helps detect accidental use of
a toy dataset or a different classification path. Exact test sample IDs and
predictions are stored in `classification_golden_v1.json`; floating-point
tolerances are explicit in the same file.

## Regression protection

The automated checks verify:

- fixture provenance, class balance, unique identifiers, and input hash;
- the ordered interactive prompt contract;
- successful execution through the public console command;
- the exact output directory and normalized file manifest;
- primary metrics and cross-validation statistics within documented
  tolerances;
- exact split membership and test predictions;
- unchanged hashes for `geochemistrypi/cli.py` and
  `geochemistrypi/data_mining/cli_pipeline.py`;
- API-router imports without database configuration;
- a clear error only when unconfigured database access is attempted;
- real database access when an explicit SQLite URL is supplied;
- exclusion of repository test modules from the production wheel.

The database and wheel checks make the full baseline reliable in CI. They do
not alter classification behavior.

## Running the baseline

From a Python 3.9 environment with the built wheel installed:

```text
python -m pytest tests/cli_contract/test_classification_cli_contract.py
```

Run the complete project verification with database configuration absent:

```text
python -m pytest
pre-commit run --all-files
```

The CI workflow additionally builds the production wheel, inspects its archive
paths, installs it, and then runs the repository tests.

## Updating the baseline

Do not update a Golden file merely to make a failing test pass. A baseline
change requires all of the following:

1. identify an intentional CLI, scientific, or pinned-dependency change;
2. run the public CLI again in an isolated Python 3.9 environment;
3. review the complete output manifest and prediction differences;
4. record the reason, dependency versions, seeds, hashes, and tolerances;
5. keep characterization evidence separate from claims of scientific
   correctness.

MCP implementation begins only after this baseline is green. Future MCP code
must operate this CLI contract instead of replacing it with a new
classification implementation.
