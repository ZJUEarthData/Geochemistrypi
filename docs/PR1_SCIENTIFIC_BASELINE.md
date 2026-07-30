# PR1 scientific baseline and leakage correction

## Purpose

PR1 separates legacy compatibility evidence from scientific correctness
evidence and corrects the confirmed supervised-learning leakage in scaling and
feature selection.

The interactive supervised CLI now performs these operations in this order:

1. select features and the target;
2. split the selected feature matrix into training and final test sets;
3. fit the configured scaler and selector on the training set only;
4. transform training, test, full display, and later application data with
   parameters reproduced from the training-only fit;
5. train and evaluate the selected model.

The unsupervised CLI behavior is unchanged because it has no train/test
evaluation split.

## Baseline responsibilities

- `tests/characterization/` freezes the old full-dataset scaling result. It is
  compatibility evidence, not a scientific target.
- `tests/scientific/` contains a reference dataset designed so that a leaky
  selector chooses `test_decoy`, while a training-only selector correctly
  chooses `signal`.
- Scientific tests also verify train-only imputation and scaler statistics,
  exact split membership, preserved row indexes, and inference schema checks.

## Golden result rules

- sample identifiers, split membership, and selected feature names use exact
  equality;
- deterministic floating-point values use explicit `rtol=1e-12` and
  `atol=1e-12`;
- every Golden file records its source dataset, generation method, random seed,
  dependency versions, and tolerances;
- a Golden result may be changed only with a documented scientific,
  dependency, or platform reason;
- legacy Characterization results must never replace Scientific reference
  results.

## Expected behavior change

Metrics can differ from historical CLI runs when scaling or feature selection
was enabled. This is intentional: the final test set no longer contributes
means, variances, ranges, or feature scores during fitting.

The new reusable preprocessor also supports train-only imputation for the
non-interactive experiment service planned in later PRs. The legacy interactive
missing-value stage still runs before feature and target selection and is
otherwise unchanged in PR1. Moving that interaction into the future
non-interactive experiment contract is intentionally outside this PR.
