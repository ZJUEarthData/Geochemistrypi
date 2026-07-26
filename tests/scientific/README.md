# Scientific correctness tests

Place independent scientific reference tests and invariants here.

For supervised learning, these tests must verify that splitting happens before
fitting imputation, scaling, feature selection, or resampling. Golden results
must record their generation method, dependency versions, and numeric
tolerances.

## PR1 reference

`data/supervised_preprocessing_reference.csv` is designed to expose leakage:
feature selection fitted on the complete dataset chooses `test_decoy`, while a
training-only fit chooses `signal`.

`golden/supervised_preprocessing_v1.json` records exact split membership,
training-only imputer and scaler statistics, selected features, dependency
versions, and explicit floating-point tolerances.
