# Scientific correctness tests

Place independent scientific reference tests and invariants here.

For supervised learning, these tests must verify that splitting happens before
fitting imputation, scaling, feature selection, or resampling. Golden results
must record their generation method, dependency versions, and numeric
tolerances.
