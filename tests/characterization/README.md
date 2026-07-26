# Characterization tests

Place tests here when their purpose is to freeze an observable legacy CLI or
`data_mining` behavior before changing it.

Characterization results must not be treated as proof of scientific
correctness. If a scientifically necessary fix changes a result, retain the old
expectation as documented compatibility evidence and add the corrected
expectation under `tests/scientific/`.

## PR1 baseline

`data/legacy_full_data_scaling.csv` and
`golden/legacy_full_data_scaling_v1.json` preserve the historical behavior in
which `StandardScaler` sees the complete dataset before the train/test split.
The outlier is deliberately placed in the eventual test set so the fixture
clearly exposes that behavior.
