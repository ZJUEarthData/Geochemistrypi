# Characterization tests

Place tests here when their purpose is to freeze an observable legacy CLI or
`data_mining` behavior before changing it.

Characterization results must not be treated as proof of scientific
correctness. If a scientifically necessary fix changes a result, retain the old
expectation as documented compatibility evidence and add the corrected
expectation under `tests/scientific/`.
