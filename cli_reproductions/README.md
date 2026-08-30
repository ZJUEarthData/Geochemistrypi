# Local CLI reproduction ledger

All runs start from the original datasets under `D:\AGU\复现`. Paper names are
used only to organize configurations and outputs; scientific execution uses the
public GeochemistryPi CLI and paper-independent framework capabilities.

On Windows, set UTF-8 before interactive or automated runs:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
```

## Validation status

| Paper | Target | Status | Input validation | Output validation |
|---|---|---|---|---|
| Ji et al. (2024) | Figure 3 | Passed | 1,375 rows; SHA-256 `433bb6c8...5458cfc` | confusion-matrix PNG equals MCP |
| Tortelli et al. (2026) | Figure 2b | Passed | 1,017 rows; SHA-256 `43a409a0...c837fa5` | PCA PNGs equal MCP |
| Tao et al. (2021) | Figure 3c coordinates | Passed | 38 rows; SHA-256 `97d7595f...5f031d2` | t-SNE PNG equals MCP |
| Liu et al. (2024) | Figure 3a | Passed | 22,623 rows; SHA-256 `0b3221d6...a17876` | numeric CSV and metrics equal MCP |
| Lu et al. (2025) | Figures 1a/1b | Long-running | 20,127 rows; SHA-256 `42589bbc...47f85b` | exact run stopped after >2 CPU hours |

## Ji et al. (2024): classification

Input sequence: identifier `1`; columns `[2,10]`; keep missing values `2`;
feature engineering `2`; mode `2`; X `[2,9]`; Y `1`; keep/encode labels `1`;
scaling `2`; feature selection `2`; test ratio `0.3`; XGBoost `1`; AutoML `2`;
estimators `100`; learning rate `0.1`; max depth `4`; accept defaults for
subsample, column subsample, alpha, and lambda.

Effective controls: stratified split; split seed 42; model seed 42; 10-fold
cross-validation; no confusion-matrix normalization. Result:
`[[128, 32], [24, 229]]`, accuracy 0.8644067797, F1 0.8910505837.

## Tortelli et al. (2026): PCA

Input sequence: identifier `2`; world map disabled; columns `[6,15]`; feature
engineering `2`; mode `4`; scaling `1`; standardization `2`; PCA `1`;
components `2`; SVD solver `2` (`full`). Explained variance ratios:
0.67357444 and 0.16968870.

## Tao et al. (2021): t-SNE

Input sequence: identifier `3`; world map disabled; columns `[4,126]`; feature
engineering `2`; mode `4`; scaling `2`; t-SNE `2`; components `2`;
perplexity `30`; learning rate Enter (`200`); iterations `500`; early
exaggeration Enter (`12`). Effective model seed: 42.

## Liu et al. (2024): subaerial-proportion time series

Non-interactive parameters: bin width 100 Ma; iterations 100; seed 2025;
age `R_AGE`; maximum age `R_MAX_AGE`; latitude/longitude `LATITUDE`/`LONGITUDE`;
probability `Estimated Proportion of Subaerial Basalts`; identifier `ROCK NAME`;
drop rows missing `MIN_AGE`, `AGE`, or `MAX_AGE`; no feature engineering; no
curve fit.
