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
| Petrelli et al. (2020) | Figures 5a/6a | Passed | train 850 rows, test 119 rows; SHA-256 `52185c7b...e7e105` / `0218d6f7...179e3d` | both external observed-predicted PNGs, feature-importance PNGs, and metrics equal MCP |
| Ji et al. (2024) | Figure 3 | Passed | 1,375 rows; SHA-256 `433bb6c8...5458cfc` | confusion-matrix PNG equals MCP |
| Liu et al. (2025) | Figure 1b | Passed | 13,569 rows; SHA-256 `37087baf...600265` | confusion-matrix PNG and classification metrics equal MCP |
| Tortelli et al. (2026) | Figure 2b | Passed | 1,017 rows; SHA-256 `43a409a0...c837fa5` | PCA PNGs equal MCP |
| Tao et al. (2021) | Figure 3c coordinates | Passed | 38 rows; SHA-256 `97d7595f...5f031d2` | t-SNE PNG equals MCP |
| Stracke et al. (2022) | Figure 5a coordinates | Passed | 2,775 rows; SHA-256 `89975795...57ba2` | t-SNE PNG equals MCP |
| Sharapatov et al. (2025) | Figure 3a | Passed | 3,112 rows; SHA-256 `5a3f7220...21cb3e` | anomaly tables numerically equal MCP; final overlay PNG equals MCP |
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

## Petrelli et al. (2020): Extra-Trees regression

Prepare `SM Tab3` from `GlobalDataset_Final_rev9_TrainValidation.xlsx` and
`SM Tab4` from `GlobalDataset_Final_rev9_Test.xlsx` using header rows `0,1`,
separator ` | `, forward-filled empty headers, suffixed duplicate headers, and
a non-null filter on the corresponding `Sample_ID` column.

Common input sequence: identifier `1`; world map disabled; process missing
values `1`; impute `2`; constant imputation `4`; fill value Enter (`0`);
feature engineering `2`; regression `1`; X `[1,22]`; Y `23`; scaling `1`;
standardization `2`; feature selection `2`; Extra-Trees `7`; AutoML `2`;
estimators `550`; max depth Enter (`None`); minimum split `2`; minimum leaf
`1`; maximum features `22`; bootstrap `2`; out-of-bag score `2`. Effective
model seed: 280; cross-validation: 10 folds; external labeled evaluation.

Pressure Figure 5a selects `[2,13]; [15,24]; 26` before segmentation and applies
`P_GPa * 10`. External result: R2 0.9294430768, RMSE 2.9941394566.
Temperature Figure 6a selects `[2,13]; [15,24]; 27` and uses no target
transformation. External result: R2 0.9326977517, RMSE 53.8767240197 K.

## Liu et al. (2025): XGBoost classification

Input sequence: identifier `1`; world map disabled; columns `[2,29]`; keep
missing values `2`; feature engineering `2`; classification `2`; X `[2,28]`;
Y `1`; keep and encode labels `1`; scaling `2`; feature selection `2`; test
ratio `0.3`; XGBoost `1`; AutoML `2`; estimators `100`; learning rate `0.6`;
maximum depth `7`; subsample Enter (`1`); column subsample Enter (`1`); L1
regularization `0.1`; L2 regularization Enter (`1`). Effective split and model
seeds: 42; random holdout; 10-fold cross-validation; no confusion-matrix
normalization.

Result: confusion matrix `[[1399, 129], [60, 2483]]`; accuracy 0.9535740604;
precision 0.9506125574; recall 0.9764058199; F1 0.9633365664.

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

## Stracke et al. (2022): t-SNE

Input sequence: identifier `1`; world map disabled; columns `[5,10]`; feature
engineering `2`; mode `4`; scaling `1`; mean normalization `3`; t-SNE `2`;
components `2`; perplexity `30`; learning rate Enter (`200`); iterations
`1000`; early exaggeration Enter (`12`). Effective model seed: 42.

The prepared input combines `Data_MORB` and `Data_OIB`, retains `SAMPLE ID`,
`Location`, `lineage`, and the six isotope variables, then removes rows lacking
`Location` or any isotope variable. The local CLI and MCP runs produced the
same primary t-SNE PNG (SHA-256
`1e00a2fee8256430d1d7eb4b4cd1d0ef20220315d9a2a12f92ad6859a616ecb0`).

## Sharapatov et al. (2025): anomaly detection and PCA overlay

Isolation Forest input sequence: identifier `1`; world map disabled; columns
`[2,139]`; feature engineering `2`; anomaly detection `5`; scaling `1`;
standardization `2`; Isolation Forest `1`; estimators `100`; contamination
`0.05`; maximum features `138`; bootstrap `2`; 2-D plot features `1`, `2`;
3-D plot features `1`, `2`, `3`. Result: 156 anomalies and 2,956 normal rows.

PCA input sequence uses the same identifier, selected columns, feature
engineering, and standardization choices; dimensional reduction `4`; PCA `1`;
components `2`; SVD solver `2` (`full`). Explained variance ratios: 0.06759721
and 0.05062610.

Final overlay command joins PCA `X Reduced.xlsx` and Isolation Forest
`X Abnormal Detection.xlsx` on `Name`; x/y columns are `Principal Axis 1` and
`Principal Axis 2`; label column is `is_abnormal`; positive anomaly value is
`-1`. The final PNG SHA-256 is
`99634c394bb62cb5201783270fad71832fc2b71e5a142271d300f824764ac01f`.
