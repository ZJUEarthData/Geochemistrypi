# PR9E–PR9G: Map, Time Series, and All-Models Parity

## Outcome

PR9E–PR9G close three public CLI gaps without moving scientific computation
into the MCP package:

- coordinate-bearing datasets now use an explicit semantic world-map request;
- Time Series has a validated, seeded, noninteractive production command;
- every machine-learning task can select the CLI's real `all_models` branch and
  return bounded parent/child results.

The MCP server continues to validate, launch, monitor, cancel, and index the
installed GeochemistryPi CLI. Model training, plotting, Time Series numerical
work, MLflow runs, and original files remain owned by the CLI.

## PR9E: Semantic world maps

All five machine-learning requests accept a discriminated `world_map` object.
Disabling the map is explicit and never waits for a prompt:

```json
{
  "world_map": {
    "enabled": false
  }
}
```

Enabling it requires semantic coordinate roles. `value_columns` may contain
zero, one, or as many as 20 unique artifact-safe columns:

```json
{
  "world_map": {
    "enabled": true,
    "longitude_column": "LONGITUDE",
    "latitude_column": "LATITUDE",
    "value_columns": ["SIO2(WT%)", "TIO2(WT%)"]
  }
}
```

Before the CLI starts, MCP scans every selected coordinate and projected value.
It rejects missing, nonnumeric, or nonfinite values, longitudes outside
`[-180, 180]`, latitudes outside `[-90, 90]`, missing columns, and empty data.
The CLI repeats the scientific validation before rendering.

Windows and Linux retain Basemap. macOS uses the declared Cartopy dependency.
Neither path invokes a package manager during analysis. Missing dependencies
and renderer failures return actionable errors instead of silently skipping a
requested map.

## PR9F: Production Time Series

The public standalone command is:

```text
geochemistrypi time-series \
  --input Data_Time_Series.xlsx \
  --bin-width 10 \
  --iterations 100 \
  --seed 2025 \
  --experiment-name "Time Series" \
  --run-name "Subaerial Proportion"
```

The command supports CSV and XLSX, explicit column roles, `Ma` or `Ga`, and
`--fit-curve` / `--no-fit-curve`. Interactive menu option 6 calls the same
validated `run_time_series_dataframe` workflow. Menu positions are mapped by
their semantic labels, so the shortened missing-value menu no longer routes
Time Series to decomposition.

The MCP request is discriminated by `task`:

```json
{
  "task": "time_series",
  "training_dataset": {
    "source": "builtin",
    "dataset_id": "builtin:time_series"
  },
  "experiment_name": "Time Series",
  "run_name": "Seeded Bootstrap",
  "bin_width": 10,
  "iterations": 100,
  "seed": 2025,
  "age_column": "R_AGE",
  "maximum_age_column": "R_MAX_AGE",
  "probability_column": "SBAP",
  "latitude_column": "LATITUDE",
  "longitude_column": "LONGITUDE",
  "age_unit": "Ma",
  "fit_curve": true
}
```

Validation covers five distinct required columns, numeric and finite values,
nonnegative ages, maximum age not less than age, probability in `[0, 1]`,
coordinate ranges, positive finite bin width, 1–10,000 bootstrap iterations,
nonnegative seed, at most 10,000 bins, and at least one populated output bin.

The implementation uses a local NumPy random state. It is reproducible for the
same data and seed and does not mutate NumPy's process-global random state. CSV
uses fixed column names, 12-significant-digit formatting, and LF line endings.
PDF metadata removes creation and modification timestamps.

The standard output hierarchy is:

```text
geopi_output/<experiment>/<run>/
├── artifacts/data/Subaerial Proportion.csv
├── artifacts/image/model_output/Subaerial Proportion.pdf
├── metrics/Time Series Metrics.json
├── parameters/Time Series Parameters.json
└── summary/...
```

Parameters record the input path and SHA-256, seed, iterations, bin width,
column roles, unit, and curve choice. MCP indexes these original files and does
not contain a duplicate Time Series calculation.

## PR9G: Exact CLI all-models execution

Each machine-learning request now has:

```json
{
  "model_selection": {
    "mode": "all",
    "tuning": "manual"
  }
}
```

`tuning` may be `manual` or `automl` for classification and regression.
Unsupervised requests reject AutoML because the public CLI does not expose that
combination. When `mode` is `all`, explicitly supplying legacy `model` or
`tuning` fields is rejected so user input is never silently ignored.

The plan compiler selects the real final “All models” menu item. Manual mode
supplies default validated settings for every public model. Supervised AutoML
is selected once at the parent level; regression still runs Linear Regression
and Polynomial Regression manually, matching the CLI's existing exclusions.

The CLI preserves the parent MLflow run, creates one nested run and output
directory per model, and restores the parent output context afterward:

```text
geopi_output/<experiment>/<run>/
├── artifacts/                     # common preprocessing outputs
├── metrics/
├── parameters/
├── summary/Aggregate Model Results.json
├── Logistic Regression/
│   ├── artifacts/
│   ├── metrics/
│   ├── parameters/
│   └── summary/
└── ... one child per public task model
```

One child exception does not discard successful siblings. The atomic aggregate
manifest records the exact ordered model list, `succeeded` or `failed` state,
relative child directory, current artifact count, and a bounded error. MCP
validates the manifest, exact model order, child containment, counts, and files.

An aggregate with no child failures returns run state `succeeded` and aggregate
state `complete`. Any child failure returns terminal run state
`partial_failure`; `get_run_result` remains available and exposes every child.
Cancellation is not caught as a child error: the existing driver still
terminates the recorded CLI process tree.

Artifact discovery now indexes parent files and all direct model-child
`artifacts`, `metrics`, `parameters`, and `summary` trees. Response references
remain bounded, while the durable artifact index retains the complete inventory
up to its existing safety limit.

## Verification boundary

Local Windows evidence covers core scientific and contract tests, all five
aggregate plan families, controlled complete/partial aggregate manifests,
recursive artifact indexing, run lifecycle, and the noninteractive Time Series
CLI. Installed-wheel direct-CLI-versus-MCP parity includes a seeded Time Series
scenario and a real anomaly-detection aggregate run whose Isolation Forest and
Local Outlier Factor child states and complete file inventories match.

Linux and macOS execution is delegated to the existing GitHub Actions OS matrix.
Those remote jobs must actually complete before a release note may claim
three-platform map, Time Series, or aggregate acceptance. The later PR9I parity
matrix remains responsible for exhaustive real execution of every model and all
five aggregate families.
