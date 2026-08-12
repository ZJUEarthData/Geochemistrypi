# Geochemistry Pi Online MVP

This document describes the first usable Online version. The current goal is a complete calculation loop, not a production deployment or final visual design.

## Current scope

The MVP provides:

- a Vue page at `/online`;
- a lightweight FastAPI application independent of the legacy Dash/auth stack;
- a dynamic chemical-modeling catalog;
- method-level `verified` and `testing` readiness states;
- structured input documentation with formulas, columns, meanings, units, types, examples, and notes;
- `.xlsx` and UTF-8 `.csv` upload with a 20 MB limit;
- a 30-minute hard limit for every calculation process;
- one site-wide calculation slot; additional calculation requests wait in a serial queue;
- synchronous calculation in an isolated job directory;
- result-file download;
- unified supervised-learning Pipelines for regression and classification;
- downloadable `trained_pipeline.joblib` model bundles with feature, target, model, software-version, and training metadata;
- independent Application Data inference through `POST /api/data-mining/inference`, using a training Job ID and an `.xlsx` or `.csv` dataset without requiring the target column;
- readable validation errors;
- backend regression tests.

The fully verified methods are:

```text
task:     algo_kinetic
methods:  first_order, second_order, radioactive_decay
columns:  c0/k/t or n0/decay_const/t, depending on the method

task:     algo_transport
methods:  fick_diffusion, chromatography
columns:  D/dc_dx or tR/sigma, depending on the method

task:     algo_thermodynamic
methods:  vanthoff, activity_coefficient
columns:  K1/dH/T1/T2 or z/ionic_strength, depending on the method
```

For `second_order`, `c0` must be greater than zero, while `k` and `t` must be greater than or equal to zero. The Online validator also rejects empty and non-numeric values before calculation.

Other catalog entries remain visible for inspection but are marked `testing` and cannot run until scientific and interface validation is complete.

### Saved Pipeline and Application Data inference

Every successful regression or classification job saves a single fitted scikit-learn Pipeline. The Pipeline contains the training-set median imputer and the selected estimator, including any estimator-specific transformations such as standardization. Application Data must contain columns whose names match every training feature. Rows with at least one numeric feature are predicted; missing feature values are imputed with the medians learned from the training split. Rows for which every required feature is missing or non-numeric are retained in the downloaded CSV and marked as excluded.

The inference API accepts only a server-created training Job ID. It deliberately does not accept uploaded `.joblib` files because loading an untrusted pickle-compatible artifact would permit arbitrary code execution. Model files should only be loaded in a trusted Python environment with compatible Geochemistry Pi and scikit-learn versions.

## Resource limits

The backend enforces the limits below for both Chemical Modeling and Data Mining. Frontend checks are only an early user-facing warning and are not the security boundary.

- Maximum uploaded dataset: `20 MiB` (`20,971,520` bytes).
- Maximum calculation runtime: `30 minutes` after the task starts running; queue time is not counted. The isolated calculation process is terminated and the API returns HTTP `504` when the deadline is reached.
- Maximum concurrent calculations: `1` for the whole Online instance. Later requests wait and start automatically, one by one, after the running task completes or times out.
- The page polls `GET /api/tasks/{task_id}` for live queue position, execution state, elapsed runtime, and progress-stage display.
- `POST /api/tasks/{task_id}/cancel` removes a queued task or terminates a running calculation process. The next queued task then starts automatically.
- Progress reflects real lifecycle stages (`queued` → `running` → terminal state). Algorithms that do not expose iteration callbacks use an indeterminate running bar rather than a fabricated percentage.
- Catalog, health, and result-download requests do not occupy the calculation slot.

## One-click startup on Windows

From the repository root, double-click:

```text
start-online.cmd
```

The launcher:

1. detects Python 3.11+ and Node.js 20+, reusing compatible installations;
2. installs Python 3.12 or Node.js LTS with Windows WinGet when a compatible version is missing;
3. locates or creates `.venv-online`;
4. installs backend and frontend dependencies when needed;
5. starts FastAPI and Vue in hidden background processes;
6. verifies that the frontend and backend belong to this checkout and the same source build;
7. safely replaces an older Geochemistry Pi instance, then opens the Online page.

If another application occupies port `5173` or `8000`, the launcher exits with the process ID and leaves that unrelated process untouched.

The first automatic software installation may still display a Windows administrator confirmation. If WinGet is unavailable, the launcher reports which prerequisite must be installed manually. To prohibit any automatic installation, run:

```powershell
.\start-online.cmd -SkipInstall
```

To stop processes started by the launcher, double-click:

```text
stop-online.cmd
```

Runtime logs are written to:

```text
runtime/logs/backend.out.log
runtime/logs/backend.err.log
runtime/logs/frontend.out.log
runtime/logs/frontend.err.log
```

## URLs

| Purpose | URL |
|---|---|
| Online page | <http://127.0.0.1:5173/online> |
| API documentation | <http://127.0.0.1:8000/docs> |
| Health check | <http://127.0.0.1:8000/api/health> |

## Manual startup

Create and prepare the Python environment from the repository root:

```powershell
python -m venv .venv-online
.\.venv-online\Scripts\python.exe -m pip install -r requirements-online.txt
```

Install frontend dependencies:

```powershell
cd geochemistrypi\frontend
pnpm install
cd ..\..
```

Start the backend in the first terminal:

```powershell
.\.venv-online\Scripts\python.exe -m uvicorn geochemistrypi.online.app:app --host 127.0.0.1 --port 8000
```

Start the frontend in the second terminal:

```powershell
cd geochemistrypi\frontend
pnpm start -- --host 127.0.0.1 --port 5173
```

## Using the verified calculation

1. Open the Online page.
2. Select `kinetic`.
3. Select `First-order kinetics`.
4. Select `Any`.
5. Upload an `.xlsx` workbook or UTF-8 `.csv` file containing `c0`, `k`, and `t` columns.
6. Select **Start calculation**.
7. Download `first_order_results.xlsx` after completion.

## Verification commands

Backend tests:

```powershell
.\.venv-online\Scripts\python.exe -m pip install -r requirements-online-dev.txt
.\.venv-online\Scripts\python.exe -m pytest tests\test_online_api.py -q
```

Frontend type check and production build:

```powershell
cd geochemistrypi\frontend
pnpm run build
```

Current expected result:

```text
backend: all Online API and task-runner tests pass
frontend: production build succeeds
```

## Known MVP limitations

- calculations still keep the HTTP request open while running, up to the enforced 30-minute deadline;
- job metadata exists only in the filesystem and is not stored in a database;
- there is no user login or per-user authorization in the new lightweight API;
- uploaded files and results have no automatic retention or cleanup policy;
- Chemical Modeling accepts `.xlsx` and comma-delimited UTF-8 `.csv` datasets;
- verified methods are currently limited to first-order kinetics, second-order kinetics, radioactive decay, Fick diffusion, chromatography plate number, the van't Hoff equation, and the current simplified activity-coefficient model;
- input documentation for methods other than the four kinetic entries is still being organized;
- method-specific optional parameters are not yet described by the catalog;
- `algo_solubility` is marked unavailable when `scikit-learn` is not installed;
- the page is functional but not the final visual design.

## Improvement backlog

### P0 — required before multi-user or public deployment

1. Define and verify input columns, optional parameters, units, and outputs for every published method.
2. Move long calculations to a background task queue with status, progress, timeout, cancellation, and retry.
3. Store datasets, jobs, status, and artifacts in persistent metadata storage.
4. Integrate authentication and enforce per-user data isolation.
5. Add file-retention, cleanup, quota, and deletion policies.
6. Add structured logs, error IDs, security checks, and production configuration.
7. Validate scientific equivalence between CLI and Online results.

### P1 — next engineering iteration

1. Add a Data Mining adapter using the same dataset/job/result model.
2. Extend the catalog with typed optional parameters and downloadable input templates.
3. Add job history, progress display, cancellation, and rerun.
4. Add Docker images, database migrations, CI tests, and a deployment environment.
5. Version the public API under `/api/v1` before external use.

### P2 — product and presentation work

1. Complete visual design and responsive interaction.
2. Add result previews, tables, figures, and scientific metadata.
3. Add Chinese/English localization and user guidance.
4. Add project sharing and reproducible calculation reports.
