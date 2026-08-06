# Geochemistry Pi Online MVP

This document describes the first usable Online version. The current goal is a complete calculation loop, not a production deployment or final visual design.

## Current scope

The MVP provides:

- a Vue page at `/online`;
- a lightweight FastAPI application independent of the legacy Dash/auth stack;
- a dynamic chemical-modeling catalog;
- method-level `verified` and `testing` readiness states;
- structured input documentation with formulas, columns, meanings, units, types, examples, and notes;
- `.xlsx` and UTF-8 `.csv` upload with a 10 MB limit;
- synchronous calculation in an isolated job directory;
- result-file download;
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
6. waits for both services to become healthy and opens the Online page.

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
backend: 44 passed
frontend: production build succeeds
```

## Known MVP limitations

- calculations are synchronous, so a long calculation keeps the HTTP request open;
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
