# PR0 test baseline

## Purpose

PR0 establishes a trustworthy starting point for the Geochemistryπ local MCP
platform. It does not change machine-learning behavior and does not implement
MCP.

## Baseline before PR0

Validated locally on 2026-07-26 with Python 3.9.20:

- `test_multiclass_regressions.py`: 33 passed and 2 failed.
- `test_data_readiness.py`: collection failed because it imported
  `data.data_readiness` as a top-level package.
- The two executable test failures imported the FastAPI router, which imported
  `database.py`; that module called `create_engine(None)` when
  `SQLALCHEMY_DATABASE_URL` was absent.
- The GitHub Actions workflow ran pre-commit but did not run pytest.

## Known scientific and determinism risks

These risks are recorded here, not fixed in PR0:

- The interactive CLI can fit scaling and feature selection before the
  train/test split.
- `WorkflowBase` stores run data in class-level attributes.
- MLflow active runs, matplotlib state, process environment variables, and
  random-number generators can affect process-level behavior.
- Feature expressions and some legacy input parsers use `eval()`.
- Fixed random seeds do not guarantee bitwise equality across platforms,
  dependency versions, BLAS implementations, or parallel algorithms.

PR1 will add characterization fixtures and independent scientific tests for the
leakage correction. Later PRs will isolate each experiment in a worker process.

## PR0 acceptance criteria

- The full test suite is collected without errors.
- Pure module and API-router imports do not require database configuration.
- Actual database access still requires an explicit
  `SQLALCHEMY_DATABASE_URL`.
- CI executes pytest on the supported Python 3.9 engine environment.
- Characterization, scientific, integration, and MCP test responsibilities are
  documented separately.
