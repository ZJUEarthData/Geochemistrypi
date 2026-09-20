# Online Summary download

## Contract (2026-09-20)

The user superseded the earlier CLI-parity requirement. Summary now bundles the
current Online job's existing output files, byte-for-byte, without creating CLI
diagnostics, retraining models, changing preprocessing, or altering parameters.

The six Data Mining result panels provide **Download Summary (.zip)** above their
existing single-file links. A classification download, for example, contains:

```text
summary.zip
└── summary/
    ├── classification_predictions.csv
    ├── classification_report.json
    └── trained_pipeline.joblib
```

Other modes contain their own existing outputs (CSV, JSON, SVG, model files,
etc.). Any existing subdirectories are preserved under `summary/`. Browser-only
exports, such as client-generated PNG renditions, are not server output files
and are not included. No additional README or synthetic results are inserted.

## Implementation

- `POST /api/data-mining/jobs/{job_id}/summary` returns
  `{artifact: {name, download_url, size_bytes}, files: [...]}`.
- Works for older jobs too, provided their output files still exist. No private
  model snapshot or database migration is needed.
- Packaging runs in a worker thread, not on the event loop or model-training
  queue. Files are streamed into ZIP; model files are never deserialized.
- Staging is outside the downloadable output directory. Only a complete ZIP is
  published. Identical contents reuse the same content-hashed URL; changed
  contents receive a new URL without replacing earlier downloads.
- Existing Summary archives are excluded from subsequent archives.
- Job IDs, output paths, symlinks and Windows reparse points are validated.
  This is filesystem isolation, not user authentication: existing job URLs
  remain accessible to anyone who knows them, as before this change.
- The UI shows loading/errors, blocks duplicate clicks, supports retry, and
  leaves a normal download link if automatic downloading is blocked.

## Local verification

From the repository root:

```powershell
.venv-online/Scripts/python.exe -m pytest tests/test_online_summary.py tests/test_artifact_paths.py -q
```

From `geochemistrypi/frontend` (Node and pnpm must be on PATH):

```text
node --test tests/summary-download.test.cjs
pnpm run build
```

Tests cover actual output files from all six modes, byte equality, previous jobs,
repeat downloads, concurrent requests, failed builds, private-file isolation,
download headers, UI loading/error/retry/unmount states and the six result gates.

## Deployment boundary

Production deployment was authorized on 2026-09-20. A Git push alone does not
deploy the server; verify the running build and a real Summary download after
updating. A production update requires both backend and frontend images. Before publishing,
back up runtime data, configuration and certificates, retain the previous images,
and let running jobs finish. Do not use `docker compose down -v`.
