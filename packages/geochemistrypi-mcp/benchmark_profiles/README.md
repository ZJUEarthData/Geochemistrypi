# Benchmark profiles

Benchmark profiles are configuration only. They do not add handlers, alter the
GeochemistryPi CLI, or dispatch on a paper name. The generic loader validates a
bounded YAML file, returns its SHA-256, and expands it into the same strict MCP
analysis request used by `validate_analysis`.

```yaml
profile_version: 1
benchmark:
  profile_id: example_classification
  title: Example classification reproduction
  citation: Replace with the verified paper citation
profile_state:
  execution_ready: true
  blocker_category: READY
  evidence_level: verified
workflow:
  family: supervised_learning
  mode: classification
  method: logistic_regression
dataset:
  source: path
  path: D:/verified-data/prepared.csv
  expected_sha256: REPLACE_WITH_64_HEX_CHARACTERS
environment_profile:
  profile_id: example-classification-py39
  python: 3.9.13
  package_versions:
    scikit-learn: REPLACE_WITH_EXACT_VERSION
  runtime_constraints:
    python_implementation: CPython
reproducibility:
  split_seed: 42
  model_seed: 42
  deterministic_policy: fixed_seed_and_dependency_required
parameters:
  experiment_name: Paper reproduction
  run_name: Classification
  identifier_column: SampleID
  feature_columns: [FeatureA, FeatureB]
  target_column: Label
  model:
    type: logistic_regression
expected_artifacts:
  - requirement_id: evaluation.holdout
    scientific_type: holdout_metrics
    output_role: evaluation.holdout
    category: metrics
    path_pattern: metrics/Model Score*.txt
    media_types: [application/json]
    required_json_keys: [accuracy, f1]
acceptance_rules:
  require_execution_ready: true
  required_artifact_ids: [evaluation.holdout]
```

Use `load_benchmark_profile(path, expected_sha256)` and retain the returned
profile-file hash with the benchmark package. Then call
`profile.to_analysis_request()` and submit that expanded request to MCP. After
planning, `attest_profile_plan(profile, plan)` checks the compiled workflow and
environment-profile identity without changing the plan. The legacy inline
`environment` block remains accepted for existing profiles; do not provide it
together with `environment_profile`.

Only create a paper-named profile after its dataset, parameters, environment,
and acceptance evidence have been verified. A profile with unknown values must
use `profile_state.execution_ready: false`, name its blocker category and
unknown fields, and remain a template; `UNKNOWN` is never passed to a runtime
request. Multi-stage workflows can declare generic stages with unique IDs,
family/method names, and input/output ports. Their graph is validated for
cycles and duplicate producers, but remains non-executable until a complete
pipeline adapter exists.
