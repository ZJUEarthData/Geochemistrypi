import json

import numpy as np

from geochemistrypi.data_mining.model import _base


def test_hyper_parameters_are_strict_json_without_changing_mlflow_logging(monkeypatch):
    saved = {}
    logged = []
    monkeypatch.setattr(
        _base,
        "save_text",
        lambda content, name, path: saved.update(content=content, name=name, path=path),
    )
    monkeypatch.setattr(_base.mlflow, "log_param", lambda key, value: logged.append((key, value)))
    parameters = {
        "missing": np.float64(np.nan),
        "positive_infinity": float("inf"),
        "nested": {
            "negative_infinity": np.float32(-np.inf),
            "sequence": [np.int64(7), (np.float32(1.25), np.float32(np.nan))],
        },
        "ordinary": {"enabled": True, "label": "unchanged", "count": 3},
    }

    _base.WorkflowBase.save_hyper_parameters(parameters, "XGBoost", "artifact-root")

    def reject_nonstandard_constant(value):
        raise AssertionError(f"non-standard JSON constant: {value}")

    persisted = json.loads(saved["content"], parse_constant=reject_nonstandard_constant)
    assert persisted == {
        "missing": "nan",
        "positive_infinity": "inf",
        "nested": {
            "negative_infinity": "-inf",
            "sequence": [7, [1.25, "nan"]],
        },
        "ordinary": {"enabled": True, "label": "unchanged", "count": 3},
    }
    assert saved["name"] == "Hyper Parameters - XGBoost"
    assert saved["path"] == "artifact-root"
    assert logged == [(key, str(value)) for key, value in parameters.items()]
