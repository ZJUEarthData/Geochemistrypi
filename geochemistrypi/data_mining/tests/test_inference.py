from types import SimpleNamespace

import pandas as pd

from geochemistrypi.data_mining.data import inference


def test_build_transform_pipeline_accepts_unsupervised_target(monkeypatch):
    """Clustering has no target, but still needs its fitted preprocessing pipeline."""

    fitted = {}

    class RecordingPipeline:
        def fit(self, X, y):
            fitted["X"] = X.copy()
            fitted["y"] = y
            return self

    pipeline = RecordingPipeline()
    monkeypatch.setattr(inference.PipelineConstrutor, "chain", lambda self, config: pipeline)
    monkeypatch.setattr(inference, "save_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, "save_model", lambda *args, **kwargs: None)

    X_train = pd.DataFrame({"SiO2": [50.0, 55.0], "MgO": [8.0, 5.0]})
    scaling = {"StandardScaler": {"copy": True, "with_mean": True, "with_std": True}}

    transformer_config, transform_pipeline = inference.build_transform_pipeline(
        {},
        scaling,
        {},
        SimpleNamespace(transformer_config={}),
        X_train,
        None,
    )

    assert transformer_config == scaling
    assert transform_pipeline is pipeline
    pd.testing.assert_frame_equal(fitted["X"], X_train)
    assert fitted["y"] is None


def test_external_regression_evaluation_binds_data_to_an_explicit_figure(monkeypatch):
    captured = {}

    monkeypatch.setattr(inference, "save_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, "save_data", lambda *args, **kwargs: None)

    def capture_figure(*args, figure, **kwargs):
        captured["figure"] = figure

    monkeypatch.setattr(inference, "save_fig", capture_figure)
    actual = pd.DataFrame({"P_kbar": [3.0, 8.0, 15.0]})
    predicted = pd.DataFrame({"Predicted Value": [2.5, 8.5, 14.0]})

    inference.save_external_regression_evaluation(
        actual,
        predicted,
        pd.Series(["A", "B", "C"], name="SampleID"),
        "Extra-Trees",
    )

    figure = captured["figure"]
    axis = figure.axes[0]
    assert axis.collections[0].get_offsets().tolist() == [
        [2.5, 3.0],
        [8.5, 8.0],
        [14.0, 15.0],
    ]
    assert axis.get_xlim() != (0.0, 1.0)
    assert axis.get_ylim() != (0.0, 1.0)
