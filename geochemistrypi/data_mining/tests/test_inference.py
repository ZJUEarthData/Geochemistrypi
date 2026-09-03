from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

from geochemistrypi.data_mining.data import inference, preprocessing


def test_build_transform_pipeline_accepts_unsupervised_target(monkeypatch):
    """Clustering has no target, but still needs its fitted preprocessing pipeline."""

    fitted = {}

    class RecordingPipeline:
        def fit(self, X, y):
            fitted["X"] = X.copy()
            fitted["y"] = y
            return self

    pipeline = RecordingPipeline()
    chained = {}

    def build_pipeline(self, config):
        chained["config"] = config
        return pipeline

    monkeypatch.setattr(inference.PipelineConstrutor, "chain", build_pipeline)
    monkeypatch.setattr(inference, "save_text", lambda *args, **kwargs: None)
    saved = {}
    monkeypatch.setattr(inference, "save_model", lambda model, *args, **kwargs: saved.update(model=model))

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
    assert chained["config"] == scaling
    assert saved["model"] is pipeline
    pd.testing.assert_frame_equal(fitted["X"], X_train)
    assert fitted["y"] is None


def test_supervised_empty_config_saves_replayable_identity_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "save_text", lambda *args, **kwargs: None)
    saved_path = tmp_path / "Transform Pipeline.joblib"

    def save_pipeline(model, *args, **kwargs):
        joblib.dump(model, saved_path)

    monkeypatch.setattr(inference, "save_model", save_pipeline)
    X_train = pd.DataFrame(
        {"SiO2": [50.0, 55.0], "Rock": ["A", "B"]},
        index=pd.Index([10, 20], name="source_row"),
    )
    y_train = pd.DataFrame({"Label": [0, 1]}, index=X_train.index)

    transformer_config, transform_pipeline = inference.build_transform_pipeline(
        {},
        {},
        {},
        SimpleNamespace(transformer_config={}),
        X_train,
        y_train,
    )

    assert transformer_config == {}
    assert saved_path.is_file()
    pd.testing.assert_frame_equal(transform_pipeline.transform(X_train), X_train)
    reloaded = joblib.load(saved_path)
    pd.testing.assert_frame_equal(reloaded.transform(X_train), X_train)


def test_unsupervised_empty_config_does_not_save_pipeline(monkeypatch):
    monkeypatch.setattr(inference, "save_text", lambda *args, **kwargs: None)
    saved = []
    monkeypatch.setattr(inference, "save_model", lambda *args, **kwargs: saved.append(args))
    X_train = pd.DataFrame({"SiO2": [50.0, 55.0]})

    transformer_config, transform_pipeline = inference.build_transform_pipeline(
        {},
        {},
        {},
        SimpleNamespace(transformer_config={}),
        X_train,
        None,
    )

    assert transformer_config == {}
    assert transform_pipeline is None
    assert saved == []


def test_scientific_scaling_and_selection_fit_only_the_training_partition(monkeypatch):
    captured = {}

    class RecordingSelectKBest(SelectKBest):
        def fit(self, X, y=None):
            captured["selector_X"] = np.asarray(X).copy()
            captured["selector_y"] = np.asarray(y).copy()
            return super().fit(X, y)

    monkeypatch.setattr(inference, "SelectKBest", RecordingSelectKBest)
    X_train = pd.DataFrame(
        {
            "SIO2": [50.0, 51.0, 53.0, 56.0, 60.0, 63.0, 67.0, 70.0],
            "TIO2": [0.4, 0.7, 0.6, 1.0, 1.2, 1.1, 1.5, 1.7],
            "AL2O3": [12.0, 13.0, 13.5, 14.0, 15.0, 16.0, 16.5, 17.0],
            "FEOT": [11.0, 10.5, 10.0, 9.0, 8.5, 8.0, 7.0, 6.5],
            "CAO": [9.0, 8.0, 8.5, 7.0, 7.5, 6.0, 6.5, 5.0],
            "NA2O": [2.0, 2.1, 2.0, 2.4, 2.5, 2.7, 2.8, 3.0],
        },
        index=pd.Index(range(100, 108), name="source_row"),
    )
    y_train = pd.DataFrame(
        {"MGO": [8.0, 7.6, 7.1, 6.2, 5.7, 4.9, 4.1, 3.4]},
        index=X_train.index,
    )
    X_holdout = pd.DataFrame(
        {column: [1000.0 + position, -1000.0 - position] for position, column in enumerate(X_train.columns)},
        index=pd.Index([900, 901], name="source_row"),
    )
    X_application = X_holdout.copy()
    X_application.index = pd.Index(["APP-1", "APP-2"], name="sample")
    transformer_config = {
        "StandardScaler": StandardScaler().get_params(),
        "SelectKBest": SelectKBest(score_func=f_regression, k=4).get_params(),
    }

    transform_pipeline = inference.fit_training_transform_pipeline(
        transformer_config,
        X_train,
        y_train,
    )
    transformed_train = inference.transform_feature_frame(transform_pipeline, X_train)
    transformed_holdout = inference.transform_feature_frame(transform_pipeline, X_holdout)
    transformed_application = inference.transform_feature_frame(transform_pipeline, X_application)
    monkeypatch.setattr(inference, "save_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "save_model",
        lambda model, _name, sample, _path: captured.update(
            saved_model=model,
            input_example=sample.copy(),
        ),
    )
    _config, saved_pipeline = inference.build_transform_pipeline(
        {},
        {"StandardScaler": transformer_config["StandardScaler"]},
        {"SelectKBest": transformer_config["SelectKBest"]},
        SimpleNamespace(transformer_config={}),
        transformed_train,
        y_train,
        prefitted_transform_pipeline=transform_pipeline,
        pipeline_input_example=X_train.iloc[[0]],
    )

    np.testing.assert_allclose(
        transform_pipeline.named_steps["standardscaler"].mean_,
        X_train.mean(axis=0).to_numpy(),
    )
    np.testing.assert_allclose(
        captured["selector_X"],
        StandardScaler().fit_transform(X_train),
    )
    assert captured["selector_X"].shape[0] == len(X_train)
    assert captured["selector_y"].shape[0] == len(y_train)
    assert list(transform_pipeline.feature_names_in_) == list(X_train.columns)
    assert transformed_train.shape == (8, 4)
    assert list(transformed_holdout.columns) == list(transformed_train.columns)
    assert list(transformed_application.columns) == list(transformed_train.columns)
    assert transformed_holdout.index.equals(X_holdout.index)
    assert transformed_application.index.equals(X_application.index)
    assert saved_pipeline is transform_pipeline
    assert captured["saved_model"] is transform_pipeline
    pd.testing.assert_frame_equal(captured["input_example"], X_train.iloc[[0]])
    assert not np.allclose(
        transform_pipeline.named_steps["standardscaler"].mean_,
        pd.concat([X_train, X_holdout]).mean(axis=0).to_numpy(),
    )


def test_scientific_feature_selector_can_defer_fit_until_after_split(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "4")
    monkeypatch.setattr(
        preprocessing.SelectKBest,
        "fit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("selector fitted before the split")),
    )
    X = pd.DataFrame({f"F{index}": range(8) for index in range(6)})
    y = pd.DataFrame({"Target": range(8)})

    config, selected = preprocessing.feature_selector(
        X,
        y,
        1,
        ["Generic Univariate Select", "Select K Best"],
        1,
        fit=False,
    )

    assert selected is None
    assert config["SelectKBest"]["k"] == 4


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
