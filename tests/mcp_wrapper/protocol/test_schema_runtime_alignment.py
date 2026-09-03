from copy import deepcopy
from typing import Any

import pytest
from geochemistrypi_mcp.api.schemas import (
    AffineTargetTransformation,
    AnalysisRequest,
    AnomalyDetectionRequest,
    ClassificationRequest,
    ClusteringRequest,
    DatasetFilterRule,
    DatasetInspectionRequest,
    DatasetPreparationContract,
    DBSCANClusteringSettings,
    DecompositionRequest,
    EvaluationContract,
    ExtraTreesSettings,
    ImputeMissingValues,
    IsolationForestAnomalyDetectionSettings,
    KNearestNeighborsSettings,
    LogisticRegressionSettings,
    PCADecompositionSettings,
    RandomForestSettings,
    RegressionRequest,
    SourceRowIdentityContract,
    StochasticGradientDescentRegressionSettings,
    StochasticGradientDescentSettings,
    SupportVectorMachineSettings,
    TimeSeriesRequest,
    TSNEDecompositionSettings,
)
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

REQUEST_MODELS = (
    ClassificationRequest,
    RegressionRequest,
    ClusteringRequest,
    DecompositionRequest,
    AnomalyDetectionRequest,
    TimeSeriesRequest,
)


def _assert_valid(model: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
    schema = model.model_json_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    return model.model_validate(payload)


def _assert_invalid(model: type[BaseModel], payload: dict[str, Any]) -> None:
    schema = model.model_json_schema()
    Draft202012Validator.check_schema(schema)
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(payload)
    with pytest.raises(PydanticValidationError):
        model.model_validate(payload)


def _builtin(dataset_id: str = "builtin:classification") -> dict[str, str]:
    return {"source": "builtin", "dataset_id": dataset_id}


def _classification() -> dict[str, Any]:
    return {
        "task": "classification",
        "training_dataset_path": "C:/data/training.csv",
        "experiment_name": "Schema alignment",
        "run_name": "Classification",
        "identifier_column": "SampleID",
        "feature_columns": ["SIO2"],
        "target_column": "Label",
    }


def _regression() -> dict[str, Any]:
    return {
        "task": "regression",
        "training_dataset_path": "C:/data/training.csv",
        "experiment_name": "Schema alignment",
        "run_name": "Regression",
        "identifier_column": "SampleID",
        "feature_columns": ["SIO2"],
        "target_column": "Target",
    }


def _clustering() -> dict[str, Any]:
    return {
        "task": "clustering",
        "training_dataset_path": "C:/data/training.csv",
        "experiment_name": "Schema alignment",
        "run_name": "Clustering",
        "identifier_column": "SampleID",
        "feature_columns": ["SIO2"],
    }


def _decomposition() -> dict[str, Any]:
    return {
        "task": "decomposition",
        "training_dataset_path": "C:/data/training.csv",
        "experiment_name": "Schema alignment",
        "run_name": "Decomposition",
        "identifier_column": "SampleID",
        "feature_columns": ["SIO2", "TIO2"],
    }


def _anomaly() -> dict[str, Any]:
    return {
        "task": "anomaly_detection",
        "training_dataset_path": "C:/data/training.csv",
        "experiment_name": "Schema alignment",
        "run_name": "Anomaly",
        "identifier_column": "SampleID",
        "feature_columns": ["SIO2"],
    }


def _time_series() -> dict[str, Any]:
    return {
        "task": "time_series",
        "training_dataset_path": "C:/data/time-series.csv",
        "bin_width": 100,
    }


REQUEST_FACTORIES = (
    _classification,
    _regression,
    _clustering,
    _decomposition,
    _anomaly,
    _time_series,
)


@pytest.mark.parametrize(
    "model,factory",
    tuple(zip(REQUEST_MODELS, REQUEST_FACTORIES, strict=True)),
)
def test_all_six_tasks_advertise_the_required_training_source_xor(
    model: type[BaseModel],
    factory,
) -> None:
    path_request = factory()
    _assert_valid(model, path_request)

    reference_request = factory()
    reference_request.pop("training_dataset_path")
    reference_request["training_dataset"] = _builtin()
    _assert_valid(model, reference_request)

    missing_request = factory()
    missing_request.pop("training_dataset_path")
    _assert_invalid(model, missing_request)

    ambiguous_request = factory()
    ambiguous_request["training_dataset"] = _builtin()
    _assert_invalid(model, ambiguous_request)


@pytest.mark.parametrize(
    "model,factory",
    ((ClassificationRequest, _classification), (RegressionRequest, _regression)),
)
def test_supervised_application_source_is_optional_but_never_ambiguous(
    model: type[BaseModel],
    factory,
) -> None:
    _assert_valid(model, factory())

    path_request = factory()
    path_request["application_dataset_path"] = "C:/data/application.csv"
    _assert_valid(model, path_request)

    reference_request = factory()
    reference_request["application_dataset"] = _builtin("builtin:application_classification")
    _assert_valid(model, reference_request)

    ambiguous_request = factory()
    ambiguous_request["application_dataset_path"] = "C:/data/application.csv"
    ambiguous_request["application_dataset"] = _builtin("builtin:application_classification")
    _assert_invalid(model, ambiguous_request)


def test_regression_advertises_exactly_one_non_empty_target_form() -> None:
    legacy = _regression()
    _assert_valid(RegressionRequest, legacy)

    multiple = _regression()
    multiple.pop("target_column")
    multiple["target_columns"] = ["TargetA", "TargetB"]
    _assert_valid(RegressionRequest, multiple)

    missing = _regression()
    missing.pop("target_column")
    _assert_invalid(RegressionRequest, missing)

    empty = _regression()
    empty.pop("target_column")
    empty["target_columns"] = []
    _assert_invalid(RegressionRequest, empty)

    ambiguous = _regression()
    ambiguous["target_columns"] = ["TargetB"]
    _assert_invalid(RegressionRequest, ambiguous)


def test_classification_binary_metric_contract_is_typed_and_self_describing() -> None:
    automatic = _assert_valid(ClassificationRequest, _classification())
    assert automatic.metric_average == "auto"
    assert automatic.positive_label is None

    binary = _classification()
    binary.update(metric_average="binary", positive_label=1)
    numeric = _assert_valid(ClassificationRequest, binary)

    binary_string = {**binary, "positive_label": "1"}
    string = _assert_valid(ClassificationRequest, binary_string)
    assert type(numeric.positive_label) is int
    assert type(string.positive_label) is str

    missing_positive = _classification()
    missing_positive["metric_average"] = "binary"
    _assert_invalid(ClassificationRequest, missing_positive)

    incompatible_positive = _classification()
    incompatible_positive.update(metric_average="weighted", positive_label=1)
    _assert_invalid(ClassificationRequest, incompatible_positive)

    properties = ClassificationRequest.model_json_schema()["properties"]
    assert "backward-compatible" in properties["metric_average"]["description"]
    assert "numeric 1 and string '1'" in properties["positive_label"]["description"]


@pytest.mark.parametrize(
    "model,factory",
    (
        (ClusteringRequest, _clustering),
        (DecompositionRequest, _decomposition),
        (AnomalyDetectionRequest, _anomaly),
    ),
)
def test_unsupervised_schemas_do_not_advertise_runtime_rejected_keep_missing_values(
    model: type[BaseModel],
    factory,
) -> None:
    request = factory()
    request["missing_values"] = {"method": "keep"}
    _assert_invalid(model, request)


def test_evaluation_dataset_and_mode_conditions_match_runtime() -> None:
    _assert_valid(EvaluationContract, {})
    _assert_valid(
        EvaluationContract,
        {
            "mode": "external_labeled",
            "evaluation_dataset_path": "C:/data/evaluation.csv",
        },
    )
    _assert_valid(
        EvaluationContract,
        {
            "mode": "external_labeled",
            "evaluation_dataset": _builtin(),
            "external_identifier_column": "SampleID",
        },
    )
    _assert_invalid(EvaluationContract, {"mode": "external_labeled"})
    _assert_invalid(
        EvaluationContract,
        {
            "mode": "external_labeled",
            "evaluation_dataset_path": "C:/data/evaluation.csv",
            "evaluation_dataset": _builtin(),
        },
    )
    _assert_invalid(
        EvaluationContract,
        {"mode": "quality_report", "evaluation_dataset_path": "C:/data/evaluation.csv"},
    )
    _assert_invalid(EvaluationContract, {"mode": "cross_validation"})
    _assert_valid(EvaluationContract, {"mode": "cross_validation", "folds": 10})
    _assert_valid(EvaluationContract, {"mode": "holdout", "folds": 10})
    _assert_invalid(EvaluationContract, {"folds": 10})
    _assert_valid(
        EvaluationContract,
        {"mode": "holdout", "split_strategy": "stratified_holdout"},
    )
    _assert_invalid(EvaluationContract, {"split_strategy": "random_holdout"})


def test_dataset_preparation_advertises_projection_header_and_worksheet_conditions() -> None:
    _assert_valid(DatasetPreparationContract, {})
    _assert_valid(DatasetPreparationContract, {"selected_columns": ["SampleID"]})
    _assert_invalid(
        DatasetPreparationContract,
        {"selected_columns": ["SampleID"], "excluded_columns": ["Label"]},
    )
    _assert_valid(DatasetPreparationContract, {"header_row_indices": [0, 1]})
    _assert_invalid(
        DatasetPreparationContract,
        {"header_row_index": 0, "header_row_indices": [0, 1]},
    )

    row_union = {
        "worksheets": ["North", "South"],
        "union_mode": "rows",
        "source_sheet_column": "Source sheet",
        "source_row_column": "Source row",
        "selected_columns": ["Source sheet", "Source row"],
    }
    _assert_valid(DatasetPreparationContract, row_union)
    _assert_invalid(DatasetPreparationContract, {**row_union, "worksheets": ["North"]})
    _assert_invalid(DatasetPreparationContract, {**row_union, "source_row_column": None})
    _assert_invalid(DatasetPreparationContract, {**row_union, "worksheet": "North"})
    _assert_invalid(DatasetPreparationContract, {"union_mode": "rows"})


def test_filter_and_row_identity_operand_conditions_match_runtime() -> None:
    _assert_valid(
        DatasetFilterRule,
        {"column": "AGE", "operator": "between", "minimum": 0, "maximum": 100},
    )
    _assert_invalid(
        DatasetFilterRule,
        {"column": "AGE", "operator": "between", "minimum": 0},
    )
    _assert_invalid(
        DatasetFilterRule,
        {
            "column": "AGE",
            "operator": "between",
            "minimum": 0,
            "maximum": 100,
            "value": 50,
        },
    )

    _assert_valid(SourceRowIdentityContract, {})
    _assert_valid(
        SourceRowIdentityContract,
        {"strategy": "column_values", "columns": ["SampleID"]},
    )
    _assert_invalid(SourceRowIdentityContract, {"strategy": "column_values"})
    _assert_invalid(SourceRowIdentityContract, {"columns": ["SampleID"]})
    _assert_valid(
        SourceRowIdentityContract,
        {
            "source_mapping_path": "C:/data/source-map.csv",
            "source_mapping_sha256": "0" * 64,
        },
    )
    _assert_invalid(
        SourceRowIdentityContract,
        {"source_mapping_path": "C:/data/source-map.csv"},
    )


def _overlay() -> dict[str, Any]:
    request = _decomposition()
    request.update(
        mode="embedding_label_overlay",
        application_dataset_path="C:/data/labels.csv",
        label_identifier_column="SampleID",
        label_column="Label",
        positive_label_values=["positive"],
        scaling="none",
    )
    return request


def test_decomposition_mode_advertises_application_and_overlay_contract() -> None:
    _assert_valid(DecompositionRequest, _decomposition())
    _assert_valid(DecompositionRequest, _overlay())

    model_with_application = _decomposition()
    model_with_application["application_dataset_path"] = "C:/data/labels.csv"
    _assert_invalid(DecompositionRequest, model_with_application)

    missing_application = _overlay()
    missing_application.pop("application_dataset_path")
    _assert_invalid(DecompositionRequest, missing_application)

    missing_scaling = _overlay()
    missing_scaling.pop("scaling")
    _assert_invalid(DecompositionRequest, missing_scaling)

    wrong_shape = _overlay()
    wrong_shape["feature_columns"] = ["X"]
    _assert_invalid(DecompositionRequest, wrong_shape)

    all_models = _overlay()
    all_models["model_selection"] = {"mode": "all"}
    _assert_invalid(DecompositionRequest, all_models)


@pytest.mark.parametrize(
    "model",
    (
        PCADecompositionSettings().model_dump(mode="json"),
        TSNEDecompositionSettings().model_dump(mode="json"),
    ),
)
def test_decomposition_modes_reject_fields_owned_by_the_other_mode(
    model: dict[str, Any],
) -> None:
    overlay_with_model = _overlay()
    overlay_with_model["model"] = model
    _assert_invalid(DecompositionRequest, overlay_with_model)

    overlay_with_selection = _overlay()
    overlay_with_selection["model_selection"] = {"mode": "single"}
    _assert_invalid(DecompositionRequest, overlay_with_selection)

    for field, value in (
        ("coordinate_sheet", "Coordinates"),
        ("label_sheet", "Labels"),
        ("label_identifier_column", "SampleID"),
        ("label_column", "Label"),
        ("positive_label_values", ["positive"]),
    ):
        model_request = _decomposition()
        model_request[field] = value
        _assert_invalid(DecompositionRequest, model_request)


def test_decomposition_standard_serialization_preserves_the_selected_mode() -> None:
    model_request = DecompositionRequest.model_validate(_decomposition())
    overlay_request = DecompositionRequest.model_validate(_overlay())

    model_payload = model_request.model_dump(mode="json")
    overlay_payload = overlay_request.model_dump(mode="json")
    assert (
        not {
            "coordinate_sheet",
            "label_sheet",
            "label_identifier_column",
            "label_column",
            "positive_label_values",
        }
        & model_payload.keys()
    )
    assert not {"model", "model_selection"} & overlay_payload.keys()
    _assert_valid(DecompositionRequest, model_payload)
    _assert_valid(DecompositionRequest, overlay_payload)


def test_all_model_selection_and_tuning_fields_match_runtime() -> None:
    all_classification = _classification()
    all_classification["model_selection"] = {"mode": "all"}
    _assert_valid(ClassificationRequest, all_classification)

    explicit_model = deepcopy(all_classification)
    explicit_model["model"] = {"type": "logistic_regression"}
    _assert_invalid(ClassificationRequest, explicit_model)

    explicit_tuning = deepcopy(all_classification)
    explicit_tuning["tuning"] = "manual"
    _assert_invalid(ClassificationRequest, explicit_tuning)

    unsupervised_automl = _clustering()
    unsupervised_automl["model_selection"] = {"mode": "all", "tuning": "automl"}
    _assert_invalid(ClusteringRequest, unsupervised_automl)

    automl_regression = _regression()
    automl_regression.update(tuning="automl", model={"type": "xgboost"})
    _assert_valid(RegressionRequest, automl_regression)

    no_model = _regression()
    no_model["tuning"] = "automl"
    _assert_invalid(RegressionRequest, no_model)

    unsupported_model = _regression()
    unsupported_model.update(tuning="automl", model={"type": "linear_regression"})
    _assert_invalid(RegressionRequest, unsupported_model)

    manual_parameters = _regression()
    manual_parameters.update(
        tuning="automl",
        model={"type": "xgboost", "maximum_depth": 4},
    )
    _assert_invalid(RegressionRequest, manual_parameters)


def test_all_four_time_series_modes_advertise_required_and_forbidden_fields() -> None:
    _assert_valid(TimeSeriesRequest, _time_series())

    missing_bin = _time_series()
    missing_bin.pop("bin_width")
    _assert_invalid(TimeSeriesRequest, missing_bin)

    continuous = _time_series()
    continuous.update(
        mode="continuous",
        minimum_age_column="R_MIN_AGE",
        value_column="VALUE",
    )
    _assert_valid(TimeSeriesRequest, continuous)
    continuous.pop("value_column")
    _assert_invalid(TimeSeriesRequest, continuous)

    element_mean = _time_series()
    element_mean.update(mode="element_mean", element_columns=["SIO2"])
    _assert_valid(TimeSeriesRequest, element_mean)
    element_mean["element_columns"] = []
    _assert_invalid(TimeSeriesRequest, element_mean)

    reference = _time_series()
    reference.pop("bin_width")
    reference.update(
        mode="reference_anomaly_series",
        time_column="DATE",
        signal_columns=["SIGNAL"],
        reference_label_column="REFERENCE",
        reference_positive_values=["event"],
    )
    _assert_valid(TimeSeriesRequest, reference)

    forbidden_bin = deepcopy(reference)
    forbidden_bin["bin_width"] = 100
    _assert_invalid(TimeSeriesRequest, forbidden_bin)

    missing_signal = deepcopy(reference)
    missing_signal["signal_columns"] = []
    _assert_invalid(TimeSeriesRequest, missing_signal)

    unmatched_comparison = deepcopy(reference)
    unmatched_comparison["comparison_label_column"] = "CALCULATED"
    _assert_invalid(TimeSeriesRequest, unmatched_comparison)

    incomplete_event = deepcopy(reference)
    incomplete_event["event_dataset_path"] = "C:/data/events.csv"
    _assert_invalid(TimeSeriesRequest, incomplete_event)


@pytest.mark.parametrize(
    ("mode", "updates", "irrelevant_field", "irrelevant_value"),
    (
        ("subaerial_proportion", {}, "element_columns", ["SIO2"]),
        (
            "continuous",
            {"minimum_age_column": "R_MIN_AGE", "value_column": "VALUE"},
            "probability_column",
            "SBAP",
        ),
        ("element_mean", {"element_columns": ["SIO2"]}, "iterations", 100),
        (
            "reference_anomaly_series",
            {
                "time_column": "DATE",
                "signal_columns": ["SIGNAL"],
                "reference_label_column": "REFERENCE",
                "reference_positive_values": ["event"],
            },
            "age_column",
            "R_AGE",
        ),
        (
            "reference_anomaly_series",
            {
                "time_column": "DATE",
                "signal_columns": ["SIGNAL"],
                "reference_label_column": "REFERENCE",
                "reference_positive_values": ["event"],
            },
            "bin_width",
            None,
        ),
    ),
)
def test_time_series_modes_reject_explicit_fields_owned_by_other_modes(
    mode: str,
    updates: dict[str, Any],
    irrelevant_field: str,
    irrelevant_value: Any,
) -> None:
    request = _time_series()
    request["mode"] = mode
    request.update(updates)
    if mode == "reference_anomaly_series":
        request.pop("bin_width")
    _assert_valid(TimeSeriesRequest, request)

    request[irrelevant_field] = irrelevant_value
    _assert_invalid(TimeSeriesRequest, request)


def test_time_series_standard_serialization_preserves_each_selected_mode() -> None:
    mode_fields = {
        "subaerial_proportion": {
            "bin_width",
            "iterations",
            "seed",
            "age_column",
            "maximum_age_column",
            "probability_column",
            "latitude_column",
            "longitude_column",
            "age_unit",
            "fit_curve",
        },
        "continuous": {
            "bin_width",
            "iterations",
            "seed",
            "age_column",
            "minimum_age_column",
            "maximum_age_column",
            "value_column",
            "latitude_column",
            "longitude_column",
            "filter_column",
            "filter_minimum",
            "filter_maximum",
            "minimum_samples_per_bin",
            "relative_value_two_sigma",
            "age_unit",
            "fit_curve",
            "compact_y_axis",
        },
        "element_mean": {
            "bin_width",
            "age_column",
            "element_columns",
            "filter_column",
            "filter_minimum",
            "filter_maximum",
            "aggregation",
            "uncertainty",
            "minimum_samples_per_bin",
        },
        "reference_anomaly_series": {
            "time_column",
            "signal_columns",
            "reference_label_column",
            "reference_positive_values",
            "reference_label_provenance",
            "comparison_label_column",
            "comparison_positive_values",
            "comparison_label_provenance",
            "event_dataset_path",
            "event_sheet",
            "event_time_column",
            "event_identifier_column",
            "event_filter_column",
            "event_filter_values",
            "association_window_days",
            "association_direction",
        },
    }
    all_mode_fields = set().union(*mode_fields.values())
    requests = []
    for mode, updates in (
        ("subaerial_proportion", {}),
        (
            "continuous",
            {"minimum_age_column": "R_MIN_AGE", "value_column": "VALUE"},
        ),
        ("element_mean", {"element_columns": ["SIO2"]}),
        (
            "reference_anomaly_series",
            {
                "time_column": "DATE",
                "signal_columns": ["SIGNAL"],
                "reference_label_column": "REFERENCE",
                "reference_positive_values": ["event"],
            },
        ),
    ):
        payload = _time_series()
        payload["mode"] = mode
        payload.update(updates)
        if mode == "reference_anomaly_series":
            payload.pop("bin_width")
        requests.append(TimeSeriesRequest.model_validate(payload))

    for request in requests:
        payload = request.model_dump(mode="json")
        assert not ((all_mode_fields - mode_fields[request.mode]) & set(payload))
        _assert_valid(TimeSeriesRequest, payload)


def test_dataset_inspection_defaults_to_names_without_samples_and_keeps_opt_in_views() -> None:
    default_request = _assert_valid(
        DatasetInspectionRequest,
        {"dataset_path": "C:/data/training.csv"},
    )
    assert default_request.detail == "names"
    assert default_request.sample_rows == 0

    full_request = _assert_valid(
        DatasetInspectionRequest,
        {"dataset_path": "C:/data/training.csv", "detail": "full"},
    )
    assert full_request.detail == "full"
    assert full_request.sample_rows == 0

    sampled_request = _assert_valid(
        DatasetInspectionRequest,
        {
            "dataset": _builtin(),
            "detail": "names",
            "sample_rows": 5,
        },
    )
    assert sampled_request.sample_rows == 5

    _assert_invalid(
        DatasetInspectionRequest,
        {
            "dataset_path": "C:/data/training.csv",
            "dataset": _builtin(),
        },
    )


def test_imputation_strategy_and_fill_value_are_advertised_as_one_contract() -> None:
    _assert_valid(ImputeMissingValues, {})
    _assert_valid(
        ImputeMissingValues,
        {"strategy": "constant", "fill_value": 0.0},
    )
    _assert_invalid(ImputeMissingValues, {"strategy": "constant"})
    _assert_invalid(
        ImputeMissingValues,
        {"strategy": "median", "fill_value": 0.0},
    )


def test_logistic_penalty_solver_and_l1_ratio_matrix_matches_runtime() -> None:
    _assert_valid(LogisticRegressionSettings, {})
    _assert_valid(
        LogisticRegressionSettings,
        {"penalty": "l1", "solver": "liblinear"},
    )
    _assert_valid(
        LogisticRegressionSettings,
        {"penalty": "l1", "solver": "saga"},
    )
    _assert_valid(
        LogisticRegressionSettings,
        {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
    )

    _assert_invalid(LogisticRegressionSettings, {"penalty": "l1"})
    _assert_invalid(
        LogisticRegressionSettings,
        {"penalty": "l1", "solver": "lbfgs"},
    )
    _assert_invalid(
        LogisticRegressionSettings,
        {"penalty": "l2", "solver": "liblinear"},
    )
    _assert_invalid(
        LogisticRegressionSettings,
        {"penalty": "elasticnet", "solver": "saga"},
    )
    _assert_invalid(
        LogisticRegressionSettings,
        {"penalty": "elasticnet", "solver": "lbfgs", "l1_ratio": 0.5},
    )
    _assert_invalid(LogisticRegressionSettings, {"l1_ratio": 0.5})


def test_knn_conditional_parameters_match_the_consumed_cli_branch() -> None:
    _assert_valid(KNearestNeighborsSettings, {})
    _assert_valid(
        KNearestNeighborsSettings,
        {"algorithm": "ball_tree", "leaf_size": 99},
    )
    _assert_valid(
        KNearestNeighborsSettings,
        {"metric": "minkowski", "power": 3},
    )
    _assert_invalid(
        KNearestNeighborsSettings,
        {"algorithm": "auto", "leaf_size": 99},
    )
    _assert_invalid(
        KNearestNeighborsSettings,
        {"algorithm": "brute", "leaf_size": 99},
    )
    _assert_invalid(
        KNearestNeighborsSettings,
        {"metric": "euclidean", "power": 3},
    )


def test_svm_conditional_parameters_match_the_consumed_cli_branch() -> None:
    _assert_valid(SupportVectorMachineSettings, {})
    _assert_valid(
        SupportVectorMachineSettings,
        {"kernel": "poly", "degree": 4, "gamma": 0.2},
    )
    _assert_valid(SupportVectorMachineSettings, {"kernel": "rbf", "gamma": 0.2})
    _assert_invalid(
        SupportVectorMachineSettings,
        {"kernel": "linear", "degree": 4},
    )
    _assert_invalid(
        SupportVectorMachineSettings,
        {"kernel": "linear", "gamma": 0.2},
    )
    _assert_invalid(
        SupportVectorMachineSettings,
        {"kernel": "rbf", "degree": 4},
    )


@pytest.mark.parametrize(
    "model",
    (StochasticGradientDescentSettings, StochasticGradientDescentRegressionSettings),
)
def test_sgd_l1_ratio_is_only_configurable_for_elasticnet(
    model: type[BaseModel],
) -> None:
    _assert_valid(model, {})
    _assert_valid(model, {"penalty": "elasticnet", "l1_ratio": 0.4})
    _assert_invalid(model, {"penalty": "l2", "l1_ratio": 0.4})


@pytest.mark.parametrize("model", (RandomForestSettings, ExtraTreesSettings))
def test_forest_bootstrap_sample_and_out_of_bag_matrix_matches_runtime(
    model: type[BaseModel],
) -> None:
    _assert_valid(model, {})
    _assert_valid(
        model,
        {
            "bootstrap": False,
            "maximum_samples": None,
            "out_of_bag_score": False,
        },
    )
    _assert_invalid(model, {"bootstrap": True, "maximum_samples": None})
    _assert_invalid(model, {"bootstrap": False})
    _assert_invalid(
        model,
        {"bootstrap": False, "maximum_samples": None},
    )
    _assert_invalid(
        model,
        {
            "bootstrap": False,
            "maximum_samples": 0.5,
            "out_of_bag_score": False,
        },
    )
    _assert_invalid(
        model,
        {
            "bootstrap": False,
            "maximum_samples": None,
            "out_of_bag_score": True,
        },
    )


def test_affine_target_scale_advertises_the_non_zero_runtime_guard() -> None:
    _assert_valid(AffineTargetTransformation, {})
    _assert_valid(AffineTargetTransformation, {"scale": -1.0, "offset": 10.0})
    _assert_invalid(AffineTargetTransformation, {"scale": 0})
    _assert_invalid(AffineTargetTransformation, {"scale": 0.0})


def test_dbscan_minkowski_power_condition_matches_runtime() -> None:
    _assert_valid(DBSCANClusteringSettings, {})
    _assert_valid(
        DBSCANClusteringSettings,
        {"metric": "minkowski", "power": 2},
    )
    _assert_invalid(DBSCANClusteringSettings, {"metric": "minkowski"})
    _assert_invalid(
        DBSCANClusteringSettings,
        {"metric": "euclidean", "power": 2},
    )


def test_isolation_forest_sample_count_is_independent_of_replacement_policy() -> None:
    _assert_valid(IsolationForestAnomalyDetectionSettings, {})
    _assert_valid(
        IsolationForestAnomalyDetectionSettings,
        {"bootstrap": True, "maximum_samples": 100},
    )
    _assert_valid(
        IsolationForestAnomalyDetectionSettings,
        {"bootstrap": True, "maximum_samples": "auto"},
    )
    _assert_valid(
        IsolationForestAnomalyDetectionSettings,
        {"bootstrap": False, "maximum_samples": 100},
    )
    _assert_valid(
        IsolationForestAnomalyDetectionSettings,
        {"bootstrap": False, "maximum_samples": "auto"},
    )
    _assert_invalid(IsolationForestAnomalyDetectionSettings, {"maximum_samples": "all"})


def test_complete_analysis_schema_remains_a_six_branch_draft_2020_12_contract() -> None:
    schema = AnalysisRequest.model_json_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["type"] == "object"
    assert len(schema["oneOf"]) == 6
    assert schema["discriminator"]["propertyName"] == "task"
    assert set(schema["discriminator"]["mapping"]) == {
        "classification",
        "regression",
        "clustering",
        "decomposition",
        "anomaly_detection",
        "time_series",
    }

    for factory in REQUEST_FACTORIES:
        Draft202012Validator(schema).validate(factory())
