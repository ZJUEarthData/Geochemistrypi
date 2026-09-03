import json
from pathlib import Path

import pandas as pd
import pytest

from geochemistrypi.scientific_execution import ScientificExecutionContract, ScientificExecutionContractError, save_scientific_execution_attestation, scientific_execution_context

CLASSIFICATION_METHODS = (
    "logistic_regression",
    "support_vector_machine",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "xgboost",
    "multi_layer_perceptron",
    "gradient_boosting",
    "k_nearest_neighbors",
    "stochastic_gradient_descent",
    "adaboost",
)
CLASSIFICATION_DEFAULT_SEEDED_METHODS = frozenset(method for method in CLASSIFICATION_METHODS if method not in {"logistic_regression", "k_nearest_neighbors"})
REGRESSION_ALWAYS_SEEDED_METHODS = (
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "xgboost",
    "multi_layer_perceptron",
)


def _contract(**updates: object) -> dict:
    value = {
        "schema_version": 4,
        "workflow_family": "supervised_learning",
        "workflow_mode": "regression",
        "method": "xgboost",
        "split_strategy": "random_holdout",
        "split_seed": 99,
        "model_seed": 0,
        "cross_validation_folds": 5,
        "evaluation_mode": "internal_holdout",
        "confusion_matrix_normalization": None,
        "external_evaluation_identifier_column": None,
        "external_evaluation_target_columns": [],
        "target_transformations": {"pressure": {"scale": 10.0, "offset": 0.0}},
        "classification_metric_average": None,
        "classification_positive_label": None,
        "model_parameters": {
            "n_estimators": 890,
            "max_depth": 19,
            "min_child_weight": 130.0,
            "base_score": 0.2,
        },
    }
    value.update(updates)
    return value


def _write_contract(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "scientific-execution.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve()


def _constructor_parameters(
    contract: ScientificExecutionContract,
    method: str,
    legacy: dict,
    *,
    class_count=None,
) -> dict:
    return contract.constructor_parameters(
        method,
        legacy,
        workflow_family=contract.workflow_family,
        workflow_mode=contract.workflow_mode,
        class_count=class_count,
    )


def test_scientific_contract_preserves_zero_seed_and_target_transform(
    tmp_path: Path,
) -> None:
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, _contract()))

    assert contract.split_seed == 99
    assert contract.model_seed == 0
    assert contract.cross_validation_folds == 5
    assert _constructor_parameters(contract, "xgboost", {"random_state": 42})["random_state"] == 0
    transformed = contract.transform_targets(pd.DataFrame({"pressure": [1.2, 2.3]}))
    assert transformed["pressure"].tolist() == pytest.approx([12.0, 23.0])


def test_constructor_parameters_bind_family_mode_and_method(
    tmp_path: Path,
) -> None:
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, _contract()))

    with pytest.raises(
        ScientificExecutionContractError,
        match="cannot configure selected CLI workflow",
    ):
        contract.constructor_parameters(
            "xgboost",
            {},
            workflow_family="supervised_learning",
            workflow_mode="classification",
        )
    with pytest.raises(
        ScientificExecutionContractError,
        match="cannot configure selected CLI method",
    ):
        contract.constructor_parameters(
            "decision_tree",
            {},
            workflow_family="supervised_learning",
            workflow_mode="regression",
        )


def test_scientific_contract_rejects_unknown_fields(tmp_path: Path) -> None:
    value = _contract(unregistered=True)
    with pytest.raises(ScientificExecutionContractError, match="Unknown"):
        ScientificExecutionContract.load(_write_contract(tmp_path, value))


def test_classification_xgboost_objective_is_resolved_from_observed_classes(tmp_path: Path) -> None:
    value = _contract(
        workflow_mode="classification",
        split_strategy="random_holdout",
        target_transformations={},
        classification_metric_average="auto",
        model_parameters={"objective": "auto", "importance_type": "gain"},
    )
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, value))

    assert _constructor_parameters(contract, "xgboost", {}, class_count=2)["objective"] == "binary:logistic"
    assert _constructor_parameters(contract, "xgboost", {}, class_count=3)["objective"] == "multi:softprob"

    value["model_parameters"]["objective"] = "multi:softprob"
    incompatible = ScientificExecutionContract.load(_write_contract(tmp_path, value))
    with pytest.raises(ScientificExecutionContractError, match="incompatible with two-class data"):
        _constructor_parameters(incompatible, "xgboost", {}, class_count=2)


@pytest.mark.parametrize("method", CLASSIFICATION_METHODS)
def test_metric_only_contract_accepts_every_manual_classification_method(tmp_path: Path, method: str) -> None:
    value = _contract(
        workflow_mode="classification",
        method=method,
        split_strategy="stratified_holdout",
        model_seed=(0 if method in CLASSIFICATION_DEFAULT_SEEDED_METHODS else None),
        target_transformations={},
        classification_metric_average="auto",
        model_parameters={},
    )

    contract = ScientificExecutionContract.load(_write_contract(tmp_path, value))

    assert contract.method == method
    assert contract.model_parameters == {}


def test_metric_only_contract_rejects_unknown_or_cross_workflow_method(tmp_path: Path) -> None:
    unknown = _contract(
        workflow_mode="classification",
        method="unknown_classifier",
        split_strategy="stratified_holdout",
        target_transformations={},
        classification_metric_average="auto",
        model_parameters={},
    )
    with pytest.raises(ScientificExecutionContractError, match="No generic CLI parameter binding"):
        ScientificExecutionContract.load(_write_contract(tmp_path, unknown))

    cross_workflow = {**unknown, "method": "local_outlier_factor"}
    with pytest.raises(ScientificExecutionContractError, match="not registered for workflow"):
        ScientificExecutionContract.load(_write_contract(tmp_path, cross_workflow))

    conditionally_inapplicable_seed = {
        **unknown,
        "method": "logistic_regression",
        "model_seed": 2025,
    }
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, conditionally_inapplicable_seed))
    with pytest.raises(ScientificExecutionContractError, match="not effective"):
        _constructor_parameters(
            contract,
            "logistic_regression",
            {"solver": "lbfgs"},
        )


@pytest.mark.parametrize(
    ("method", "legacy"),
    (
        ("logistic_regression", {"solver": "liblinear"}),
        (
            "stochastic_gradient_descent",
            {"shuffle": True, "early_stopping": False},
        ),
    ),
)
def test_conditional_classification_seed_is_injected_and_preserves_zero(
    tmp_path: Path,
    method: str,
    legacy: dict,
) -> None:
    value = _contract(
        workflow_mode="classification",
        method=method,
        split_strategy="stratified_holdout",
        model_seed=0,
        target_transformations={},
        classification_metric_average="auto",
        model_parameters={},
    )
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, value))

    assert _constructor_parameters(contract, method, legacy)["random_state"] == 0


@pytest.mark.parametrize(
    ("method", "legacy"),
    (
        ("logistic_regression", {"solver": "lbfgs"}),
        (
            "stochastic_gradient_descent",
            {"shuffle": False, "early_stopping": False},
        ),
    ),
)
def test_conditional_classification_seed_is_rejected_when_algorithm_ignores_it(
    tmp_path: Path,
    method: str,
    legacy: dict,
) -> None:
    value = _contract(
        workflow_mode="classification",
        method=method,
        split_strategy="stratified_holdout",
        model_seed=0,
        target_transformations={},
        classification_metric_average="auto",
        model_parameters={},
    )
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, value))

    with pytest.raises(ScientificExecutionContractError, match="not effective"):
        _constructor_parameters(contract, method, legacy)


@pytest.mark.parametrize("method", REGRESSION_ALWAYS_SEEDED_METHODS)
def test_regression_stochastic_models_bind_and_attest_zero_seed(
    tmp_path: Path,
    method: str,
) -> None:
    contract_path = _write_contract(
        tmp_path,
        _contract(
            method=method,
            model_seed=0,
            target_transformations={},
            model_parameters={},
        ),
    )
    contract = ScientificExecutionContract.load(contract_path)
    assert _constructor_parameters(contract, method, {})["random_state"] == 0

    output = tmp_path / "parameters"
    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _workflow_estimator("regression", method, {"random_state": 0}),
            str(output),
        )
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert "random_state" in record["verified_parameter_names"]


def test_multioutput_regression_attestation_verifies_every_wrapped_estimator(
    tmp_path: Path,
) -> None:
    from sklearn.multioutput import MultiOutputRegressor

    contract_path = _write_contract(tmp_path, _contract())
    parameters = {
        "n_estimators": 890,
        "max_depth": 19,
        "min_child_weight": 130.0,
        "base_score": 0.2,
        "random_state": 0,
    }
    children = [
        _workflow_estimator("regression", "xgboost", parameters),
        _workflow_estimator("regression", "xgboost", parameters),
    ]
    estimator = MultiOutputRegressor(children[0])
    estimator.estimators_ = children
    output = tmp_path / "parameters"

    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(estimator, str(output))
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["verification_status"] == "matched"
    assert record["estimator_identity"]["observed"]["wrapper"] == {
        "module": "sklearn.multioutput",
        "qualname": "MultiOutputRegressor",
        "fitted_estimator_count": 2,
    }

    estimator.estimators_[1] = _Estimator(parameters)
    with pytest.raises(ScientificExecutionContractError, match="estimator identity"):
        with scientific_execution_context(contract_path):
            save_scientific_execution_attestation(
                estimator,
                str(tmp_path / "wrong-parameters"),
            )


@pytest.mark.parametrize(
    ("method", "stochastic", "deterministic"),
    (
        (
            "lasso_regression",
            {"selection": "random"},
            {"selection": "cyclic"},
        ),
        (
            "elastic_net",
            {"selection": "random"},
            {"selection": "cyclic"},
        ),
        (
            "stochastic_gradient_descent",
            {"shuffle": True},
            {"shuffle": False},
        ),
    ),
)
def test_conditional_regression_seed_depends_on_effective_parameters(
    tmp_path: Path,
    method: str,
    stochastic: dict,
    deterministic: dict,
) -> None:
    value = _contract(
        method=method,
        model_seed=0,
        target_transformations={},
        model_parameters={},
    )
    contract_path = _write_contract(tmp_path, value)
    contract = ScientificExecutionContract.load(contract_path)

    assert _constructor_parameters(contract, method, stochastic)["random_state"] == 0
    with pytest.raises(ScientificExecutionContractError, match="not effective"):
        _constructor_parameters(contract, method, deterministic)
    output = tmp_path / "parameters"
    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _workflow_estimator(
                "regression",
                method,
                {**stochastic, "random_state": 0},
            ),
            str(output),
        )
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["estimator_identity"]["expected"]["class_name"] == _workflow_estimator("regression", method, {}).__class__.__name__


@pytest.mark.parametrize("method", ("lasso_regression", "elastic_net"))
def test_deterministic_coordinate_descent_attests_with_an_omitted_model_seed(
    tmp_path: Path,
    method: str,
) -> None:
    contract_path = _write_contract(
        tmp_path,
        _contract(
            method=method,
            model_seed=None,
            target_transformations={},
            model_parameters={},
        ),
    )
    contract = ScientificExecutionContract.load(contract_path)
    assert "random_state" not in _constructor_parameters(
        contract,
        method,
        {"selection": "cyclic"},
    )

    output = tmp_path / "parameters"
    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _workflow_estimator(
                "regression",
                method,
                {"selection": "cyclic"},
            ),
            str(output),
        )
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["verification_status"] == "matched"
    assert record["contract"]["model_seed"] is None


@pytest.mark.parametrize(
    ("workflow_family", "workflow_mode", "method", "evaluation_mode"),
    (
        ("clustering", "clustering", "kmeans", "training_clustering"),
        (
            "clustering",
            "clustering",
            "affinity_propagation",
            "training_clustering",
        ),
        ("dimension_reduction", "embedding", "tsne", "fit_transform"),
        ("dimension_reduction", "embedding", "mds", "fit_transform"),
        (
            "anomaly_detection",
            "outlier_detection",
            "isolation_forest",
            "training_outlier",
        ),
    ),
)
def test_unsupervised_stochastic_models_bind_and_attest_zero_seed(
    tmp_path: Path,
    workflow_family: str,
    workflow_mode: str,
    method: str,
    evaluation_mode: str,
) -> None:
    contract_path = _write_contract(
        tmp_path,
        _contract(
            workflow_family=workflow_family,
            workflow_mode=workflow_mode,
            method=method,
            split_seed=None,
            split_strategy=None,
            model_seed=0,
            evaluation_mode=evaluation_mode,
            target_transformations={},
            model_parameters={},
        ),
    )
    contract = ScientificExecutionContract.load(contract_path)
    assert _constructor_parameters(contract, method, {})["random_state"] == 0

    output = tmp_path / "parameters"
    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _workflow_estimator(workflow_mode, method, {"random_state": 0}),
            str(output),
        )
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["contract"]["evaluation_mode"] == evaluation_mode
    assert "random_state" in record["verified_parameter_names"]


def test_pca_seed_is_bound_only_for_explicit_stochastic_solver(
    tmp_path: Path,
) -> None:
    value = _contract(
        workflow_family="dimension_reduction",
        workflow_mode="embedding",
        method="pca",
        split_seed=None,
        split_strategy=None,
        model_seed=0,
        evaluation_mode="fit_transform",
        target_transformations={},
        model_parameters={},
    )
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, value))

    assert _constructor_parameters(contract, "pca", {"svd_solver": "randomized"})["random_state"] == 0
    with pytest.raises(ScientificExecutionContractError, match="not effective"):
        _constructor_parameters(contract, "pca", {"svd_solver": "full"})
    with pytest.raises(ScientificExecutionContractError, match="not effective"):
        _constructor_parameters(contract, "pca", {"svd_solver": "auto"})
    output = tmp_path / "parameters"
    with scientific_execution_context(_write_contract(tmp_path, value)):
        save_scientific_execution_attestation(
            _workflow_estimator(
                "embedding",
                "pca",
                {"svd_solver": "randomized", "random_state": 0},
            ),
            str(output),
        )
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["estimator_identity"]["expected"]["class_name"] == "PCA"


def test_pca_auto_attests_the_v4_contract_with_an_omitted_model_seed(
    tmp_path: Path,
) -> None:
    contract_path = _write_contract(
        tmp_path,
        _contract(
            workflow_family="dimension_reduction",
            workflow_mode="embedding",
            method="pca",
            split_seed=None,
            split_strategy=None,
            model_seed=None,
            evaluation_mode="fit_transform",
            target_transformations={},
            model_parameters={},
        ),
    )
    contract = ScientificExecutionContract.load(contract_path)
    assert "random_state" not in _constructor_parameters(
        contract,
        "pca",
        {"svd_solver": "auto"},
    )

    output = tmp_path / "parameters"
    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _workflow_estimator(
                "embedding",
                "pca",
                {"svd_solver": "auto"},
            ),
            str(output),
        )
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["verification_status"] == "matched"
    assert record["contract"]["model_seed"] is None


def test_extra_trees_constructor_parameters_bind_structured_values_and_model_seed(tmp_path: Path) -> None:
    value = _contract(
        workflow_mode="classification",
        method="extra_trees",
        split_strategy="stratified_holdout",
        model_seed=2025,
        target_transformations={},
        classification_metric_average="auto",
        model_parameters={
            "n_estimators": 321,
            "max_depth": 9,
            "bootstrap": True,
            "max_samples": 0.75,
        },
    )
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, value))

    observed = _constructor_parameters(
        contract,
        "extra_trees",
        {"n_estimators": 100, "max_depth": 4, "bootstrap": False},
    )

    assert observed == {
        "n_estimators": 321,
        "max_depth": 9,
        "bootstrap": True,
        "max_samples": 0.75,
        "random_state": 2025,
    }


def test_isolation_forest_auto_samples_bind_with_bootstrap_false_and_attest(
    tmp_path: Path,
) -> None:
    contract_path = _write_contract(
        tmp_path,
        _contract(
            workflow_family="anomaly_detection",
            workflow_mode="outlier_detection",
            method="isolation_forest",
            split_seed=None,
            split_strategy=None,
            model_seed=2024,
            evaluation_mode="training_outlier",
            target_transformations={},
            model_parameters={
                "n_estimators": 100,
                "contamination": 0.3,
                "max_features": 1,
                "bootstrap": False,
                "max_samples": "auto",
            },
        ),
    )
    contract = ScientificExecutionContract.load(contract_path)
    effective = _constructor_parameters(
        contract,
        "isolation_forest",
        {
            "n_estimators": 10,
            "contamination": 0.1,
            "max_features": 1,
            "bootstrap": True,
            "max_samples": 8,
        },
    )

    assert effective["bootstrap"] is False
    assert effective["max_samples"] == "auto"
    assert effective["random_state"] == 2024

    output = tmp_path / "parameters"
    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _workflow_estimator("outlier_detection", "isolation_forest", effective),
            str(output),
        )
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["effective_model_parameters"]["bootstrap"] is False
    assert record["effective_model_parameters"]["max_samples"] == "auto"
    assert record["verification_status"] == "matched"


def test_lof_contract_requires_explicit_native_evaluation_semantics(
    tmp_path: Path,
) -> None:
    value = _contract(
        workflow_family="anomaly_detection",
        workflow_mode="outlier_detection",
        method="local_outlier_factor",
        split_seed=None,
        model_seed=None,
        evaluation_mode="external_labeled",
        external_evaluation_target_columns=["label"],
        target_transformations={},
        model_parameters={"n_neighbors": 20, "contamination": 0.08},
    )
    with pytest.raises(
        ScientificExecutionContractError,
        match="training_outlier or novelty_detection",
    ):
        ScientificExecutionContract.load(_write_contract(tmp_path, value))


def test_lof_attestation_verifies_estimator_identity_and_novelty_mode(
    tmp_path: Path,
) -> None:
    contract_path = _write_contract(
        tmp_path,
        _contract(
            workflow_family="anomaly_detection",
            workflow_mode="outlier_detection",
            method="local_outlier_factor",
            split_seed=None,
            split_strategy=None,
            model_seed=None,
            evaluation_mode="training_outlier",
            target_transformations={},
            model_parameters={},
        ),
    )
    output = tmp_path / "parameters"
    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _workflow_estimator(
                "outlier_detection",
                "local_outlier_factor",
                {"novelty": False},
            ),
            str(output),
        )
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["estimator_identity"]["expected"]["class_name"] == "LocalOutlierFactor"
    assert "novelty" in record["verified_parameter_names"]


def test_classification_only_confusion_normalization_is_fail_closed(
    tmp_path: Path,
) -> None:
    value = _contract(confusion_matrix_normalization="true")
    with pytest.raises(
        ScientificExecutionContractError,
        match="only for classification",
    ):
        ScientificExecutionContract.load(_write_contract(tmp_path, value))


def test_external_labeled_regression_has_no_internal_split_seed(
    tmp_path: Path,
) -> None:
    value = _contract(
        split_seed=None,
        split_strategy=None,
        evaluation_mode="external_labeled",
        external_evaluation_identifier_column="ExternalSampleID",
        external_evaluation_target_columns=["pressure"],
    )
    contract = ScientificExecutionContract.load(_write_contract(tmp_path, value))

    assert contract.workflow_mode == "regression"
    assert contract.evaluation_mode == "external_labeled"
    assert contract.external_evaluation_identifier_column == "ExternalSampleID"
    assert contract.split_seed is None

    value["split_seed"] = 99
    with pytest.raises(
        ScientificExecutionContractError,
        match="complete training cohort.*split_seed",
    ):
        ScientificExecutionContract.load(_write_contract(tmp_path, value))


class _Estimator:
    def __init__(self, parameters: dict, classes=None):
        self.parameters = parameters
        if classes is not None:
            self.classes_ = classes

    def get_params(self, deep: bool = False) -> dict:
        assert deep is False
        return self.parameters


def _classification_estimator(
    method: str,
    parameters: dict,
    classes=(0, 1),
):
    identities = {
        "logistic_regression": ("sklearn.linear_model._logistic", "LogisticRegression"),
        "support_vector_machine": ("sklearn.svm._classes", "SVC"),
        "decision_tree": ("sklearn.tree._classes", "DecisionTreeClassifier"),
        "random_forest": ("sklearn.ensemble._forest", "RandomForestClassifier"),
        "extra_trees": ("sklearn.ensemble._forest", "ExtraTreesClassifier"),
        "xgboost": ("xgboost.sklearn", "XGBClassifier"),
        "multi_layer_perceptron": ("sklearn.neural_network._multilayer_perceptron", "MLPClassifier"),
        "gradient_boosting": ("sklearn.ensemble._gb", "GradientBoostingClassifier"),
        "k_nearest_neighbors": ("sklearn.neighbors._classification", "KNeighborsClassifier"),
        "stochastic_gradient_descent": ("sklearn.linear_model._stochastic_gradient", "SGDClassifier"),
        "adaboost": ("sklearn.ensemble._weight_boosting", "AdaBoostClassifier"),
    }
    module, class_name = identities[method]
    estimator_type = type(class_name, (_Estimator,), {"__module__": module})
    return estimator_type(parameters, classes=classes)


def _workflow_estimator(workflow_mode: str, method: str, parameters: dict):
    identities = {
        ("regression", "decision_tree"): ("sklearn.tree._classes", "DecisionTreeRegressor"),
        ("regression", "random_forest"): ("sklearn.ensemble._forest", "RandomForestRegressor"),
        ("regression", "extra_trees"): ("sklearn.ensemble._forest", "ExtraTreesRegressor"),
        ("regression", "gradient_boosting"): ("sklearn.ensemble._gb", "GradientBoostingRegressor"),
        ("regression", "xgboost"): ("xgboost.sklearn", "XGBRegressor"),
        ("regression", "multi_layer_perceptron"): ("sklearn.neural_network._multilayer_perceptron", "MLPRegressor"),
        ("regression", "lasso_regression"): ("sklearn.linear_model._coordinate_descent", "Lasso"),
        ("regression", "elastic_net"): ("sklearn.linear_model._coordinate_descent", "ElasticNet"),
        ("regression", "stochastic_gradient_descent"): ("sklearn.linear_model._stochastic_gradient", "SGDRegressor"),
        ("clustering", "kmeans"): ("sklearn.cluster._kmeans", "KMeans"),
        ("clustering", "affinity_propagation"): ("sklearn.cluster._affinity_propagation", "AffinityPropagation"),
        ("embedding", "pca"): ("sklearn.decomposition._pca", "PCA"),
        ("embedding", "tsne"): ("sklearn.manifold._t_sne", "TSNE"),
        ("embedding", "mds"): ("sklearn.manifold._mds", "MDS"),
        ("outlier_detection", "isolation_forest"): ("sklearn.ensemble._iforest", "IsolationForest"),
        ("outlier_detection", "local_outlier_factor"): ("sklearn.neighbors._lof", "LocalOutlierFactor"),
    }
    module, class_name = identities[(workflow_mode, method)]
    estimator_type = type(class_name, (_Estimator,), {"__module__": module})
    return estimator_type(parameters)


def _binary_metric_configuration(average: str = "binary") -> dict:
    aggregate_semantic = {"type": "string", "value": "basalt"} if average == "binary" else None
    aggregate_encoded = 0 if average == "binary" else None
    return {
        "schema_version": 2,
        "requested_average": average,
        "effective_average": average,
        "requested_positive_label": ({"type": "string", "value": "basalt"} if average == "binary" else None),
        "aggregate_semantic_positive_label": aggregate_semantic,
        "aggregate_encoded_positive_label": aggregate_encoded,
        "curve_semantic_positive_label": aggregate_semantic or {"type": "string", "value": "granite"},
        "curve_encoded_positive_label": aggregate_encoded if aggregate_encoded is not None else 1,
        "curve_probability_column_index": aggregate_encoded if aggregate_encoded is not None else 1,
        "consumers": {
            "holdout_score": {
                "consumer_kind": "aggregate_metric",
                "effective_average": average,
                "aggregate_encoded_positive_label": aggregate_encoded,
            },
            "cross_validation": {
                "consumer_kind": "aggregate_metric",
                "effective_average": average,
                "aggregate_encoded_positive_label": aggregate_encoded,
            },
            **{
                name: {
                    "consumer_kind": "binary_curve",
                    "curve_encoded_positive_label": aggregate_encoded if aggregate_encoded is not None else 1,
                    "probability_column_index": aggregate_encoded if aggregate_encoded is not None else 1,
                }
                for name in ("precision_recall", "precision_recall_threshold", "roc")
            },
        },
    }


@pytest.mark.parametrize("method", CLASSIFICATION_METHODS)
def test_metric_only_attestation_verifies_each_registered_estimator_identity(tmp_path: Path, method: str) -> None:
    seeded = method in CLASSIFICATION_DEFAULT_SEEDED_METHODS
    contract_path = _write_contract(
        tmp_path,
        _contract(
            workflow_mode="classification",
            method=method,
            split_strategy="stratified_holdout",
            model_seed=0 if seeded else None,
            target_transformations={},
            classification_metric_average="macro",
            model_parameters={},
        ),
    )
    output = tmp_path / "parameters"
    observed_parameters = {"random_state": 0} if seeded else {}
    if method == "stochastic_gradient_descent":
        observed_parameters.update({"shuffle": True, "early_stopping": False})

    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _classification_estimator(
                method,
                observed_parameters,
            ),
            str(output),
            _binary_metric_configuration("macro"),
        )

    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["verification_status"] == "matched"
    assert record["estimator_identity"]["observed"]["qualname"] == record["estimator_identity"]["expected"]["class_name"]


def test_attestation_verifies_actual_estimator_parameters(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path, _contract())
    output = tmp_path / "parameters"
    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _workflow_estimator(
                "regression",
                "xgboost",
                {
                    "n_estimators": 890,
                    "max_depth": 19,
                    "min_child_weight": 130.0,
                    "base_score": 0.2,
                    "random_state": 0,
                },
            ),
            str(output),
        )
    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["verification_status"] == "matched"
    assert record["contract"]["model_seed"] == 0
    assert record["attestation_sha256"]


def test_attestation_rejects_effective_parameter_mismatch(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path, _contract())
    with pytest.raises(
        ScientificExecutionContractError,
        match="does not match",
    ):
        with scientific_execution_context(contract_path):
            save_scientific_execution_attestation(
                _Estimator(
                    {
                        "n_estimators": 100,
                        "max_depth": 19,
                        "min_child_weight": 130.0,
                        "base_score": 0.2,
                        "random_state": 0,
                    }
                ),
                str(tmp_path / "parameters"),
            )


def test_classification_attestation_verifies_every_binary_metric_consumer(tmp_path: Path) -> None:
    contract_path = _write_contract(
        tmp_path,
        _contract(
            workflow_mode="classification",
            split_strategy="stratified_holdout",
            target_transformations={},
            classification_metric_average="binary",
            classification_positive_label={"type": "string", "value": "basalt"},
            model_parameters={"objective": "auto", "importance_type": "gain"},
        ),
    )
    configuration = _binary_metric_configuration()
    output = tmp_path / "parameters"

    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _classification_estimator(
                "xgboost",
                {"objective": "binary:logistic", "importance_type": "gain", "random_state": 0},
            ),
            str(output),
            configuration,
        )

    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    assert record["classification_metric_semantics"]["aggregate_encoded_positive_label"] == 0
    assert record["classification_metric_semantics"]["curve_probability_column_index"] == 0
    assert record["estimator_identity"]["expected"]["class_name"] == "XGBClassifier"

    wrong_consumer = json.loads(json.dumps(configuration))
    wrong_consumer["consumers"]["roc"]["probability_column_index"] = 1
    with pytest.raises(ScientificExecutionContractError, match="inconsistent probability column"):
        with scientific_execution_context(contract_path):
            save_scientific_execution_attestation(
                _classification_estimator(
                    "xgboost",
                    {"objective": "binary:logistic", "importance_type": "gain", "random_state": 0},
                ),
                str(output),
                wrong_consumer,
            )


def test_classification_attestation_rejects_unconsumed_binary_semantics(tmp_path: Path) -> None:
    contract_path = _write_contract(
        tmp_path,
        _contract(
            workflow_mode="classification",
            split_strategy="stratified_holdout",
            target_transformations={},
            classification_metric_average="binary",
            classification_positive_label={"type": "string", "value": "basalt"},
            model_parameters={"objective": "auto", "importance_type": "gain"},
        ),
    )
    incomplete = _binary_metric_configuration()
    incomplete["consumers"] = {}

    with pytest.raises(ScientificExecutionContractError, match="not consumed"):
        with scientific_execution_context(contract_path):
            save_scientific_execution_attestation(
                _classification_estimator(
                    "xgboost",
                    {"objective": "binary:logistic", "importance_type": "gain", "random_state": 0},
                ),
                str(tmp_path / "parameters"),
                incomplete,
            )


def test_metric_only_attestation_rejects_the_wrong_estimator_identity(tmp_path: Path) -> None:
    contract_path = _write_contract(
        tmp_path,
        _contract(
            workflow_mode="classification",
            method="logistic_regression",
            split_strategy="stratified_holdout",
            model_seed=None,
            target_transformations={},
            classification_metric_average="macro",
            model_parameters={},
        ),
    )

    with pytest.raises(ScientificExecutionContractError, match="estimator identity"):
        with scientific_execution_context(contract_path):
            save_scientific_execution_attestation(
                _Estimator({}, classes=[0, 1]),
                str(tmp_path / "parameters"),
                _binary_metric_configuration("macro"),
            )


@pytest.mark.parametrize(
    ("updates", "parameters"),
    (
        (
            {},
            {
                "n_estimators": 890,
                "max_depth": 19,
                "min_child_weight": 130.0,
                "base_score": 0.2,
                "random_state": 0,
            },
        ),
        (
            {
                "workflow_family": "clustering",
                "workflow_mode": "clustering",
                "method": "kmeans",
                "split_seed": None,
                "split_strategy": None,
                "model_seed": 0,
                "evaluation_mode": "training_clustering",
                "target_transformations": {},
                "model_parameters": {},
            },
            {"random_state": 0},
        ),
        (
            {
                "workflow_family": "dimension_reduction",
                "workflow_mode": "embedding",
                "method": "tsne",
                "split_seed": None,
                "split_strategy": None,
                "model_seed": 0,
                "evaluation_mode": "fit_transform",
                "target_transformations": {},
                "model_parameters": {},
            },
            {"random_state": 0},
        ),
        (
            {
                "workflow_family": "anomaly_detection",
                "workflow_mode": "outlier_detection",
                "method": "isolation_forest",
                "split_seed": None,
                "split_strategy": None,
                "model_seed": 0,
                "evaluation_mode": "training_outlier",
                "target_transformations": {},
                "model_parameters": {},
            },
            {"random_state": 0},
        ),
    ),
    ids=("regression", "clustering", "embedding", "outlier-detection"),
)
def test_non_classification_attestation_rejects_the_wrong_estimator_identity(
    tmp_path: Path,
    updates: dict,
    parameters: dict,
) -> None:
    contract_path = _write_contract(tmp_path, _contract(**updates))
    output = tmp_path / "parameters"

    with pytest.raises(ScientificExecutionContractError, match="estimator identity"):
        with scientific_execution_context(contract_path):
            save_scientific_execution_attestation(
                _Estimator(parameters),
                str(output),
            )
    assert not (output / "Scientific Execution Attestation.json").exists()


def test_non_binary_aggregate_attestation_keeps_binary_curve_semantics_separate(tmp_path: Path) -> None:
    contract_path = _write_contract(
        tmp_path,
        _contract(
            workflow_mode="classification",
            method="logistic_regression",
            split_strategy="stratified_holdout",
            model_seed=None,
            target_transformations={},
            classification_metric_average="macro",
            model_parameters={},
        ),
    )
    output = tmp_path / "parameters"
    configuration = _binary_metric_configuration("macro")

    with scientific_execution_context(contract_path):
        save_scientific_execution_attestation(
            _classification_estimator("logistic_regression", {}),
            str(output),
            configuration,
        )

    record = json.loads((output / "Scientific Execution Attestation.json").read_text(encoding="utf-8"))
    semantics = record["classification_metric_semantics"]
    assert semantics["effective_average"] == "macro"
    assert semantics["aggregate_encoded_positive_label"] is None
    assert semantics["curve_encoded_positive_label"] == 1
    assert semantics["curve_probability_column_index"] == 1


def test_every_seeded_model_wrapper_preserves_zero() -> None:
    """A valid zero seed must reach each estimator instead of falling back to 42."""

    from geochemistrypi.data_mining.model.classification import (
        AdaBoostClassification,
        DecisionTreeClassification,
        GradientBoostingClassification,
        LogisticRegressionClassification,
        MLPClassification,
        RandomForestClassification,
        SGDClassification,
        SVMClassification,
    )
    from geochemistrypi.data_mining.model.clustering import AffinityPropagationClustering, KMeansClustering
    from geochemistrypi.data_mining.model.decomposition import MDSDecomposition, PCADecomposition, TSNEDecomposition
    from geochemistrypi.data_mining.model.detection import IsolationForestAnomalyDetection
    from geochemistrypi.data_mining.model.regression import (
        DecisionTreeRegression,
        ElasticNetRegression,
        GradientBoostingRegression,
        LassoRegression,
        MLPRegression,
        RandomForestRegression,
        SGDRegression,
    )

    wrappers = (
        SVMClassification(random_state=0),
        DecisionTreeClassification(random_state=0),
        RandomForestClassification(random_state=0),
        LogisticRegressionClassification(solver="saga", random_state=0),
        MLPClassification(random_state=0),
        GradientBoostingClassification(random_state=0),
        AdaBoostClassification(random_state=0),
        SGDClassification(random_state=0),
        DecisionTreeRegression(random_state=0),
        RandomForestRegression(random_state=0),
        MLPRegression(random_state=0),
        GradientBoostingRegression(random_state=0),
        LassoRegression(selection="random", random_state=0),
        ElasticNetRegression(selection="random", random_state=0),
        SGDRegression(random_state=0),
        KMeansClustering(random_state=0),
        AffinityPropagationClustering(random_state=0),
        PCADecomposition(svd_solver="randomized", random_state=0),
        TSNEDecomposition(random_state=0),
        MDSDecomposition(random_state=0),
        IsolationForestAnomalyDetection(random_state=0),
    )

    assert {type(wrapper).__name__: wrapper.model.get_params(deep=False)["random_state"] for wrapper in wrappers} == {type(wrapper).__name__: 0 for wrapper in wrappers}
