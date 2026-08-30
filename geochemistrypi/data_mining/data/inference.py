import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import mlflow
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import GenericUnivariateSelect, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from ..constants import MLFLOW_ARTIFACT_DATA_PATH
from ..utils.base import save_data, save_fig, save_model, save_text
from .data_readiness import np2pd
from .preprocessing import MeanNormalScaler


class PipelineConstrutor:
    """Construct a sklearn pipeline from a dictionary of transformers."""

    @property
    def transformer_dict(self) -> Dict:
        """A dictionary of transformers. Need to be updated when new transformers in the customized automated ML pipeline is added."""
        return {
            "SimpleImputer": SimpleImputer,
            "MinMaxScaler": MinMaxScaler,
            "StandardScaler": StandardScaler,
            "MeanNormalScaler": MeanNormalScaler,
            "PolynomialFeatures": PolynomialFeatures,
            "RandomOverSampler": RandomOverSampler,
            "RandomUnderSampler": RandomUnderSampler,
            "GenericUnivariateSelect": GenericUnivariateSelect,
            "SelectKBest": SelectKBest,
        }

    def chain(self, transformer_config: Dict) -> object:
        """Chain transformers together into a sklearn pipeline.

        Parameters
        ----------
        transformer_config : Dict
            A dictionary of transformers and their parameters.

        Returns
        -------
        object
            A sklearn pipeline.
        """
        transformers = []
        for transformer_name, transformer_params in transformer_config.items():
            transformers.append(self.transformer_dict[transformer_name](**transformer_params))
        return make_pipeline(*transformers)


def build_transform_pipeline(
    imputation_config: Dict,
    feature_scaling_config: Dict,
    feature_selection_config: Dict,
    run: object,
    X_train: pd.DataFrame,
    y_train: Optional[pd.DataFrame],
    prefitted_transform_pipeline: Optional[object] = None,
) -> Tuple[Dict, object]:
    """Build the transform pipeline.

    Parameters
    ----------
    imputation_config : Dict
        The imputation configuration.

    feature_scaling_config : Dict
        The feature scaling configuration.

    feature_selection_config : Dict
        The feature selection configuration.

    run : object
        The model selection object.

    X_train : pd.DataFrame
        The training data.

    y_train : pd.DataFrame, optional
        The target data. Unsupervised workflows pass ``None``.

    Returns
    -------
    Tuple[Dict, object]
        The transform pipeline configuration and the transform pipeline object.
    """
    print("Build the transform pipeline according to the previous operations.")

    # Aggregate transformer configuration
    transformer_config = {}
    transformer_config.update(imputation_config)
    transformer_config.update(feature_scaling_config)
    transformer_config.update(feature_selection_config)
    transformer_config.update(run.transformer_config)

    # =====================================================================
    # Multi-output Regression Handling:
    # Skip feature selection when y_train has multiple columns, because
    # univariate feature selection methods (GenericUnivariateSelect,
    # SelectKBest with f_regression) do not support multi-output targets.
    # =====================================================================
    if y_train is not None and y_train.shape[1] > 1:
        print(f"[Multi-output Regression] Detected {y_train.shape[1]} target columns. " "Skipping feature selection (GenericUnivariateSelect / SelectKBest) " "to avoid dimension mismatch.")
        # Remove feature selection transformers from configuration
        transformer_config = {k: v for k, v in transformer_config.items() if k not in ["GenericUnivariateSelect", "SelectKBest"]}
    # =====================================================================

    # Save transformer configuration as JSON for logging
    transformer_config_str = copy.deepcopy(transformer_config)
    for key, value in transformer_config_str.items():
        for k, v in value.items():
            if callable(v):
                transformer_config_str[key][k] = v.__name__
    transformer_config_str = json.dumps(transformer_config_str, indent=4)

    GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
    save_text(transformer_config_str, "Transform Pipeline Configuration", GEOPI_OUTPUT_ARTIFACTS_PATH, "root")

    # If transformer_config is not empty, build and fit the transform pipeline
    if transformer_config:
        # Create the transform pipeline
        transform_pipeline = prefitted_transform_pipeline
        if transform_pipeline is None:
            transform_pipeline = PipelineConstrutor().chain(transformer_config)
            # Fit the transform pipeline with the training data
            transform_pipeline.fit(X_train, y_train)

        # Save the transform pipeline
        GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH")
        save_model(transform_pipeline, "Transform Pipeline", X_train.iloc[[0]], GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH)
    else:
        transform_pipeline = None

    return transformer_config, transform_pipeline


def model_inference(
    inference_data: pd.DataFrame,
    inference_name_column: str,
    is_inference: bool,
    run: object,
    transformer_config: Dict,
    transform_pipeline: Optional[object] = None,
    y_columns: Optional[List[str]] = None,
):
    """Run the model inference.

    Parameters
    ----------
    inference_data : pd.DataFrame
        The inference data.

    inference_name_column: str
        The name of inference_data

    is_inference : bool
        Whether to run the model inference.

    run : object
        The model selection object.

    transformer_config : Dict
        The transformer configuration.

    transform_pipeline : Optional[object], optional
        The transform pipeline object. The default is None.

    y_columns : Optional[List[str]], optional
        The column names of the target variables. The default is None.
    """
    # If is_inference is True, then run the model inference.
    if is_inference is True:
        print("Use the trained model to make predictions on the application data.")

        # If transformer_config is not empty, transform the inference data
        if transformer_config:
            inference_data_transformed = transform_pipeline.transform(inference_data)
        else:
            inference_data_transformed = inference_data

        # Load the trained model from MLflow
        loaded_model = mlflow.sklearn.load_model(f"runs:/{mlflow.active_run().info.run_id}/{run.model_name}")
        inference_data_predicted_np = loaded_model.predict(inference_data_transformed)

        # Support multi-output: generate column names based on the prediction shape
        if y_columns is not None and len(y_columns) > 0:
            prediction_width = 1 if inference_data_predicted_np.ndim == 1 else inference_data_predicted_np.shape[1]
            if len(y_columns) != prediction_width:
                raise ValueError("The number of regression target names does not match the model prediction width: " f"{len(y_columns)} names for {prediction_width} outputs.")
            # Use the original Y column names as the base
            predicted_columns = [f"Predicted_{col}" for col in y_columns]
        else:
            # Generate generic column names based on the prediction shape
            if inference_data_predicted_np.ndim == 1:
                predicted_columns = ["Predicted Value"]
            else:
                predicted_columns = [f"Predicted_Value_{i+1}" for i in range(inference_data_predicted_np.shape[1])]

        # Convert predictions to DataFrame and save
        inference_data_predicted = np2pd(inference_data_predicted_np, predicted_columns)
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(inference_data_predicted, inference_name_column, "Application Data Predicted", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        return inference_data_predicted
    return None


def save_external_regression_evaluation(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    identifiers: pd.Series,
    algorithm_name: str,
) -> None:
    """Save native metrics, joined predictions, and a figure for labeled unseen data."""

    from ..model.func.algo_regression._common import plot_predicted_vs_actual, score

    if actual.shape[1] != predicted.shape[1]:
        raise ValueError("External evaluation target width does not match prediction width.")
    actual_values = actual.reset_index(drop=True).copy()
    predicted_values = predicted.reset_index(drop=True).copy()
    predicted_values.columns = actual_values.columns
    metrics = score(actual_values, predicted_values)
    metrics_path = os.getenv("GEOPI_OUTPUT_METRICS_PATH")
    save_text(
        json.dumps(metrics, indent=4),
        f"External Evaluation Model Score - {algorithm_name}",
        metrics_path,
    )
    joined_predictions = pd.concat(
        [
            actual_values,
            predicted_values.rename(columns=lambda column: f"Predicted_{column}"),
        ],
        axis=1,
    )
    artifacts_path = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
    save_data(
        joined_predictions,
        identifiers.reset_index(drop=True),
        f"External Evaluation Predictions - {algorithm_name}",
        artifacts_path,
        MLFLOW_ARTIFACT_DATA_PATH,
    )
    residuals = actual_values - predicted_values
    residuals.columns = [f"Residual_{column}" for column in actual_values.columns]
    save_data(
        residuals,
        identifiers.reset_index(drop=True),
        f"External Evaluation Residuals - {algorithm_name}",
        artifacts_path,
        MLFLOW_ARTIFACT_DATA_PATH,
    )
    if actual_values.shape[1] == 1:
        figure = plot_predicted_vs_actual(
            predicted_values,
            actual_values,
            algorithm_name,
        )
        image_path = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
        save_fig(
            f"External Predicted vs. Actual - {algorithm_name}",
            image_path,
            "image/model_output",
            figure=figure,
        )
