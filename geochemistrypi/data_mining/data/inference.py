import copy
import json
import os
from typing import Dict, Optional, Tuple

import mlflow
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import GenericUnivariateSelect, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from ..constants import MLFLOW_ARTIFACT_DATA_PATH
from ..utils.base import save_data, save_model, save_text
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

    def chain(self, transformer_config: Dict, fitted_steps: Optional[Dict] = None) -> object:
        """Chain transformers together into a sklearn pipeline.

        Parameters
        ----------
        transformer_config : Dict
            A dictionary of transformers and their parameters.

        fitted_steps : Dict, optional
            Already-fitted transformer instances keyed by their class name.
            Reusing them avoids a second stateful fit and keeps training and
            inference on the same fitted state.

        Returns
        -------
        object
            A sklearn pipeline.
        """
        transformers = []
        fitted_steps = fitted_steps or {}
        for transformer_name, transformer_params in transformer_config.items():
            if transformer_name in fitted_steps:
                transformers.append(fitted_steps[transformer_name])
            else:
                transformers.append(self.transformer_dict[transformer_name](**transformer_params))
        return make_pipeline(*transformers)


def build_transform_pipeline(
    imputation_config: Dict,
    feature_scaling_config: Dict,
    feature_selection_config: Dict,
    run: object,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    fitted_steps: Optional[Dict] = None,
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

    fitted_steps : Dict, optional
        Already-fitted transformer instances reused for inference. When every
        configured step is already fitted, the pipeline is not refit and the
        training state is shared by model training and inference.

    Returns
    -------
    Tuple[Dict, object]
        The transform pipeline configuration and the transform pipeline object.
    """
    print("Build the transform pipeline according to the previous operations.")
    # Aggregate transformer configuartion.
    transformer_config = {}
    transformer_config.update(imputation_config)
    transformer_config.update(feature_scaling_config)
    transformer_config.update(feature_selection_config)
    transformer_config.update(run.transformer_config)
    transformer_config_str = copy.deepcopy(transformer_config)
    for key, value in transformer_config_str.items():
        for k, v in value.items():
            if callable(v):
                transformer_config_str[key][k] = v.__name__
    transformer_config_str = json.dumps(transformer_config_str, indent=4)
    GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
    save_text(transformer_config_str, "Transform Pipeline Configuration", GEOPI_OUTPUT_ARTIFACTS_PATH, "root")
    # If transformer_config is not {}, then create the transform pipeline.
    if transformer_config:
        # Create the transform pipeline.
        transform_pipeline = PipelineConstrutor().chain(transformer_config, fitted_steps)
        # Fit the transform pipeline with the training data only when at least
        # one configured step has not been fitted on the training rows yet.
        if fitted_steps is None or any(name not in fitted_steps for name in transformer_config):
            transform_pipeline.fit(X_train, y_train)
        # Save the transform pipeline.
        GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH")
        save_model(transform_pipeline, "Transform Pipeline", X_train.iloc[[0]], GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH)
    else:
        transform_pipeline = None
    return transformer_config, transform_pipeline


def model_inference(inference_data: pd.DataFrame, inference_name_column: str, is_inference: bool, run: object, transformer_config: Dict, transform_pipeline: Optional[object] = None):
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
    """
    # If is_inference is True, then run the model inference.
    if is_inference is True:
        print("Use the trained model to make predictions on the application data.")
        # If transformer_config is not {}, then transform the inference data with the transform pipeline.
        if transformer_config:
            inference_data_transformed = transform_pipeline.transform(inference_data)
        else:
            inference_data_transformed = inference_data
        loaded_model = mlflow.sklearn.load_model(f"runs:/{mlflow.active_run().info.run_id}/{run.model_name}")  # noqa: E231
        inference_data_predicted_np = loaded_model.predict(inference_data_transformed)
        inference_data_predicted = np2pd(inference_data_predicted_np, ["Predicted Value"])
        # Keep the source identity so the prediction rows are aligned by
        # explicit identity instead of by position.
        inference_data_predicted.index = inference_data.index
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(inference_data_predicted, inference_name_column, "Application Data Predicted", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
