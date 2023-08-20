import json
import os
from typing import Dict, Optional, Tuple

import mlflow
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from ..constants import MLFLOW_ARTIFACT_DATA_PATH
from ..utils.base import save_data, save_model, save_text
from .data_readiness import np2pd


class PipelineConstrutor:
    """Construct a sklearn pipeline from a dictionary of transformers."""

    @property
    def transformer_dict(self) -> Dict:
        """A dictionary of transformers. Need to be updated when new transformers in the customized automated ML pipeline is added."""
        return {
            "SimpleImputer": SimpleImputer,
            "MinMaxScaler": MinMaxScaler,
            "StandardScaler": StandardScaler,
            "PolynomialFeatures": PolynomialFeatures,
            "RandomOverSampler": RandomOverSampler,
            "RandomUnderSampler": RandomUnderSampler,
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


def build_transform_pipeline(imputation_config: Dict, feature_scaling_config: Dict, run: object, X_train: pd.DataFrame) -> Tuple[Dict, object]:
    """Build the transform pipeline.

    Parameters
    ----------
    imputation_config : Dict
        The imputation configuration.

    feature_scaling_config : Dict
        The feature scaling configuration.

    run : object
        The model selection object.

    X_train : pd.DataFrame
        The training data.

    Returns
    -------
    Tuple[Dict, object]
        The transform pipeline configuration and the transform pipeline object.
    """
    print("-*-*- Transform Pipeline -*-*-")
    # Aggregate transformer configuartion.
    transformer_config = {}
    transformer_config.update(imputation_config)
    transformer_config.update(feature_scaling_config)
    transformer_config.update(run.transformer_config)
    transformer_config_str = json.dumps(transformer_config, indent=4)
    GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
    save_text(transformer_config_str, "Transform Pipeline Configuration", GEOPI_OUTPUT_ARTIFACTS_PATH, "root")
    # If transformer_config is not {}, then create the transform pipeline.
    if transformer_config:
        # Create the transform pipeline.
        transform_pipeline = PipelineConstrutor().chain(transformer_config)
        # Fit the transform pipeline with the training data.
        transform_pipeline.fit(X_train)
        # Save the transform pipeline.
        GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH")
        save_model(transform_pipeline, "Transform Pipeline", X_train.iloc[[0]], GEOPI_OUTPUT_ARTIFACTS_MODEL_PATH)
    else:
        transform_pipeline = None
    return transformer_config, transform_pipeline


def model_inference(inference_data: pd.DataFrame, is_inference: bool, feature_engineering_config: Dict, run: object, transformer_config: Dict, transform_pipeline: Optional[object] = None):
    """Run the model inference.

    Parameters
    ----------
    inference_data : pd.DataFrame
        The inference data.

    is_inference : bool
        Whether to run the model inference.

    feature_engineering_config : Dict
        The feature engineering configuration.

    run : object
        The model selection object.

    transformer_config : Dict
        The transformer configuration.

    transform_pipeline : Optional[object], optional
        The transform pipeline object. The default is None.
    """
    # If inference_data is not None and is_inference is True, then run the model inference.
    if (inference_data is not None) and (is_inference is True):
        print("-*-*- Model Inference -*-*-")
        # If transformer_config is not {}, then transform the inference data with the transform pipeline.
        if transformer_config:
            inference_data_transformed = transform_pipeline.transform(inference_data)
        else:
            inference_data_transformed = inference_data
        loaded_model = mlflow.sklearn.load_model(f"runs:/{mlflow.active_run().info.run_id}/{run.model_name}")
        inference_data_predicted_np = loaded_model.predict(inference_data_transformed)
        inference_data_predicted = np2pd(inference_data_predicted_np, ["Predicted Value"])
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        save_data(inference_data_predicted, "Inference Data Predicted", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
