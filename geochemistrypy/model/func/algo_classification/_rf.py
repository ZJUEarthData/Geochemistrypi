# -*- coding: utf-8 -*-
import pandas as pd


def feature_importances(X_train: pd.DataFrame, trained_model: object) -> None:
    """Draw the feature importance bar diagram.

    Parameters
    ----------
    X_train : pd.DataFrame (n_samples, n_components)
        The training feature data.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.
    """
    importances_values = trained_model.feature_importances_
    importances = pd.DataFrame(importances_values, columns=["importance"])
    feature_data = pd.DataFrame(X_train.columns, columns=["feature"])
    importance = pd.concat([feature_data, importances], axis=1)
    importance = importance.sort_values(["importance"], ascending=True)
    importance["importance"] = (importance["importance"]).astype(float)
    importance = importance.sort_values(["importance"])
    importance.set_index('feature', inplace=True)
    importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(8, 8))
