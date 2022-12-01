import matplotlib.pyplot as plt
import xgboost
import pandas as pd

# print the feature importance value orderly
def feature_importance_value(data: pd.DataFrame, trained_model: any, algorithm_name: str,):
    """
    Draw the feature importance value orderly for analysis.

    Parameters
    ----------
    data: pd.DataFrame (n_samples, n_components)
        Data for silhouette.

    trained_model: any
        The algorithm which to be used

    algorithm_name : str
        the name of the algorithm

    References
    ----------
    XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, f
    lexible and portable. It implements machine learning algorithms under the Gradient Boosting fram
    ework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data
    science problems in a fast and accurate way.

    https://xgboost.readthedocs.io/en/stable/
    """
    # columns_name = ClassificationWorkflowBase.X.columns
    columns_name = data.columns
    for feature_name, score in zip(list(columns_name), trained_model.feature_importances_):
        print(feature_name, ":", score)

# print histograms present feature weights for XGBoost predictions
def feature_weights_histograms(data: pd.DataFrame, trained_model: any, algorithm_name: str,):
    """
    Draw the histograms of feature weights for XGBoost predictions.

    Parameters
    ----------
    data: pd.DataFrame (n_samples, n_components)
        Data for silhouette.

    trained_model: any
        The algorithm which to be used

    algorithm_name : str
        the name of the algorithm

    References
    ----------
    XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, f
    lexible and portable. It implements machine learning algorithms under the Gradient Boosting fram
    ework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data
    science problems in a fast and accurate way.

    https://xgboost.readthedocs.io/en/stable/
    """
    columns_name = data.columns
    plt.figure(figsize=(16, 8))
    plt.bar(range(len(columns_name)), trained_model.feature_importances_, tick_label=columns_name)
    plt.title(f'{algorithm_name} - feature-weights-histograms-plot')

# print feature importance map ranked by importance
def feature_importance_map(trained_model: any, algorithm_name: str,):
    """
    Draw the diagram of feature importance map ranked by importance for analysis.

    Parameters
    ----------
    trained_model: any
        The algorithm which to be used

    algorithm_name : str
        the name of the algorithm

    References
    ----------
    XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, f
    lexible and portable. It implements machine learning algorithms under the Gradient Boosting fram
    ework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data
    science problems in a fast and accurate way.

    https://xgboost.readthedocs.io/en/stable/
    """
    plt.rcParams["figure.figsize"] = (14, 8)
    xgboost.plot_importance(trained_model)
    plt.title(f'{algorithm_name} - feature-importance-map-plot')