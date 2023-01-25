import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def logistic_importance_plot(data: pd.DataFrame, trained_model: any, algorithm_name: str) -> None:
    """
    Draw the feature importance diagram for analysis.

    Parameters
    ----------
    data: pd.DataFrame (n_samples, n_components)
        Data for silhouette.

    trained_model: any
        The algorithm which to be used.

    algorithm_name : str
        The name of the algorithm.

    References
    ----------
    Logistic regression, despite its name, is a linear model for classification rather than regression.
    Logistic regression is also known in the literature as logit regression, maximum-entropy classific
    ation (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible
    outcomes of a single trial are modeled using a logistic function.

    https://scikit-learn.org/stable/modules/linear_model.html/logistic-regression
    """
    columns_name = data.columns
    for feature_name, score in zip(list(columns_name), trained_model.coef_.flatten()):
        print(feature_name, ":", score)

    # feature importance map ranked by coefficient
    coef_lr = pd.DataFrame({
        'var': columns_name,
        'coef': trained_model.coef_.flatten()
    })
    index_sort = np.abs(coef_lr['coef']).sort_values().index
    coef_lr_sort = coef_lr.loc[index_sort, :]

    # Horizontal column chart plot
    fig, ax = plt.subplots(figsize=(14 ,8))
    x, y = coef_lr_sort['var'], coef_lr_sort['coef']
    rects = plt.barh(x, y, color='dodgerblue')
    plt.grid(linestyle="-.", axis='y', alpha=0.4)
    plt.tight_layout()

    # Add data labels
    for rect in rects:
        w = rect.get_width()
        ax.text(w, rect.get_y() + rect.get_height() / 2, '%.2f' % w, ha='left', va='center')
        plt.title(f'{algorithm_name} - importance - plot')