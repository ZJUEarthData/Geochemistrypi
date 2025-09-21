from typing import Dict

from rich import print

from ....constants import SECTION
from ....data.data_readiness import float_input, num_input


def xgboost_manual_hyper_parameters() -> Dict:
    """Manually set hyperparameters.

    Returns
    -------
    hyper_parameters : dict
    """
    print("N Estimators: The number of trees in the forest.")
    print("Please specify the number of trees in the forest. A good starting range could be between 50 and 500, such as 100.")
    n_estimators = num_input(SECTION[2], "@N Estimators: ")
    print("Learning Rate: It controls the step-size in updating the weights. It shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.")
    print("Please specify the initial learning rate of Xgboost, such as 0.1.")
    learning_rate = float_input(0.01, SECTION[2], "@Learning Rate: ")
    print("Max Depth: The maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.")
    print("Please specify the maximum depth of a tree. A good starting range could be between 3 and 15, such as 4.")
    max_depth = num_input(SECTION[2], "@Max Depth: ")
    print("Subsample: The fraction of samples to be used for fitting the individual base learners.")
    print("Please specify the fraction of samples to be used for fitting the individual base learners. A good starting range could be between 0.5 and 1.0, such as 0.8.")
    subsample = float_input(1, SECTION[2], "@Subsample: ")
    print("Colsample Bytree: The fraction of features to be used for fitting the individual base learners.")
    print("Please specify the fraction of features to be used for fitting the individual base learners. A good starting range could be between 0.5 and 1.0, such as 1.")
    colsample_bytree = float_input(1, SECTION[2], "@Colsample Bytree: ")
    print("Alpha: L1 regularization term on weights.")
    print("Please specify the L1 regularization term on weights. A good starting range could be between 0 and 1.0, such as 0.")
    alpha = float_input(0, SECTION[2], "@Alpha: ")
    print("Lambda: L2 regularization term on weights.")
    print("Please specify the L2 regularization term on weights. A good starting range could be between 0 and 1.0, such as 1.")
    lambd = float_input(1, SECTION[2], "@Lambda: ")
    hyper_parameters = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "alpha": alpha,
        "lambd": lambd,
    }
    return hyper_parameters


# def feature_importance_value(data: pd.DataFrame, trained_model: any) -> None:
#     """
#     Draw the feature importance value orderly for analysis.

#     Parameters
#     ----------
#     data: pd.DataFrame (n_samples, n_components)
#         Data for silhouette.

#     trained_model: any
#         The algorithm which to be used

#     algorithm_name : str
#         the name of the algorithm

#     References
#     ----------
#     XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, f
#     lexible and portable. It implements machine learning algorithms under the Gradient Boosting fram
#     ework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data
#     science problems in a fast and accurate way.

#     https://xgboost.readthedocs.io/en/stable/
#     """
#     columns_name = data.columns
#     for feature_name, score in zip(list(columns_name), trained_model.feature_importances_):
#         print(feature_name, ":", score)


# def feature_weights_histograms(trained_model: any, image_config: dict, algorithm_name: str) -> None:
#     """
#     Draw the histograms of feature weights for XGBoost predictions.

#     Parameters
#     ----------
#     data: pd.DataFrame (n_samples, n_components)
#         Data for silhouette.

#     trained_model: any
#         The algorithm which to be used

#     algorithm_name : str
#         the name of the algorithm

#     References
#     ----------
#     XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, f
#     lexible and portable. It implements machine learning algorithms under the Gradient Boosting fram
#     ework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data
#     science problems in a fast and accurate way.

#     https://xgboost.readthedocs.io/en/stable/
#     """

#     # create drawing canvas
#     fig, ax = plt.subplots(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])

#     ax.bar(
#         image_config["bar_x"],
#         trained_model.feature_importances_,
#         tick_label=image_config["bar_label"],
#         width=image_config["bar_width"],
#         bottom=image_config["bottom"],
#     )

#     # automatically optimize picture layout structure
#     fig.tight_layout()
#     xmin, xmax = ax.get_xlim()
#     ymin, ymax = ax.get_ylim()
#     x_adjustment = (xmax - xmin) * 0.1
#     y_adjustment = (ymax - ymin) * 0.1
#     ax.axis([xmin - x_adjustment, xmax + x_adjustment, ymin - y_adjustment, ymax + y_adjustment])

#     # convert the font of the axes
#     plt.tick_params(labelsize=image_config["labelsize"])  # adjust the font size of the axis label
#     # plt.setp(ax.get_xticklabels(), rotation=image_config['xrotation'], ha=image_config['xha'],
#     #          rotation_mode="anchor")  # axis label rotation Angle
#     # plt.setp(ax.get_yticklabels(), rotation=image_config['rot'], ha=image_config['yha'],
#     #          rotation_mode="anchor")  # axis label rotation Angle
#     x1_label = ax.get_xticklabels()  # adjust the axis label font
#     [x1_label_temp.set_fontname(image_config["axislabelfont"]) for x1_label_temp in x1_label]
#     y1_label = ax.get_yticklabels()
#     [y1_label_temp.set_fontname(image_config["axislabelfont"]) for y1_label_temp in y1_label]

#     ax.set_title(
#         label=algorithm_name,
#         fontdict={
#             "size": image_config["title_size"],
#             "color": image_config["title_color"],
#             "family": image_config["title_font"],
#         },
#         loc=image_config["title_location"],
#         pad=image_config["title_pad"],
#     )


# def feature_importance_map(trained_model: any, image_config: dict, algorithm_name: str) -> None:
#     """
#     Draw the diagram of feature importance map ranked by importance for analysis.

#     Parameters
#     ----------
#     trained_model: any
#         The algorithm which to be used

#     algorithm_name : str
#         the name of the algorithm

#     References
#     ----------
#     XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, f
#     lexible and portable. It implements machine learning algorithms under the Gradient Boosting fram
#     ework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data
#     science problems in a fast and accurate way.

#     https://xgboost.readthedocs.io/en/stable/
#     """
#     # create drawing canvas
#     fig, ax = plt.subplots(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])

#     # draw the main content
#     xgboost.plot_importance(booster=trained_model, ax=image_config["ax"], height=image_config["bar_width"])

#     # automatically optimize picture layout structure
#     fig.tight_layout()
#     xmin, xmax = ax.get_xlim()
#     ymin, ymax = ax.get_ylim()
#     x_adjustment = (xmax - xmin) * 0.1
#     y_adjustment = (ymax - ymin) * 0.1
#     ax.axis([xmin - x_adjustment, xmax + x_adjustment, ymin - y_adjustment, ymax + y_adjustment])

#     # convert the font of the axes
#     plt.tick_params(labelsize=image_config["labelsize"])  # adjust the font size of the axis label
#     # plt.setp(ax.get_xticklabels(), rotation=image_config['xrotation'], ha=image_config['xha'],
#     #          rotation_mode="anchor")  # axis label rotation Angle
#     # plt.setp(ax.get_yticklabels(), rotation=image_config['rot'], ha=image_config['yha'],
#     #          rotation_mode="anchor")  # axis label rotation Angle
#     x1_label = ax.get_xticklabels()  # adjust the axis label font
#     [x1_label_temp.set_fontname(image_config["axislabelfont"]) for x1_label_temp in x1_label]
#     y1_label = ax.get_yticklabels()
#     [y1_label_temp.set_fontname(image_config["axislabelfont"]) for y1_label_temp in y1_label]

#     ax.set_title(
#         label=algorithm_name,
#         fontdict={
#             "size": image_config["title_size"],
#             "color": image_config["title_color"],
#             "family": image_config["title_font"],
#         },
#         loc=image_config["title_location"],
#         pad=image_config["title_pad"],
#     )
