import matplotlib.pyplot as plt
import xgboost
import pandas as pd


def feature_importance_value(data: pd.DataFrame, trained_model: any) -> None:
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
    columns_name = data.columns
    for feature_name, score in zip(list(columns_name), trained_model.feature_importances_):
        print(feature_name, ":", score)


def feature_weights_histograms(trained_model: any, image_config: dict, algorithm_name: str) -> None:
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

    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config['width'], image_config['height']), dpi=image_config['dpi'])

    ax.bar(image_config['bar_x'], trained_model.feature_importances_, tick_label=image_config['bar_label'], width=image_config['bar_width'], bottom=image_config['bottom'])

    # automatically optimize picture layout structure
    fig.tight_layout()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_adjustment = (xmax - xmin) * 0.1
    y_adjustment = (ymax - ymin) * 0.1
    ax.axis([xmin - x_adjustment, xmax + x_adjustment, ymin - y_adjustment, ymax + y_adjustment])

    # convert the font of the axes
    plt.tick_params(labelsize=image_config['labelsize'])  # adjust the font size of the axis label
    # plt.setp(ax.get_xticklabels(), rotation=image_config['xrotation'], ha=image_config['xha'],
    #          rotation_mode="anchor")  # axis label rotation Angle
    # plt.setp(ax.get_yticklabels(), rotation=image_config['rot'], ha=image_config['yha'],
    #          rotation_mode="anchor")  # axis label rotation Angle
    x1_label = ax.get_xticklabels()  # adjust the axis label font
    [x1_label_temp.set_fontname(image_config['axislabelfont']) for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname(image_config['axislabelfont']) for y1_label_temp in y1_label]

    ax.set_title(label=algorithm_name, fontdict={"size": image_config['title_size'], "color": image_config['title_color'],
                "family": image_config['title_font']}, loc=image_config['title_location'],
                pad=image_config['title_pad'])


def feature_importance_map(trained_model: any, image_config: dict, algorithm_name: str) -> None:
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
    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config['width'], image_config['height']), dpi=image_config['dpi'])

    # draw the main content
    xgboost.plot_importance(booster=trained_model, ax=image_config['ax'], height=image_config['bar_width'])

    # automatically optimize picture layout structure
    fig.tight_layout()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_adjustment = (xmax - xmin) * 0.1
    y_adjustment = (ymax - ymin) * 0.1
    ax.axis([xmin - x_adjustment, xmax + x_adjustment, ymin - y_adjustment, ymax + y_adjustment])

    # convert the font of the axes
    plt.tick_params(labelsize=image_config['labelsize'])  # adjust the font size of the axis label
    # plt.setp(ax.get_xticklabels(), rotation=image_config['xrotation'], ha=image_config['xha'],
    #          rotation_mode="anchor")  # axis label rotation Angle
    # plt.setp(ax.get_yticklabels(), rotation=image_config['rot'], ha=image_config['yha'],
    #          rotation_mode="anchor")  # axis label rotation Angle
    x1_label = ax.get_xticklabels()  # adjust the axis label font
    [x1_label_temp.set_fontname(image_config['axislabelfont']) for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname(image_config['axislabelfont']) for y1_label_temp in y1_label]

    ax.set_title(label=algorithm_name,
                 fontdict={"size": image_config['title_size'], "color": image_config['title_color'],
                           "family": image_config['title_font']}, loc=image_config['title_location'],
                 pad=image_config['title_pad'])
