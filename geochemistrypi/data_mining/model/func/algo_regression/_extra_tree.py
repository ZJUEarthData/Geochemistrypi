# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

def feature_importances(X_train: pd.DataFrame, trained_model: object, image_config: dict) -> None:
    """Draw the feature importance bar diagram.

    Parameters
    ----------
    X_train : pd.DataFrame (n_samples, n_components)
        The training feature data.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.
    """
    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config['width'], image_config['height']), dpi=image_config['dpi'])

    # draw the main content
    importances_values = trained_model.feature_importances_
    importances = pd.DataFrame(importances_values, columns=["importance"])
    feature_data = pd.DataFrame(X_train.columns, columns=["feature"])
    importance = pd.concat([feature_data, importances], axis=1)
    importance = importance.sort_values(["importance"], ascending=True)
    importance["importance"] = (importance["importance"]).astype(float)
    importance = importance.sort_values(["importance"])
    importance.set_index('feature', inplace=True)
    rects = importance.plot.barh(alpha=image_config['alpha2'], rot=0)

    # automatically optimize picture layout structure
    fig.tight_layout()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_adjustment = (xmax - xmin) * 0.01
    y_adjustment = (ymax - ymin) * 0.01
    ax.axis([xmin - x_adjustment, xmax + x_adjustment, ymin - y_adjustment, ymax + y_adjustment])

    # convert the font of the axes
    x1_label = ax.get_xticklabels()  # adjust the axis label font
    [x1_label_temp.set_fontname(image_config['axislabelfont']) for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname(image_config['axislabelfont']) for y1_label_temp in y1_label]

    ax.set_title(label=image_config['title_label'],
                 fontdict={"size": image_config['title_size'], "color": image_config['title_color'],
                           "family": image_config['title_font']}, loc=image_config['title_location'],
                 pad=image_config['title_pad'])


