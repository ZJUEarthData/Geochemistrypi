import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def dbscan_result_plot(data: pd.DataFrame, trained_model: any, image_config: dict, algorithm_name: str) -> None:
    """
    Draw the clustering result diagram for analysis.

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
    The DBSCAN algorithm is deterministic, always generating the same clusters when given the same data in the same order.

    https://scikit-learn.org/stable/modules/clustering.html/dbscan

    """
    db = trained_model.fit(data)
    labels = trained_model.labels_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Estimated number of clusters: %d\n" % n_clusters_)
    unique_labels = set(labels)

    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config['width'], image_config['height']), dpi=image_config['dpi'])


    # draw the main content
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        xy = data[class_member_mask & core_samples_mask]
        ax.plot(xy.iloc[:, 0], xy.iloc[:, 1], image_config['marker_angle'], markerfacecolor=tuple(col),
                markeredgecolor=image_config['edgecolor'], markersize=image_config['markersize1'],
                alpha=image_config['alpha1'])
        xy = data[class_member_mask & ~core_samples_mask]
        ax.plot(xy.iloc[:, 0], xy.iloc[:, 1], image_config['marker_circle'], markerfacecolor=tuple(col),
                markeredgecolor=image_config['edgecolor'], markersize=image_config['markersize2'],
                alpha=image_config['alpha2'])

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
                           "family": image_config['title_font']}, loc=image_config['title_location'], pad=image_config['title_pad'])
