# -*- coding: utf-8 -*-
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from rich import print
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, make_scorer, precision_recall_curve, precision_score, recall_score, roc_curve
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

from ....constants import CALCULATION_METHOD_OPTION, SECTION
from ....data.data_readiness import limit_num_input, num2option, num_input


def score(y_true: pd.DataFrame, y_predict: pd.DataFrame) -> tuple[str, Dict]:
    """Calculate the scores of the classification model.

    Parameters
    ----------
    y_true : pd.DataFrame (n_samples, n_components)
        The true target values.

    y_predict : pd.DataFrame (n_samples, n_components)
        The predicted target values.

    Returns
    -------
    average : str
        Metric parameters.

    scores : dict
        The scores of the classification model.
    """
    average = "binary"
    if int(y_true.nunique().values) > 2:
        print("Please select calculation method:")
        print("[bold green]Micro[/bold green]: Calculate metrics globally by counting the total true positives, false negatives and false positives.")
        print("[bold green]Macro[/bold green]: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.")
        print("[bold green]Weighted[/bold green]: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).")
        num2option(CALCULATION_METHOD_OPTION)
        average_num = limit_num_input(CALCULATION_METHOD_OPTION, SECTION[0], num_input)
        if average_num == 1:
            average = "micro"
        elif average_num == 2:
            average = "macro"
        elif average_num == 3:
            average = "weighted"
    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict, average=average)
    recall = recall_score(y_true, y_predict, average=average)
    f1 = f1_score(y_true, y_predict, average=average)
    print("Accuracy: ", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    scores = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }
    return average, scores


def plot_confusion_matrix(y_test: pd.DataFrame, y_test_predict: pd.DataFrame, trained_model: object) -> np.ndarray:
    """Plot the confusion matrix.

    Parameters
    ----------
    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    y_test_predict : pd.DataFrame (n_samples, n_components)
        The predicted target values.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.

    Returns
    -------
    cm : np.ndarray
        The confusion matrix.
    """
    cm = confusion_matrix(y_test, y_test_predict)
    print(cm)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trained_model.classes_)
    disp.plot()
    return cm


def display_cross_validation_scores(scores: np.ndarray, score_name: str) -> Dict:
    """Display the scores of cross-validation.

    Parameters
    ----------
    scores : np.ndarray
        The scores of cross-validation.

    score_name : str
        The name of the score.

    Returns
    -------
    cv_scores : dict
        The scores of cross-validation.
    """
    cv_scores = {
        "Fold Scores": str(scores.tolist()),
        "Mean": scores.mean(),
        "Standard Deviation": scores.std(),
    }
    print("Scores:", cv_scores["Fold Scores"])
    print("Mean:", cv_scores["Mean"])
    print("Standard deviation:", cv_scores["Standard Deviation"])
    mlflow.log_metric(f"CV - {score_name} - Mean", cv_scores["Mean"])
    mlflow.log_metric(f"CV - {score_name} - Standard Deviation", cv_scores["Standard Deviation"])
    return cv_scores


def cross_validation(trained_model: object, X_train: pd.DataFrame, y_train: pd.DataFrame, average: str, cv_num: int = 10) -> Dict:
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Parameters
    ----------
    trained_model : object
        The model trained.

    X_train : pd.DataFrame (n_samples, n_components)
        The training feature data.

    y_train : pd.DataFrame (n_samples, n_components)
        The training target values.

    average : str
        Metric parameters.

    cv_num : int
        Determines the cross-validation splitting strategy.

    Returns
    -------
    scores_result : dict
        The scores of cross-validation.
    """
    if average == "binary":
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="binary"),
            "recall": make_scorer(recall_score, average="binary"),
            "f1": make_scorer(f1_score, average="binary"),
        }
    elif average == "micro":
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="micro"),
            "recall": make_scorer(recall_score, average="micro"),
            "f1": make_scorer(f1_score, average="micro"),
        }
    elif average == "macro":
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="macro"),
            "recall": make_scorer(recall_score, average="macro"),
            "f1": make_scorer(f1_score, average="macro"),
        }
    elif average == "weighted":
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="weighted"),
            "recall": make_scorer(recall_score, average="weighted"),
            "f1": make_scorer(f1_score, average="weighted"),
        }
    scores = cross_validate(trained_model, X_train, y_train, scoring=scoring, cv=cv_num)
    del scores["fit_time"]
    del scores["score_time"]
    # the keys follow the returns of cross_validate in scikit-learn
    scores2display = {
        "test_accuracy": "Accuracy",
        "test_precision": "Precision",
        "test_recall": "Recall",
        "test_f1": "F1 Score",
    }
    scores_result = {"K-Fold": cv_num}
    for key, values in scores.items():
        print("*", scores2display[key], "*")
        cv_scores = display_cross_validation_scores(values, scores2display[key])
        scores_result[scores2display[key]] = cv_scores
        print("-------------")
    return scores_result


def plot_precision_recall(X_test, y_test, trained_model: object, algorithm_name: str) -> tuple:
    """Plot the precision-recall curve.

    Parameters
    ----------
    X_test : pd.DataFrame (n_samples, n_components)
        The testing feature data.

    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    trained_model : object
        The model trained.

    algorithm_name : str
        The name of the algorithm.

    Returns
    -------
    y_probs : np.ndarray
        The probabilities of the model.

    precisions : np.ndarray
        The precision of the model.

    recalls : np.ndarray
        The recall of the model.

    thresholds : np.ndarray
        The thresholds of the model.
    """
    #  Predict probabilities for the positive class
    y_probs = trained_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    plt.figure()
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend(labels=["Precision", "Recall"], loc="best")
    plt.title(f"Precision Recall Curve - {algorithm_name}")
    return y_probs, precisions, recalls, thresholds


def plot_ROC(X_test: pd.DataFrame, y_test: pd.DataFrame, trained_model: object, algorithm_name: str) -> tuple:
    """Plot the ROC curve.

    Parameters
    ----------
    X_test : pd.DataFrame (n_samples, n_components)
        The testing feature data.

    y_test : pd.DataFrame (n_samples, n_components)
        The testing target values.

    trained_model : object
        The model trained.

    algorithm_name : str
        The name of the algorithm.

    Returns
    -------
    y_probs : np.ndarray
        The probabilities of the model.

    fpr : np.ndarray
        The false positive rate of the model.

    tpr : np.ndarray
        The true positive rate of the model.

    thresholds : np.ndarray
        The thresholds of the model.
    """
    y_probs = trained_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve - {algorithm_name}")
    return y_probs, fpr, tpr, thresholds


def plot_2d_decision_boundary(X: pd.DataFrame, X_test: pd.DataFrame, trained_model: object, image_config: Dict) -> None:
    """Plot the decision boundary of the trained model with the testing data set below.

    Parameters
    ----------
    X : pd.DataFrame (n_samples, n_components)
        The complete feature data.

    X_test : pd.DataFrame (n_samples, n_components)
        The testing feature data.

    trained_model : sklearn algorithm model
        The sklearn algorithm model trained with X_train data.

    image_config : dict
        The configuration of the image.
    """

    # Prepare the data for contour plot

    x0s = np.arange(X.iloc[:, 0].min(), X.iloc[:, 0].max(), (X.iloc[:, 0].max() - X.iloc[:, 0].min()) / 50)
    x1s = np.arange(X.iloc[:, 1].min(), X.iloc[:, 1].max(), (X.iloc[:, 1].max() - X.iloc[:, 1].min()) / 50)
    x0, x1 = np.meshgrid(x0s, x1s)
    input_array = np.c_[x0.ravel(), x1.ravel()]
    labels = trained_model.predict(input_array).reshape(x0.shape)

    # Use the code below when the dimensions of X are greater than 2

    # create drawing canvas
    fig, ax = plt.subplots(figsize=(image_config["width"], image_config["height"]), dpi=image_config["dpi"])
    ax.patch.set_facecolor("cornsilk")

    # draw the main content
    ax.contourf(x0, x1, labels, cmap=image_config["cmap2"], alpha=image_config["alpha1"])
    ax.scatter(
        X_test.iloc[:, 0],
        X_test.iloc[:, 1],
        alpha=image_config["alpha2"],
        marker=image_config["marker_angle"],
        cmap=image_config["cmap"],
    )
    # automatically optimize picture layout structure
    fig.tight_layout()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_adjustment = (xmax - xmin) * 0.01
    y_adjustment = (ymax - ymin) * 0.01
    ax.axis([xmin - x_adjustment, xmax + x_adjustment, ymin - y_adjustment, ymax + y_adjustment])

    # convert the font of the axes
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    plt.tick_params(labelsize=image_config["labelsize"])  # adjust the font size of the axis label
    # plt.setp(ax.get_xticklabels(), rotation=image_config['xrotation'], ha=image_config['xha'],
    #          rotation_mode="anchor")  # axis label rotation Angle
    # plt.setp(ax.get_yticklabels(), rotation=image_config['rot'], ha=image_config['yha'],
    #          rotation_mode="anchor")  # axis label rotation Angle
    x1_label = ax.get_xticklabels()  # adjust the axis label font
    [x1_label_temp.set_fontname(image_config["axislabelfont"]) for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname(image_config["axislabelfont"]) for y1_label_temp in y1_label]

    ax.set_title(
        label=image_config["title_label"],
        fontdict={
            "size": image_config["title_size"],
            "color": image_config["title_color"],
            "family": image_config["title_font"],
        },
        loc=image_config["title_location"],
        pad=image_config["title_pad"],
    )


def resampler(X_train: pd.DataFrame, y_train: pd.DataFrame, method: List[str], method_idx: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Use this method when the classification dataset has an unbalanced number of categories.

    Parameters
    ----------
    X_train : pd.DataFrame (n_samples, n_components)
        The train feature data.

    y_train : pd.DataFrame (n_samples, n_label)
        The train label data.

    method : list
        The specific method of resampling.

    method_idx : int
        The index corresponding to the specific method of resampling.

    Returns
    -------
    sample_balance_config : dict
        The configuration of the resampling method.

    X_train_resampled : pd.DataFrame (n_samples, n_components)
        The resampled train feature data.

    y_train_resampled : pd.DataFrame (n_samples, n_label)
        The resampled train label data.
    """

    if method[method_idx] == "Over Sampling":
        resampler = RandomOverSampler()
        X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
        sample_balance_config = {type(resampler).__name__: resampler.get_params()}
    elif method[method_idx] == "Under Sampling":
        resampler = RandomUnderSampler()
        X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
        sample_balance_config = {type(resampler).__name__: resampler.get_params()}
    elif method[method_idx] == "Oversampling and Undersampling":
        over = RandomOverSampler()
        under = RandomUnderSampler()
        over_under_steps = [("oversample", over), ("undersample", under)]
        resampler = Pipeline(steps=over_under_steps)
        X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
        sample_balance_config = {type(resampler[0]).__name__: resampler[0].get_params(), type(resampler[1]).__name__: resampler[1].get_params()}
    return sample_balance_config, X_train_resampled, y_train_resampled


def reset_label(y: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, method: List[str], method_idx: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reset label.

    Parameters
    ----------
    y : pd.DataFrame
        The label data set.

    y_train : pd.DataFrame
        The label data on train set.

    y_test : pd.DataFrame
        The label data on test set.

    method : list
        The specific method of customizing label.

    method_idx : int
        The index corresponding to the specific method of customizing label.

    Returns
    -------
    y_reset : pd.DataFrame
        The label data set after reseting label.

    y_train_reset : pd.DataFrame
        The label data on train set after reseting label.

    y_test_reset : pd.DataFrame
        The label data on test set after reseting label.
    """
    y_colunm_name = list(y)[0]
    label_range = int(y.nunique().values)
    print("----Originla label----\n", y.drop_duplicates().reset_index(drop=True))
    if method[method_idx] == "Automatic Coding":
        label_encoder = LabelEncoder()
        label_encoder.fit(y.squeeze())
        y_reset = label_encoder.transform(y.squeeze())
        y_reset = pd.DataFrame({y_colunm_name: y_reset})
        y_train_reset = label_encoder.transform(y_train.squeeze())
        y_train_reset = pd.DataFrame({y_colunm_name: y_train_reset})
        y_test_reset = label_encoder.transform(y_test.squeeze())
        y_test_reset = pd.DataFrame({y_colunm_name: y_test_reset})
    elif method[method_idx] == "Custom Numeric Labels":
        print("When doing customization, the new label should start from 0.")
        print("Case 1: For binary classification, the range of the new label must be from 0 to 1.")
        print("For example, if your original labels are 2 and 4, you can replace them with 0 and 1 respectively.")
        print("Input format: [bold green][original label,[/bold green] [bold green]new label][/bold green], such as [bold green][2, 0]; [4, 1][/bold green].")
        print("This means you want to replace 2 with 0 and 4 with 1.")
        print(f"Case 2: For multi-classification, the range of the new label must be from 0 to {label_range}(the number of category - 1).")
        print("For example, if your original labels are 2, 4 and 6, you can replace them with 0, 1 and 2 respectively.")
        print("Input format: [bold green][original label,[/bold green] [bold green]new label][/bold green], such as [bold green][2, 0]; [4, 1]; [6, 2][/bold green].")
        print("This means you want to replace 2 with 0, 4 with 1 and 6 with 2.")
        original_new_label = input("Select the label range you want to process.\n" "@input: ")
        label_sequence = re.findall(r"\[(\d+)\,\s*(\d+)\]", original_new_label)
        customize_label_mapping = {int(label): int(value) for label, value in label_sequence}
        y_reset = y.squeeze().map(customize_label_mapping)
        y_reset = pd.DataFrame({y_colunm_name: y_reset})
        y_train_reset = y_train.squeeze().map(customize_label_mapping)
        y_train_reset = pd.DataFrame({y_colunm_name: y_train_reset})
        y_test_reset = y_test.squeeze().map(customize_label_mapping)
        y_test_reset = pd.DataFrame({y_colunm_name: y_test_reset})
    elif method[method_idx] == "Custom Non-numeric Labels":
        print("When doing customization, the new label should start from 0.")
        print("Case 1: For binary classification, the range of the new label must be from 0 to 1.")
        print("For example, if your original labels are 'A' and 'B', you can replace them with 0 and 1 respectively.")
        print("Input format: [bold green]['original label', new label][/bold green], such as [bold green]['A', 0]; ['B', 1][/bold green].")
        print("This means you want to replace A with 0 and B with 1.")
        print(f"Case 2: For multi-classification, the range of the new label must be from 0 to {label_range}(the number of category - 1).")
        print("For example, if your original labels are A, B and C, you can replace them with 0, 1 and 2 respectively.")
        print("Input format: [bold green]['original label', new label][/bold green], such as [bold green]['A', 2]; ['B', 0]; ['C', 1][/bold green].")
        print("This means you want to replace A with 2, B with 0 and C with 1.")
        original_new_label = input("Select the label range you want to process.\n" "@input: ")
        label_sequence = re.findall(r"\[\'(.*?)\',\s*(\d+)\]", original_new_label)
        customize_label_mapping = {label: int(value) for label, value in label_sequence}
        y_reset = y.squeeze().map(customize_label_mapping)
        y_reset = pd.DataFrame({y_colunm_name: y_reset})
        y_train_reset = y_train.squeeze().map(customize_label_mapping)
        y_train_reset = pd.DataFrame({y_colunm_name: y_train_reset})
        y_test_reset = y_test.squeeze().map(customize_label_mapping)
        y_test_reset = pd.DataFrame({y_colunm_name: y_test_reset})

    return y_reset, y_train_reset, y_test_reset
