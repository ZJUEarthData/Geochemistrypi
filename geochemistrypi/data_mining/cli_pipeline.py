# -*- coding: utf-8 -*-
import os
from time import sleep

import mlflow
from rich import print
from rich.console import Console
from rich.prompt import Confirm, Prompt

from .constants import (
    CLASSIFICATION_MODELS,
    CLUSTERING_MODELS,
    DATASET_OUTPUT_PATH,
    DECOMPOSITION_MODELS,
    FEATURE_SCALING_STRATEGY,
    GEO_IMAGE_PATH,
    IMPUTING_STRATEGY,
    MAP_IMAGE_PATH,
    MLFLOW_ARTIFACT_DATA_PATH,
    MODE_OPTION,
    MODEL_OUTPUT_IMAGE_PATH,
    MODEL_PATH,
    NON_AUTOML_MODELS,
    OPTION,
    OUTPUT_PATH,
    REGRESSION_MODELS,
    SECTION,
    STATISTIC_IMAGE_PATH,
    TEST_DATA_OPTION,
    WORKING_PATH,
)
from .data.data_readiness import basic_info, create_sub_data_set, data_split, float_input, limit_num_input, np2pd, num2option, num_input, read_data, show_data_columns
from .data.feature_engineering import FeatureConstructor
from .data.imputation import imputer
from .data.preprocessing import feature_scaler
from .data.statistic import monte_carlo_simulator
from .plot.map_plot import process_world_map
from .plot.statistic_plot import basic_statistic, correlation_plot, distribution_plot, is_imputed, is_null_value, logged_distribution_plot, probability_plot, ratio_null_vs_filled
from .process.classify import ClassificationModelSelection
from .process.cluster import ClusteringModelSelection
from .process.decompose import DecompositionModelSelection
from .process.regress import RegressionModelSelection
from .utils.base import clear_output, log, save_data, show_warning
from .utils.mlflow_utils import retrieve_previous_experiment_id

# create the directories if they didn't exist yet
os.makedirs(MODEL_OUTPUT_IMAGE_PATH, exist_ok=True)
os.makedirs(STATISTIC_IMAGE_PATH, exist_ok=True)
os.makedirs(DATASET_OUTPUT_PATH, exist_ok=True)
os.makedirs(MAP_IMAGE_PATH, exist_ok=True)
os.makedirs(GEO_IMAGE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)


def cli_pipeline(file_name: str) -> None:
    """The command line interface for Geochemistry π."""

    # TODO: If the argument is False, hide all Python level warnings. Developers can turn it on by setting the argument to True.
    show_warning(False)

    logger = log(OUTPUT_PATH, "inner_test.log")
    logger.info("Geochemistry Pi is running.")

    # Display the interactive splash screen when launching the CLI software
    console = Console()
    console.print("\n[bold blue]Welcome to Geochemistry Pi![/bold blue]")
    console.print("[bold]Initializing...[/bold]")
    with console.status("[bold green]Data Loading...[/bold green]", spinner="dots"):
        sleep(2)
        if file_name:
            # If the user provides file name, then load the data from the file.
            data = read_data(file_name=file_name, is_own_data=1)
        else:
            console.print("[bold red]No Data File Provided![/bold red]")
            console.print("[bold green]Built-in Data Loading...[/bold green]")

    # <--- Experiment Setup --->
    logger.debug("Experiment Setup")
    console.print("✨ Input Template [bold magenta][Option1/Option2][/bold magenta] [bold cyan](Default Value)[/bold cyan]: Input Value")
    # Create a new experiment or use the previous experiment
    is_used_previous_experiment = Confirm.ask("✨ Use Previous Experiment", default=False)
    # Set the tracking uri to the local directory, in the future, we can set it to the remote server.
    artifact_localtion = f"file:{WORKING_PATH}/geopi_tracking"
    mlflow.set_tracking_uri(artifact_localtion)
    # print("tracking uri:", mlflow.get_tracking_uri())
    if is_used_previous_experiment:
        old_experiment_id = None
        # If the user doesn't provide the correct experiment name, then ask the user to input again.
        while not old_experiment_id:
            old_experiment_name = Prompt.ask("✨ Previous Experiment Name")
            old_experiment_id = retrieve_previous_experiment_id(old_experiment_name)
        mlflow.set_experiment(experiment_id=old_experiment_id)
        experiment = mlflow.get_experiment(experiment_id=old_experiment_id)
    else:
        new_experiment_name = Prompt.ask("✨ New Experiment", default="GeoPi - Rock Classification")
        new_experiment_tag = Prompt.ask("✨ Experiment Tag Version", default="E - v1.0.0")
        try:
            new_experiment_id = mlflow.create_experiment(name=new_experiment_name, artifact_location=artifact_localtion, tags={"version": new_experiment_tag})
        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                console.print("   The experiment name already exists.", style="bold red")
                console.print("   Use the existing experiment.", style="bold red")
                console.print(f"   '{new_experiment_name}' is activated.", style="bold red")
                new_experiment_id = mlflow.get_experiment_by_name(name=new_experiment_name).experiment_id
            else:
                raise e
        experiment = mlflow.get_experiment(experiment_id=new_experiment_id)
    # print("Artifact Location: {}".format(experiment.artifact_location))
    run_name = Prompt.ask("✨ Run Name", default="Xgboost Algorithm")
    run_tag = Prompt.ask("✨ Run Tag Version", default="R - v1.0.0")
    run_description = Prompt.ask("✨ Run Description", default="Use xgboost for GeoPi classification.")
    mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id, tags={"version": run_tag, "description": run_description})
    clear_output()

    # <--- Built-in Data Loading --->
    logger.debug("Built-in Data Loading")
    # If the user doesn't provide the file name, then load the built-in data set.
    if not file_name:
        print("-*-*- Built-in Data Option-*-*-")
        num2option(TEST_DATA_OPTION)
        test_data_num = limit_num_input(TEST_DATA_OPTION, SECTION[0], num_input)
        if test_data_num == 1:
            file_name = "Data_Regression.xlsx"
        elif test_data_num == 2:
            file_name = "Data_Classification.xlsx"
        elif test_data_num == 3:
            file_name = "Data_Clustering.xlsx"
        elif test_data_num == 4:
            file_name = "Data_Decomposition.xlsx"
        data = read_data(file_name=file_name)
        print(f"Successfully loading the built-in data set '{file_name}'.")
        show_data_columns(data.columns)
        clear_output()

    # <--- World Map Projection --->
    logger.debug("World Map Projection")
    process_world_map(data)

    # <--- Data Selection --->
    logger.debug("Data Selection")
    print("-*-*- Data Selection -*-*-")
    show_data_columns(data.columns)
    data_processed = create_sub_data_set(data)
    clear_output()
    print("The Selected Data Set:")
    print(data_processed)
    clear_output()
    print("Basic Statistical Information: ")
    basic_info(data_processed)
    basic_statistic(data_processed)
    correlation_plot(data_processed.columns, data_processed)
    distribution_plot(data_processed.columns, data_processed)
    logged_distribution_plot(data_processed.columns, data_processed)
    save_data(data_processed, "Data Selected", DATASET_OUTPUT_PATH, MLFLOW_ARTIFACT_DATA_PATH)
    clear_output()

    # <--- Imputation --->
    logger.debug("Imputation")
    print("-*-*- Imputation -*-*-")
    is_null_value(data_processed)
    ratio_null_vs_filled(data_processed)
    imputed_flag = is_imputed(data_processed)
    clear_output()
    if imputed_flag:
        print("-*-*- Strategy for Missing Values -*-*-")
        num2option(IMPUTING_STRATEGY)
        print("Which strategy do you want to apply?")
        strategy_num = limit_num_input(IMPUTING_STRATEGY, SECTION[1], num_input)
        data_processed_imputed_np = imputer(data_processed, IMPUTING_STRATEGY[strategy_num - 1])
        data_processed_imputed = np2pd(data_processed_imputed_np, data_processed.columns)
        del data_processed_imputed_np
        clear_output()
        print("-*-*- Hypothesis Testing on Imputation Method -*-*-")
        print("Null Hypothesis: The distributions of the data set before and after imputing remain the same.")
        print("Thoughts: Check which column rejects null hypothesis.")
        print("Statistics Test Method: Wilcoxon Test")
        monte_carlo_simulator(
            data_processed,
            data_processed_imputed,
            sample_size=data_processed_imputed.shape[0] // 2,
            iteration=100,
            test="wilcoxon",
            confidence=0.05,
        )
        # TODO(sany sanyhew1097618435@163.com): Kruskal Wallis Test - P value - why near 1?
        # print("The statistics test method: Kruskal Wallis Test")
        # monte_carlo_simulator(data_processed, data_processed_imputed, sample_size=50,
        #                       iteration=100, test='kruskal', confidence=0.05)
        probability_plot(data_processed.columns, data_processed, data_processed_imputed)
        basic_info(data_processed_imputed)
        basic_statistic(data_processed_imputed)
        del data_processed
        clear_output()
    else:
        # if the selected data set doesn't need imputation, which means there are no missing values.
        data_processed_imputed = data_processed

    # <--- Feature Engineering --->
    logger.debug("Feature Engineering")
    feature_built = FeatureConstructor(data_processed_imputed)
    feature_built.process_feature_engineering()
    data_processed_imputed = feature_built.data

    # <--- Mode Selection --->
    logger.debug("Mode Selection")
    print("-*-*- Mode Selection -*-*-")
    num2option(MODE_OPTION)
    mode_num = limit_num_input(MODE_OPTION, SECTION[2], num_input)
    clear_output()

    # <--- Data Segmentation --->
    # divide X and y data set when it is supervised learning
    logger.debug("Data Split")
    if mode_num == 1 or mode_num == 2:
        print("-*-*- Data Split - X Set and Y Set -*-*-")
        print("Divide the processing data set into X (feature value) and Y (target value) respectively.")
        # create X data set
        print("Selected sub data set to create X data set:")
        show_data_columns(data_processed_imputed.columns)
        print("The selected X data set:")
        X = create_sub_data_set(data_processed_imputed)
        print("Successfully create X data set.")
        print("The Selected Data Set:")
        print(X)
        print("Basic Statistical Information: ")
        basic_statistic(X)
        save_data(X, "X Without Scaling", DATASET_OUTPUT_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        clear_output()

        # <--- Feature Scaling --->
        print("-*-*- Feature Scaling on X Set -*-*-")
        num2option(OPTION)
        is_feature_scaling = limit_num_input(OPTION, SECTION[1], num_input)
        if is_feature_scaling == 1:
            print("Which strategy do you want to apply?")
            num2option(FEATURE_SCALING_STRATEGY)
            feature_scaling_num = limit_num_input(FEATURE_SCALING_STRATEGY, SECTION[1], num_input)
            X_scaled_np = feature_scaler(X, FEATURE_SCALING_STRATEGY, feature_scaling_num - 1)
            X = np2pd(X_scaled_np, X.columns)
            del X_scaled_np
            print("Data Set After Scaling:")
            print(X)
            print("Basic Statistical Information: ")
            basic_statistic(X)
            save_data(X, "X With Scaling", DATASET_OUTPUT_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        clear_output()

        # create Y data set
        print("-*-*- Data Split - X Set and Y Set-*-*-")
        print("Selected sub data set to create Y data set:")
        show_data_columns(data_processed_imputed.columns)
        print("The selected Y data set:")
        print("Notice: Normally, please choose only one column to be tag column Y, not multiple columns.")
        print("Notice: For classification model training, please choose the label column which has distinctive integers.")
        y = create_sub_data_set(data_processed_imputed)
        print("Successfully create Y data set.")
        print("The Selected Data Set:")
        print(y)
        print("Basic Statistical Information: ")
        basic_statistic(y)
        save_data(y, "y", DATASET_OUTPUT_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        clear_output()

        # create training data and testing data
        print("-*-*- Data Split - Train Set and Test Set -*-*-")
        print("Notice: Normally, set 20% of the dataset aside as test set, such as 0.2")
        test_ratio = float_input(default=0.2, prefix=SECTION[1], slogan="@Test Ratio: ")
        train_test_data = data_split(X, y, test_ratio)
        for key, value in train_test_data.items():
            print("-" * 25)
            print(f"The Selected Data Set: {key}")
            print(value)
            print(f"Basic Statistical Information: {key}")
            basic_statistic(value)
            save_data(value, key, DATASET_OUTPUT_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        X_train, X_test = train_test_data["X train"], train_test_data["X test"]
        y_train, y_test = train_test_data["y train"], train_test_data["y test"]
        del data_processed_imputed
        clear_output()
    else:
        # unsupervised learning
        X = data_processed_imputed
        X_train = data_processed_imputed
        y, X_test, y_train, y_test = None, None, None, None

    # <--- Model Selection --->
    logger.debug("Model Selection")
    print("-*-*- Model Selection -*-*-:")
    Modes2Models = {1: REGRESSION_MODELS, 2: CLASSIFICATION_MODELS, 3: CLUSTERING_MODELS, 4: DECOMPOSITION_MODELS}
    Modes2Initiators = {
        1: RegressionModelSelection,
        2: ClassificationModelSelection,
        3: ClusteringModelSelection,
        4: DecompositionModelSelection,
    }
    MODELS = Modes2Models[mode_num]
    num2option(MODELS)
    all_models_num = len(MODELS) + 1
    print(str(all_models_num) + " - All models above to be trained")
    print("Which model do you want to apply?(Enter the Corresponding Number)")
    MODELS.append("all_models")
    model_num = limit_num_input(MODELS, SECTION[2], num_input)
    clear_output()

    # AutoML-training
    is_automl = False
    model = MODELS[model_num - 1]
    if mode_num == 1 or mode_num == 2:
        if model not in NON_AUTOML_MODELS:
            print("Do you want to employ automated machine learning with respect to this algorithm?" "(Enter the Corresponding Number):")
            num2option(OPTION)
            automl_num = limit_num_input(OPTION, SECTION[2], num_input)
            if automl_num == 1:
                is_automl = True
            clear_output()

    # Model trained selection
    logger.debug("Model Training")
    if model_num != all_models_num:
        # run the designated model
        run = Modes2Initiators[mode_num](model)
        if not is_automl:
            run.activate(X, y, X_train, X_test, y_train, y_test)
        else:
            run.activate(X, y, X_train, X_test, y_train, y_test, is_automl)
    else:
        # gain all models result in the specific mode
        for i in range(len(MODELS) - 1):
            run = Modes2Initiators[mode_num](MODELS[i])
            run.activate(X, y, X_train, X_test, y_train, y_test)
            clear_output()

    mlflow.end_run()
