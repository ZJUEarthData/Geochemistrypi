# -*- coding: utf-8 -*-
import os
from time import sleep
from typing import Optional

import mlflow
from rich import print
from rich.console import Console
from rich.prompt import Confirm, Prompt

from .constants import (
    CLASSIFICATION_MODELS,
    CLUSTERING_MODELS,
    DECOMPOSITION_MODELS,
    FEATURE_SCALING_STRATEGY,
    FEATURE_SELECTION_STRATEGY,
    IMPUTING_STRATEGY,
    MLFLOW_ARTIFACT_DATA_PATH,
    MODE_OPTION,
    NON_AUTOML_MODELS,
    OPTION,
    OUTPUT_PATH,
    REGRESSION_MODELS,
    SECTION,
    TEST_DATA_OPTION,
    WORKING_PATH,
)
from .data.data_readiness import basic_info, create_sub_data_set, data_split, float_input, limit_num_input, np2pd, num2option, num_input, read_data, show_data_columns
from .data.feature_engineering import FeatureConstructor
from .data.imputation import imputer
from .data.inference import build_transform_pipeline, model_inference
from .data.preprocessing import feature_scaler, feature_selector
from .data.statistic import monte_carlo_simulator
from .plot.map_plot import process_world_map
from .plot.statistic_plot import basic_statistic, correlation_plot, distribution_plot, is_imputed, is_null_value, log_distribution_plot, probability_plot, ratio_null_vs_filled
from .process.classify import ClassificationModelSelection
from .process.cluster import ClusteringModelSelection
from .process.decompose import DecompositionModelSelection
from .process.regress import RegressionModelSelection
from .utils.base import check_package, clear_output, create_geopi_output_dir, get_os, install_package, log, save_data, show_warning
from .utils.mlflow_utils import retrieve_previous_experiment_id


def cli_pipeline(training_data_path: str, inference_data_path: Optional[str] = None) -> None:
    """The command line interface for Geochemistry π.

    Parameters
    ----------
    training_data_path : str
        The path of the training data.

    inference_data_path : str, optional
        The path of the inference data, by default None
    """

    # TODO: If the argument is False, hide all Python level warnings. Developers can turn it on by setting the argument to True.
    show_warning(False)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    logger = log(OUTPUT_PATH, "geopi_inner_test.log")
    logger.info("Geochemistry Pi is running.")

    # Display the interactive splash screen when launching the CLI software
    console = Console()
    print("\n[bold blue]Welcome to Geochemistry π![/bold blue]")
    print("[bold]Initializing...[/bold]")

    # <-- User Training Data Loading -->
    with console.status("[bold green]Training Data Loading...[/bold green]", spinner="dots"):
        sleep(0.75)
    if training_data_path:
        # If the user provides file name, then load the training data from the file.
        data = read_data(file_path=training_data_path, is_own_data=1)
        print("[bold green]Successfully Loading Own Training Data![bold green]")
    else:
        print("[bold red]No Training Data File Provided![/bold red]")
        print("[bold green]Built-in Data Loading.[/bold green]")

    # <-- User Inference Data Loading -->
    with console.status("[bold green]Inference Data Loading...[/bold green]", spinner="dots"):
        sleep(0.75)
    is_built_in_inference_data = False
    if training_data_path and inference_data_path:
        # If the user provides file name, then load the inference data from the file.
        inference_data = read_data(file_path=inference_data_path, is_own_data=1)
        print("[bold green]Successfully Loading Own Inference Data![bold green]")
    elif training_data_path and (not inference_data_path):
        # If the user doesn't provide the inference data path, it means that the user doesn't want to run the model inference.
        inference_data = None
        print("[bold red]No Inference Data File Provided![/bold red]")
    elif (not training_data_path) and (not inference_data_path):
        is_built_in_inference_data = True
        print("[bold red]No Inference Data File Provided![/bold red]")
        print("[bold green]Built-in Inference Data Loading.[/bold green]")

    # <-- Dependency Checking -->
    with console.status("[bold green]Denpendency Checking...[/bold green]", spinner="dots"):
        sleep(0.75)
    my_os = get_os()
    # Check the dependency of the basemap or cartopy to project the data on the world map later.
    if my_os == "Windows" or my_os == "Linux":
        if not check_package("basemap"):
            print("[bold red]Downloading Basemap...[/bold red]")
            install_package("basemap")
            print("[bold green]Successfully downloading![/bold green]")
            print("[bold green]Download happens only once![/bold green]")
            clear_output()
    elif my_os == "macOS":
        if not check_package("cartopy"):
            print("[bold red]Downloading Cartopy...[/bold red]")
            install_package("cartopy")
            print("[bold green]Successfully downloading![/bold green]")
            print("[bold green]Downloading happens only once![/bold green]")
            clear_output()
    else:
        print("[bold red]Unsupported Operating System![/bold red]")
        print("[bold red]Please use Windows, Linux or macOS.[/bold red]")
        exit(1)

    # <--- Experiment Setup --->
    logger.debug("Experiment Setup")
    console.print("✨ Press [bold magenta]Ctrl + C[/bold magenta] to exit our software at any time.")
    console.print("✨ Input Template [bold magenta][Option1/Option2][/bold magenta] [bold cyan](Default Value)[/bold cyan]: Input Value")
    # Create a new experiment or use the previous experiment
    is_used_previous_experiment = Confirm.ask("✨ Use Previous Experiment", default=False)
    # Set the tracking uri to the local directory, in the future, we can set it to the remote server.
    artifact_localtion = f"file:{WORKING_PATH}/geopi_tracking"
    mlflow.set_tracking_uri(artifact_localtion)
    # Print the tracking uri for debugging.
    # print("tracking uri:", mlflow.get_tracking_uri())
    if is_used_previous_experiment:
        # List all existing experiment names
        existing_experiments = mlflow.search_experiments()
        print("   [underline]Experiment Index: Experiment Name[/underline]")
        for idx, exp in enumerate(existing_experiments):
            print(f"   [bold underline magenta]Experiment {idx}: {exp.name}[/bold underline magenta]")
        old_experiment_id = None
        # If the user doesn't provide the correct experiment name, then ask the user to input again.
        while not old_experiment_id:
            old_experiment_name = Prompt.ask("✨ Previous Experiment Name")
            old_experiment_id = retrieve_previous_experiment_id(old_experiment_name)
        mlflow.set_experiment(experiment_id=old_experiment_id)
        experiment = mlflow.get_experiment(experiment_id=old_experiment_id)
    else:
        new_experiment_name = Prompt.ask("✨ New Experiment", default="GeoPi - Rock Classification")
        # new_experiment_tag = Prompt.ask("✨ Experiment Tag Version", default="E - v1.0.0")
        try:
            # new_experiment_id = mlflow.create_experiment(name=new_experiment_name, artifact_location=artifact_localtion, tags={"version": new_experiment_tag})
            new_experiment_id = mlflow.create_experiment(name=new_experiment_name, artifact_location=artifact_localtion)
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
    run_name = Prompt.ask("✨ Run Name", default="Xgboost Algorithm - Test 1")
    # run_tag = Prompt.ask("✨ Run Tag Version", default="R - v1.0.0")
    # run_description = Prompt.ask("✨ Run Description", default="Use xgboost for GeoPi classification.")
    # mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id, tags={"version": run_tag, "description": run_description})
    mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id)
    create_geopi_output_dir(experiment.name, run_name)
    clear_output()

    # <--- Built-in Training Data Loading --->
    logger.debug("Built-in Training Data Loading")
    # If the user doesn't provide the training data path, then use the built-in training data.
    if not training_data_path:
        print("-*-*- Built-in Training Data Option-*-*-")
        num2option(TEST_DATA_OPTION)
        built_in_training_data_num = limit_num_input(TEST_DATA_OPTION, SECTION[0], num_input)
        if built_in_training_data_num == 1:
            training_data_path = "Data_Regression.xlsx"
        elif built_in_training_data_num == 2:
            training_data_path = "Data_Classification.xlsx"
        elif built_in_training_data_num == 3:
            training_data_path = "Data_Clustering.xlsx"
        elif built_in_training_data_num == 4:
            training_data_path = "Data_Decomposition.xlsx"
        data = read_data(file_path=training_data_path)
        print(f"Successfully loading the built-in training data set '{training_data_path}'.")
        show_data_columns(data.columns)
        clear_output()

    # <--- Built-in Inference Data Loading --->
    logger.debug("Built-in Inference Data Loading")
    # If the user doesn't provide training data path and inference data path, then use the built-in inference data.
    if is_built_in_inference_data:
        print("-*-*- Built-in Inference Data Option-*-*-")
        num2option(TEST_DATA_OPTION)
        built_in_inference_data_num = limit_num_input(TEST_DATA_OPTION, SECTION[0], num_input)
        if built_in_inference_data_num == 1:
            inference_data_path = "InferenceData_Regression.xlsx"
        elif built_in_inference_data_num == 2:
            inference_data_path = "InferenceData_Classification.xlsx"
        elif built_in_inference_data_num == 3:
            inference_data_path = "InferenceData_Clustering.xlsx"
        elif built_in_inference_data_num == 4:
            inference_data_path = "InferenceData_Decomposition.xlsx"
        inference_data = read_data(file_path=inference_data_path)
        print(f"Successfully loading the built-in inference data set '{inference_data_path}'.")
        show_data_columns(inference_data.columns)
        clear_output()

    # <--- World Map Projection --->
    logger.debug("World Map Projection")
    print("-*-*- World Map Projection -*-*-")
    process_world_map(data)

    # <--- Data Selection --->
    logger.debug("Data Selection")
    print("-*-*- Data Selection -*-*-")
    show_data_columns(data.columns)
    data_selected = create_sub_data_set(data)
    clear_output()
    print("The Selected Data Set:")
    print(data_selected)
    clear_output()
    print("Basic Statistical Information: ")
    basic_info(data_selected)
    basic_statistic(data_selected)
    correlation_plot(data_selected.columns, data_selected)
    distribution_plot(data_selected.columns, data_selected)
    log_distribution_plot(data_selected.columns, data_selected)
    GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
    save_data(data, "Data Original", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
    save_data(data_selected, "Data Selected", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
    clear_output()

    # <--- Imputation --->
    logger.debug("Imputation")
    print("-*-*- Imputation -*-*-")
    is_null_value(data_selected)
    ratio_null_vs_filled(data_selected)
    imputed_flag = is_imputed(data_selected)
    clear_output()
    if imputed_flag:
        print("-*-*- Strategy for Missing Values -*-*-")
        num2option(IMPUTING_STRATEGY)
        print("Which strategy do you want to apply?")
        strategy_num = limit_num_input(IMPUTING_STRATEGY, SECTION[1], num_input)
        imputation_config, data_selected_imputed_np = imputer(data_selected, IMPUTING_STRATEGY[strategy_num - 1])
        data_selected_imputed = np2pd(data_selected_imputed_np, data_selected.columns)
        del data_selected_imputed_np
        clear_output()
        print("-*-*- Hypothesis Testing on Imputation Method -*-*-")
        print("Null Hypothesis: The distributions of the data set before and after imputing remain the same.")
        print("Thoughts: Check which column rejects null hypothesis.")
        print("Statistics Test Method: Kruskal Test")
        monte_carlo_simulator(
            data_selected,
            data_selected_imputed,
            sample_size=data_selected_imputed.shape[0] // 2,
            iteration=100,
            test="kruskal",
            confidence=0.05,
        )
        probability_plot(data_selected.columns, data_selected, data_selected_imputed)
        basic_info(data_selected_imputed)
        basic_statistic(data_selected_imputed)
        save_data(data_selected_imputed, "Data Selected Imputed", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        del data_selected
        clear_output()
    else:
        # If the selected data set doesn't need imputation, which means there are no missing values.
        imputation_config = {}
        data_selected_imputed = data_selected

    # <--- Feature Engineering --->
    logger.debug("Feature Engineering")
    print("-*-*- Feature Engineering -*-*-")
    feature_builder = FeatureConstructor(data_selected_imputed)
    data_selected_imputed_fe = feature_builder.build()
    # feature_engineering_config is possible to be {}
    feature_engineering_config = feature_builder.config
    del data_selected_imputed

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
        # Supervised learning
        print("-*-*- Data Split - X Set and Y Set -*-*-")
        print("Divide the processing data set into X (feature value) and Y (target value) respectively.")
        # create X data set
        print("Selected sub data set to create X data set:")
        show_data_columns(data_selected_imputed_fe.columns)
        print("The selected X data set:")
        X = create_sub_data_set(data_selected_imputed_fe)
        print("Successfully create X data set.")
        print("The Selected Data Set:")
        print(X)
        print("Basic Statistical Information: ")
        basic_statistic(X)
        save_data(X, "X Without Scaling", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        clear_output()

        # <--- Feature Scaling --->
        print("-*-*- Feature Scaling on X Set -*-*-")
        num2option(OPTION)
        is_feature_scaling = limit_num_input(OPTION, SECTION[1], num_input)
        if is_feature_scaling == 1:
            print("Which strategy do you want to apply?")
            num2option(FEATURE_SCALING_STRATEGY)
            feature_scaling_num = limit_num_input(FEATURE_SCALING_STRATEGY, SECTION[1], num_input)
            feature_scaling_config, X_scaled_np = feature_scaler(X, FEATURE_SCALING_STRATEGY, feature_scaling_num - 1)
            X = np2pd(X_scaled_np, X.columns)
            del X_scaled_np
            print("Data Set After Scaling:")
            print(X)
            print("Basic Statistical Information: ")
            basic_statistic(X)
            save_data(X, "X With Scaling", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        else:
            feature_scaling_config = {}
        clear_output()

        # create Y data set
        print("-*-*- Data Split - X Set and Y Set-*-*-")
        print("Selected sub data set to create Y data set:")
        show_data_columns(data_selected_imputed_fe.columns)
        print("The selected Y data set:")
        print("Notice: Normally, please choose only one column to be tag column Y, not multiple columns.")
        print("Notice: For classification model training, please choose the label column which has distinctive integers.")
        y = create_sub_data_set(data_selected_imputed_fe)
        print("Successfully create Y data set.")
        print("The Selected Data Set:")
        print(y)
        print("Basic Statistical Information: ")
        basic_statistic(y)
        save_data(y, "Y", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        clear_output()

        # <--- Feature Selection --->
        print("-*-*- Feature Selection -*-*-")
        num2option(OPTION)
        is_feature_selection = limit_num_input(OPTION, SECTION[1], num_input)
        if is_feature_selection == 1:
            print("Which strategy do you want to apply?")
            num2option(FEATURE_SELECTION_STRATEGY)
            feature_selection_num = limit_num_input(FEATURE_SELECTION_STRATEGY, SECTION[1], num_input)
            feature_selection_config, X = feature_selector(X, y, mode_num, FEATURE_SELECTION_STRATEGY, feature_selection_num - 1)
            print("--Selected Features-")
            show_data_columns(X.columns)
            save_data(X, "X After Feature Selection", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        else:
            feature_selection_config = {}
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
            save_data(value, key, GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        X_train, X_test = train_test_data["X Train"], train_test_data["X Test"]
        y_train, y_test = train_test_data["Y Train"], train_test_data["Y Test"]
        del data_selected_imputed_fe
        clear_output()
    else:
        # Unsupervised learning
        # Create X data set without data split because it is unsupervised learning
        X = data_selected_imputed_fe
        # <--- Feature Scaling --->
        print("-*-*- Feature Scaling on X Set -*-*-")
        num2option(OPTION)
        is_feature_scaling = limit_num_input(OPTION, SECTION[1], num_input)
        if is_feature_scaling == 1:
            print("Which strategy do you want to apply?")
            num2option(FEATURE_SCALING_STRATEGY)
            feature_scaling_num = limit_num_input(FEATURE_SCALING_STRATEGY, SECTION[1], num_input)
            feature_scaling_config, X_scaled_np = feature_scaler(X, FEATURE_SCALING_STRATEGY, feature_scaling_num - 1)
            X = np2pd(X_scaled_np, X.columns)
            del X_scaled_np
            print("Data Set After Scaling:")
            print(X)
            print("Basic Statistical Information: ")
            basic_statistic(X)
            save_data(X, "X With Scaling", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        else:
            feature_scaling_config = {}
        clear_output()

        feature_selection_config = {}
        # Create training data without data split because it is unsupervised learning
        X_train = X
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
    # Add the option of all models
    all_models_num = len(MODELS) + 1
    print(str(all_models_num) + " - All models above to be trained")
    print("Which model do you want to apply?(Enter the Corresponding Number)")
    MODELS.append("all_models")
    model_num = limit_num_input(MODELS, SECTION[2], num_input)
    clear_output()

    # AutoML hyper parameter tuning control
    is_automl = False
    model_name = MODELS[model_num - 1]
    # If the model is supervised learning, then allow the user to use AutoML.
    if mode_num == 1 or mode_num == 2:
        # If the model is not in the NON_AUTOML_MODELS, then ask the user whether to use AutoML.
        if model_name not in NON_AUTOML_MODELS:
            print("Do you want to employ automated machine learning with respect to this algorithm?" "(Enter the Corresponding Number):")
            num2option(OPTION)
            automl_num = limit_num_input(OPTION, SECTION[2], num_input)
            if automl_num == 1:
                is_automl = True
            clear_output()

    # Model inference control
    is_inference = False
    # If the model is supervised learning, then allow the user to use model inference.
    if mode_num == 1 or mode_num == 2:
        print("-*-*- Feature Engineering on Inference Data -*-*-")
        is_inference = True
        selected_columns = X_train.columns
        if inference_data is not None:
            if feature_engineering_config:
                # If inference_data is not None and feature_engineering_config is not {}, then apply feature engineering with the same operation to the input data.
                print("The same feature engineering operation will be applied to the inference data.")
                new_feature_builder = FeatureConstructor(inference_data)
                inference_data_fe = new_feature_builder.batch_build(feature_engineering_config)
                inference_data_fe_selected = inference_data_fe[selected_columns]
                save_data(inference_data, "Inference Data Original", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
                save_data(inference_data_fe, "Inference Data Feature-Engineering", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
                save_data(inference_data_fe_selected, "Inference Data Feature-Engineering Selected", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
            else:
                print("You have not applied feature engineering to the training data.")
                print("Hence, no feature engineering operation will be applied to the inference data.")
                inference_data_fe_selected = inference_data[selected_columns]
                save_data(inference_data, "Inference Data Original", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
                save_data(inference_data_fe_selected, "Inference Data Selected", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        else:
            # If the user doesn't provide the inference data path, it means that the user doesn't want to run the model inference.
            print("You did not enter inference data.")
            inference_data_fe_selected = None
        clear_output()
    else:
        # If the model is unsupervised learning, then don't allow the user to use model inference.
        inference_data_fe_selected = None

    # <--- Model Training --->
    logger.debug("Model Training")
    # If the user doesn't choose all models, then run the designated model.
    if model_num != all_models_num:
        # run the designated model
        run = Modes2Initiators[mode_num](model_name)
        # If is_automl is False, then run the model without AutoML.
        if not is_automl:
            run.activate(X, y, X_train, X_test, y_train, y_test)
        else:
            run.activate(X, y, X_train, X_test, y_train, y_test, is_automl)
        clear_output()

        # <--- Transform Pipeline --->
        logger.debug("Transform Pipeline")
        transformer_config, transform_pipeline = build_transform_pipeline(imputation_config, feature_scaling_config, feature_selection_config, run, X_train, y_train)
        clear_output()

        # <--- Model Inference --->
        logger.debug("Model Inference")
        if inference_data_fe_selected is not None:
            model_inference(inference_data_fe_selected, is_inference, run, transformer_config, transform_pipeline)
            clear_output()
    else:
        # Run all models
        for i in range(len(MODELS) - 1):
            # Start a nested MLflow run within the current MLflow run
            with mlflow.start_run(run_name=MODELS[i], experiment_id=experiment.experiment_id, nested=True):
                create_geopi_output_dir(experiment.name, run_name, MODELS[i])
                run = Modes2Initiators[mode_num](MODELS[i])
                # If is_automl is False, then run all models without AutoML.
                if not is_automl:
                    run.activate(X, y, X_train, X_test, y_train, y_test)
                else:
                    # If is_automl is True, but MODELS[i] is in the NON_AUTOML_MODELS, then run the model without AutoML.
                    if MODELS[i] in NON_AUTOML_MODELS:
                        run.activate(X, y, X_train, X_test, y_train, y_test)
                    else:
                        # If is_automl is True, and MODELS[i] is not in the NON_AUTOML_MODELS, then run the model with AutoML.
                        run.activate(X, y, X_train, X_test, y_train, y_test, is_automl)

                # <--- Transform Pipeline --->
                logger.debug("Transform Pipeline")
                transformer_config, transform_pipeline = build_transform_pipeline(imputation_config, feature_scaling_config, feature_selection_config, run, X_train, y_train)

                # <--- Model Inference --->
                logger.debug("Model Inference")
                if inference_data_fe_selected is not None:
                    model_inference(inference_data_fe_selected, is_inference, run, transformer_config, transform_pipeline)
                    clear_output()
    mlflow.end_run()
