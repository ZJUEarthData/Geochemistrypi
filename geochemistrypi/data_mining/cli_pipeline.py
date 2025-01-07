# -*- coding: utf-8 -*-
import os
from time import sleep
from typing import Optional

import mlflow
import pandas as pd
from rich import print
from rich.console import Console
from rich.prompt import Confirm, Prompt

from .constants import (
    ANOMALYDETECTION_MODELS,
    BUILT_IN_DATASET_PATH,
    CLASSIFICATION_MODELS,
    CLASSIFICATION_MODELS_WITH_MISSING_VALUES,
    CLUSTERING_MODELS,
    CLUSTERING_MODELS_WITH_MISSING_VALUES,
    DECOMPOSITION_MODELS,
    DROP_MISSING_VALUE_STRATEGY,
    FEATURE_SCALING_STRATEGY,
    FEATURE_SELECTION_STRATEGY,
    IMPUTING_STRATEGY,
    MISSING_VALUE_STRATEGY,
    MLFLOW_ARTIFACT_DATA_PATH,
    MODE_OPTION,
    MODE_OPTION_WITH_MISSING_VALUES,
    NON_AUTOML_MODELS,
    OPTION,
    REGRESSION_MODELS,
    REGRESSION_MODELS_WITH_MISSING_VALUES,
    SECTION,
    TEST_DATA_OPTION,
)
from .data.data_readiness import (
    basic_info,
    create_sub_data_set,
    data_split,
    float_input,
    limit_num_input,
    np2pd,
    num2option,
    num_input,
    read_data,
    select_column_name,
    show_data_columns,
    show_excel_columns,
)
from .data.feature_engineering import FeatureConstructor
from .data.imputation import imputer
from .data.inference import build_transform_pipeline, model_inference
from .data.preprocessing import feature_scaler, feature_selector
from .data.statistic import monte_carlo_simulator
from .enum import DataSource
from .plot.map_plot import process_world_map
from .plot.statistic_plot import basic_statistic, check_missing_value, correlation_plot, distribution_plot, is_null_value, log_distribution_plot, probability_plot, ratio_null_vs_filled
from .process.classify import ClassificationModelSelection
from .process.cluster import ClusteringModelSelection
from .process.decompose import DecompositionModelSelection
from .process.detect import AnomalyDetectionModelSelection
from .process.regress import RegressionModelSelection
from .utils.base import (
    check_package,
    clear_output,
    copy_files,
    copy_files_from_source_dir_to_dest_dir,
    create_geopi_output_dir,
    get_os,
    install_package,
    list_excel_files,
    log,
    save_data,
    show_warning,
)
from .utils.mlflow_utils import retrieve_previous_experiment_id


def cli_pipeline(training_data_path: str, application_data_path: Optional[str] = None, data_source: Optional[DataSource] = None) -> None:
    """The command line interface software for Geochemistry π.
    The business logic of this CLI software can be found in the figures in the README.md file.
    It provides three  MLOps core functionalities:
        1. Continuous Training
        2. Machine Learning Lifecycle Management
        3. Model Inference

    Parameters
    ----------
    training_data_path : str
        The path of the training data.

    application_data_path : str, optional
        The path of the application data, by default None
    """

    # Local test: If the argument is False, hide all Python level warnings. Developers can turn it on by setting the argument to True.
    show_warning(False)

    # Display the interactive splash screen when launching the CLI software
    console = Console()
    print("\n[bold blue]Welcome to Geochemistry π![/bold blue]")
    print("[bold blue]Three cores components:[/bold blue]")
    print("✨ [bold blue]Continuous Training[/bold blue]")
    print("✨ [bold blue]Model Inference[/bold blue]")
    print("✨ [bold blue]Machine Learning Lifecycle Management[/bold blue]")
    print("[bold green]Initializing...[/bold green]")

    # Set the working path based on the data source
    # If the user uses the built-in data, the working path is the desktop, the output path is the desktop.
    # If the user uses the desktop data, the working path is the desktop, the output path is the desktop.
    # If the user uses the any path, the working path is the current working directory, the output path is the current working directory.
    if data_source == DataSource.BUILT_IN:
        # If the user uses the built-in data, the working path is the desktop, the output path is the desktop.
        WORKING_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
    elif data_source == DataSource.DESKTOP:
        # If the user uses the desktop data, the working path is the desktop, the output path is the desktop.
        WORKING_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
        INPUT_PATH = os.path.join(WORKING_PATH, "geopi_input")

        with console.status("[bold green] Data Direcotry Checking ...[/bold green]", spinner="dots"):
            sleep(1)

        def _data_requirement_print():
            print("[bold green]Please restart the software after putting the data in the 'geopi_input' directory.[/bold green]")
            print("[bold green]Currently, the data file format only supports '.xlsx', '.xls', '.csv'.[/bold green]")
            print("[bold green]If you want to activate the model inference, please put the 'application data' in it as well.[/bold green]")
            print("[bold green]Check our online documentation for more information on the format of the 'application data'.[/bold green]")

        if not os.path.exists(INPUT_PATH):
            print("[bold red]The 'geopi_input' directory is not found on the desktop.[/bold red]")
            os.makedirs(INPUT_PATH, exist_ok=True)
            print("[bold green]Creating the 'geopi_input' directory ...[/bold green]")
            print("[bold green]Successfully create 'geopi_input' directory on the desktop.[/bold green]")
            # Copy the built-in datasets to the 'geopi_input' directory on the desktop.
            copy_files_from_source_dir_to_dest_dir(BUILT_IN_DATASET_PATH, INPUT_PATH)
            print("[bold green]Successfully copy the built-in datasets to the 'geopi_input' directory on the desktop.[/bold green]")

        with console.status("[bold green]Data Loading ...[/bold green]", spinner="dots"):
            sleep(1)

        # List all existing Excel files in the 'geopi_input' directory on the desktop.
        existing_excel_files = list_excel_files(INPUT_PATH)
        if len(existing_excel_files) == 0:
            print("[bold red]No data files found in the 'geopi_input' directory on the desktop.[/bold red]")
            _data_requirement_print()
            clear_output("(Press Enter key to exit)")
            exit(1)
        else:
            print("[bold green]Data files are found in the 'geopi_input' directory on the desktop.[/bold green]")
            print(f"[bold green]Total Number of Data Files: {len(existing_excel_files)}[/bold green]")
        show_excel_columns(existing_excel_files)

        # Read the training data from the Excel file.
        print("Please select the training data by index:")
        # Limit the user input to a number within the range of available files and assign the result to training_data_path
        training_data_path = existing_excel_files[limit_num_input(range(1, len(existing_excel_files) + 1), SECTION[0], num_input) - 1]
        is_application_data = Confirm.ask("Do you want to activate the inference functionality", default=False)
        if is_application_data:
            # Read the application data from the Excel file.
            print("Please select the application data by index:")
            # Limit the user input to a number within the range of available files and assign the result to application_data_path
            application_data_path = existing_excel_files[limit_num_input(range(1, len(existing_excel_files) + 1), SECTION[0], num_input) - 1]
    elif data_source == DataSource.ANY_PATH:
        # If the user uses the any path, the working path is the current working directory, the output path is the current working directory.
        WORKING_PATH = os.getcwd()

    # Set the output path to the working path
    OUTPUT_PATH = os.path.join(WORKING_PATH, "geopi_output")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Set the log file path
    logger = log(OUTPUT_PATH, "geopi_inner_test.log")
    logger.info("Geochemistry Pi is running.")

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

    # <-- User Application Data Loading -->
    with console.status("[bold green]Application Data Loading...[/bold green]", spinner="dots"):
        sleep(0.75)
    # Three scenarios for the application data loading:
    # 1. The user provides the training data path and the application data path.
    #   - The user wants to use the model inference.
    # 2. The user provides the training data path but doesn't provide the application data path.
    #   - The user doesn't want to use the model inference.
    # 3. The user doesn't provide the training data path and the application data path.
    #   - The continuous training and model inference will use the built-in data.
    is_built_in_inference_data = False
    if training_data_path and application_data_path:
        # If the user provides file name, then load the inference data from the file.
        inference_data = read_data(file_path=application_data_path, is_own_data=1)
        print("[bold green]Successfully Loading Own Application Data![bold green]")
    elif training_data_path and (not application_data_path):
        # If the user doesn't provide the inference data path, it means that the user doesn't want to run the model inference.
        inference_data = None
        print("[bold red]No Application Data File Provided![/bold red]")
    elif (not training_data_path) and (not application_data_path):
        is_built_in_inference_data = True
        print("[bold red]No Application Data File Provided![/bold red]")
        print("[bold green]Built-in Application Data Loading.[/bold green]")

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
    experiments_location = "file:///" + os.path.join(WORKING_PATH, "geopi_tracking")
    mlflow.set_tracking_uri(experiments_location)
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
            new_experiment_id = mlflow.create_experiment(name=new_experiment_name)
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
    run_name = Prompt.ask("✨ Run Name", default="XGBoost Algorithm - Test 1")
    # run_tag = Prompt.ask("✨ Run Tag Version", default="R - v1.0.0")
    # run_description = Prompt.ask("✨ Run Description", default="Use xgboost for GeoPi classification.")
    # mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id, tags={"version": run_tag, "description": run_description})
    mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id)
    create_geopi_output_dir(OUTPUT_PATH, experiment.name, run_name)
    clear_output()

    # <--- Built-in Training Data Loading --->
    logger.debug("Built-in Training Data Loading")
    # If the user doesn't provide the training data path, then use the built-in training data.
    if not training_data_path:
        print("[bold green]-*-*- Built-in Training Data Option -*-*-[/bold green]")
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
        elif built_in_training_data_num == 5:
            training_data_path = "Data_AnomalyDetection.xlsx"
        data = read_data(file_path=training_data_path)
        print(f"Successfully loading the built-in training data set '{training_data_path}'.")
        show_data_columns(data.columns)
        clear_output()

    # <--- Built-in Application Data Loading --->
    logger.debug("Built-in Application Data Loading")
    # If the user doesn't provide training data path and inference data path, then use the built-in inference data.
    # There are two scenarios for the built-in inference data loading:
    # 1. The user chooses the built-in training data for regression or classification.
    #   - Only the supervised learning mode supports model inference.
    # 2. The user chooses the built-in training data for clustering, decomposition or anomaly detection.
    #   - The unsupervised learning mode doesn't support model inference.
    if is_built_in_inference_data and built_in_training_data_num == 1:
        application_data_path = "ApplicationData_Regression.xlsx"
        inference_data = read_data(file_path=application_data_path)
        print(f"Successfully loading the built-in application data set '{application_data_path}'.")
        show_data_columns(inference_data.columns)
        clear_output()
    elif is_built_in_inference_data and built_in_training_data_num == 2:
        application_data_path = "ApplicationData_Classification.xlsx"
        inference_data = read_data(file_path=application_data_path)
        print(f"Successfully loading the built-in application data set '{application_data_path}'.")
        show_data_columns(inference_data.columns)
        clear_output()
    elif is_built_in_inference_data and built_in_training_data_num == 3:
        inference_data = None
    elif is_built_in_inference_data and built_in_training_data_num == 4:
        inference_data = None
    elif is_built_in_inference_data and built_in_training_data_num == 5:
        inference_data = None

    # <--- Name Selection --->
    logger.debug("Output Data Identifier Column Selection")
    print("[bold green]-*-*- Output Data Identifier Column Selection -*-*-[/bold green]")
    show_data_columns(data.columns)
    NAME = select_column_name(data)
    clear_output()
    name_column_origin = []
    name_column_select = data[NAME]

    # <--- World Map Projection --->
    logger.debug("World Map Projection")
    print("[bold green]-*-*- World Map Projection -*-*-[/bold green]")
    process_world_map(data, name_column_select)

    # <--- Data Selection --->
    logger.debug("Data Selection")
    print("[bold green]-*-*- Data Selection -*-*-[/bold green]")
    show_data_columns(data.columns)
    data_selected = create_sub_data_set(data, allow_empty_columns=False)
    clear_output()
    print("The Selected Data Set:")
    print(data_selected)
    clear_output()
    print("[bold green]-*-*- Basic Statistical Information -*-*-[/bold green]")
    basic_info(data_selected)
    basic_statistic(data_selected)
    correlation_plot(data_selected.columns, data_selected, name_column_select)
    distribution_plot(data_selected.columns, data_selected, name_column_select)
    log_distribution_plot(data_selected.columns, data_selected, name_column_select)
    GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
    save_data(data, name_column_origin, "Data Original", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
    save_data(data_selected, name_column_select, "Data Selected", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
    data_selected_name = pd.concat([name_column_select, data_selected], axis=1)
    clear_output()

    # <--- Missing Value Process --->
    # When detecting no missing values in the selected data, this section will be skipped.
    # Otherwise, there are three scenarios to deal with the missing values.
    # 1. Keep the missing values. Subsequently, in the following section, only the models that support missing values are available.
    # 2. Drop the rows with missing values. It means the impuation is not applied.
    # 3. Impute the missing values with one of the imputation techniques.
    # Reference: https://scikit-learn.org/stable/modules/impute.html
    logger.debug("Missing Value")
    print("[bold green]-*-*- Missing Value Check -*-*-[/bold green]")
    is_null_value(data_selected)
    ratio_null_vs_filled(data_selected)
    # missing_value_flag and process_missing_value_flag will be used in mode selection and model selection to differeniate two scenarios.
    # 1. the selected data set is with missing values -> the user can choose regression, classification and clustering modes and the models support missing values.
    # 2. the selected data set is without missing values -> the user can choose all modes and all models.
    missing_value_flag = check_missing_value(data_selected)
    process_missing_value_flag = False
    # drop_rows_with_missing_value_flag will be used in the inference section to differeniate two scenarios.
    # 1. Dropping the rows with missing values, before implementing the model inference, the missing value in the inference data set should be dropped as well.
    # 2. Don't drop the rows with missing values, before implementing the model inference, the inference data set should be imputed as well.
    # Because dropping the rows with missing values use pandas.DataFrame.dropna() method, while imputing the missing values use sklearn.impute.SimpleImputer() method.
    drop_rows_with_missing_value_flag = False
    # clear_output()
    if missing_value_flag:
        clear_output()
        # Ask the user whether to use imputation techniques to deal with the missing values.
        print("[bold green]-*-*- Missing Values Process -*-*-[/bold green]")
        print("[bold red]Caution: Only some algorithms can process the data with missing value, such as XGBoost for regression and classification![/bold red]")
        print("Do you want to deal with the missing values?")
        num2option(OPTION)
        is_process_missing_value = limit_num_input(OPTION, SECTION[1], num_input)
        if is_process_missing_value == 1:
            process_missing_value_flag = True
            # If the user wants to deal with the missing values, then ask the user which strategy to use.
            clear_output()
            print("[bold green]-*-*- Strategy for Missing Values -*-*-[/bold green]")
            num2option(MISSING_VALUE_STRATEGY)
            print("Notice: Drop the rows with missing values may lead to a significant loss of data if too many features are chosen.")
            print("Which strategy do you want to apply?")
            missing_value_strategy_num = limit_num_input(MISSING_VALUE_STRATEGY, SECTION[1], num_input)
            clear_output()
            if missing_value_strategy_num == 1:
                print("[bold green]-*-*- Drop the rows with Missing Values -*-*-[/bold green]")
                num2option(DROP_MISSING_VALUE_STRATEGY)
                print("Notice: Drop the rows with missing values may lead to a significant loss of data if too many features are chosen.")
                print("Which strategy do you want to apply?")
                drop_missing_value_strategy_num = limit_num_input(DROP_MISSING_VALUE_STRATEGY, SECTION[1], num_input)
                if drop_missing_value_strategy_num == 1:
                    # Drop the rows with missing values
                    data_selected_dropped = data_selected.dropna()
                    # Reset the index of the data set after dropping the rows with missing values.
                    data_selected_dropped = data_selected_dropped.reset_index(drop=True)
                    # Drop the rows with missing values
                    data_selected_dropped_name = data_selected_name.dropna()
                    # Reset the index of the data set after dropping the rows with missing values.
                    data_selected_dropped_name = data_selected_dropped_name.reset_index(drop=True)
                    print("Successfully drop the rows with missing values.")
                    print("The Selected Data Set After Dropping:")
                    print(data_selected_dropped)
                    print("Basic Statistical Information:")
                    drop_name_column = data_selected_dropped_name.iloc[:, 0]
                    save_data(data_selected_dropped, drop_name_column, "Data Selected Dropped-Imputed", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
                    drop_rows_with_missing_value_flag = True
                    imputed_flag = False
                elif drop_missing_value_strategy_num == 2:
                    is_null_value(data_selected)
                    show_data_columns(data_selected.columns)
                    print("Note: The data set schema will remain the same after dropping the rows with missing values by specific columns.")
                    drop_data_selected = create_sub_data_set(data_selected)
                    data_selected_dropped = data_selected
                    data_selected_dropped_name = data_selected_name
                    for column_name in drop_data_selected.columns:
                        # Drop the rows with missing values
                        data_selected_dropped = data_selected_dropped.dropna(subset=[column_name])
                        # Reset the index of the data set after dropping the rows with missing values.
                        data_selected_dropped = data_selected_dropped.reset_index(drop=True)
                        # Drop the rows with missing values
                        data_selected_dropped_name = data_selected_dropped_name.dropna(subset=[column_name])
                        # Reset the index of the data set after dropping the rows with missing values.
                        data_selected_dropped_name = data_selected_dropped_name.reset_index(drop=True)
                    print("Successfully drop the rows with missing values.")
                    print("The Selected Data Set After Dropping:")
                    print(data_selected_dropped)
                    print("Basic Statistical Information:")
                    drop_name_column = data_selected_dropped_name.iloc[:, 0]
                    save_data(data_selected_dropped, drop_name_column, "Data Selected Dropped-Imputed", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
                    drop_rows_with_missing_value_flag = True
                    imputed_flag = False
                    missing_value_flag = check_missing_value(data_selected_dropped)
                    if missing_value_flag:
                        process_missing_value_flag = False
                clear_output()
            elif missing_value_strategy_num == 2:
                # Don't drop the rows with missing values but use imputation techniques to deal with the missing values later.
                # No need to save the data set here because it will be saved after imputation.
                imputed_flag = True
        else:
            # Don't deal with the missing values, which means neither drop the rows with missing values nor use imputation techniques.
            imputed_flag = False
            save_data(data_selected, name_column_select, "Data Selected Dropped-Imputed", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
            clear_output()
    else:
        # If the selected data set doesn't have missing values, then don't deal with the missing values.
        imputed_flag = False
        save_data(data_selected, name_column_select, "Data Selected Dropped-Imputed", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        clear_output()
    data_selected = data_selected_dropped if drop_rows_with_missing_value_flag else data_selected
    process_name_column = data_selected_dropped_name.iloc[:, 0] if drop_rows_with_missing_value_flag else name_column_select
    # If the selected data set contains missing values and the user wants to deal with the missing values and choose not to drop the rows with missing values,
    # then use imputation techniques to deal with the missing values.
    if imputed_flag:
        print("[bold green]-*-*- Imputation Method Option -*-*-[/bold green]")
        num2option(IMPUTING_STRATEGY)
        print("Which method do you want to apply?")
        strategy_num = limit_num_input(IMPUTING_STRATEGY, SECTION[1], num_input)
        imputation_config, data_selected_imputed_np = imputer(data_selected, IMPUTING_STRATEGY[strategy_num - 1])
        data_selected_imputed = np2pd(data_selected_imputed_np, data_selected.columns)
        del data_selected_imputed_np
        clear_output()
        print("[bold green]-*-*- Hypothesis Testing on Imputation Method -*-*-[/bold green]")
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
        probability_plot(data_selected.columns, data_selected, data_selected_imputed, process_name_column)
        basic_info(data_selected_imputed)
        basic_statistic(data_selected_imputed)
        save_data(data_selected_imputed, process_name_column, "Data Selected Dropped-Imputed", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        del data_selected
        clear_output()
    else:
        # If the selected data set doesn't need imputation, which means there are no missing values.
        imputation_config = {}
        data_selected_imputed = data_selected

    # <--- Feature Engineering --->
    logger.debug("Feature Engineering")
    print("[bold green]-*-*- Feature Engineering -*-*-[/bold green]")
    feature_builder = FeatureConstructor(data_selected_imputed, process_name_column)
    data_selected_imputed_fe = feature_builder.build()
    # feature_engineering_config is possible to be {}
    feature_engineering_config = feature_builder.config
    del data_selected_imputed

    # <--- Mode Selection --->
    logger.debug("Mode Selection")
    print("[bold green]-*-*- Mode Selection -*-*-[/bold green]")
    # The following scenarios support three modes (regression, classification and clustering) with the models that support missing values.
    # Because finally, the data contains missing values.
    # 1. missing value flag = True, process_missing_value_flag = False, drop rows with missing values flag = Flase, imputed flag = False
    # The following scenarios support four modes with all models.
    # Because finally, the data is complete.
    # 1. missing value flag = True, process_missing_value_flag = True, drop rows with missing values flag = True, imputed flag = False
    # 2. missing value flag = True, process_missing_value_flag = True, drop rows with missing values flag = False, imputed flag = True
    # 3. missing value flag = False, process_missing_value_flag = False, drop rows with missing values flag = False, imputed flag = False
    # If the selected data set is with missing values and is not been imputed, then only allow the user to choose regression, classification and clustering models.
    # Otherwise, allow the user to choose decomposition models.
    if missing_value_flag and not process_missing_value_flag:
        # The anomaly detection mode and decomposition mode don't support missing values.
        num2option(MODE_OPTION_WITH_MISSING_VALUES)
        mode_num = limit_num_input(MODE_OPTION_WITH_MISSING_VALUES, SECTION[2], num_input)
    else:
        num2option(MODE_OPTION)
        mode_num = limit_num_input(MODE_OPTION, SECTION[2], num_input)
    clear_output()

    # <--- Data Segmentation --->
    # divide X and y data set when it is supervised learning
    logger.debug("Data Divsion")
    name_all = process_name_column
    if mode_num == 1 or mode_num == 2:
        # Supervised learning
        print("[bold green]-*-*- Data Segmentation - X Set and Y Set -*-*-[/bold green]")
        print("Divide the processing data set into X (feature value) and Y (target value) respectively.")
        # create X data set
        print("Selected sub data set to create X data set:")
        show_data_columns(data_selected_imputed_fe.columns)
        print("The selected X data set:")
        X = create_sub_data_set(data_selected_imputed_fe, allow_empty_columns=False)
        print("Successfully create X data set.")
        print("The Selected Data Set:")
        print(X)
        print("Basic Statistical Information: ")
        basic_statistic(X)
        save_data(X, name_all, "X Without Scaling", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        clear_output()

        # Create Y data set
        print("[bold green]-*-*- Data Segmentation - X Set and Y Set -*-*-[/bold green]")
        print("Selected sub data set to create Y data set:")
        show_data_columns(data_selected_imputed_fe.columns)
        print("The selected Y data set:")
        print("Notice: Normally, please choose only one column to be tag column Y, not multiple columns.")
        print("Notice: For classification model training, please choose the label column which has distinctive integers.")
        y = create_sub_data_set(data_selected_imputed_fe, allow_empty_columns=False)
        print("Successfully create Y data set.")
        print("The Selected Data Set:")
        print(y)
        print("Basic Statistical Information: ")
        basic_statistic(y)
        save_data(y, name_all, "Y", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        clear_output()

        # <--- Feature Scaling --->
        print("[bold green]-*-*- Feature Scaling on X Set -*-*-[/bold green]")
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
            save_data(X, name_all, "X With Scaling", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        else:
            feature_scaling_config = {}
        clear_output()

        # <--- Feature Selection --->
        print("[bold green]-*-*- Feature Selection on X set -*-*-[/bold green]")
        num2option(OPTION)
        is_feature_selection = limit_num_input(OPTION, SECTION[1], num_input)
        if is_feature_selection == 1:
            print("Which strategy do you want to apply?")
            num2option(FEATURE_SELECTION_STRATEGY)
            feature_selection_num = limit_num_input(FEATURE_SELECTION_STRATEGY, SECTION[1], num_input)
            feature_selection_config, X = feature_selector(X, y, mode_num, FEATURE_SELECTION_STRATEGY, feature_selection_num - 1)
            print("--Selected Features-")
            show_data_columns(X.columns)
            save_data(X, name_all, "X After Feature Selection", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        else:
            feature_selection_config = {}
        clear_output()

        # Create training data and testing data
        print("[bold green]-*-*- Data Split - Train Set and Test Set -*-*-[/bold green]")
        print("Notice: Normally, set 20% of the dataset aside as test set, such as 0.2.")
        test_ratio = float_input(default=0.2, prefix=SECTION[1], slogan="@Test Ratio: ")
        train_test_data = data_split(X, y, process_name_column, test_ratio)
        for key, value in train_test_data.items():
            if key in ["Name Train", "Name Test"]:
                continue
            print("-" * 25)
            print(f"The Selected Data Set: {key}")
            print(value)
            print(f"Basic Statistical Information: {key}")
            basic_statistic(value)
            if key == "X Train" or key == "Y Train":
                data_name_column = train_test_data["Name Train"]
            else:
                data_name_column = train_test_data["Name Test"]
            save_data(value, data_name_column, key, GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        X_train, X_test = train_test_data["X Train"], train_test_data["X Test"]
        y_train, y_test = train_test_data["Y Train"], train_test_data["Y Test"]
        name_train, name_test = train_test_data["Name Train"], train_test_data["Name Test"]
        del data_selected_imputed_fe
        clear_output()
    else:
        # Unsupervised learning
        # Create X data set without data split because it is unsupervised learning
        X = data_selected_imputed_fe
        # <--- Feature Scaling --->
        print("[bold green]-*-*- Feature Scaling on X Set -*-*-[/bold green]")
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
            save_data(X, name_all, "X With Scaling", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        else:
            feature_scaling_config = {}
        clear_output()

        feature_selection_config = {}
        # Create training data without data split because it is unsupervised learning
        X_train = X
        y, X_test, y_train, y_test, name_train, name_test = None, None, None, None, None, None
        name_all = process_name_column
    # <--- Model Selection --->
    logger.debug("Model Selection")
    print("[bold green]-*-*- Model Selection -*-*-[/bold green]")
    # The following scenarios support three modes (regression, classification and clustering) with the models that support missing values.
    # Because finally, the data contains missing values.
    # 1. missing value flag = True, process_missing_value_flag = False, drop rows with missing values flag = Flase, imputed flag = False
    # The following scenarios support four modes with all models.
    # Because finally, the data is complete.
    # 1. missing value flag = True, process_missing_value_flag = True, drop rows with missing values flag = True, imputed flag = False
    # 2. missing value flag = True, process_missing_value_flag = True, drop rows with missing values flag = False, imputed flag = True
    # 3. missing value flag = False, process_missing_value_flag = False, drop rows with missing values flag = False, imputed flag = False
    # If the selected data set is with missing values and is not been imputed, then only allow the user to choose regression, classification and clustering models.
    # Otherwise, allow the user to choose decomposition models.
    if missing_value_flag and not process_missing_value_flag:
        Modes2Models = {1: REGRESSION_MODELS_WITH_MISSING_VALUES, 2: CLASSIFICATION_MODELS_WITH_MISSING_VALUES, 3: CLUSTERING_MODELS_WITH_MISSING_VALUES}
        Modes2Initiators = {1: RegressionModelSelection, 2: ClassificationModelSelection, 3: ClusteringModelSelection}
    else:
        Modes2Models = {1: REGRESSION_MODELS, 2: CLASSIFICATION_MODELS, 3: CLUSTERING_MODELS, 4: DECOMPOSITION_MODELS, 5: ANOMALYDETECTION_MODELS}
        Modes2Initiators = {
            1: RegressionModelSelection,
            2: ClassificationModelSelection,
            3: ClusteringModelSelection,
            4: DecompositionModelSelection,
            5: AnomalyDetectionModelSelection,
        }
    MODELS = Modes2Models[mode_num]
    num2option(MODELS)
    # Add the option of all models
    all_models_num = len(MODELS) + 1
    print(str(all_models_num) + " - All models above to be trained")
    print("Which model do you want to apply?")
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
            print("Do you want to employ automated machine learning with respect to this algorithm?")
            num2option(OPTION)
            automl_num = limit_num_input(OPTION, SECTION[2], num_input)
            if automl_num == 1:
                is_automl = True
            clear_output()

    # Model inference control
    is_inference = False
    # If the model is supervised learning, then allow the user to use model inference.
    if mode_num == 1 or mode_num == 2:
        print("[bold green]-*-*- Feature Engineering on Application Data -*-*-[/bold green]")
        is_inference = True
        selected_columns = X_train.columns
        if inference_data is not None:
            if feature_engineering_config:
                # If inference_data is not None and feature_engineering_config is not {}, then apply feature engineering with the same operation to the input data.
                print("The same feature engineering operation will be applied to the inference data.")
                new_feature_builder = FeatureConstructor(inference_data, name_column_origin)
                inference_data_fe = new_feature_builder.batch_build(feature_engineering_config)
                inference_data_fe_selected = inference_data_fe[selected_columns]
                inference_name_column = inference_data[NAME]
                save_data(inference_data, name_column_origin, "Application Data Original", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
                save_data(inference_data_fe, name_column_origin, "Application Data Feature-Engineering", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
                save_data(inference_data_fe_selected, inference_name_column, "Application Data Feature-Engineering Selected", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
            else:
                print("You have not applied feature engineering to the training data.")
                print("Hence, no feature engineering operation will be applied to the inference data.")
                inference_data_fe_selected = inference_data[selected_columns]
                inference_name_column = inference_data[NAME]
                save_data(inference_data, name_column_origin, "Application Data Original", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
                save_data(inference_data_fe_selected, inference_name_column, "Application Data Selected", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
        else:
            # If the user doesn't provide the inference data path, it means that the user doesn't want to run the model inference.
            print("You did not provide application data.")
            print("Hence, this part will be skipped.")
            inference_data_fe_selected = None
        clear_output()
    else:
        # If the model is unsupervised learning, then don't allow the user to use model inference.
        inference_data_fe_selected = None

    # <--- Model Training --->
    # In this section, there are two scenarios which are either choosing one model to run or choosing all models.
    # In both scenarios, after the model training is finished, the transform pipeline will be created.
    # Subsequently, the model inference will be executed if the user provides the inference data.
    # Technically, the transform pipeline contains the operations on the training data and it will be applied to the inference data in the same order.
    logger.debug("Model Training")
    # If the user doesn't choose all models, then run the designated model.
    if model_num != all_models_num:
        # run the designated model
        run = Modes2Initiators[mode_num](model_name)
        # If is_automl is False, then run the model without AutoML.
        if not is_automl:
            run.activate(X, y, X_train, X_test, y_train, y_test, name_train, name_test, name_all)
        else:
            run.activate(X, y, X_train, X_test, y_train, y_test, name_train, name_test, name_all, is_automl)
        clear_output()

        # <--- Transform Pipeline --->
        # Construct the transform pipeline using sklearn.pipeline.make_pipeline method.
        logger.debug("Transform Pipeline")
        print("[bold green]-*-*- Transform Pipeline Construction -*-*-[/bold green]")
        transformer_config, transform_pipeline = build_transform_pipeline(imputation_config, feature_scaling_config, feature_selection_config, run, X_train, y_train)
        clear_output()

        # <--- Model Inference --->
        # If the user provides the inference data, then run the model inference.
        # If the user chooses to drop the rows with missing values, then before running the model inference, need to drop the rows with missing values in inference data either.
        logger.debug("Model Inference")
        if inference_data_fe_selected is not None:
            print("[bold green]-*-*- Model Inference -*-*-[/bold green]")
            if drop_rows_with_missing_value_flag:
                inference_name_column = inference_data[NAME]
                inference_data_name = pd.concat([inference_name_column, inference_data_fe_selected], axis=1)
                inference_data_fe_selected_dropped = inference_data_fe_selected.dropna()
                inference_data_fe_selected_dropped_name = inference_data_name.dropna()
                inference_name_column_drop = inference_data_fe_selected_dropped_name[NAME]
                model_inference(inference_data_fe_selected_dropped, inference_name_column_drop, is_inference, run, transformer_config, transform_pipeline)
                save_data(
                    inference_data_fe_selected_dropped,
                    inference_name_column_drop,
                    "Application Data Feature-Engineering Selected Dropped-Imputed",
                    GEOPI_OUTPUT_ARTIFACTS_DATA_PATH,
                    MLFLOW_ARTIFACT_DATA_PATH,
                )
            else:
                inference_name_column = inference_data[NAME]
                model_inference(inference_data_fe_selected, inference_name_column, is_inference, run, transformer_config, transform_pipeline)
            clear_output()

        # <--- Data Dumping --->
        # In this section, convert the data in the output to the summary.
        GEOPI_OUTPUT_SUMMARY_PATH = os.getenv("GEOPI_OUTPUT_SUMMARY_PATH")
        GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
        GEOPI_OUTPUT_METRICS_PATH = os.getenv("GEOPI_OUTPUT_METRICS_PATH")
        GEOPI_OUTPUT_PARAMETERS_PATH = os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH")
        copy_files(GEOPI_OUTPUT_ARTIFACTS_PATH, GEOPI_OUTPUT_METRICS_PATH, GEOPI_OUTPUT_PARAMETERS_PATH, GEOPI_OUTPUT_SUMMARY_PATH)

    else:
        # Run all models
        for i in range(len(MODELS) - 1):
            # Start a nested MLflow run within the current MLflow run
            with mlflow.start_run(run_name=MODELS[i], experiment_id=experiment.experiment_id, nested=True):
                create_geopi_output_dir(OUTPUT_PATH, experiment.name, run_name, MODELS[i])
                run = Modes2Initiators[mode_num](MODELS[i])
                # If is_automl is False, then run all models without AutoML.
                if not is_automl:
                    run.activate(X, y, X_train, X_test, y_train, y_test, name_train, name_test, name_all)
                else:
                    # If is_automl is True, but MODELS[i] is in the NON_AUTOML_MODELS, then run the model without AutoML.
                    if MODELS[i] in NON_AUTOML_MODELS:
                        run.activate(X, y, X_train, X_test, y_train, y_test, name_train, name_test, name_all)
                    else:
                        # If is_automl is True, and MODELS[i] is not in the NON_AUTOML_MODELS, then run the model with AutoML.
                        run.activate(X, y, X_train, X_test, y_train, y_test, name_train, name_test, name_all, is_automl)

                # <--- Transform Pipeline --->
                # Construct the transform pipeline using sklearn.pipeline.make_pipeline method.
                logger.debug("Transform Pipeline")
                print("[bold green]-*-*- Transform Pipeline Construction -*-*-[/bold green]")
                transformer_config, transform_pipeline = build_transform_pipeline(imputation_config, feature_scaling_config, feature_selection_config, run, X_train, y_train)

                # <--- Model Inference --->
                # If the user provides the inference data, then run the model inference.
                # If the user chooses to drop the rows with missing values, then before running the model inference, need to drop the rows with missing values in inference data either.
                logger.debug("Model Inference")
                if inference_data_fe_selected is not None:
                    print("[bold green]-*-*- Model Inference -*-*-[/bold green]")
                    if drop_rows_with_missing_value_flag:
                        inference_name_column = inference_data[NAME]
                        inference_data_name = pd.concat([inference_name_column, inference_data_fe_selected], axis=1)
                        inference_data_fe_selected_dropped = inference_data_fe_selected.dropna()
                        inference_data_fe_selected_dropped_name = inference_data_name.dropna()
                        inference_name_column_drop = inference_data_fe_selected_dropped_name[NAME]
                        model_inference(inference_data_fe_selected_dropped, inference_name_column_drop, is_inference, run, transformer_config, transform_pipeline)
                        save_data(
                            inference_data_fe_selected_dropped,
                            inference_name_column_drop,
                            "Application Data Feature-Engineering Selected Dropped-Imputed",
                            GEOPI_OUTPUT_ARTIFACTS_DATA_PATH,
                            MLFLOW_ARTIFACT_DATA_PATH,
                        )
                    else:
                        inference_name_column = inference_data[NAME]
                        model_inference(inference_data_fe_selected, inference_name_column, is_inference, run, transformer_config, transform_pipeline)
                    clear_output()

            # <--- Data Dumping --->
            # In this section, convert the data in the output to the summary.
            GEOPI_OUTPUT_SUMMARY_PATH = os.getenv("GEOPI_OUTPUT_SUMMARY_PATH")
            GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
            GEOPI_OUTPUT_METRICS_PATH = os.getenv("GEOPI_OUTPUT_METRICS_PATH")
            GEOPI_OUTPUT_PARAMETERS_PATH = os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH")
            copy_files(GEOPI_OUTPUT_ARTIFACTS_PATH, GEOPI_OUTPUT_METRICS_PATH, GEOPI_OUTPUT_PARAMETERS_PATH, GEOPI_OUTPUT_SUMMARY_PATH)

    mlflow.end_run()
