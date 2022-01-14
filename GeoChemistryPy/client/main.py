# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from global_variable import *
from data.data_readiness import *
from data.imputation import *
from data.feature_engineering import *
from plot.statistic_plot import *
from plot.map_plot import map_projected
from core.base import *
from process.regress import RegressionModelSelection
from process.classify import ClassificationModelSelection
from process.cluster import ClusteringModelSelection
from process.decompose import DecompositionModelSelection


def main():
    print("GeoChemistryPy - User Behaviour Testing Demo")
    print(".......")

    # read the data
    # TODO: seperate the input into outside
    # file_name = input("Upload the data set.(Enter the name of data set) ")
    print("-*-*- Data Uploaded -*-*-")
    file_name = 'Data_Regression.xlsx'
    data = read_data(file_name)
    show_data_columns(data.columns)
    clear_output()


    # world map projection for a specific element
    map_flag = 0
    while True:
        if map_flag != 1:
            print("World Map Projection for A Specific Element Option:")
            num2option(OPTION)
            is_map_projection = int(input("@Number: "))
            clear_output()
        if is_map_projection == 1:
            print("-*-*- Distribution in World Map -*-*-")
            print("Select one of the elements below to be projected in the World Map: ")
            show_data_columns(data.columns)
            elm_num = int(input("@number: "))
            map_projected(data.iloc[:, elm_num-1], data)
            clear_output()
        else:
            break
        print("Do you want to continue to project a new element in the World Map?")
        num2option(OPTION)
        map_flag = int(input("@Number: "))
        if map_flag == 1:
            clear_output()
            continue
        else:
            print('Exit Map Projection Mode.')
            clear_output()
            break


    # create the processing data set
    print("-*-*- Data Selected -*-*-")
    show_data_columns(data.columns)
    data_processed = create_sub_data_set(data)
    clear_output()
    print("The Selected Data Set:")
    print(data_processed)
    clear_output()
    print('Basic Statistical Information: ')
    basic_info(data_processed)
    basic_statistic(data_processed)
    correlation_plot(data_processed.columns, data_processed)
    distribution_plot(data_processed.columns, data_processed)
    is_null_value(data_processed)
    clear_output()


    # imputing
    print("-*-*- Strategy for Missing Values -*-*-")
    num2option(IMPUTING_STRATEGY)
    strategy_num = int(input("Which strategy do you want to apply?(Enter the Corresponding Number): "))
    data_processed_imputed_np = imputer(data_processed, IMPUTING_STRATEGY[strategy_num - 1])
    data_processed_imputed = np2pd(data_processed_imputed_np, data_processed.columns)
    basic_info(data_processed_imputed)
    basic_statistic(data_processed_imputed)
    probability_plot(data_processed.columns, data_processed, data_processed_imputed)
    clear_output()

    # feature engineering
    print("The Selected Data Set:")
    show_data_columns(data_processed.columns)
    fe_flag = 0
    while True:
        if fe_flag != 1:
            print("Feature Engineering Option:")
            num2option(OPTION)
            is_feature_engineering = int(input("@Number: "))
            clear_output()
        if is_feature_engineering == 1:
            print("-*-*- Feature Engineering -*-*-")
            feature_built = FeatureConstructor(data_processed_imputed)
            feature_built.index2name()
            feature_built.name_feature()
            feature_built.input_expression()
            feature_built.infix_expr2postfix_expr()
            feature_built.eval_expression()
            clear_output()
            # update the original data with a new feature
            data_processed_imputed = feature_built.create_data_set()
            clear_output()
            basic_info(data_processed_imputed)
            basic_statistic(data_processed_imputed)
            clear_output()
        else:
            break
        print("Do you want to continue to construct a new feature?")
        num2option(OPTION)
        fe_flag = int(input("@Number: "))
        if fe_flag == 1:
            clear_output()
            continue
        else:
            print('Exit Feature Engineering Mode.')
            clear_output()
            break


    print("-*-*- Mode Options -*-*-")
    num2option(MODE_OPTION)
    mode_num = int(input("@Number: "))
    clear_output()
    # divide X and y data set when it is supervised learning
    if mode_num == 1 or mode_num == 2:
        print("-*-*- Data Split -*-*-")
        print("Divide the processing data set into X (feature value) and Y (target value) respectively.")
        # create X data set
        print("Selected sub data set to create X data set:")
        show_data_columns(data_processed_imputed.columns)
        print('The selected X data set:')
        X = create_sub_data_set(data_processed_imputed)
        print('Successfully create X data set.')
        clear_output()
        # create Y data set
        print("Selected sub data set to create Y data set:")
        show_data_columns(data_processed_imputed.columns)
        print('The selected Y data set:')
        y = create_sub_data_set(data_processed_imputed)
        print('Successfully create Y data set.')
        clear_output()
    else:
        # unsupervised learning
        X = data_processed_imputed
        y = None


    # model option for users
    print("-*-*- Model Selection -*-*-:")
    Modes2Models = {1: REGRESSION_MODELS, 2: CLASSIFICATION_MODELS,
                    3: CLUSTERING_MODELS, 4: DECOMPOSITION_MODELS}
    Modes2Initiators = {1: RegressionModelSelection, 2: ClassificationModelSelection,
                        3: ClusteringModelSelection, 4: DecompositionModelSelection}
    MODELS = Modes2Models[mode_num]
    num2option(MODELS)
    all_models_num = len(MODELS) + 1
    print(str(all_models_num) + " - All models above to be trained")
    model_num = int(input("Which model do you want to apply?(Enter the Corresponding Number): "))
    clear_output()

    # model trained selection
    if model_num != all_models_num:
        # run the designated model
        model = MODELS[int(model_num) - 1]
        run = Modes2Initiators[mode_num](model)
        run.activate(X, y)
    else:
        # gain all models result in the specific mode
        for i in range(len(MODELS)):
            run = Modes2Initiators[mode_num](MODELS[i])
            run.activate(X, y)


if __name__ == "__main__":
    main()
    