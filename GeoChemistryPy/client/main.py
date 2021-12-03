# -*- coding: utf-8 -*-
import sys, os
sys.path.append("..")
from process.regress import ModelSelection
from global_variable import *
from data.data_readiness import *
import re
from plot.statistic_plot import *

def main():
    print("User Behaviour Testing Demo on Regression Model")
    print(".......")

    # read the data
    #file_name = input("Upload the data set.(Enter the name of data set) ")
    file_name = 'Data_Example_for_Geochemistry_Py.xlsx'
    data = read_data(file_name)
    show_data_columns(data.columns)

    # create the processing data set
    print("-*-*- Data Selected -*-*-")
    data_processed = create_sub_data_set(data)
    print("the Processing Data Set:")
    print(data_processed)
    basic_info(data_processed)
    basic_statistic(data_processed)
    is_null_value(data_processed)

    # imputing
    print("-*-*- Strategy for Missing Values -*-*-")
    num2option(IMPUTING_STRATEGY)
    strategy_num = int(input("Which strategy do you want to apply?(Enter the Corresponding Number): "))
    data_processed_imputed_np = imputer(data_processed, IMPUTING_STRATEGY[strategy_num - 1])
    data_processed_imputed = np2pd(data_processed_imputed_np, data_processed.columns)

    # feature engineering
    print("Selected sub data set to be processed...")
    show_data_columns(data_processed)
    num2option(OPTION)
    is_feature_engineering = int(input("Feature Engineering Option: "))
    if is_feature_engineering == 1:
        #print("")
        pass

    # divide features set and target set
    print("-*-*- Mode Options -*-*-")
    num2option(MODE_OPTION)
    mode_num = int(input("@Number: "))
    if mode_num == 1:
        # create X data set
        print("-*-*- Data Split -*-*-")
        print("Divide X and Y in the processing data set. ")
        print("X data:")
        X = create_sub_data_set(data_processed_imputed)
        # create Y data set
        print("Y data")
        y = create_sub_data_set(data_processed_imputed)
    else:
        # unsupervised learning
        pass

    print("\n")

    # imputing
    # print("-*-*- Strategy for Missing Values -*-*-")
    # num2option(IMPUTING_STRATEGY)
    # strategy_num = int(input("Which strategy do you want to apply?(Enter the Corresponding Number): "))
    # X_imputed_np = imputer(X, IMPUTING_STRATEGY[int(strategy_num) - 1])
    # X_imputed = np2pd(X_imputed_np, X.columns)
    # print(f'X data set: \n {X_imputed}')
    # y_imputed_np = imputer(y, IMPUTING_STRATEGY[int(strategy_num) - 1])
    # y_imputed = np2pd(y_imputed_np, y.columns)
    # print(f'Y data set: \n {y_imputed}')

    # feature engineering


    print("\n")

    # model option for users
    print("-*-*- Model Selection -*-*-:")
    num2option(REGRESSION_MODELS)
    # for i in range(len(REGRESSION_MODELS)):
    #     print(str(i+1) + " - " + REGRESSION_MODELS[i])
    all_models_num = len(REGRESSION_MODELS) + 1
    print(str(all_models_num) + " - All models above trained")
    model_num = int(input("Which model do you want to apply?(Enter the Corresponding Number): "))

    # model trained selection
    if model_num != all_models_num:
        # gain the designated model'result
        model = REGRESSION_MODELS[int(model_num) - 1]
        run = ModelSelection(model)
        run.activate(X_imputed, y_imputed)
    else:
        # gain all models result
        for i in range(len(REGRESSION_MODELS)):
            run = ModelSelection(REGRESSION_MODELS[i])
            run.activate(X_imputed, y_imputed)


if __name__ == "__main__":
    main()