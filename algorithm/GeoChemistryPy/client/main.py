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

    basic_info(data)

    basic_statistic(data)

    is_null_value(data)

    show_data_columns(data.columns)

    # create X data set
    print("Data Split")
    print("X data:")
    X = create_sub_data_set(data)
    # X_columns_range = input('Select the data range as X you want to process.\n'
    #                         'Input format:\n'
    #                         'Method 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" '
    #                         '--> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13 \n'
    #                         'Method 2: "xx", such as "7" --> you want to deal with the columns 7 \n'
    #                         '@input: ')
    # X_columns_selected = select_columns(X_columns_range)
    # X = data.iloc[:, X_columns_selected]
    # show_data_columns(X.columns, X_columns_selected)

    # create Y data set
    print("Y data")
    y = create_sub_data_set(data)
    # y_columns_range = input('Select the data range as Y you want to process.\n'
    #                         'Input format "xx", such as "7",'
    #                         'ignore double quotation marks.\n'
    #                         'It means you want to deal with the columns 7.\n'
    #                         '@input: ')
    # y_columns_selected = select_columns(y_columns_range)
    # y = data.iloc[:, y_columns_selected]
    # show_data_columns(y.columns, y_columns_selected)

    print("\n")

    # imputing
    print("Strategy for Missing Values:")
    num2option(IMPUTING_STRATEGY)
    strategy_num = int(input("Which strategy do you want to apply?(Enter the Corresponding Number): "))
    X_imputed_np = imputer(X, IMPUTING_STRATEGY[int(strategy_num) - 1])
    X_imputed = np2pd(X_imputed_np, X.columns)
    print(f'X data set: \n {X_imputed}')
    y_imputed_np = imputer(y, IMPUTING_STRATEGY[int(strategy_num) - 1])
    y_imputed = np2pd(y_imputed_np, y.columns)
    print(f'Y data set: \n {y_imputed}')

    # feature engineering


    print("\n")

    # model option for users
    print("Model Selection:")
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