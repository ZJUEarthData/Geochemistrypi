# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from process.regress import RegressionModelSelection
from global_variable import *
from data.data_readiness import *
from data.imputation import *
from data.feature_engineering import *
from plot.statistic_plot import *
from core.base import *


def main():
    print("User Behaviour Testing Demo on Regression Model")
    print(".......")

    # read the data
    # TODO: seperate the input outside
    # file_name = input("Upload the data set.(Enter the name of data set) ")
    print("-*-*- Data Uploaded -*-*-")
    file_name = 'Data_Regression.xlsx'
    data = read_data(file_name)
    show_data_columns(data.columns)
    clear_output()

    # create the processing data set
    print("-*-*- Data Selected -*-*-")
    data_processed = create_sub_data_set(data)
    print("the Processing Data Set:")
    print(data_processed)
    clear_output()
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
    probability_plot(data_processed.columns, data_processed, data_processed_imputed)
    clear_output()

    # feature engineering
    print("Data set to be processed...")
    show_data_columns(data_processed.columns)
    flag = 0
    while True:
        if flag != 1:
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
            # update the original data with a new feature
            data_processed_imputed = feature_built.create_data_set()
            basic_info(data_processed_imputed)
            basic_statistic(data_processed_imputed)
            clear_output()
        else:
            clear_output()
            break
        print("Do you want to continue to construct a new feature")
        num2option(OPTION)
        flag = int(input("@Number: "))
        if flag == 1:
            continue
        else:
            print('Exit Feature Engineering Mode.')
            break
    clear_output()

    # divide features set and target set
    print("-*-*- Mode Options -*-*-")
    num2option(MODE_OPTION)
    mode_num = int(input("@Number: "))
    if mode_num == 1:
        # create X data set
        print("-*-*- Data Split -*-*-")
        print("Divide X and Y in the processing data set. ")
        print("Selected sub data set to be processed...")
        print("X data:")
        show_data_columns(data_processed_imputed.columns)
        X = create_sub_data_set(data_processed_imputed)
        clear_output()
        # create Y data set
        print("Y data")
        show_data_columns(data_processed_imputed.columns)
        y = create_sub_data_set(data_processed_imputed)
    else:
        # unsupervised learning
        pass

    # model option for users
    print("-*-*- Model Selection -*-*-:")
    num2option(REGRESSION_MODELS)
    all_models_num = len(REGRESSION_MODELS) + 1
    print(str(all_models_num) + " - All models above trained")
    model_num = int(input("Which model do you want to apply?(Enter the Corresponding Number): "))

    # model trained selection
    if model_num != all_models_num:
        # gain the designated model result
        model = REGRESSION_MODELS[int(model_num) - 1]
        run = RegressionModelSelection(model)
        run.activate(X, y)
    else:
        # gain all models result
        for i in range(len(REGRESSION_MODELS)):
            run = RegressionModelSelection(REGRESSION_MODELS[i])
            run.activate(X, y)


if __name__ == "__main__":
    main()