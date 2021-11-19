# -*- coding: utf-8 -*-
import sys, os
sys.path.append("..")
from process.regress import ModelSelection
from global_variable import *
from data.data_readiness import *
import re

def main():
    print("User Behaviour Testing Demo on Regression Model")
    print(".......")

    # read the data
    #file_name = input("Upload the data set.(Enter the name of data set) ")
    file_name = 'Data_Example_for_Geochemistry_Py.xlsx'
    data = read_data(file_name)
    show_data_columns(data)
    columns_range = input('Select the data range you want to process.\n'
                          'Input format "(xx: xx); xx; (xx, xx)", such as "(1: 3); 7; (10: 13)"\n'
                          'It means you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13\n'
                          '@input: ')
    columns_selected = select_columns(columns_range)
    # reindex by subtract 1 due to python list traits
    columns_selected = [columns_selected[i]-1 for i in range(len(columns_selected))]
    data_selected = data.iloc[:, columns_selected]
    for i, j in zip(columns_selected, data_selected.columns):
        print(i+1, "-", j)
    #print(re.sub(r'\[|\]', '', str(list(data_selected.columns))))



    # model option for users
    print("Model Selection:")
    for i in range(len(MODELS)):
        print(str(i+1) + " - " + MODELS[i])
    all_models_num = len(MODELS) + 1
    print(str(all_models_num) + " - All models above trained")
    model_num = int(input("Which model do you want to apply?(Enter the Corresponding Number): "))

    # model trained selection
    if model_num != all_models_num:
        # gain the designated model'result
        model = MODELS[int(model_num) - 1]
        run = ModelSelection(model)
        run.activate(data)
    else:
        # gain all models'result
        for i in range(len(MODELS)):
            run = ModelSelection(MODELS[i])
            run.activate(data)


if __name__ == "__main__":
    main()