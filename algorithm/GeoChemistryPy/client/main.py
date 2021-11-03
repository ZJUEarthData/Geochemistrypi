# -*- coding: utf-8 -*-
import sys, os
sys.path.append("..")
from process.regress import ModelSelection
from global_variable import *
import pandas as pd

def main():
    print("User Behaviour Testing Demo on Regression Model")
    print(".......")

    # read the data
    #data_name = input("Upload the data set.(Enter the name of data set) ")
    data_name = 'Data_Example_for_Geochemistry_Py.xlsx'
    data_path = os.path.join(DATASET_PATH, data_name)
    data = pd.read_excel(data_path, engine="openpyxl")

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