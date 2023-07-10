# -*- coding: utf-8 -*-
import string

import numpy as np
import pandas as pd
from rich import print

from ..global_variable import DATASET_OUTPUT_PATH, OPTION, SECTION
from ..plot.statistic_plot import basic_statistic
from ..utils.base import clear_output, save_data
from .data_readiness import basic_info, limit_num_input, num2option, num_input, show_data_columns


class FeatureConstructor(object):

    oper = "+-*/^(),."
    # parenthesis = ['(', ')']
    alphabet = string.ascii_letters
    cal_words = ["pow", "sin", "cos", "tan", "pi", "mean", "std", "var", "log"]

    def __init__(self, data):
        self.feature_name = None
        self.data = data
        self._infix_expr = []
        self._postfix_expr = []
        self.map_dict = {}
        self._result = None

    def index2name(self):
        """Pattern: [letter : column name], e.g. a : 1st column name; b : 2nd column name

        :return: index : column name, dict
        """
        columns_name = self.data.columns
        print("Selected data set:")
        for i in range(len(columns_name)):
            print(FeatureConstructor.alphabet[i] + " - " + columns_name[i])
            self.map_dict[FeatureConstructor.alphabet[i]] = columns_name[i]

    def _get_column(self, index):
        return self.map_dict[index]

    def name_feature(self):
        while True:
            self.feature_name = input("Name the constructed feature (column name), like 'NEW-COMPOUND': \n" "@input: ")
            if len(self.feature_name) == 0:
                print("Sorry!You haven't named it yet!")
            else:
                break

    def input_expression(self):
        expression = input(
            "Build up new feature with the combination of 4 basic arithmatic operator,"
            " including '+', '-', '*', '/', '()'.\n"
            "Input example 1: a * b - c \n"
            "--> Step 1: Multiply a column with b column; \n"
            "--> Step 2: Subtract c from the result of Step 1; \n"
            "Input example 2: (d + 5 * f) / g \n"
            "--> Step 1: Multiply 5 with f; \n"
            "--> Step 2: Plus d column with the result of Step 1;\n"
            "--> Step 3: Divide the result of Step 1 by g; \n"
            "Now we've added new computing capabilities:\n"
            "You can use pow(x,y) to perform power operations.\n"
            "You can use mean(x) to calculate the average value.\n"
            "You can use log(x) for logarithms with base e.\n"
            # "Input example 3: (h ** 3) / (i ** (1/2)) \n"
            # "--> Step 1: Exponent calculation, h raised to the power of 3; \n"
            # "--> Step 2: Root calculation, find the square root of i; \n"
            # "--> Step 3: Divide the result of Step 1 by the result of Step 2; \n"
            "@input: "
        )
        while True:
            self._infix_expr = expression.replace(" ", "")
            if not all(c.isdigit() or c.isspace() or c in FeatureConstructor.oper or c in FeatureConstructor.alphabet for c in self._infix_expr):
                print("There's something wrong with the input !\n")
                expression = input(
                    "Build up new feature with the combination of 4 basic arithmatic operator,"
                    " including '+', '-', '*', '/', '()'.\n"
                    "Input example 1: a * b - c \n"
                    "--> Step 1: Multiply a column with b column; \n"
                    "--> Step 2: Subtract c from the result of Step 1; \n"
                    "Input example 2: (d + 5 * f) / g \n"
                    "--> Step 1: Multiply 5 with f; \n"
                    "--> Step 2: Plus d column with the result of Step 1;\n"
                    "--> Step 3: Divide the result of Step 1 by g; \n"
                    "Input example 3: pow(a, b) + c * d \n"
                    "--> Step 1: Raise the base a to the power of the exponent b; \n"
                    "--> Step 2: Multiply the value of c by the value of d; \n"
                    "--> Step 3: Add the result of Step 1 to the result of Step 2; \n"
                    "Input example 4: Example: log(a)/b - c \n"
                    "--> Step 1: Take the logarithm of the value a; \n"
                    "--> Step 2: Divide the result of Step 1 by the value of b; \n"
                    "--> Step 3: Subtract the value of c from the result of Step 2; \n"
                    "You can use mean(x) to calculate the average value.\n"
                    # "Input example 3: (h ** 3) / (i ** (1/2)) \n"
                    # "--> Step 1: Exponent calculation, h raised to the power of 3; \n"
                    # "--> Step 2: Root calculation, find the square root of i; \n"
                    # "--> Step 3: Divide the result of Step 1 by the result of Step 2; \n"
                    "@input: "
                )
            else:
                break

    def evaluate(self):
        self.letter_map()
        np.array(["dummy"])  # dummy array to skip the flake8 warning - F401 'numpy as np' imported but unused'
        self._infix_expr = self._infix_expr.replace("sin", "np.sin")
        self._infix_expr = self._infix_expr.replace("cos", "np.cos")
        self._infix_expr = self._infix_expr.replace("tan", "np.tan")
        self._infix_expr = self._infix_expr.replace("pi", "np.pi")
        self._infix_expr = self._infix_expr.replace("pow", "np.power")
        self._infix_expr = self._infix_expr.replace("mean", "np.mean")
        self._infix_expr = self._infix_expr.replace("std", "np.std")
        self._infix_expr = self._infix_expr.replace("var", "np.var")
        self._infix_expr = self._infix_expr.replace("log", "np.log")
        try:
            self._result = eval(self._infix_expr)
            if isinstance(self._result, pd.DataFrame) or isinstance(self._result, pd.Series):
                self._result.name = self.feature_name
            else:
                self._result = pd.Series([self._result for i in range(self.data.shape[0])])
                self._result.name = self.feature_name
        except SyntaxError:
            print("The expression contains a syntax error.")
        except ZeroDivisionError:
            print("The expression contains a division by zero.")

    def letter_map(self):
        new_text = ""
        test_text = "".join(ch for ch in self._infix_expr if ch not in set(" "))
        for words in FeatureConstructor.cal_words:
            if words in test_text:
                test_text = test_text.replace(words, ";" + words + ";")
        new_text = test_text
        self._infix_expr = ""
        if new_text[0] == ";":
            new_text = new_text[1:]
        for word in new_text.split(";"):
            if word in FeatureConstructor.cal_words:
                self._infix_expr += word
            else:
                for ww in word:
                    if ww in FeatureConstructor.alphabet:
                        self._infix_expr += str("self.data['" + self._get_column(ww) + "']")
                    else:
                        self._infix_expr += ww

    def create_data_set(self):
        print(f'Successfully construct a new feature "{self.feature_name}".')
        print(self._result)
        return pd.concat([self.data, self._result], axis=1)

    def process_feature_engineering(self):
        print("-*-*- Feature Engineering -*-*-")
        print("The Selected Data Set:")
        show_data_columns(self.data.columns)
        fe_flag = 0
        is_feature_engineering = 0
        while True:
            if fe_flag != 1:
                print("Feature Engineering Option:")
                num2option(OPTION)
                is_feature_engineering = limit_num_input(OPTION, SECTION[1], num_input)
            if is_feature_engineering == 1:
                feature_built = FeatureConstructor(self.data)
                feature_built.index2name()
                feature_built.name_feature()
                feature_built.input_expression()
                feature_built.evaluate()
                # feature_built.infix_expr2postfix_expr()
                # feature_built.eval_expression()
                clear_output()
                # update the original data with a new feature
                data_processed_imputed = feature_built.create_data_set()
                clear_output()
                basic_info(data_processed_imputed)
                basic_statistic(data_processed_imputed)
                clear_output()
                print("Do you want to continue to construct a new feature?")
                num2option(OPTION)
                fe_flag = limit_num_input(OPTION, SECTION[1], num_input)
                if fe_flag == 1:
                    clear_output()
                    continue
                else:
                    save_data(data_processed_imputed, "Data Before Splitting", DATASET_OUTPUT_PATH)
                    print("Exit Feature Engineering Mode.")
                    clear_output()
                    break
            else:
                save_data(data_processed_imputed, "Data Before Splitting", DATASET_OUTPUT_PATH)
                clear_output()
                break

    # TODO: Is the scope of input right?
    def check_data_scope(self):
        pass
