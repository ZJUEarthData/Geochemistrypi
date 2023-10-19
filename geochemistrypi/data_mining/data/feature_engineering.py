# -*- coding: utf-8 -*-
import os
import string
import time
from typing import Dict

import pandas as pd
from rich import print

from ..constants import MLFLOW_ARTIFACT_DATA_PATH, OPTION, SECTION
from ..plot.statistic_plot import basic_statistic
from ..utils.base import clear_output, save_data
from .data_readiness import basic_info, limit_num_input, num2option, num_input, show_data_columns


class FeatureConstructor(object):
    """Construct new feature based on the existing data set."""

    oper = "+-*/^(),."
    # parenthesis = ['(', ')']
    cal_words = ["pow", "sin", "cos", "tan", "pi", "mean", "std", "var", "log"]

    def __init__(self, data: pd.DataFrame) -> None:
        self.feature_name = None
        self.data = data
        self.alphabet = string.ascii_lowercase
        self._infix_expr = []
        self._postfix_expr = []
        self.map_dict = {}
        self._result = None
        self.config = {}

    def index2name(self) -> None:
        """Show the index of columns in the data set. The display pattern is [letter : column name], e.g. a : 1st column name; b : 2nd column name."""
        columns_name = self.data.columns
        print("Selected data set:")
        for i in range(len(columns_name)):
            print(self.alphabet[i] + " - " + columns_name[i])
            self.map_dict[self.alphabet[i]] = columns_name[i]

    def _get_column(self, index: str) -> str:
        return self.map_dict[index]

    def input_feature_name(self) -> None:
        """Name the constructed feature (column name), like 'NEW-COMPOUND'."""
        while True:
            self.feature_name = input("Name the constructed feature (column name), like 'NEW-COMPOUND': \n" "@input: ")
            if len(self.feature_name) == 0:
                print("Sorry!You haven't named it yet!")
            else:
                break
        return self.feature_name

    def input_expression(self) -> None:
        """Input the expression of the constructed feature."""
        expression = input(
            "Build up new feature with the combination of basic arithmatic operators,"
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
            "Input example 4: log(a)/b - c \n"
            "--> Step 1: Take the logarithm of the value a; \n"
            "--> Step 2: Divide the result of Step 1 by the value of b; \n"
            "--> Step 3: Subtract the value of c from the result of Step 2; \n"
            "You can use mean(x) to calculate the average value.\n"
            "@input: "
        )
        while True:
            self._infix_expr = expression.replace(" ", "")
            if len(self._infix_expr) == 0:
                print("You haven't built any new features yet!")
                time.sleep(0.5)
                expression = input("-----* Please enter again *-----\n@input: ")
            elif not all(c.isdigit() or c.isspace() or c in FeatureConstructor.oper or c in self.alphabet for c in self._infix_expr):
                print("There's something wrong with the input !")
                time.sleep(0.5)
                expression = input("-----* Please enter again *-----\n@input: ")
            else:
                try:
                    self.letter_map()
                except Exception as e:
                    print("Your input contains the following error:", e)
                    time.sleep(0.5)
                    expression = input("-----* Please enter again *-----\n@input: ")
                else:
                    break
        return self._infix_expr

    def _evaluate_expression(self, feature_name, feature_expression) -> None:
        feature_expression = feature_expression.replace("sin", "np.sin")
        feature_expression = feature_expression.replace("cos", "np.cos")
        feature_expression = feature_expression.replace("tan", "np.tan")
        feature_expression = feature_expression.replace("pi", "np.pi")
        feature_expression = feature_expression.replace("pow", "np.power")
        feature_expression = feature_expression.replace("mean", "np.mean")
        feature_expression = feature_expression.replace("std", "np.std")
        feature_expression = feature_expression.replace("var", "np.var")
        feature_expression = feature_expression.replace("log", "np.log")
        try:
            new_feature_column = eval(feature_expression)
            if isinstance(new_feature_column, pd.Series) or isinstance(new_feature_column, pd.DataFrame):
                new_feature_column.name = feature_name
            else:
                new_feature_column = pd.Series([new_feature_column for _ in range(self.data.shape[0])], name=feature_name)
        except SyntaxError:
            print("The expression contains a syntax error.")
        except ZeroDivisionError:
            print("The expression contains a division by zero.")
        return new_feature_column

    def letter_map(self) -> None:
        """Map the letter to the column name."""
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
                    if ww in self.alphabet:
                        self._infix_expr += str("self.data['" + self._get_column(ww) + "']")
                    else:
                        self._infix_expr += ww

    def build(self) -> None:
        """Build the new feature."""
        print("The Selected Data Set:")
        show_data_columns(self.data.columns)
        fe_flag = 0
        is_feature_engineering = 0
        GEOPI_OUTPUT_ARTIFACTS_DATA_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_DATA_PATH")
        while True:
            if fe_flag != 1:
                print("Feature Engineering Option:")
                num2option(OPTION)
                is_feature_engineering = limit_num_input(OPTION, SECTION[1], num_input)
            if is_feature_engineering == 1:
                self.index2name()
                new_feature_name = self.input_feature_name()
                new_feature_expression = self.input_expression()
                instruction = {new_feature_name: new_feature_expression}
                self.config.update(instruction)
                new_feature_column = self._evaluate_expression(new_feature_name, new_feature_expression)
                clear_output()
                # Append the new feature to the original data and update the data
                self.data = self.append_feature(new_feature_column)
                clear_output()
                basic_info(self.data)
                basic_statistic(self.data)
                clear_output()
                print("Do you want to continue to build a new feature? ")
                num2option(OPTION)
                fe_flag = limit_num_input(OPTION, SECTION[1], num_input)
                if fe_flag == 1:
                    clear_output()
                    continue
                else:
                    save_data(self.data, "Data Selected Imputed Feature-Engineering", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
                    print("Exit Feature Engineering Mode.")
                    clear_output()
                    break
            else:
                save_data(self.data, "Data Selected Imputed Feature-Engineering", GEOPI_OUTPUT_ARTIFACTS_DATA_PATH, MLFLOW_ARTIFACT_DATA_PATH)
                clear_output()
                break
        return self.data

    def batch_build(self, feature_engineering_config: Dict) -> None:
        for feature_name, feature_expression in feature_engineering_config.items():
            new_feature_column = self._evaluate_expression(feature_name, feature_expression)
            self.data = self.append_feature(new_feature_column)
        return self.data

    def append_feature(self, new_feature_column: pd.Series) -> None:
        """Append the new feature to the original data."""
        print(f"Successfully construct a new feature {new_feature_column.name}.")
        print(new_feature_column)
        return pd.concat([self.data, new_feature_column], axis=1)
