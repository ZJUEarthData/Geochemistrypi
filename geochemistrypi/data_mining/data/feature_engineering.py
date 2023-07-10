# -*- coding: utf-8 -*-
import string

import numpy as np
import pandas as pd
from rich import print

# class Stack(object):
#     """Create a stack."""
#     Error = -1
#
#     def __init__(self, MaxSize):
#         # The size of the stack.
#         self.MaxSize = MaxSize
#         # Pointer indicates the position of the top element.
#         self.Top = -1
#         self.Data = [None for _ in range(self.MaxSize)]
#
#     def is_empty(self):
#         return self.Top == -1
#
#     def push(self, item):
#         if self.Top == self.MaxSize:
#             print("The stack is full")
#             return
#         else:
#             self.Top += 1
#             self.Data[self.Top] = item
#
#     def pop(self):
#         if self.Top == -1:
#             print("The stack is empty")
#             return Stack.Error
#         else:
#             item = self.Data[self.Top]
#             self.Data[self.Top] = None
#             self.Top -= 1
#             return item


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

    # @staticmethod
    # def _oper_priority_out(oper):
    #     return {
    #         '+': 1, '-': 1,
    #         '*': 2, '/': 2,
    #         '(': 3, ')': 0,
    #     }[oper]
    #
    # @staticmethod
    # def _oper_priority_in(oper):
    #     return {
    #         '+': 1, '-': 1,
    #         '*': 2, '/': 2,
    #         '(': 0, ')': 0
    #     }[oper]
    #
    # @staticmethod
    # def _get_operator_fn(oper):
    #     return {
    #         '+': operator.add,
    #         '-': operator.sub,
    #         '*': operator.mul,
    #         '/': operator.truediv,
    #         '%': operator.mod,
    #         '^': operator.xor,
    #     }[oper]

    # @staticmethod
    # def _eval_binary_expr(op1, oper, op2):
    #     return FeatureConstructor._get_operator_fn(oper)(op1, op2)
    #
    # def infix_expr2postfix_expr(self):
    #     oper_stack = Stack(100)
    #
    #     for i in range(len(self._infix_expr)):
    #         if self._infix_expr[i] in FeatureConstructor.oper:
    #             # deal with the operators in the expression
    #
    #             # Compare the operators' priority inside and outside the operator stack.
    #             # If the priority of the inside operator is greater than that of the outside,
    #             # then pop out the current top item in the stack and append it to the postfix expression.
    #             # Outside the operator stack, '(' is the highest priority
    #             # while inside the operator stack, it is the lowest priority.
    #             while not oper_stack.is_empty() and \
    #                     FeatureConstructor._oper_priority_out(self._infix_expr[i]) < \
    #                     FeatureConstructor._oper_priority_in(oper_stack.Data[oper_stack.Top]):
    #                 # When the operator is ')', pop out all the items in the operator stack
    #                 # until meeting up '(' in the stack.
    #                 if self._infix_expr[i] == ')':
    #                     top_item = oper_stack.pop()
    #                     while top_item != '(':
    #                         self._postfix_expr.append(top_item)
    #                         top_item = oper_stack.pop()
    #                     # release '(' parenthesis variable in the stack
    #                     del top_item
    #                     gc.collect()
    #                     break
    #                 else:
    #                     self._postfix_expr.append(oper_stack.pop())
    #
    #             # When the operator is ')', don't need to store it in the operator stack
    #             if self._infix_expr[i] == ')':
    #                 continue
    #             else:
    #                 oper_stack.push(self._infix_expr[i])
    #         else:
    #             # deal with the operands in the expression
    #             self._postfix_expr.append(self._infix_expr[i])
    #
    #     # pop up the rest of items in the stack until it's empty
    #     while not oper_stack.is_empty():
    #         self._postfix_expr.append(oper_stack.pop())
    #
    # def eval_expression(self):
    #     expr_stack = Stack(100)
    #     for i in range(len(self._postfix_expr)):
    #         # when the top item is not an operator
    #         if self._postfix_expr[i] not in FeatureConstructor.oper:
    #             expr_stack.push(self._postfix_expr[i])
    #         else:
    #             op2 = expr_stack.pop()
    #             op1 = expr_stack.pop()
    #             oper = self._postfix_expr[i]
    #             # check the type of the items
    #             if isinstance(op2, pandas.core.series.Series):
    #                 pass
    #             else:
    #                 if op2 in FeatureConstructor.alphabet:
    #                     op2 = self._get_column(op2)
    #                 else:
    #                     op2 = float(op2)
    #             if isinstance(op1, pandas.core.series.Series):
    #                 pass
    #             else:
    #                 if op1 in FeatureConstructor.alphabet:
    #                     op1 = self._get_column(op1)
    #                 else:
    #                     op1 = float(op1)
    #             temp = FeatureConstructor._eval_binary_expr(op1, oper, op2)
    #             expr_stack.push(temp)
    #     self._result = expr_stack.pop()
    #     self._result.name = self.feature_name

    def create_data_set(self):
        print(f'Successfully construct a new feature "{self.feature_name}".')
        print(self._result)
        return pd.concat([self.data, self._result], axis=1)

    # TODO: Is the scope of input right?
    def check_data_scope(self):
        pass
