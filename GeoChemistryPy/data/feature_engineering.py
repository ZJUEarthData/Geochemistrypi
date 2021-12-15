# -*- coding: utf-8 -*-
import string
from global_variable import *
import operator

def create_index_vs_name(columns_name):
    """pattern: letter : column name, e.g. a : 1st column name; b : 2nd column name

    :param columns_name: the name of each column
    :return: columns index, list
    """
    map_dict = {}
    for i in range(len(columns_name)):
        print("Selected data set:")
        print(string.ascii_letters[i] + ' - ' + columns_name[i])
        map_dict[string.ascii_letters[i]] = columns_name[i]
    return map_dict


# def create_columns_index(columns_name):
#     """pattern: C[num], e.g. C1 -> 1st column
#
#      :param columns_name: the name of each column
#      :return: columns index, list
#     """
#     columns_index = []
#     for i, j in enumerate(columns_name):
#         index = "C{}".format(str(i+1))
#         columns_index.append(index)
#         print(index + " - " + j)
#     return columns_index
#
#
# def index2name(columns_index, columns_name):
#     map_dict = dict([(i, j) for i, j in zip(columns_index, columns_name)])
#     return map_dict


def input_expression():
    expression = input("Build up new feature with the combination of 4 basic arithmatic operator.\n"
                       "Input example 1: C1 * C2 - C3 \n"
                       "--> Step 1: Multiply C1 column with C2 column; \n "
                       "--> Step 2: Subtract C3 from the result of Step 1; \n"
                       "Input example 2: (C4 + 5 * C6) / C5 \n"
                       "--> Step 1: multiply 5 with C6"
                       "--> Step 2: Plus C4 column with the result of Step 1;\n"
                       "--> Step 3: Divide the result of Step 1 by C5 \n"
                       "@input: ")
    temp = list(expression.replace(' ', ''))
    return temp


class Stack(object):

    Error = -1

    def __init__(self):
        # the size of the stack
        self._MaxSize = 100
        # pointer indicates the position of the top element
        self._Top = -1
        self._Data = [None for _ in range(self._MaxSize)]

    def is_empty(self):
        return self._Top == -1

    def push(self, item):
        if self._Top == self._MaxSize:
            print("The stack is full")
            return
        else:
            self._Top += 1
            self._Data[self._Top] = item

    def pop(self):
        if self._Top == -1:
            print("The stack is empty")
            return Stack.Error
        else:
            item = self._Data[self._Top]
            self._Data[self._Top] = None
            self._Top -= 1
            return item


class FeatureConstructor(object):

    operator_priority_out = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 3, ')': 3}
    operator_priority_in = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0, ')': 3}

    def __init__(self, feature_name):
        self.feature_name = feature_name
        self._expression = None

    def input_expression(self):
        expression = input("Build up new feature with the combination of 4 basic arithmatic operator.\n"
                           "Input example 1: C1 * C2 - C3 \n"
                           "--> Step 1: Multiply C1 column with C2 column; \n "
                           "--> Step 2: Subtract C3 from the result of Step 1; \n"
                           "Input example 2: (C4 + 5 * C6) / C5 \n"
                           "--> Step 1: multiply 5 with C6"
                           "--> Step 2: Plus C4 column with the result of Step 1;\n"
                           "--> Step 3: Divide the result of Step 1 by C5 \n"
                           "@input: ")
        self._expression = list(expression.replace(' ', ''))

    # TODO: str -> operator
    def _get_operator_fn(self, oper):
        return {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '%': operator.mod,
            '^': operator.xor,
            }[oper]

    # TODO: infix expression -> postfix expression
    def _infix_exp2postfix_exp(self):
        pass

















