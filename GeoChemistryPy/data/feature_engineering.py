# -*- coding: utf-8 -*-
import string
from global_variable import *
import operator
import gc


def create_index_vs_name(columns_name):
    """pattern: letter : column name, e.g. a : 1st column name; b : 2nd column name

    :param columns_name: the name of each column
    :return: columns index, list
    """
    map_dict = {}
    print("Selected data set:")
    for i in range(len(columns_name)):
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



class Stack(object):
    Error = -1

    def __init__(self, MaxSize):
        # the size of the stack
        self.MaxSize = MaxSize
        # pointer indicates the position of the top element
        self.Top = -1
        self.Data = [None for _ in range(self.MaxSize)]

    def is_empty(self):
        return self.Top == -1

    def push(self, item):
        if self.Top == self.MaxSize:
            print("The stack is full")
            return
        else:
            self.Top += 1
            self.Data[self.Top] = item

    def pop(self):
        if self.Top == -1:
            print("The stack is empty")
            return Stack.Error
        else:
            item = self.Data[self.Top]
            self.Data[self.Top] = None
            self.Top -= 1
            return item


class FeatureConstructor(object):

    oper = ['-', '+', '*', '/', '(', ')']
    parenthesis = ['(', ')']

    def __init__(self, feature_name):
        self.feature_name = feature_name
        self._infix_expr = []
        self._postfix_expr = []

    def input_expression(self):
        expression = input("Build up new feature with the combination of 4 basic arithmatic operator.\n"
                           "Input example 1: a * b - c \n"
                           "--> Step 1: Multiply a column with b column; \n "
                           "--> Step 2: Subtract c from the result of Step 1; \n"
                           "Input example 2: (d + 5 * f) / g \n"
                           "--> Step 1: multiply 5 with f"
                           "--> Step 2: Plus d column with the result of Step 1;\n"
                           "--> Step 3: Divide the result of Step 1 by g \n"
                           "@input: ")
        self._infix_expr = list(expression.replace(' ', ''))

    @staticmethod
    def _oper_priority_out(oper):
        return {
            '+': 1, '-': 1,
            '*': 2, '/': 2,
            '(': 3, ')': 0,
        }[oper]

    @staticmethod
    def _oper_priority_in(oper):
        return {
            '+': 1, '-': 1,
            '*': 2, '/': 2,
            '(': 0, ')': 0
        }[oper]

    @staticmethod
    def _get_operator_fn(oper):
        return {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '%': operator.mod,
            '^': operator.xor,
        }[oper]

    @staticmethod
    def _eval_binary_expr(op1, oper, op2):
        op1, op2 = float(op1), float(op2)
        return FeatureConstructor._get_operator_fn(oper)(op1, op2)

    # TODO: infix expression -> postfix expression

    def infix_expr2postfix_expr(self):
        op_stack = Stack(100)
        oper_stack = Stack(100)

        for i in range(len(self._infix_expr)):
            if self._infix_expr[i] in FeatureConstructor.oper:
                # deal with the operators in the expression

                # Compare the operators' priority inside and outside the operator stack.
                # If the priority of the inside operator is greater than that of the outside,
                # then pop out the current top item in the stack and append it to the postfix expression.
                # Outside the operator stack, '(' is the highest priority
                # while inside the operator stack, it is the lowest priority.
                while oper_stack.Top != -1 and \
                        FeatureConstructor._oper_priority_out(self._infix_expr[i]) < \
                        FeatureConstructor._oper_priority_in(oper_stack.Data[oper_stack.Top]):
                    # When the operator is ')', pop out all the items in the operator stack
                    # until meeting up '(' in the stack.
                    if self._infix_expr[i] == ')':
                        top_item = oper_stack.pop()
                        while top_item != '(':
                            self._postfix_expr.append(top_item)
                            top_item = oper_stack.pop()
                        # release '(' parenthesis variable in the stack
                        del top_item
                        gc.collect()
                        break
                    else:
                        self._postfix_expr.append(oper_stack.pop())

                # When the operator is ')', don't need to store it in the operator stack
                if self._infix_expr[i] == ')':
                    continue
                else:
                    oper_stack.push(self._infix_expr[i])
            else:
                # deal with the operands in the expression
                self._postfix_expr.append(self._infix_expr[i])

        # pop up the rest of items in the stack until it's empty
        while oper_stack.Top != -1:
            self._postfix_expr.append(oper_stack.pop())

    def show_postfix_expr(self):
        print(self._infix_expr)
        print(self._postfix_expr)

