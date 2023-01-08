# -*- coding: utf-8 -*-
import numpy as np


def show_formula(coef, intercept, features_name):
    term = []
    coef = np.around(coef, decimals=3).tolist()[0]

    for i in range(len(coef)):
        # the first value stay the same
        if i == 0:
            # not append if zero
            if coef[i] != 0:
                temp = str(coef[i]) + features_name[i]
                term.append(temp)
        else:
            # add plus symbol if positive, maintain if negative, not append if zero
            if coef[i] > 0:
                temp = '+' + str(coef[i]) + features_name[i]
                term.append(temp)
            elif coef[i] < 0:
                temp = str(coef[i]) + features_name[i]
                term.append(temp)
    if intercept[0] >= 0:
        # formula of polynomial regression
        formula = ''.join(term) + '+' + str(intercept[0])
    else:
        formula = ''.join(term) + str(intercept[0])
    print('y =', formula)
