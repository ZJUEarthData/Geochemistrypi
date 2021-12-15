# -*- coding: utf-8 -*-
import os


def clear_output():
    flag = input("(Press Enter key to move forward.)")
    if flag == '':
        os.system('clear')

