# -*- coding: utf-8 -*-
class InvalidFileError(Exception):
    def __init__(self, value):
        self._value = value

    def __str__(self):
        return repr(self._value)
