# -*- coding: utf-8 -*-
from data.data_readiness import num2option, num_input, limit_num_input
import pytest
from pytest import MonkeyPatch


def test_num2option() -> None:
    array = ['a', 'b', 'c']
    num2option(array)


def test_empty_num2option() -> None:
    array = []
    num2option(array)


def test_valid_num_input_one_time(monkeypatch: MonkeyPatch) -> None:
    inputs = ['1']
    prefix = 'Data'
    slogan = '@Number'
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    assert num_input(prefix, slogan) == 1


def test_invalid_num_input_two_time(monkeypatch: MonkeyPatch) -> None:
    inputs = ['string', '1']
    prefix = 'Data'
    slogan = '@Number'
    # when the input is string type, then move on until the function get a int value
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    assert num_input(prefix, slogan) == 1


def test_invalid_num_input_equal(monkeypatch: MonkeyPatch) -> None:
    with pytest.raises(AssertionError):
        inputs = ['2']
        prefix = 'Data'
        slogan = '@Number'
        monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
        assert num_input(prefix, slogan) == 1


# @pytest.fixture
# def input_func(monkeypatch: MonkeyPatch) -> num_input:
#     inputs = ['3']
#     prefix = 'Data'
#     slogan = '@Number'
#     monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
#     return num_input(prefix, slogan)


# def test_valid_limit_num_input(input_func: num_input) -> None:
#     option_list = ['Option1', 'Option2', 'Option3']
#     prefix = 'Data'
#     assert limit_num_input(option_list, prefix, input_func) == 3


def test_valid_limit_num_input_one_time(monkeypatch: MonkeyPatch) -> None:
    option_list = ['Option1', 'Option2', 'Option3']
    inputs = ['3']
    prefix = 'Data'
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    assert limit_num_input(option_list, prefix, num_input) == 3


def test_valid_limit_num_input_two_time(monkeypatch: MonkeyPatch) -> None:
    option_list = ['Option1', 'Option2', 'Option3']
    inputs = ['4', '1']
    prefix = 'Data'
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    assert limit_num_input(option_list, prefix, num_input) == 1


def test_invalid_limit_num_input(monkeypatch: MonkeyPatch) -> None:
    with pytest.raises(AssertionError):
        option_list = ['Option1', 'Option2', 'Option3']
        inputs = ['4', '1']
        prefix = 'Data'
        monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
        assert limit_num_input(option_list, prefix, num_input) == 4
