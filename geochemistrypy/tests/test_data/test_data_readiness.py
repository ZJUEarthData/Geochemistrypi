# -*- coding: utf-8 -*-
from data.data_readiness import num2option, num_input
import pytest
from pytest import MonkeyPatch


def test_num2option() -> None:
    array = ['a', 'b', 'c']
    num2option(array)


def test_empty_num2option() -> None:
    array = []
    num2option(array)


def test_valid_num_input(monkeypatch: MonkeyPatch) -> None:
    inputs = ['1']
    prefix = 'Data'
    slogan = '@Number'
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    assert num_input(prefix, slogan) == 1


def test_invalid_num_input_equal(monkeypatch: MonkeyPatch) -> None:
    with pytest.raises(AssertionError):
        inputs = ['2']
        prefix = 'Data'
        slogan = '@Number'
        monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
        assert num_input(prefix, slogan) == 1


def test_invalid_num_input_type(monkeypatch: MonkeyPatch) -> None:
    inputs = ['string', '1']
    prefix = 'Data'
    slogan = '@Number'
    # when the input is string type, then move on until the function get a int value
    monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))
    assert num_input(prefix, slogan) == 1

