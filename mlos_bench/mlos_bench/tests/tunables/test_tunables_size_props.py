#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Unit tests for checking tunable size properties.
"""

import numpy as np
import pytest

from mlos_bench.tunables.tunable import Tunable


def test_tunable_int_size_props() -> None:
    """Test tunable int size properties"""
    tunable = Tunable(
        name="test",
        config={
            "type": "int",
            "range": [1, 5],
            "default": 3,
        })
    assert tunable.span == 4
    assert tunable.cardinality == 5


def test_tunable_float_size_props() -> None:
    """Test tunable float size properties"""
    tunable = Tunable(
        name="test",
        config={
            "type": "float",
            "range": [1.5, 5],
            "default": 3,
        })
    assert tunable.span == 3.5
    assert tunable.cardinality == np.inf


def test_tunable_categorical_size_props() -> None:
    """Test tunable categorical size properties"""
    tunable = Tunable(
        name="test",
        config={
            "type": "categorical",
            "values": ["a", "b", "c"],
            "default": "a",
        })
    with pytest.raises(AssertionError):
        _ = tunable.span
    assert tunable.cardinality == 3


def test_tunable_quantized_int_size_props() -> None:
    """Test quantized tunable int size properties"""
    tunable = Tunable(
        name="test",
        config={
            "type": "int",
            "range": [100, 1000],
            "default": 100,
            "quantization": 100
        })
    assert tunable.span == 900
    assert tunable.cardinality == 10


def test_tunable_quantized_float_size_props() -> None:
    """Test quantized tunable float size properties"""
    tunable = Tunable(
        name="test",
        config={
            "type": "float",
            "range": [0, 1],
            "default": 0,
            "quantization": .1
        })
    assert tunable.span == 1
    assert tunable.cardinality == 11
