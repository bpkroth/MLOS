#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Tunable parameter definition.
"""
from enum import Enum
import copy
import collections
import logging

from abc import abstractmethod

from typing import Generic, List, NamedTuple, Optional, Sequence, Type, TypedDict, TypeVar, Union, SupportsInt, SupportsFloat

_LOG = logging.getLogger(__name__)


# Underlying tunable var types can be int, float, or str.
TunableValueTypes = Union[int, float, str]
TunableValueBaseTypes = Union[Type[int], Type[float], Type[str]]
TunableNumberValueTypes = Union[int, float]
TunableNumberValueBaseTypes = Union[Type[int], Type[float]]
TunableTypeVar = TypeVar("TunableTypeVar", int, float, str)
TunableNumberTypeVar = TypeVar("TunableNumberTypeVar", int, float)
TunableValuesTypes = Union[int, "TunableValue[int]", float, "TunableValue[float]", str, "TunableValue[str]"]
TunableNumberValuesTypes = Union[int, "TunableValue[int]", float, "TunableValue[float]"]


class TunableType(Enum):
    """Acceptable types of tunables as an enum."""

    INT = int
    FLOAT = float
    CATEGORICAL = str

    @classmethod
    def from_type(cls, from_type: Type[TunableTypeVar]) -> "TunableType":
        """Gets the type of the tunable.

        Returns
        -------
        Union[Type[int], Type[float], Type[str]]
        """
        if from_type is int:
            return TunableType.INT
        elif from_type is float:
            return TunableType.FLOAT
        elif from_type is str:
            return TunableType.CATEGORICAL
        raise NotImplementedError(f"Unhandled TunableType: {from_type}")

    @classmethod
    def from_string(cls, type_name: str) -> "TunableType":
        """Convert a string to a TunableType.

        Returns
        -------
        TunableType
        """
        if type_name == "int":
            return TunableType.INT
        if type_name == "float":
            return TunableType.FLOAT
        if type_name == "categorical":
            return TunableType.CATEGORICAL
        raise ValueError(f"Invalid TunableType: {str}")

    @property
    def to_type(self) -> TunableValueBaseTypes:
        """Gets the type of the tunable.

        Returns
        -------
        Union[Type[int], Type[float], Type[str]]
        """
        if self == TunableType.INT:
            return int
        elif self == TunableType.FLOAT:
            return float
        elif self == TunableType.CATEGORICAL:
            return str
        raise NotImplementedError(f"Unhandled TunableType: {self}")

    @property
    def is_categorical(self) -> bool:
        """Checks if the tunable is categorical.

        Returns
        -------
        bool
        """
        return self == TunableType.CATEGORICAL

    @property
    def is_numerical(self) -> bool:
        """Checks if the tunable is numerical.

        Returns
        -------
        bool
        """
        return self in [TunableType.INT, TunableType.FLOAT]


def _coerce_value(value: TunableValuesTypes, to_type: Type[TunableTypeVar]) -> TunableTypeVar:
    """Attempts to convert the value to the specified type.

    Parameters
    ----------
    value: Union[int, float, str]
        The value to convert.

    type_name: int, float, str
        The type to attempt to convert to.

    Returns
    -------
    Union[int, float, str]
    """
    # We need this coercion for the values produced by some optimizers
    # (e.g., scikit-optimize) and for data restored from certain storage
    # systems (where values can be strings).
    to_tunable_type = TunableType.from_type(to_type)
    if isinstance(value, TunableValue):
        value = value.value
    assert isinstance(value, (int, str, float))
    try:
        coerced_value = to_type(value)
    except ValueError:
        _LOG.error("Impossible conversion: %s <- %s %s",
                   to_type, type(value), value)
        raise

    if to_tunable_type == TunableType.INT and isinstance(value, float) and value != coerced_value:
        _LOG.error("Loss of precision: %s <- %s %s",
                   to_type, type(value), value)
        raise ValueError(f"Loss of precision {coerced_value}!={value}")
    return coerced_value


class TunableValue(Generic[TunableTypeVar]):
    """A tunable parameter value type generic.
    Useful to support certain types of operator overloading and encapsulate type checking.
    """

    def __init__(self, tunable_type: Type[TunableTypeVar], value: TunableValuesTypes):
        assert tunable_type in (int, float, str)
        self._type: Type[TunableTypeVar] = tunable_type
        self._tunable_type = TunableType.from_type(tunable_type)
        self._value: TunableTypeVar = _coerce_value(value, self._type)

    def __repr__(self) -> str:
        return f"TunableValue[{self._type}]({self._value})"

    def __str__(self) -> str:
        return str(self._value)

    @property
    def tunable_type(self) -> TunableType:
        """Gets the type of the tunable."""
        return self._tunable_type

    @property
    def type(self) -> Type[TunableTypeVar]:
        """Gets the type of the tunable."""
        return self._type

    @property
    def value(self) -> TunableTypeVar:
        """Gets the value of the tunable."""
        return self._value

    @value.setter
    def value(self, new_value: TunableValuesTypes) -> TunableTypeVar:
        self._value = _coerce_value(new_value, self._type)
        return self._value

    def _try_convert_other(self, other: object) -> Optional["TunableValue[TunableTypeVar]"]:
        """Attempts to covert the object to a TunableValue of this same type or returns None."""
        if isinstance(other, TunableValue):
            other = other.value
        if not isinstance(other, (int, float, str)):
            return None
        try:
            new = copy.deepcopy(self)
            new._value = _coerce_value(other, self._type)   # pylint: disable=protected-access
            return new
        except ValueError:
            return None

    def __eq__(self, other: object) -> bool:
        other_tunable_value = self._try_convert_other(other)
        if other_tunable_value is None:
            return False
        return self._value == other_tunable_value._value

    def __lt__(self, other: object) -> bool:
        other_tunable_value = self._try_convert_other(other)
        if other_tunable_value is None:
            return False
        return self._value < other_tunable_value._value


class TunableNumberValue(TunableValue[TunableNumberTypeVar]):
    """Tunable value for numerical types."""

    def __init__(self, number_type: Type[TunableNumberTypeVar], value: TunableValuesTypes):
        super().__init__(number_type, value)
        self._numerical_type: Type[TunableNumberTypeVar] = number_type
        assert self._numerical_type in (int, float)

    def _other_to_type(self, other: object) -> TunableNumberValuesTypes:
        """Attempts to convert the other object to the same type as this TunableNumberValue."""
        if isinstance(other, str):
            other = _coerce_value(other, self._numerical_type)
        elif isinstance(other, TunableNumberValue):
            other = other.value
        if isinstance(other, (int, float)):
            return other
        raise ValueError(f"Cannot convert other to TunableNumberValue: {other}")

    def __int__(self) -> int:
        return int(self._value)

    def __float__(self) -> float:
        return float(self._value)

    def __add__(self, other: object) -> TunableNumberTypeVar:
        return self._value + self._numerical_type(self._other_to_type(other))

    def __iadd__(self, other: object) -> "TunableNumberValue[TunableNumberTypeVar]":
        self._value += self._other_to_type(other)
        return self

    def __sub__(self, other: object) -> TunableNumberTypeVar:
        return self._value - self._other_to_type(other)

    def __isub__(self, other: object) -> "TunableNumberValue[TunableNumberTypeVar]":
        self._value -= self._other_to_type(other)
        return self

    def __mul__(self, other: object) -> TunableNumberTypeVar:
        return self._value * self._other_to_type(other)

    def __imul__(self, other: object) -> "TunableNumberValue[TunableNumberTypeVar]":
        self._value *= self._other_to_type(other)
        return self

    def __truediv__(self, other: object) -> float:
        return float(self._value / float(self._other_to_type(other)))

    def __floordiv__(self, other: object) -> int:
        return int(self._value / self._other_to_type(other))


class TunableIntValue(TunableNumberValue[int], SupportsInt):
    """TunableValue for integer types."""

    def __init__(self, value: Union[int, TunableValue[int]]):
        super().__init__(int, value)


class TunableFloatValue(TunableNumberValue[float], SupportsFloat):
    """TunableValue for float types."""

    def __init__(self, value: Union[float, TunableValue[float]]):
        super().__init__(float, value)


class TunableCategoricalValue(TunableValue[str]):
    """TunableValue for categorical types."""

    def __init__(self, value: Union[str, TunableValue[str]]):
        super().__init__(str, value)


class TunableDict(TypedDict, total=False):
    """
    A typed dict for tunable parameters.

    Mostly used for mypy type checking.

    These are the types expected to be received from the json config.
    """

    type: str
    description: Optional[str]
    default: TunableValueTypes
    values: Optional[List[str]]
    range: Optional[Union[Sequence[int], Sequence[float]]]
    special: Optional[Union[List[int], List[str]]]


class Tunable(Generic[TunableTypeVar]):
    """
    A tunable parameter definition and its current value.
    """

    def __init__(self, name: str, config: TunableDict, tunable_type: Type[TunableTypeVar]):
        """
        Create an instance of a new tunable parameter.

        Parameters
        ----------
        name : str
            Human-readable identifier of the tunable parameter.
        config : dict
            Python dict that represents a Tunable (e.g., deserialized from JSON)
        """
        self._name = name
        self._description = config.get("description")

        self._type = TunableType.from_string(config["type"])  # required
        assert self._type.to_type == tunable_type

        self._default: TunableValue[TunableTypeVar]
        self._current_value: TunableValue[TunableTypeVar]

    def copy(self) -> "Tunable[TunableTypeVar]":
        """
        Deep copy of the Tunable object.

        Returns
        -------
        tunable : Tunable
            A new Tunable object that is a deep copy of the original one.
        """
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        """
        Produce a human-readable version of the Tunable (mostly for logging).

        Returns
        -------
        string : str
            A human-readable version of the Tunable.
        """
        return f"{self._name}[{self._type}]={self._current_value}"

    def __eq__(self, other: object) -> bool:
        """
        Check if two Tunable objects are equal.

        Parameters
        ----------
        other : Tunable
            A tunable object to compare to.

        Returns
        -------
        is_equal : bool
            True if the Tunables correspond to the same parameter and have the same value and type.
            NOTE: ranges and special values are not currently considered in the comparison.
        """
        if not isinstance(other, Tunable):
            return False
        return bool(
            self._name == other._name and
            self._type == other._type and
            self._current_value == other._current_value
        )

    @property
    def name(self) -> str:
        """
        Get the name / string ID of the tunable.
        """
        return self._name

    @property
    def description(self) -> Optional[str]:
        """
        Get the description of the tunable.
        """
        return self._description

    @property
    def type(self) -> TunableType:
        """
        Get the data type of the tunable.

        Returns
        -------
        type : TunableType
        """
        return self._type

    @property
    def default(self) -> TunableValue[TunableTypeVar]:
        """
        Get the default value of the tunable.
        """
        return self._default

    @property
    def current_value(self) -> TunableValue[TunableTypeVar]:
        """
        Get the current value of the tunable.
        """
        return self._current_value

    @current_value.setter
    def current_value(self, new_value: TunableValuesTypes) -> TunableValue[TunableTypeVar]:
        """
        Set the current value of the tunable.
        """
        self._current_value.value = new_value
        return self._current_value

    @abstractmethod
    def is_valid(self, value: TunableValueTypes) -> bool:
        """
        Check if the value can be assigned to the tunable.

        Parameters
        ----------
        value : Union[int, float, str]
            Value to validate.

        Returns
        -------
        is_valid : bool
            True if the value is valid, False otherwise.
        """


class RangeTuple(NamedTuple, Generic[TunableNumberTypeVar]):
    """Tuple representing a range of values."""

    min: TunableNumberTypeVar
    max: TunableNumberTypeVar


class TunableNumber(Tunable, Generic[TunableNumberTypeVar]):
    """Tunable number."""

    def __init__(self, name: str, config: TunableDict, number_type: Type[TunableNumberTypeVar]):
        assert number_type in (int, float)
        if number_type == int:
            super().__init__(name, config, number_type)
        elif number_type == float:
            super().__init__(name, config, float)
        else:
            raise ValueError(f"Invalid number type: {number_type}")

        self._special = config.get("special")

        config_range = config.get("range")
        assert config_range is not None
        assert len(config_range) == 2, \
            f"Invalid range: {config_range}.  All ranges must be 2-tuples."
        assert set(type(v) for v in config_range) == {number_type}, \
            f"Invalid range: {config_range}.  All types must match."
        if config_range[0] >= config_range[1]:
            raise ValueError(f"Invalid range: {config_range}.  Min must be less than max.")
        self._range: RangeTuple[TunableNumberTypeVar] = RangeTuple[TunableNumberTypeVar](
            min=number_type(config_range[0]), max=number_type(config_range[1]))

    @property
    def range(self) -> RangeTuple[TunableNumberTypeVar]:
        """
        Get the range of the tunable if it is numerical, None otherwise.

        Returns
        -------
        range : (min=number, max=number)
            A 2-tuple of numbers that represents the range of the tunable.
            Numbers can be int or float, depending on the type of the tunable.
        """
        return self._range

    def is_valid(self, value: TunableValueTypes) -> bool:
        """
        Check if the value can be assigned to the tunable.

        Parameters
        ----------
        value : Union[int, float, str]
            Value to validate.

        Returns
        -------
        is_valid : bool
            True if the value is valid, False otherwise.
        """
        if isinstance(value, TunableValue):
            value = value.value

        numerical_value: Union[int, float]
        if isinstance(value, str):
            if self._type == TunableType.INT:
                numerical_value = _coerce_value(value, int)
            elif self._type == TunableType.FLOAT:
                numerical_value = _coerce_value(value, float)
            else:
                raise ValueError(f"Invalid type: {self._type}")
        else:
            numerical_value = value
        assert isinstance(numerical_value, (int, float))
        return bool(self._range[0] <= numerical_value <= self._range[1]) or numerical_value == self._default


class TunableInt(TunableNumber[int]):
    """Discrete Integer Tunable."""

    def __init__(self, name: str, config: TunableDict):
        super().__init__(name, config, int)
        assert self._type == TunableType.INT
        self._default = TunableIntValue(_coerce_value(config["default"], int))
        self._current_value = self._default


class TunableFloat(TunableNumber[float]):
    """Continuous Float Tunable."""

    def __init__(self, name: str, config: TunableDict):
        super().__init__(name, config, float)
        assert self._type == TunableType.FLOAT
        self._default = TunableFloatValue(_coerce_value(config["default"], float))
        self._current_value = self._default


class TunableCategorical(Tunable[str]):
    """Categorical Tunable."""

    def __init__(self, name: str, config: TunableDict):
        super().__init__(name, config, str)
        assert self._type == TunableType.CATEGORICAL
        self._default = TunableCategoricalValue(str(config["default"]))
        self._current_value = self._default

        values = config["values"]
        if not (values and isinstance(values, collections.abc.Iterable)):
            raise ValueError("Must specify values for the categorical type")
        self._values: List[str] = values

        if config.get("range") is not None:
            raise ValueError("Range must be None for the categorical type")
        if config.get("special") is not None:
            raise ValueError("Special values must be None for the categorical type")

    @property
    def categorical_values(self) -> List[str]:
        """
        Get the list of all possible values of a categorical tunable.
        Return None if the tunable is not categorical.

        Returns
        -------
        values : List[str]
            List of all possible values of a categorical tunable.
        """
        assert self._type.is_categorical
        assert self._values is not None
        return self._values

    def is_valid(self, value: TunableValueTypes) -> bool:
        """
        Check if the value can be assigned to the tunable.

        Parameters
        ----------
        value : Union[int, float, str]
            Value to validate.

        Returns
        -------
        is_valid : bool
            True if the value is valid, False otherwise.
        """
        if isinstance(value, TunableValue):
            value = value.value
        return value in self._values


class TunableFactory:
    """Generates an appropriate Tunable type from the given config."""

    @classmethod
    def from_config(cls, name: str, config: TunableDict) -> Union[TunableInt, TunableFloat, TunableCategorical]:
        """Generates an appropriate Tunable type from the given config.

        Parameters
        ----------
        name : str
        config : TunableDict

        Returns
        -------
        Union[TunableInt, TunableFloat, TunableCategorical]
        """
        tunable_type = TunableType.from_string(config["type"])  # required
        if tunable_type == TunableType.INT:
            return TunableInt(name, config)
        elif tunable_type == TunableType.FLOAT:
            return TunableFloat(name, config)
        elif tunable_type == TunableType.CATEGORICAL:
            return TunableCategorical(name, config)
        else:
            raise NotImplementedError(f"Unhandled parameter type: {tunable_type}")


if __name__ == "__main__":
    tunableInt = TunableFactory.from_config(name="int",
        config={"type": "int", "default": 1, "range": [0, 10]})
    assert isinstance(tunableInt, TunableInt)
    assert tunableInt.range == (0, 10)
    assert isinstance(tunableInt.current_value, TunableIntValue)
    assert tunableInt.default == 1
    tunableInt.current_value += 2
    assert isinstance(tunableInt.current_value.value, int)
    assert tunableInt.current_value == 3
    print(tunableInt)

    tunableFloat = TunableFactory.from_config(name="float",
        config={"type": "float", "default": 1.0, "range": [0.0, 10.0]})
    assert isinstance(tunableFloat, TunableFloat)
    assert tunableFloat.range == (0.0, 10.0)
    assert isinstance(tunableFloat.current_value, TunableFloatValue)
    assert tunableFloat.default == 1
    tunableFloat.current_value += 1.1
    assert tunableFloat.current_value == 2.1
    print(tunableFloat)

    tunableCategorical = TunableFactory.from_config(name="categorical",
        config={"type": "categorical", "default": "red", "values": ["red", "blue", "green"]})
    assert isinstance(tunableCategorical, TunableCategorical)
    assert tunableCategorical.categorical_values == ["red", "blue", "green"]
    assert isinstance(tunableCategorical.current_value, TunableCategoricalValue)
    assert tunableCategorical.default == "red"
    #tunableCategorical.current_value += "foo"
    print(tunableCategorical)
