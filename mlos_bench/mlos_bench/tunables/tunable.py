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

from typing import Generic, List, Optional, Sequence, Tuple, Type, TypedDict, TypeVar, Union

_LOG = logging.getLogger(__name__)


class TunableDict(TypedDict, total=False):
    """
    A typed dict for tunable parameters.

    Mostly used for mypy type checking.

    These are the types expected to be received from the json config.
    """

    type: str
    description: Optional[str]
    default: Union[int, float, str]
    values: Optional[List[str]]
    range: Optional[Union[Sequence[int], Sequence[float]]]
    special: Optional[Union[List[int], List[str]]]


# Underlying tunable var types can be int, float, or str.
TunableTypeVar = TypeVar("TunableTypeVar", int, float, str)


class TunableType(Enum):
    """Acceptable types of tunables as an enum."""

    INT: int
    FLOAT: float
    CATEGORICAL: str

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
    def to_type(self) -> Union[Type[int], Type[float], Type[str]]:
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


def _coerce_value(value: Union[int, float, str], to_type: Type[TunableTypeVar]) -> TunableTypeVar:
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
    try:
        coerced_value = to_type(value)
    except Exception:
        _LOG.error("Impossible conversion: %s <- %s %s",
                   to_type, type(value), value)
        raise

    if to_tunable_type == TunableType.INT and isinstance(value, float) and value != coerced_value:
        _LOG.error("Loss of precision: %s <- %s %s",
                   to_type, type(value), value)
        raise ValueError(f"Loss of precision {coerced_value}!={value}")
    return coerced_value


class TunableValue(Generic[TunableTypeVar]):
    """A tunable parameter value type alias.
    Useful to support certain types of operator overloading and encapsulate type checking.
    """

    def __init__(self, tunable_type: TunableType, value: Union[int, float, str]):
        self._tunable_type = tunable_type
        to_type = self._tunable_type.to_type
        assert to_type in (int, float, str)
        self._type: Type[TunableTypeVar] = to_type  # type: ignore[assignment]
        self._value: TunableTypeVar = _coerce_value(value, self._type)

    def __repr__(self) -> str:
        return f"TunableValue({self._value})"

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
    def value(self, new_value: Union["TunableValue", int, float, str]) -> TunableTypeVar:
        if isinstance(new_value, TunableValue):
            self._value = _coerce_value(new_value.value, self._type)
        else:
            self._value = _coerce_value(new_value, self._type)
        return self._value

    def _other_to_tunable_value(self, other: object) -> Optional["TunableValue[TunableTypeVar]"]:
        """Attempts to covert the object to a TunableValue."""
        if isinstance(other, TunableValue):
            return other
        try:
            if not isinstance(other, (int, float, str)):
                return None
            return TunableValue(self._tunable_type, other)
        except Exception:
            return None

    def __eq__(self, other: object) -> bool:
        other_tunable_value = self._other_to_tunable_value(other)
        if other_tunable_value is None:
            return False
        return self._value == other_tunable_value._value

    def __lt__(self, other: object) -> bool:
        other_tunable_value = self._other_to_tunable_value(other)
        if other_tunable_value is None:
            return False
        return self._value < other_tunable_value._value

    def __add__(self, other: object) -> TunableTypeVar:
        if not self.tunable_type.is_numerical:
            raise ValueError("Cannot add to categorical tunable")
        other_tunable_value = self._other_to_tunable_value(other)
        if other_tunable_value is None:
            raise ValueError(f"Cannot convert other to TunableValue: {other}")
        self._value += other_tunable_value._value
        return self._value


class TunableIntValue(TunableValue[int]):
    pass


class TunableFloatValue(TunableValue[float]):
    pass


class TunableCategoricalValue(TunableValue[str]):
    pass


class Tunable:  # pylint: disable=too-many-instance-attributes
    """
    A tunable parameter definition and its current value.
    """

    def __init__(self, name: str, config: TunableDict):
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
        self._type = TunableType.from_string(config["type"])  # required
        self._description = config.get("description")
        self._default = config["default"]
        self._values = config.get("values")
        self._range: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
        config_range = config.get("range")
        if config_range is not None:
            assert len(config_range) == 2, f"Invalid range: {config_range}"
            config_range = (config_range[0], config_range[1])
            self._range = config_range
        self._special = config.get("special")
        self._current_value = TunableValue(self._type, self._default)
        if self._type.is_categorical:
            if not (self._values and isinstance(self._values, collections.abc.Iterable)):
                raise ValueError("Must specify values for the categorical type")
            if self._range is not None:
                raise ValueError("Range must be None for the categorical type")
            if self._special is not None:
                raise ValueError("Special values must be None for the categorical type")
        elif self._type.is_numerical:
            if not self._range or len(self._range) != 2 or self._range[0] >= self._range[1]:
                raise ValueError(f"Invalid range: {self._range}")
        else:
            raise ValueError(f"Invalid parameter type: {self._type}")

    def __repr__(self) -> str:
        """
        Produce a human-readable version of the Tunable (mostly for logging).

        Returns
        -------
        string : str
            A human-readable version of the Tunable.
        """
        return f"{self._name}={self._current_value}"

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

    def copy(self) -> "Tunable":
        """
        Deep copy of the Tunable object.

        Returns
        -------
        tunable : Tunable
            A new Tunable object that is a deep copy of the original one.
        """
        return copy.deepcopy(self)

    @property
    def value(self) -> TunableValue:
        """
        Get the current value of the tunable.
        """
        return self._current_value

    @value.setter
    def value(self, new_value: Union[TunableValue, int, float, str]) -> TunableValue:
        """
        Set the current value of the tunable.
        """
        if isinstance(new_value, TunableValue):
            self._current_value = new_value.value
        else:
            self._current_value.value = new_value
        return self._current_value

    def is_valid(self, value: Union[TunableValue, int, float, str]) -> bool:
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
        if self._type.is_categorical and self._values:
            return value in self._values
        elif self._type.is_numerical and self._range:
            assert isinstance(value, (int, float))
            return bool(self._range[0] <= value <= self._range[1]) or value == self._default
        else:
            raise ValueError(f"Invalid parameter type: {self._type}")

    @property
    def categorical_value(self) -> str:
        """
        Get the current value of the categorical tunable as a string.
        """
        return str(self._current_value.value)

    @property
    def numerical_value(self) -> Union[int, float]:
        """
        Get the current value of the numerical tunable as a number.
        """
        return self._current_value.value

    @property
    def name(self) -> str:
        """
        Get the name / string ID of the tunable.
        """
        return self._name

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
    def range(self) -> Union[Tuple[int, int], Tuple[float, float]]:
        """
        Get the range of the tunable if it is numerical, None otherwise.

        Returns
        -------
        range : (number, number)
            A 2-tuple of numbers that represents the range of the tunable.
            Numbers can be int or float, depending on the type of the tunable.
        """
        assert self._type.is_numerical
        assert self._range is not None
        return self._range

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
