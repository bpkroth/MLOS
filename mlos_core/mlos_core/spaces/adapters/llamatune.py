#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Implementation of LlamaTune space adapter.

LlamaTune is a technique that transforms the original parameter space into a
lower-dimensional space to try and improve the sample efficiency of the underlying
optimizer by making use of the inherent parameter sensitivity correlations in most
systems.

See Also: `LlamaTune: Sample-Efficient DBMS Configuration Tuning
<https://www.microsoft.com/en-us/research/publication/llamatune-sample-efficient-dbms-configuration-tuning>`_.
"""
import os
from typing import Any
from warnings import warn

import ConfigSpace
import ConfigSpace.exceptions
import numpy as np
import numpy.typing as npt
import pandas as pd
from ConfigSpace.hyperparameters import NumericalHyperparameter
from sklearn.preprocessing import MinMaxScaler

from mlos_core.spaces.adapters.adapter import BaseSpaceAdapter
from mlos_core.util import normalize_config


class LlamaTuneAdapter(BaseSpaceAdapter):  # pylint: disable=too-many-instance-attributes
    """Implementation of LlamaTune, a set of parameter space transformation techniques,
    aimed at improving the sample-efficiency of the underlying optimizer.
    """

    DEFAULT_NUM_LOW_DIMS = 16
    """Default number of dimensions in the low-dimensional search space, generated by
    HeSBO projection.
    """

    DEFAULT_SPECIAL_PARAM_VALUE_BIASING_PERCENTAGE = 0.2
    """Default percentage of bias for each special parameter value."""

    DEFAULT_MAX_UNIQUE_VALUES_PER_PARAM = 10000
    """Default number of (max) unique values of each parameter, when space
    discretization is used.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        orig_parameter_space: ConfigSpace.ConfigurationSpace,
        num_low_dims: int = DEFAULT_NUM_LOW_DIMS,
        special_param_values: dict | None = None,
        max_unique_values_per_param: int | None = DEFAULT_MAX_UNIQUE_VALUES_PER_PARAM,
        use_approximate_reverse_mapping: bool = False,
    ):
        """
        Create a space adapter that employs LlamaTune's techniques.

        Parameters
        ----------
        orig_parameter_space : ConfigSpace.ConfigurationSpace
            The original (user-provided) parameter space to optimize.
        num_low_dims : int
            Number of dimensions used in the low-dimensional parameter search space.
        special_param_values_dict : dict | None
            Dictionary of special
        max_unique_values_per_param : int | None
            Number of unique values per parameter. Used to discretize the parameter space.
            If `None` space discretization is disabled.
        """
        super().__init__(orig_parameter_space=orig_parameter_space)

        if num_low_dims >= len(orig_parameter_space):
            raise ValueError(
                "Number of target config space dimensions should be "
                "less than those of original config space."
            )

        # Validate input special param values dict
        special_param_values = special_param_values or {}
        self._validate_special_param_values(special_param_values)

        # Create low-dimensional parameter search space
        self._construct_low_dim_space(num_low_dims, max_unique_values_per_param)

        # Initialize config values scaler: from (-1, 1) to (0, 1) range
        config_scaler = MinMaxScaler(feature_range=(0, 1))
        ones_vector = np.ones(len(list(self.orig_parameter_space.values())))
        config_scaler.fit(np.array([-ones_vector, ones_vector]))
        self._config_scaler = config_scaler

        # Generate random mapping from low-dimensional space to original config space
        num_orig_dims = len(list(self.orig_parameter_space.values()))
        self._h_matrix = self._random_state.choice(range(num_low_dims), num_orig_dims)
        self._sigma_vector = self._random_state.choice([-1, 1], num_orig_dims)

        # Used to retrieve the low-dim point, given the high-dim one
        self._suggested_configs: dict[ConfigSpace.Configuration, ConfigSpace.Configuration] = {}
        self._pinv_matrix: npt.NDArray
        self._use_approximate_reverse_mapping = use_approximate_reverse_mapping

    @property
    def target_parameter_space(self) -> ConfigSpace.ConfigurationSpace:
        """Get the parameter space, which is explored by the underlying optimizer."""
        return self._target_config_space

    def inverse_transform(self, configuration: pd.Series) -> pd.Series:
        config = ConfigSpace.Configuration(
            self.orig_parameter_space,
            values=configuration.dropna().to_dict(),
        )

        target_config = self._suggested_configs.get(config, None)
        # NOTE: HeSBO is a non-linear projection method, and does not inherently
        # support inverse projection
        # To (partly) support this operation, we keep track of the suggested
        # low-dim point(s) along with the respective high-dim point; this way we
        # can retrieve the low-dim point, from its high-dim counterpart.
        if target_config is None:
            # Inherently it is not supported to register points, which were not
            # suggested by the optimizer.
            if config == self.orig_parameter_space.get_default_configuration():
                # Default configuration should always be registerable.
                pass
            elif not self._use_approximate_reverse_mapping:
                raise ValueError(
                    f"{repr(config)}\n"
                    "The above configuration was not suggested by the optimizer. "
                    "Approximate reverse mapping is currently disabled; "
                    "thus *only* configurations suggested "
                    "previously by the optimizer can be registered."
                )

            target_config = self._try_inverse_transform_config(config)

        return pd.Series(target_config, index=list(self.target_parameter_space.keys()))

    def _try_inverse_transform_config(
        self,
        config: ConfigSpace.Configuration,
    ) -> ConfigSpace.Configuration:
        """
        Attempts to generate an inverse mapping of the given configuration that wasn't
        previously registered.

        Parameters
        ----------
        configuration : ConfigSpace.Configuration
            Configuration in the original high-dimensional space.

        Returns
        -------
        ConfigSpace.Configuration
            Configuration in the low-dimensional space.

        Raises
        ------
        ValueError
            On conversion errors.
        """
        # ...yet, we try to support that by implementing an approximate
        # reverse mapping using pseudo-inverse matrix.
        if getattr(self, "_pinv_matrix", None) is None:
            self._try_generate_approx_inverse_mapping()

        # Replace NaNs with zeros for inactive hyperparameters
        config_vector = np.nan_to_num(config.get_array(), nan=0.0)
        # Perform approximate reverse mapping
        # NOTE: applying special value biasing is not possible
        vector: npt.NDArray = self._config_scaler.inverse_transform(np.array([config_vector]))[0]
        target_config_vector: npt.NDArray = self._pinv_matrix.dot(vector)
        # Clip values to to [-1, 1] range of the low dimensional space.
        for idx, value in enumerate(target_config_vector):
            target_config_vector[idx] = np.clip(value, -1, 1)
        if self._q_scaler is not None:
            # If the max_unique_values_per_param is set, we need to scale
            # the low dimension space back to the discretized space as well.
            target_config_vector = self._q_scaler.inverse_transform(
                np.array([target_config_vector])
            )[0]
            assert isinstance(target_config_vector, np.ndarray)
            # Clip values to [1, max_value] range (floating point errors may occur).
            for idx, value in enumerate(target_config_vector):
                target_config_vector[idx] = int(np.clip(value, 1, self._q_scaler.data_max_[idx]))
            target_config_vector = target_config_vector.astype(int)
        # Convert the vector to a dictionary.
        target_config_dict = dict(
            zip(
                self.target_parameter_space.keys(),
                target_config_vector,
            )
        )
        target_config = ConfigSpace.Configuration(
            self.target_parameter_space,
            values=target_config_dict,
            # This method results in hyperparameter type conversion issues
            # (e.g., float instead of int), so we use the values dict instead.
            # vector=target_config_vector,
        )

        # Check to see if the approximate reverse mapping looks OK.
        # Note: we know this isn't 100% accurate, so this is just a warning and
        # mostly meant for internal debugging.
        configuration_dict = dict(config)
        double_checked_config = self._transform(dict(target_config))
        double_checked_config = {
            # Skip the special values that aren't in the original space.
            k: v
            for k, v in double_checked_config.items()
            if k in configuration_dict
        }
        if double_checked_config != configuration_dict and (
            os.environ.get("MLOS_DEBUG", "false").lower() in {"1", "true", "y", "yes"}
        ):
            warn(
                (
                    f"Note: Configuration {configuration_dict} was inverse transformed to "
                    f"{dict(target_config)} and then back to {double_checked_config}. "
                    "This is an approximate reverse mapping for previously unregistered "
                    "configurations, so this is just a warning."
                ),
                UserWarning,
            )

        # But the inverse mapping should at least be valid in the target space.
        try:
            ConfigSpace.Configuration(
                self.target_parameter_space,
                values=target_config,
            ).check_valid_configuration()
        except ConfigSpace.exceptions.IllegalValueError as err:
            raise ValueError(
                f"Invalid configuration {target_config} generated by "
                f"inverse mapping of {config}:\n{err}"
            ) from err

        return target_config

    def transform(self, configuration: pd.Series) -> pd.Series:
        target_values_dict = configuration.to_dict()
        target_configuration = ConfigSpace.Configuration(
            self.target_parameter_space,
            values=target_values_dict,
        )

        orig_values_dict = self._transform(target_values_dict)
        orig_configuration = normalize_config(self.orig_parameter_space, orig_values_dict)

        # Validate that the configuration is in the original space.
        try:
            ConfigSpace.Configuration(
                self.orig_parameter_space,
                values=orig_configuration,
            ).check_valid_configuration()
        except ConfigSpace.exceptions.IllegalValueError as err:
            raise ValueError(
                f"Invalid configuration {orig_configuration} generated by "
                f"transformation of {target_configuration}:\n{err}"
            ) from err

        # Add to inverse dictionary -- needed for registering the performance later
        self._suggested_configs[orig_configuration] = target_configuration

        ret: pd.Series = pd.Series(
            list(orig_configuration.values()), index=list(orig_configuration.keys())
        )
        return ret

    def _construct_low_dim_space(
        self,
        num_low_dims: int,
        max_unique_values_per_param: int | None,
    ) -> None:
        """
        Constructs the low-dimensional parameter (potentially discretized) search space.

        Parameters
        ----------
        num_low_dims : int
            Number of dimensions used in the low-dimensional parameter search space.

        max_unique_values_per_param: int | None:
            Number of unique values per parameter. Used to discretize the parameter space.
            If `None` space discretization is disabled.
        """
        # Define target space parameters
        q_scaler = None
        hyperparameters: list[
            ConfigSpace.UniformFloatHyperparameter | ConfigSpace.UniformIntegerHyperparameter
        ]
        if max_unique_values_per_param is None:
            hyperparameters = [
                ConfigSpace.UniformFloatHyperparameter(name=f"dim_{idx}", lower=-1, upper=1)
                for idx in range(num_low_dims)
            ]
        else:
            # Currently supported optimizers do not support defining a discretized
            # space (like ConfigSpace does using `q` kwarg).
            # Thus, to support space discretization, we define the low-dimensional
            # space using integer hyperparameters.
            # We also employ a scaler, which scales suggested values to [-1, 1]
            # range, used by HeSBO projection.
            hyperparameters = [
                ConfigSpace.UniformIntegerHyperparameter(
                    name=f"dim_{idx}",
                    lower=1,
                    upper=max_unique_values_per_param,
                )
                for idx in range(num_low_dims)
            ]

            # Initialize quantized values scaler:
            # from [0, max_unique_values_per_param] to (-1, 1) range
            q_scaler = MinMaxScaler(feature_range=(-1, 1))
            ones_vector = np.ones(num_low_dims)
            max_value_vector = ones_vector * max_unique_values_per_param
            q_scaler.fit(np.array([ones_vector, max_value_vector]))

        self._q_scaler = q_scaler

        # Construct low-dimensional parameter search space
        config_space = ConfigSpace.ConfigurationSpace(name=self.orig_parameter_space.name)
        # use same random state as in original parameter space
        config_space.random = self._random_state
        config_space.add(hyperparameters)
        self._target_config_space = config_space

    def _transform(self, configuration: dict) -> dict:
        """
        Projects a low-dimensional point (configuration) to the high-dimensional
        original parameter space, and then biases the resulting parameter values towards
        their special value(s) (if any).

        Parameters
        ----------
        configuration : dict
            Configuration in the low-dimensional space.

        Returns
        -------
        configuration : dict
            Projected configuration in the high-dimensional original search space.
        """
        original_parameters = list(self.orig_parameter_space.values())
        low_dim_config_values = list(configuration.values())

        if self._q_scaler is not None:
            # Scale parameter values from [1, max_value] to [-1, 1]
            low_dim_config_values = self._q_scaler.transform(np.array([low_dim_config_values]))[0]

        # Project low-dim point to original parameter space
        original_config_values = [
            self._sigma_vector[idx] * low_dim_config_values[self._h_matrix[idx]]
            for idx in range(len(original_parameters))
        ]
        # Scale parameter values to [0, 1]
        original_config_values = self._config_scaler.transform(np.array([original_config_values]))[
            0
        ]

        original_config = {}
        for param, norm_value in zip(original_parameters, original_config_values):
            # Clip value to force it to fall in [0, 1]
            # NOTE: HeSBO projection ensures that theoretically but due to
            #       floating point ops nuances this is not always guaranteed
            value = np.clip(norm_value, 0, 1)

            if isinstance(param, ConfigSpace.CategoricalHyperparameter):
                index = int(value * len(param.choices))  # truncate integer part
                index = max(0, min(len(param.choices) - 1, index))
                # NOTE: potential rounding here would be unfair to first & last values
                orig_value = param.choices[index]
            elif isinstance(param, NumericalHyperparameter):
                if param.name in self._special_param_values_dict:
                    value = self._special_param_value_scaler(param, value)

                orig_value = param.to_value(value)
                orig_value = np.clip(orig_value, param.lower, param.upper)
            else:
                raise NotImplementedError(
                    "Only Categorical, Integer, and Float hyperparameters are currently supported."
                )

            original_config[param.name] = orig_value

        return original_config

    def _special_param_value_scaler(
        self,
        param: NumericalHyperparameter,
        input_value: float,
    ) -> float:
        """
        Biases the special value(s) of this parameter, by shifting the normalized
        `input_value` towards those.

        Parameters
        ----------
        param: NumericalHyperparameter
            Parameter of the original parameter space.

        input_value: float
            Normalized value for this parameter, as suggested by the underlying optimizer.

        Returns
        -------
        biased_value: float
            Normalized value after special value(s) biasing is applied.
        """
        special_values_list = self._special_param_values_dict[param.name]

        # Check if input value corresponds to some special value
        perc_sum = 0.0
        for special_value, biasing_perc in special_values_list:
            perc_sum += biasing_perc
            if input_value < perc_sum:
                return float(param.to_vector(special_value))

        # Scale input value uniformly to non-special values
        return float(param.to_vector((input_value - perc_sum) / (1 - perc_sum)))

    # pylint: disable=too-complex,too-many-branches
    def _validate_special_param_values(self, special_param_values_dict: dict) -> None:
        """
        Checks that the user-provided dict of special parameter values is valid. And
        assigns it to the corresponding attribute.

        Parameters
        ----------
        special_param_values_dict: dict
            User-provided dict of special parameter values.

        Raises
        ------
            ValueError: if dictionary key, valid, or structure is invalid.
            NotImplementedError: if special value is defined for a non-integer parameter
        """
        error_prefix = "Validation of special parameter values dict failed."

        all_parameters = list(self.orig_parameter_space.keys())
        sanitized_dict = {}

        for param, value in special_param_values_dict.items():
            if param not in all_parameters:
                raise ValueError(error_prefix + f"Parameter '{param}' does not exist.")

            hyperparameter = self.orig_parameter_space[param]
            if not isinstance(hyperparameter, ConfigSpace.UniformIntegerHyperparameter):
                raise NotImplementedError(
                    error_prefix + f"Parameter '{param}' is not supported. "
                    "Only Integer Hyperparameters are currently supported."
                )

            if isinstance(value, int):
                # User specifies a single special value -- default biasing percentage is used
                tuple_list = [(value, self.DEFAULT_SPECIAL_PARAM_VALUE_BIASING_PERCENTAGE)]
            elif isinstance(value, tuple) and [type(v) for v in value] == [int, float]:
                # User specifies both special value and biasing percentage
                tuple_list = [value]
            elif isinstance(value, list) and value:
                if all(isinstance(t, int) for t in value):
                    # User specifies list of special values
                    tuple_list = [
                        (v, self.DEFAULT_SPECIAL_PARAM_VALUE_BIASING_PERCENTAGE) for v in value
                    ]
                elif all(
                    isinstance(t, tuple) and [type(v) for v in t] == [int, float] for t in value
                ):
                    # User specifies list of tuples; each tuple defines the special
                    # value and the biasing percentage
                    tuple_list = value
                else:
                    raise ValueError(
                        error_prefix + f"Invalid format in value list for parameter '{param}'. "
                        f"Special value list should contain either integers, "
                        "or (special value, biasing %) tuples."
                    )
            else:
                raise ValueError(
                    error_prefix + f"Invalid format for parameter '{param}'. Dict value should be "
                    "an int, a (int, float) tuple, a list of integers, "
                    "or a list of (int, float) tuples."
                )

            # Are user-specified special values valid?
            if not all(hyperparameter.lower <= v <= hyperparameter.upper for v, _ in tuple_list):
                raise ValueError(
                    error_prefix
                    + "One (or more) special values are outside of parameter "
                    + f"'{param}' value domain."
                )
            # Are user-provided special values unique?
            if len({v for v, _ in tuple_list}) != len(tuple_list):
                raise ValueError(
                    error_prefix
                    + "One (or more) special values are defined more than once "
                    + f"for parameter '{param}'."
                )
            # Are biasing percentages valid?
            if not all(0 < perc < 1 for _, perc in tuple_list):
                raise ValueError(
                    error_prefix
                    + f"One (or more) biasing percentages for parameter '{param}' are invalid: "
                    "i.e., fall outside (0, 1) range."
                )

            total_percentage = sum(perc for _, perc in tuple_list)
            if total_percentage >= 1.0:
                raise ValueError(
                    error_prefix
                    + f"Total special values percentage for parameter '{param}' surpass 100%."
                )
            # ... and reasonable?
            if total_percentage >= 0.5:
                warn(
                    f"Total special values percentage for parameter '{param}' exceeds 50%.",
                    UserWarning,
                )

            sanitized_dict[param] = tuple_list

        self._special_param_values_dict = sanitized_dict

    def _try_generate_approx_inverse_mapping(self) -> None:
        """Tries to generate an approximate reverse mapping:
        i.e., from high-dimensional space to the low-dimensional one.

        Reverse mapping is generated using the pseudo-inverse matrix, of original
        HeSBO projection matrix.
        This mapping can be potentially used to register configurations that were
        *not* previously suggested by the optimizer.

        NOTE: This method is experimental, and there is currently no guarantee that
        it works as expected.

        Raises
        ------
            RuntimeError: if reverse mapping computation fails.
        """
        from scipy.linalg import (  # pylint: disable=import-outside-toplevel
            LinAlgError,
            pinv,
        )

        warn(
            (
                "Trying to register a configuration that was not "
                "previously suggested by the optimizer.\n"
                "This inverse configuration transformation is typically not supported.\n"
                "However, we will try to register this configuration "
                "using an *experimental* method."
            ),
            UserWarning,
        )

        orig_space_num_dims = len(list(self.orig_parameter_space.values()))
        target_space_num_dims = len(list(self.target_parameter_space.values()))

        # Construct dense projection matrix from sparse repr
        proj_matrix = np.zeros(shape=(orig_space_num_dims, target_space_num_dims))
        for row, col in enumerate(self._h_matrix):
            proj_matrix[row][col] = self._sigma_vector[row]

        # Compute pseudo-inverse matrix
        try:
            _inv = pinv(proj_matrix)
            assert _inv is not None and not isinstance(_inv, tuple)
            inv_matrix: npt.NDArray[np.floating[Any]] = _inv
            self._pinv_matrix = inv_matrix
        except LinAlgError as err:
            raise RuntimeError(
                f"Unable to generate reverse mapping using pseudo-inverse matrix: {repr(err)}"
            ) from err
        assert self._pinv_matrix.shape == (target_space_num_dims, orig_space_num_dims)
