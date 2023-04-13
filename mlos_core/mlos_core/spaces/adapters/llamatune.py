#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Implementation of LlamaTune space adapter.
"""
from typing import Dict, Optional
from warnings import warn

import ConfigSpace
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mlos_core.spaces.adapters.adapter import BaseSpaceAdapter


class LlamaTuneAdapter(BaseSpaceAdapter):   # pylint: disable=too-many-instance-attributes
    """
    Implementation of LlamaTune, a set of parameter space transformation techniques,
    aimed at improving the sample-efficiency of the underlying optimizer.
    """

    # pylint: disable=consider-alternative-union-syntax,too-many-arguments

    DEFAULT_NUM_LOW_DIMS = 16
    """Default number of dimensions in the low-dimensional search space, generated by HeSBO projection"""

    DEFAULT_SPECIAL_PARAM_VALUE_BIASING_PERCENTAGE = .2
    """Default percentage of bias for each special parameter value"""

    DEFAULT_MAX_UNIQUE_VALUES_PER_PARAM = 10000
    """Default number of (max) unique values of each parameter, when space discretization is used"""

    def __init__(
        self,
        orig_config_space: ConfigSpace.ConfigurationSpace,
        num_low_dims: int = DEFAULT_NUM_LOW_DIMS,
        special_param_values: Optional[dict] = None,
        max_unique_values_per_param: Optional[int] = DEFAULT_MAX_UNIQUE_VALUES_PER_PARAM,
        use_approximate_reverse_mapping: bool = False,
    ) -> None:
        """Create a space adapter that employs LlamaTune's techniques.

        Parameters
        ----------
        parameter_space : ConfigSpace.ConfigurationSpace
            The original (user-provided) parameter space to optimize.

        num_low_dims: int
            Number of dimensions used in the low-dimensional parameter search space.

        special_param_values_dict: Optional[dict]
            Dictionary of special

        max_unique_values_per_param: Optional[int]:
            Number of unique values per parameter. Used to discretize the parameter space.
            If `None` space discretization is disabled.
        """
        super().__init__(orig_config_space)

        if num_low_dims >= len(orig_config_space):
            raise ValueError("Number of target config space dimensions should be less than those of original config space.")

        # Validate input special param values dict
        special_param_values = special_param_values or {}
        self._validate_special_param_values(special_param_values)

        # Create low-dimensional parameter search space
        self._construct_low_dim_space(num_low_dims, max_unique_values_per_param)

        # Initialize config values scaler: from (-1, 1) to (0, 1) range
        config_scaler = MinMaxScaler(feature_range=(0, 1))
        ones_vector = np.ones(len(self.orig_parameter_space.get_hyperparameters()))
        config_scaler.fit([-ones_vector, ones_vector])
        self._config_scaler = config_scaler

        # Generate random mapping from low-dimensional space to original config space
        num_orig_dims = len(self.orig_parameter_space.get_hyperparameters())
        self._h_matrix = self._random_state.choice(range(num_low_dims), num_orig_dims)
        self._sigma_vector = self._random_state.choice([-1, 1], num_orig_dims)

        # Used to retrieve the low-dim point, given the high-dim one
        self._suggested_configs: Dict[ConfigSpace.Configuration, ConfigSpace.Configuration] = {}
        self._pinv_matrix: npt.NDArray
        self._use_approximate_reverse_mapping = use_approximate_reverse_mapping

    @property
    def target_parameter_space(self) -> ConfigSpace.ConfigurationSpace:
        """Get the parameter space, which is explored by the underlying optimizer."""
        return self._target_config_space

    def inverse_transform(self, configurations: pd.DataFrame) -> pd.DataFrame:
        target_configurations = []
        for (_, config) in configurations.iterrows():
            configuration = ConfigSpace.Configuration(self.orig_parameter_space, values=config.to_dict())

            target_config = self._suggested_configs.get(configuration, None)
            # NOTE: HeSBO is a non-linear projection method, and does not inherently support inverse projection
            # To (partly) support this operation, we keep track of the suggested low-dim point(s) along with the
            # respective high-dim point; this way we can retrieve the low-dim point, from its high-dim counterpart.
            if target_config is None:
                # Inherently it is not supported to register points, which were not suggested by the optimizer.
                if not self._use_approximate_reverse_mapping:
                    raise ValueError(f"{repr(configuration)}\n" "The above configuration was not suggested by the optimizer. "
                                     "Approximate reverse mapping is currently disabled; thus *only* configurations suggested "
                                     "previously by the optimizer can be registered.")

                # ...yet, we try to support that by implementing an approximate reverse mapping using pseudo-inverse matrix.
                if getattr(self, '_pinv_matrix', None) is None:
                    self._try_generate_approx_inverse_mapping()

                # Perform approximate reverse mapping
                # NOTE: applying special value biasing is not possible
                vector = self._config_scaler.inverse_transform([configuration.get_array()])[0]
                target_config_vector = self._pinv_matrix.dot(vector)
                target_config = ConfigSpace.Configuration(self.target_parameter_space, vector=target_config_vector)

            target_configurations.append(target_config)

        return pd.DataFrame(target_configurations, columns=self.target_parameter_space.get_hyperparameter_names())

    def transform(self, configuration: pd.DataFrame) -> pd.DataFrame:
        if len(configuration) != 1:
            raise ValueError("Configuration dataframe must contain exactly 1 row. "
                             f"Found {len(configuration)} rows.")

        target_values_dict = configuration.iloc[0].to_dict()
        target_configuration = ConfigSpace.Configuration(self.target_parameter_space, values=target_values_dict)

        orig_values_dict = self._transform(target_values_dict)
        orig_configuration = ConfigSpace.Configuration(self.orig_parameter_space, values=orig_values_dict)

        # Add to inverse dictionary -- needed for registering the performance later
        self._suggested_configs[orig_configuration] = target_configuration

        return pd.DataFrame([orig_values_dict.values()], columns=self.orig_parameter_space.get_hyperparameter_names())

    def _construct_low_dim_space(self, num_low_dims: int, max_unique_values_per_param: Optional[int]) -> None:
        """Constructs the low-dimensional parameter (potentially discretized) search space.

        Parameters
        ----------
        num_low_dims : int
            Number of dimensions used in the low-dimensional parameter search space.

        max_unique_values_per_param: Optional[int]:
            Number of unique values per parameter. Used to discretize the parameter space.
            If `None` space discretization is disabled.
        """
        # Define target space parameters
        q_scaler = None
        if max_unique_values_per_param is None:
            hyperparameters = [
                ConfigSpace.UniformFloatHyperparameter(name=f'dim_{idx}', lower=-1, upper=1)
                for idx in range(num_low_dims)
            ]
        else:
            # Currently supported optimizers do not support defining a discretized space (like ConfigSpace does using `q` kwarg).
            # Thus, to support space discretization, we define the low-dimensional space using integer hyperparameters.
            # We also employ a scaler, which scales suggested values to [-1, 1] range, used by HeSBO projection.
            hyperparameters = [
                ConfigSpace.UniformIntegerHyperparameter(name=f'dim_{idx}', lower=1, upper=max_unique_values_per_param)
                for idx in range(num_low_dims)
            ]

            # Initialize quantized values scaler: from [0, max_unique_values_per_param] to (-1, 1) range
            q_scaler = MinMaxScaler(feature_range=(-1, 1))
            ones_vector = np.ones(num_low_dims)
            max_value_vector = ones_vector * max_unique_values_per_param
            q_scaler.fit([ones_vector, max_value_vector])

        self._q_scaler = q_scaler

        # Construct low-dimensional parameter search space
        config_space = ConfigSpace.ConfigurationSpace(name=self.orig_parameter_space.name)
        config_space.random = self._random_state    # use same random state as in original parameter space
        config_space.add_hyperparameters(hyperparameters)
        self._target_config_space = config_space

    def _transform(self, configuration: dict) -> dict:
        """Projects a low-dimensional point (configuration) to the high-dimensional original parameter space,
        and then biases the resulting parameter values towards their special value(s) (if any).

        Parameters
        ----------
        configuration : dict
            Configuration in the low-dimensional space.

        Returns
        -------
        configuration : dict
            Projected configuration in the high-dimensional original search space.
        """
        original_parameters = self.orig_parameter_space.get_hyperparameters()
        low_dim_config_values = list(configuration.values())

        if self._q_scaler is not None:
            # Scale parameter values from [1, max_value] to [-1, 1]
            low_dim_config_values = self._q_scaler.transform([low_dim_config_values])[0]

        # Project low-dim point to original parameter space
        original_config_values = [
            self._sigma_vector[idx] * low_dim_config_values[self._h_matrix[idx]]
            for idx in range(len(original_parameters))
        ]
        # Scale parameter values to [0, 1]
        original_config_values = self._config_scaler.transform([original_config_values])[0]

        original_config = {}
        for param, norm_value in zip(original_parameters, original_config_values):
            # Clip value to force it to fall in [0, 1]
            # NOTE: HeSBO projection ensures that theoretically but due to
            #       floating point ops nuances this is not always guaranteed
            value = max(0., min(1., norm_value))    # pylint: disable=redefined-loop-name

            if isinstance(param, ConfigSpace.CategoricalHyperparameter):
                index = int(value * len(param.choices))     # truncate integer part
                index = max(0, min(len(param.choices) - 1, index))
                # NOTE: potential rounding here would be unfair to first & last values
                orig_value = param.choices[index]
            elif isinstance(param, ConfigSpace.hyperparameters.NumericalHyperparameter):
                if param.name in self._special_param_values_dict:
                    value = self._special_param_value_scaler(param, value)

                orig_value = param._transform(value)    # pylint: disable=protected-access
                orig_value = max(param.lower, min(param.upper, orig_value))
            else:
                raise NotImplementedError("Only Categorical, Integer, and Float hyperparameters are currently supported.")

            original_config[param.name] = orig_value

        return original_config

    def _special_param_value_scaler(self, param: ConfigSpace.UniformIntegerHyperparameter, input_value: float) -> float:
        """Biases the special value(s) of this parameter, by shifting the normalized `input_value` towards those.

        Parameters
        ----------
        param: ConfigSpace.UniformIntegerHyperparameter
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
        perc_sum = 0.
        ret: float
        for special_value, biasing_perc in special_values_list:
            perc_sum += biasing_perc
            if input_value < perc_sum:
                ret = param._inverse_transform(special_value)  # pylint: disable=protected-access
                return ret

        # Scale input value uniformly to non-special values
        ret = param._inverse_transform(                                         # pylint: disable=protected-access
            param._transform_scalar((input_value - perc_sum) / (1 - perc_sum)))  # pylint: disable=protected-access
        return ret

    # pylint: disable=too-complex,too-many-branches
    def _validate_special_param_values(self, special_param_values_dict: dict) -> None:
        """Checks that the user-provided dict of special parameter values is valid.
        And assigns it to the corresponding attribute.

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

        all_parameters = self.orig_parameter_space.get_hyperparameter_names()
        sanitized_dict = {}

        for param, value in special_param_values_dict.items():
            if param not in all_parameters:
                raise ValueError(error_prefix + f"Parameter '{param}' does not exist.")

            hyperparameter = self.orig_parameter_space.get_hyperparameter(param)
            if not isinstance(hyperparameter, ConfigSpace.UniformIntegerHyperparameter):
                raise NotImplementedError(error_prefix + f"Parameter '{param}' is not supported. "
                                          "Only Integer Hyperparameters are currently supported.")

            if isinstance(value, int):
                # User specifies a single special value -- default biasing percentage is used
                tuple_list = [(value, self.DEFAULT_SPECIAL_PARAM_VALUE_BIASING_PERCENTAGE)]
            elif isinstance(value, tuple) and [type(v) for v in value] == [int, float]:
                # User specifies both special value and biasing percentage
                tuple_list = [value]    # type: ignore
            elif isinstance(value, list) and value:
                if all(isinstance(t, int) for t in value):
                    # User specifies list of special values
                    tuple_list = [(v, self.DEFAULT_SPECIAL_PARAM_VALUE_BIASING_PERCENTAGE) for v in value]
                elif all(isinstance(t, tuple) and [type(v) for v in t] == [int, float] for t in value):
                    # User specifies list of tuples; each tuple defines the special value and the biasing percentage
                    tuple_list = value
                else:
                    raise ValueError(error_prefix + f"Invalid format in value list for parameter '{param}'. "
                                     f"Special value list should contain either integers, or (special value, biasing %) tuples.")
            else:
                raise ValueError(error_prefix + f"Invalid format for parameter '{param}'. Dict value should be "
                                 "an int, a (int, float) tuple, a list of integers, or a list of (int, float) tuples.")

            # Are user-specified special values valid?
            if not all(hyperparameter.lower <= v <= hyperparameter.upper for v, _ in tuple_list):
                raise ValueError(error_prefix + f"One (or more) special values are outside of parameter '{param}' value domain.")
            # Are user-provided special values unique?
            if len(set(v for v, _ in tuple_list)) != len(tuple_list):
                raise ValueError(error_prefix + f"One (or more) special values are defined more than once for parameter '{param}'.")
            # Are biasing percentages valid?
            if not all(0 < perc < 1 for _, perc in tuple_list):
                raise ValueError(error_prefix + f"One (or more) biasing percentages for parameter '{param}' are invalid: "
                                 "i.e., fall outside (0, 1) range.")

            total_percentage = sum(perc for _, perc in tuple_list)
            if total_percentage >= 1.:
                raise ValueError(error_prefix + f"Total special values percentage for parameter '{param}' surpass 100%.")
            # ... and reasonable?
            if total_percentage >= 0.5:
                warn(f"Total special values percentage for parameter '{param}' exceeds 50%.", UserWarning)

            sanitized_dict[param] = tuple_list

        self._special_param_values_dict = sanitized_dict

    def _try_generate_approx_inverse_mapping(self) -> None:
        """Tries to generate an approximate reverse mapping: i.e., from high-dimensional space to the low-dimensional one.
        Reverse mapping is generated using the pseudo-inverse matrix, of original HeSBO projection matrix.
        This mapping can be potentially used to register configurations that were *not* previously suggested by the optimizer.

        NOTE: This method is experimental, and there is currently no guarantee that it works as expected.

        Raises
        ------
            RuntimeError: if reverse mapping computation fails.
        """
        from scipy.linalg import pinv, LinAlgError  # pylint: disable=import-outside-toplevel

        warn("Trying to register a configuration that was not previously suggested by the optimizer. " +
             "This inverse configuration transformation is typically not supported. " +
             "However, we will try to register this configuration using an *experimental* method.", UserWarning)

        orig_space_num_dims = len(self.orig_parameter_space.get_hyperparameters())
        target_space_num_dims = len(self.target_parameter_space.get_hyperparameters())

        # Construct dense projection matrix from sparse repr
        proj_matrix = np.zeros(shape=(orig_space_num_dims, target_space_num_dims))
        for row, col in enumerate(self._h_matrix):
            proj_matrix[row][col] = self._sigma_vector[row]

        # Compute pseudo-inverse matrix
        try:
            self._pinv_matrix = pinv(proj_matrix)
        except LinAlgError as err:
            raise RuntimeError(f"Unable to generate reverse mapping using pseudo-inverse matrix: {repr(err)}") from err
        assert self._pinv_matrix.shape == (target_space_num_dims, orig_space_num_dims)
