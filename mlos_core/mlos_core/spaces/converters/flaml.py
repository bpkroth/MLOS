#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""Contains space converters for
:py:class:`~mlos_core.optimizers.flaml_optimizer.FlamlOptimizer`
"""

from typing import TYPE_CHECKING, TypeAlias

import ConfigSpace
import flaml.tune
import flaml.tune.sample
import numpy as np
from flaml.tune.sample import Domain

if TYPE_CHECKING:
    from ConfigSpace.hyperparameters import Hyperparameter

FlamlDomain: TypeAlias = Domain
"""Flaml domain type alias."""

FlamlSpace: TypeAlias = dict[str, Domain]
"""Flaml space type alias - a `dict[str, FlamlDomain]`"""


def configspace_to_flaml_space(
    config_space: ConfigSpace.ConfigurationSpace,
) -> dict[str, FlamlDomain]:
    """
    Converts a ConfigSpace.ConfigurationSpace to dict.

    Parameters
    ----------
    config_space : ConfigSpace.ConfigurationSpace
        Input configuration space.

    Returns
    -------
    flaml_space : dict
        A dictionary of flaml.tune.sample.Domain objects keyed by parameter name.
    """
    flaml_numeric_type = {
        (ConfigSpace.UniformIntegerHyperparameter, False): flaml.tune.randint,
        (ConfigSpace.UniformIntegerHyperparameter, True): flaml.tune.lograndint,
        (ConfigSpace.UniformFloatHyperparameter, False): flaml.tune.uniform,
        (ConfigSpace.UniformFloatHyperparameter, True): flaml.tune.loguniform,
    }

    def _one_parameter_convert(parameter: "Hyperparameter") -> FlamlDomain:
        if isinstance(parameter, ConfigSpace.UniformFloatHyperparameter):
            # FIXME: upper isn't included in the range
            return flaml_numeric_type[(type(parameter), parameter.log)](
                parameter.lower,
                parameter.upper,
            )
        elif isinstance(parameter, ConfigSpace.UniformIntegerHyperparameter):
            return flaml_numeric_type[(type(parameter), parameter.log)](
                parameter.lower,
                parameter.upper + 1,
            )
        elif isinstance(parameter, ConfigSpace.CategoricalHyperparameter):
            if len(np.unique(parameter.probabilities)) > 1:
                raise ValueError(
                    "FLAML doesn't support categorical parameters with non-uniform probabilities."
                )
            return flaml.tune.choice(parameter.choices)  # TODO: set order?
        raise ValueError(f"Type of parameter {parameter} ({type(parameter)}) not supported.")

    return {param.name: _one_parameter_convert(param) for param in config_space.values()}
