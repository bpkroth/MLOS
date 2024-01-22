#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Small wrapper functions for dabl plotting functions via mlos_bench data.
"""
import warnings

import dabl

from mlos_bench.storage.base_experiment_data import ExperimentData


def plot(exp_data: ExperimentData) -> None:
    """
    Plots the Experiment results data using dabl.

    Parameters
    ----------
    exp_data : ExperimentData
        The ExperimentData (e.g., obtained from the storage layer) to plot.
    """
    for objective in exp_data.objectives:
        objective_column = ExperimentData.RESULT_COLUMN_PREFIX + objective
        results_df = exp_data.results_df
        assert objective_column in results_df.columns
        dabl.plot(X=results_df, target_col=objective_column)


def ignore_plotter_warnings() -> None:
    """
    Add some filters to ignore warnings from the plotter.
    """
    warnings.filterwarnings("ignore", module="dabl", category=UserWarning, message="Could not infer format")
    warnings.filterwarnings("ignore", module="dabl", category=UserWarning, message="(Dropped|Discarding) .* outliers")
    warnings.filterwarnings("ignore", module="dabl", category=UserWarning, message="Not plotting highly correlated")
    warnings.filterwarnings("ignore", module="dabl", category=UserWarning,
                            message="Missing values in target_col have been removed for regression")
    from sklearn.exceptions import UndefinedMetricWarning   # pylint: disable=import-outside-toplevel
    warnings.filterwarnings("ignore", module="sklearn", category=UndefinedMetricWarning, message="Recall is ill-defined")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="dabl",    # but actually comes from pandas
                            message="is_categorical_dtype is deprecated and will be removed in a future version.")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn",
                            message="is_sparse is deprecated and will be removed in a future version.")
