#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
mlos_viz is a framework to help visualizing, explain, and gain insights from results
from the mlos_bench framework for benchmarking and optimization automation.
"""

from typing import Any, Callable, Dict

import warnings

from importlib.metadata import version
from matplotlib import pyplot as plt
import seaborn as sns

from mlos_bench.storage.base_experiment_data import ExperimentData


_SEABORN_VERS = version('seaborn')


def get_kwarg_defaults(target: Callable, **kwargs: Any) -> Dict[str, Any]:
    """
    Assembles a smaller kwargs dict for the specified target function.

    Note: this only works with non-positional kwargs (e.g., those after a * arg).
    """
    target_kwargs = {}
    for kword in target.__kwdefaults__:     # or {} # intentionally omitted for now
        if kword in kwargs:
            target_kwargs[kword] = kwargs[kword]
    return target_kwargs


def ignore_plotter_warnings() -> None:
    """
    Suppress some annoying warnings from third-party data visualization packages by
    adding them to the warnings filter.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    if _SEABORN_VERS <= '0.13.1':
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="seaborn",    # but actually comes from pandas
                                message="is_categorical_dtype is deprecated and will be removed in a future version.")


def plot_optimizer_trends(exp_data: ExperimentData) -> None:
    """
    Plots the optimizer trends for the Experiment.

    Intended to be used from a Jupyter notebook.

    Parameters
    ----------
    exp_data: ExperimentData
        The experiment data to plot.
    """
    # TODO: Provide a utility function in `mlos_bench` to process the results and
    # return a specialized dataframe first?
    # e.g., incumbent results up to N-th iteration?
    # Could be useful in conducting numerical analyses of optimizer policies as well.
    for (objective, direction) in exp_data.objectives.items():
        objective_column = ExperimentData.RESULT_COLUMN_PREFIX + objective
        groupby_column = "tunable_config_trial_group_id"
        results_df = exp_data.results_df
        # Needs to be a string for boxplot and lineplot to be on the same axis.
        results_df[groupby_column] = results_df[groupby_column].astype(str)

        # calculate the incumbent as a mean of each group to match the boxplot
        incumbent_performance_df = results_df.groupby(groupby_column)[objective_column].mean().reset_index()
        if direction == "min":
            incumbent_performance_df["incumbent_performance"] = incumbent_performance_df[objective_column].cummin()
        elif direction == "max":
            incumbent_performance_df["incumbent_performance"] = incumbent_performance_df[objective_column].cummax()
        else:
            raise NotImplementedError(f"Unhandled direction {direction} for target {objective}")

        plt.rcParams["figure.figsize"] = (10, 5)
        (_fig, ax) = plt.subplots()

        # Result of each set of trials for a config
        sns.boxplot(
            data=results_df,
            x=groupby_column,
            y=objective_column,
            ax=ax,
        )
        # Result of the best so far.
        ax = sns.lineplot(
            data=incumbent_performance_df,
            x=groupby_column,
            y="incumbent_performance",
            alpha=0.7,
            label="Incumbent",
            ax=ax,
        )

        plt.yscale('log')
        plt.ylabel(objective)

        plt.xlabel("Config Trial Group")
        plt.xticks(rotation=90)

        plt.title("Optimizer Trends for Experiment: " + exp_data.experiment_id)
        plt.grid()
        plt.show()  # type: ignore[no-untyped-call]


def plot_top_n_configs(exp_data: ExperimentData, with_scatter_plot: bool = False, **kwargs: Any) -> None:
    """
    Plots the top-N configs along with the default config for the given ExperimentData.

    Intended to be used from a Jupyter notebook.

    Parameters
    ----------
    exp_data: ExperimentData
        The experiment data to plot.
    with_scatter_plot : bool
        Whether to also add scatter plot to the output figure.
    kwargs : dict
        Remaining keyword arguments are passed along to the ExperimentData.top_n_configs.
    """
    top_n_config_args = get_kwarg_defaults(ExperimentData.top_n_configs, **kwargs)
    (top_n_config_results_df, opt_target, opt_direction) = exp_data.top_n_configs(**top_n_config_args)
    top_n = len(top_n_config_results_df["config_id"].unique()) - 1
    target_column = ExperimentData.RESULT_COLUMN_PREFIX + opt_target
    (_fig, ax) = plt.subplots()
    sns.boxplot(
        data=top_n_config_results_df,
        y=target_column,
    )
    if with_scatter_plot:
        sns.scatterplot(
            data=top_n_config_results_df,
            y=target_column,
            legend=None,
            ax=ax,
        )
    plt.grid()
    (xticks, xlabels) = plt.xticks()
    # default should be in the first position based on top_n_configs() return
    xlabels[0] = "default"          # type: ignore[call-overload]
    plt.xticks(xticks, xlabels)     # type: ignore[arg-type]
    plt.xlabel("Configuration")
    plt.xticks(rotation=90)
    plt.ylabel(opt_target)
    extra_title = "(higher is better)" if opt_direction == "max" else "(lower is better)"
    plt.title(f"Top {top_n} configs {opt_target} {extra_title}")
    plt.show()  # type: ignore[no-untyped-call]
