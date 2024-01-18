#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Unit tests for mlos_viz.dabl.plot.
"""

import warnings

from mlos_bench.storage.base_experiment_data import ExperimentData

from mlos_viz import dabl


def test_dabl_plot(exp_data: ExperimentData) -> None:
    """Tests plotting via dabl."""
    # For now, just ensure that no errors are thrown.
    # TODO: Check that a plot was actually produced matching our specifications.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dabl.ignore_plotter_warnings()
        dabl.plot(exp_data)