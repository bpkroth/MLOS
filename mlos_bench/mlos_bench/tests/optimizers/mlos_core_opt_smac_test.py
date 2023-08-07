#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Unit tests for mock mlos_bench optimizer.
"""

import os

import pytest

from mlos_bench.optimizers.mlos_core_optimizer import MlosCoreOptimizer
from mlos_bench.tunables.tunable_groups import TunableGroups

from mlos_core.optimizers.bayesian_optimizers.smac_optimizer import SmacOptimizer

# pylint: disable=redefined-outer-name


def test_init_mlos_core_smac_opt_bad_trial_count(tunable_groups: TunableGroups) -> None:
    """
    Test invalid max_trials initialization of mlos_core SMAC optimizer.
    """
    test_opt_config = {
        'optimizer_type': 'SMAC',
        'max_trials': 10,
        'max_iterations': 11,
    }
    with pytest.raises(AssertionError):
        opt = MlosCoreOptimizer(tunable_groups, test_opt_config)
        assert opt is None


def test_init_mlos_core_smac_opt_max_trials(tunable_groups: TunableGroups) -> None:
    """
    Test max_trials initialization of mlos_core SMAC optimizer.
    """
    test_opt_config = {
        'optimizer_type': 'SMAC',
        'max_iterations': 123,
    }
    opt = MlosCoreOptimizer(tunable_groups, test_opt_config)
    # pylint: disable=protected-access
    assert isinstance(opt._opt, SmacOptimizer)
    assert opt._opt.base_optimizer.scenario.n_trials == test_opt_config['max_iterations']


def test_init_mlos_core_smac_absolute_output_directory(tunable_groups: TunableGroups) -> None:
    """
    Test absolute path output directory initialization of mlos_core SMAC optimizer.
    """
    test_opt_config = {
        'optimizer_type': 'SMAC',
        'output_directory': '/tmp/test_output_dir',
    }
    opt = MlosCoreOptimizer(tunable_groups, test_opt_config)
    assert isinstance(opt, MlosCoreOptimizer)
    # pylint: disable=protected-access
    assert isinstance(opt._opt, SmacOptimizer)
    # Final portions of the path are generated by SMAC when run_name is not specified.
    assert str(opt._opt.base_optimizer.scenario.output_directory).startswith(test_opt_config['output_directory'])


def test_init_mlos_core_smac_relative_output_directory(tunable_groups: TunableGroups) -> None:
    """
    Test relative path output directory initialization of mlos_core SMAC optimizer.
    """
    test_opt_config = {
        'optimizer_type': 'SMAC',
        'output_directory': '/tmp/test_output_dir',
    }
    opt = MlosCoreOptimizer(tunable_groups, test_opt_config)
    assert isinstance(opt, MlosCoreOptimizer)
    # pylint: disable=protected-access
    assert isinstance(opt._opt, SmacOptimizer)
    assert str(opt._opt.base_optimizer.scenario.output_directory).startswith(
        os.path.join(os.getcwd(), test_opt_config['output_directory']))


def test_init_mlos_core_smac_relative_output_directory_with_run_name(tunable_groups: TunableGroups) -> None:
    """
    Test relative path output directory initialization of mlos_core SMAC optimizer.
    """
    test_opt_config = {
        'optimizer_type': 'SMAC',
        'output_directory': '/tmp/test_output_dir',
        'run_name': 'test_run',
    }
    opt = MlosCoreOptimizer(tunable_groups, test_opt_config)
    assert isinstance(opt, MlosCoreOptimizer)
    # pylint: disable=protected-access
    assert isinstance(opt._opt, SmacOptimizer)
    assert str(opt._opt.base_optimizer.scenario.output_directory).startswith(
        os.path.join(os.getcwd(), test_opt_config['output_directory'], test_opt_config['run_name']))


def test_init_mlos_core_smac_relative_output_directory_with_experiment_id(tunable_groups: TunableGroups) -> None:
    """
    Test relative path output directory initialization of mlos_core SMAC optimizer.
    """
    test_opt_config = {
        'optimizer_type': 'SMAC',
        'output_directory': '/tmp/test_output_dir',
    }
    global_config = {
        'experiment_id': 'experiment_id',
    }
    opt = MlosCoreOptimizer(tunable_groups, test_opt_config, global_config)
    assert isinstance(opt, MlosCoreOptimizer)
    # pylint: disable=protected-access
    assert isinstance(opt._opt, SmacOptimizer)
    assert str(opt._opt.base_optimizer.scenario.output_directory).startswith(
        os.path.join(os.getcwd(), test_opt_config['output_directory'], global_config['experiment_id']))


def test_init_mlos_core_smac_temp_output_directory(tunable_groups: TunableGroups) -> None:
    """
    Test relative path output directory initialization of mlos_core SMAC optimizer.
    """
    test_opt_config = {
        'optimizer_type': 'SMAC',
        'output_directory': None,
    }
    opt = MlosCoreOptimizer(tunable_groups, test_opt_config)
    assert isinstance(opt, MlosCoreOptimizer)
    # pylint: disable=protected-access
    assert isinstance(opt._opt, SmacOptimizer)
    assert opt._opt.base_optimizer.scenario.output_directory is not None
