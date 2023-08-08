#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Toy optimization loop to test the optimizers on mock benchmark environment.
"""

from typing import List, Tuple

import pytest
from pytest import FixtureRequest
from pytest_lazyfixture import lazy_fixture, LazyFixture

from mlos_core.tests import get_all_concrete_subclasses
from mlos_core.optimizers import OptimizerType

from mlos_bench.environments.base_environment import Environment
from mlos_bench.environments.mock_env import MockEnv
from mlos_bench.tunables.tunable_groups import TunableGroups
from mlos_bench.optimizers.base_optimizer import Optimizer
from mlos_bench.optimizers.one_shot_optimizer import OneShotOptimizer
from mlos_bench.optimizers.mlos_core_optimizer import MlosCoreOptimizer


def _optimize(env: Environment, opt: Optimizer) -> Tuple[float, TunableGroups]:
    """
    Toy optimization loop.
    """
    assert opt.not_converged()

    while opt.not_converged():

        with env as env_context:

            tunables = opt.suggest()
            assert env_context.setup(tunables)

            (status, output) = env_context.run()
            assert status.is_succeeded()
            assert output is not None
            score = output['score']
            assert 60 <= score <= 120

            opt.register(tunables, status, score)

    (best_score, best_tunables) = opt.get_best_observation()
    assert isinstance(best_score, float) and isinstance(best_tunables, TunableGroups)
    return (best_score, best_tunables)


# (optimizer, expected_score, expected_tunable_values)
OPTIMIZER_TEST_CASES: List[Tuple[LazyFixture, float, dict]] = [
    (lazy_fixture("mock_opt"), 64.9, {
        "vmSize": "Standard_B2ms",
        "idle": "halt",
        "kernel_sched_migration_cost_ns": 117025,
        "kernel_sched_latency_ns": 149827706,
    }),
    (lazy_fixture("mock_opt_no_defaults"), 60.97, {
        "vmSize": "Standard_B2s",
        "idle": "halt",
        "kernel_sched_migration_cost_ns": 49122,
        "kernel_sched_latency_ns": 234760738,
    }),
    (lazy_fixture("flaml_opt"), 75.0, {
        "vmSize": "Standard_B4ms",
        "idle": "halt",
        "kernel_sched_migration_cost_ns": -1,
        "kernel_sched_latency_ns": 2000000,
    }),
    (lazy_fixture("smac_opt"), 67.6, {
        "vmSize": "Standard_B2ms",
        "idle": "mwait",
        "kernel_sched_migration_cost_ns": 37322,
        "kernel_sched_latency_ns": 40128951,
    }),
]


@pytest.mark.parametrize("opt, expected_score, expected_tunable_values", OPTIMIZER_TEST_CASES)
def test_optimization_loop(mock_env_no_noise: MockEnv,
                           opt: Optimizer,
                           expected_score: float,
                           expected_tunable_values: dict) -> None:
    """
    Toy optimization loop with mock environment and optimizer.
    """
    (score, tunables) = _optimize(mock_env_no_noise, opt)
    assert score == pytest.approx(expected_score, 0.01)
    assert tunables.get_param_values() == expected_tunable_values


def test_optimizer_coverage(request: FixtureRequest) -> None:
    """
    Make sure that all optimizers are tested.
    """
    mlos_core_optimizer_types: List[OptimizerType] = [
        opt_type for opt_type in OptimizerType
        if opt_type not in [OptimizerType.RANDOM]     # Ignore RANDOM
    ]
    mlos_bench_optimizer_types = [
        opt_cls for opt_cls in get_all_concrete_subclasses(Optimizer)
        if opt_cls not in [OneShotOptimizer]    # Ignore OneShotOptimizer
    ]
    for (lazy_opt, _, _) in OPTIMIZER_TEST_CASES:
        opt = request.getfixturevalue(lazy_opt.name)
        assert isinstance(opt, Optimizer)
        if opt.__class__ in mlos_bench_optimizer_types:
            mlos_bench_optimizer_types.remove(opt.__class__)
        if isinstance(opt, MlosCoreOptimizer):
            if opt.optimizer_type in mlos_core_optimizer_types:
                mlos_core_optimizer_types.remove(opt.optimizer_type)
    assert len(mlos_core_optimizer_types) == 0, f"Missing tests for {mlos_core_optimizer_types}"
    assert len(mlos_bench_optimizer_types) == 0, f"Missing tests for {mlos_bench_optimizer_types}"
