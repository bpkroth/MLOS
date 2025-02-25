#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
r"""
mlos_bench is a framework to help automate benchmarking and OS/application parameter
autotuning and the data management of the results.

.. contents:: Table of Contents
   :depth: 3

Overview
++++++++

``mlos_bench`` can be installed from `pypi <https://pypi.org/project/mlos-bench>`_
via ``pip install mlos-bench`` and executed using the ``mlos_bench`` `command
<../../mlos_bench.run.usage.html>`_ using a collection of `json` `configs
<https://github.com/microsoft/MLOS/tree/main/mlos_bench/mlos_bench/config/>`_
which are managed and documented in the :py:mod:`~mlos_bench.config` module and
elsewhere in this package documentation.

It is intended to be used with :py:mod:`mlos_core` via
:py:class:`~mlos_bench.optimizers.mlos_core_optimizer.MlosCoreOptimizer` to help
navigate complex parameter spaces more efficiently, though other
:py:mod:`~mlos_bench.optimizers` are also available to help customize the search
process easily by simply swapping out the
:py:class:`~mlos_bench.optimizers.base_optimizer.Optimizer` class in the associated
json configs. For instance,
:py:class:`~mlos_bench.optimizers.grid_search_optimizer.GridSearchOptimizer` can be
used to perform a grid search over the parameter space instead, or no tuning at all
and ``mlos_bench`` can be used as only a repeatable benchmarking tool.

Goals
^^^^^

The overall goal of the MLOS project is to enable *reproducible* and *trackable*
benchmarking and *efficient* autotuning for systems software.

In this, automation of the benchmarking process is a key component that
``mlos_bench`` seeks to enable.

Interaction
^^^^^^^^^^^

Users are expected to provide JSON :py:mod:`mlos_bench.config` s that instruct the
framework how to automate their benchmark, autotuning, or other Experiment.

This may involve several steps such as

1. deploying a VM
2. installing some software
3. loading a dataset
4. running a benchmark
5. collecting and storing the results
6. repeating for statistical and noise measures (we call each iteration a ``Trial``)
7. (optionally) repeating with a different configuration (e.g., for autotuning purposes)
8. analyzing the results

Since many of these phases are common across different benchmarks, the framework is
intended to be modular and composable to allow reuse and faster development of new
benchmarking environments or autotuning experiments.

Where possible, the framework will provide common configs for reuse (e.g., deploying
a VM on Azure, or run `benchbase <https://github.com/cmu-db/benchbase>`_ against
a database system) to allow users to focus on the specifics of their
experiments.

Where none are currently available, one can create them external to MLOS, however
users are also encouraged to `submit PRs or Issues
<https://github.com/microsoft/MLOS/CONTRIBUTING.md>`_ to add new classes or config
and script snippets for others to use as well.

For more details on the configuration files, please see the documentation in the
:py:mod:`mlos_bench.config` module.

Classes Overview
++++++++++++++++

The other core classes in this package are:

- :py:mod:`~mlos_bench.environments` which provide abstractions for representing an
  execution environment.

  These are generally the target of the optimization process and are used to
  evaluate the performance of a given configuration, though can also be used to
  simply run a single benchmark. They can be used, for instance, to provision a
  :py:mod:`VM <mlos_bench.environments.remote.vm_env>`, run benchmarks or execute
  any other arbitrary code on a :py:mod:`remote machine
  <mlos_bench.environments.remote.remote_env>`, and many other things.

- Environments are often associated with :py:mod:`~mlos_bench.tunables` which
  provide a language for specifying the set of configuration parameters that can be
  optimized or searched over with the :py:mod:`~mlos_bench.optimizers`.

- :py:mod:`~mlos_bench.services` provide the necessary abstractions to run interact
  with the :py:mod:`~mlos_bench.environments` in different settings.

  For instance, the
  :py:class:`~mlos_bench.services.remote.azure.azure_vm_services.AzureVMService` can
  be used to run commands on Azure VMs for a remote
  :py:mod:`~mlos_bench.environments.remote.vm_env.VMEnv`.

  Alternatively, one could swap out that service for
  :py:class:`~mlos_bench.services.remote.ssh.ssh_host_service.SshHostService` in
  order to target a different VM without having to change the
  :py:class:`~mlos_bench.environments.base_environment.Environment` configuration at
  all since they both implement the same
  :py:class:`~mlos_bench.services.types.remote_exec_type.SupportsRemoteExec`
  :py:mod:`Services type<mlos_bench.services.types>` interfaces.

  This is particularly useful when running the same benchmark on different
  ecosystems and makes the configs more modular and composable.

- :py:mod:`~mlos_bench.storage` which provides abstractions for storing and
  retrieving data from the experiments.

  For instance, nearly any :py:mod:`SQL <mlos_bench.storage.sql>` backend that
  `sqlalchemy <https://www.sqlalchemy.org>`_ supports can be used.

The data management and automation portions of experiment data is a key component of
MLOS as it provides a unified way to manage experiment data across different
Environments, enabling more reusable visualization and analysis by mapping benchmark
metrics into common semantic types (e.g., via `OpenTelemetry
<https://opentelemetry.io>`_).

Without this most experiments are effectively siloed and require custom, and more
critically, non-reusable scripts to setup and later parse results and are hence
harder to scale to many users.

With these features as a part of the MLOS ecosystem, benchmarking can become a
*service* that any developer, admin, researcher, etc. can use and adapt.

See below for more information on the classes in this package.

Notes
-----
Note that while the docstrings in this package are generated from the source
code and hence sometimes more focused on the implementation details, we do try
to provide some (testable) examples here, but most user interactions with the
package will be through the `json configs
<https://github.com/microsoft/MLOS/tree/main/mlos_bench/mlos_bench/config/>`_.
Even so it may be useful to look at the documentation here (perhaps especially
starting with :py:mod:`mlos_bench.config`) and, if necessary, the `source code
<https://github.com/microsoft/MLOS>`_ to understand how those are interpreted.

Examples
--------
Here is an example that shows how to run a simple benchmark using the command line.

The entry point for these configs can be found `here
<https://github.com/microsoft/MLOS/blob/main/mlos_bench/mlos_bench/tests/config/cli/test-cli-local-env-bench.jsonc>`_.

>>> from subprocess import run
>>> # Note: we show the command wrapped in python here for testing purposes.
>>> # Alternatively replace test-cli-local-env-bench.jsonc with
>>> # test-cli-local-env-opt.jsonc for one that does an optimization loop.
>>> cmd = "mlos_bench \
... --config mlos_bench/mlos_bench/tests/config/cli/test-cli-local-env-bench.jsonc \
... --globals experiment_test_local.jsonc \
... --tunable_values tunable-values/tunable-values-local.jsonc"
>>> print(f"Here's the shell command you'd actually run:\n# {cmd}")
Here's the shell command you'd actually run:
# mlos_bench --config mlos_bench/mlos_bench/tests/config/cli/test-cli-local-env-bench.jsonc --globals experiment_test_local.jsonc --tunable_values tunable-values/tunable-values-local.jsonc
>>> # Now we run the command and check the output.
>>> result = run(cmd, shell=True, capture_output=True, text=True, check=True)
>>> assert result.returncode == 0
>>> lines = result.stderr.splitlines()
>>> first_line = lines[0]
>>> last_line = lines[-1]
>>> expected = "INFO Launch: mlos_bench"
>>> assert first_line.endswith(expected)
>>> expected = "INFO Final score: {'score': 123.4, 'total_time': 123.4, 'throughput': 1234567.0}"
>>> assert last_line.endswith(expected)

Notes
-----
- `mlos_bench/README.md <https://github.com/microsoft/MLOS/tree/main/mlos_bench/>`_
  for additional documentation and examples in the source tree.

- `mlos_bench/DEVNOTES.md <https://github.com/microsoft/MLOS/tree/main/mlos_bench/mlos_bench/DEVNOTES.md>`_
  for additional developer notes in the source tree.

- There is also a working example of using ``mlos_bench`` in a *separate config
  repo* (the more expected case for most users) in the `sqlite-autotuning
  <https://github.com/Microsoft-CISL/sqlite-autotuning>`_ repo.
"""  # pylint: disable=line-too-long # noqa: E501
from mlos_bench.version import VERSION

__version__ = VERSION
