#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Base interface for saving and restoring the benchmark data.

See Also
--------
mlos_bench.storage.base_storage.Storage.experiments :
    Retrieves a dictionary of the Experiments' data.
mlos_bench.storage.base_experiment_data.ExperimentData.results_df :
    Retrieves a pandas DataFrame of the Experiment's trials' results data.
mlos_bench.storage.base_experiment_data.ExperimentData.trials :
    Retrieves a dictionary of the Experiment's trials' data.
mlos_bench.storage.base_experiment_data.ExperimentData.tunable_configs :
    Retrieves a dictionary of the Experiment's sampled configs data.
mlos_bench.storage.base_experiment_data.ExperimentData.tunable_config_trial_groups :
    Retrieves a dictionary of the Experiment's trials' data, grouped by shared
    tunable config.
mlos_bench.storage.base_trial_data.TrialData :
    Base interface for accessing the stored benchmark trial data.
"""

import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from types import TracebackType
from typing import Any, ContextManager, Literal

from mlos_bench.config.schemas import ConfigSchema
from mlos_bench.dict_templater import DictTemplater
from mlos_bench.environments.status import Status
from mlos_bench.services.base_service import Service
from mlos_bench.storage.base_experiment_data import ExperimentData
from mlos_bench.tunables.tunable_groups import TunableGroups
from mlos_bench.util import get_git_info

_LOG = logging.getLogger(__name__)


class Storage(metaclass=ABCMeta):
    """An abstract interface between the benchmarking framework and storage systems
    (e.g., SQLite or MLFLow).
    """

    def __init__(
        self,
        config: dict[str, Any],
        global_config: dict | None = None,
        service: Service | None = None,
    ):
        """
        Create a new storage object.

        Parameters
        ----------
        config : dict
            Free-format key/value pairs of configuration parameters.
        """
        _LOG.debug("Storage config: %s", config)
        self._validate_json_config(config)
        self._service = service
        self._config = config.copy()
        self._global_config = global_config or {}

    @abstractmethod
    def update_schema(self) -> None:
        """Update the schema of the storage backend if needed."""

    def _validate_json_config(self, config: dict) -> None:
        """Reconstructs a basic json config that this class might have been instantiated
        from in order to validate configs provided outside the file loading
        mechanism.
        """
        json_config: dict = {
            "class": self.__class__.__module__ + "." + self.__class__.__name__,
        }
        if config:
            json_config["config"] = config
        ConfigSchema.STORAGE.validate(json_config)

    @property
    @abstractmethod
    def experiments(self) -> dict[str, ExperimentData]:
        """
        Retrieve the experiments' data from the storage.

        Returns
        -------
        experiments : dict[str, ExperimentData]
            A dictionary of the experiments' data, keyed by experiment id.
        """

    @abstractmethod
    def experiment(  # pylint: disable=too-many-arguments
        self,
        *,
        experiment_id: str,
        trial_id: int,
        root_env_config: str,
        description: str,
        tunables: TunableGroups,
        opt_targets: dict[str, Literal["min", "max"]],
    ) -> "Storage.Experiment":
        """
        Create a new experiment in the storage.

        We need the `opt_target` parameter here to know what metric to retrieve
        when we load the data from previous trials. Later we will replace it with
        full metadata about the optimization direction, multiple objectives, etc.

        Parameters
        ----------
        experiment_id : str
            Unique identifier of the experiment.
        trial_id : int
            Starting number of the trial.
        root_env_config : str
            A path to the root JSON configuration file of the benchmarking environment.
        description : str
            Human-readable description of the experiment.
        tunables : TunableGroups
        opt_targets : dict[str, Literal["min", "max"]]
            Names of metrics we're optimizing for and the optimization direction {min, max}.

        Returns
        -------
        experiment : Storage.Experiment
            An object that allows to update the storage with
            the results of the experiment and related data.
        """

    class Experiment(ContextManager, metaclass=ABCMeta):
        # pylint: disable=too-many-instance-attributes
        """
        Base interface for storing the results of the experiment.

        This class is instantiated in the `Storage.experiment()` method.
        """

        def __init__(  # pylint: disable=too-many-arguments
            self,
            *,
            tunables: TunableGroups,
            experiment_id: str,
            trial_id: int,
            root_env_config: str,
            description: str,
            opt_targets: dict[str, Literal["min", "max"]],
        ):
            self._tunables = tunables.copy()
            self._trial_id = trial_id
            self._experiment_id = experiment_id
            (self._git_repo, self._git_commit, self._root_env_config) = get_git_info(
                root_env_config
            )
            self._description = description
            self._opt_targets = opt_targets
            self._in_context = False

        def __enter__(self) -> "Storage.Experiment":
            """
            Enter the context of the experiment.

            Override the `_setup` method to add custom context initialization.
            """
            _LOG.debug("Starting experiment: %s", self)
            assert not self._in_context
            self._setup()
            self._in_context = True
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> Literal[False]:
            """
            End the context of the experiment.

            Override the `_teardown` method to add custom context teardown logic.
            """
            is_ok = exc_val is None
            if is_ok:
                _LOG.debug("Finishing experiment: %s", self)
            else:
                assert exc_type and exc_val
                _LOG.warning(
                    "Finishing experiment: %s",
                    self,
                    exc_info=(exc_type, exc_val, exc_tb),
                )
            assert self._in_context
            self._teardown(is_ok)
            self._in_context = False
            return False  # Do not suppress exceptions

        def __repr__(self) -> str:
            return self._experiment_id

        def _setup(self) -> None:
            """
            Create a record of the new experiment or find an existing one in the
            storage.

            This method is called by `Storage.Experiment.__enter__()`.
            """

        def _teardown(self, is_ok: bool) -> None:
            """
            Finalize the experiment in the storage.

            This method is called by `Storage.Experiment.__exit__()`.

            Parameters
            ----------
            is_ok : bool
                True if there were no exceptions during the experiment, False otherwise.
            """

        @property
        def experiment_id(self) -> str:
            """Get the Experiment's ID."""
            return self._experiment_id

        @property
        def trial_id(self) -> int:
            """Get the current Trial ID."""
            return self._trial_id

        @property
        def description(self) -> str:
            """Get the Experiment's description."""
            return self._description

        @property
        def root_env_config(self) -> str:
            """Get the Experiment's root Environment config file path."""
            return self._root_env_config

        @property
        def tunables(self) -> TunableGroups:
            """Get the Experiment's tunables."""
            return self._tunables

        @property
        def opt_targets(self) -> dict[str, Literal["min", "max"]]:
            """Get the Experiment's optimization targets and directions."""
            return self._opt_targets

        @abstractmethod
        def merge(self, experiment_ids: list[str]) -> None:
            """
            Merge in the results of other (compatible) experiments trials. Used to help
            warm up the optimizer for this experiment.

            Parameters
            ----------
            experiment_ids : list[str]
                List of IDs of the experiments to merge in.
            """

        @abstractmethod
        def load_tunable_config(self, config_id: int) -> dict[str, Any]:
            """Load tunable values for a given config ID."""

        @abstractmethod
        def load_telemetry(self, trial_id: int) -> list[tuple[datetime, str, Any]]:
            """
            Retrieve the telemetry data for a given trial.

            Parameters
            ----------
            trial_id : int
                Trial ID.

            Returns
            -------
            metrics : list[tuple[datetime.datetime, str, Any]]
                Telemetry data.
            """

        @abstractmethod
        def load(
            self,
            last_trial_id: int = -1,
        ) -> tuple[list[int], list[dict], list[dict[str, Any] | None], list[Status]]:
            """
            Load (tunable values, benchmark scores, status) to warm-up the optimizer.

            If `last_trial_id` is present, load only the data from the (completed) trials
            that were scheduled *after* the given trial ID. Otherwise, return data from ALL
            merged-in experiments and attempt to impute the missing tunable values.

            Parameters
            ----------
            last_trial_id : int
                (Optional) Trial ID to start from.

            Returns
            -------
            (trial_ids, configs, scores, status) : ([int], [dict], [dict] | None, [Status])
                Trial ids, Tunable values, benchmark scores, and status of the trials.
            """

        @abstractmethod
        def pending_trials(
            self,
            timestamp: datetime,
            *,
            running: bool,
        ) -> Iterator["Storage.Trial"]:
            """
            Return an iterator over the pending trials that are scheduled to run on or
            before the specified timestamp.

            Parameters
            ----------
            timestamp : datetime.datetime
                The time in UTC to check for scheduled trials.
            running : bool
                If True, include the trials that are already running.
                Otherwise, return only the scheduled trials.

            Returns
            -------
            trials : Iterator[Storage.Trial]
                An iterator over the scheduled (and maybe running) trials.
            """

        def new_trial(
            self,
            tunables: TunableGroups,
            ts_start: datetime | None = None,
            config: dict[str, Any] | None = None,
        ) -> "Storage.Trial":
            """
            Create a new experiment run in the storage.

            Parameters
            ----------
            tunables : TunableGroups
                Tunable parameters to use for the trial.
            ts_start : datetime.datetime | None
                Timestamp of the trial start (can be in the future).
            config : dict
                Key/value pairs of additional non-tunable parameters of the trial.

            Returns
            -------
            trial : Storage.Trial
                An object that allows to update the storage with
                the results of the experiment trial run.
            """
            # Check that `config` is json serializable (e.g., no callables)
            if config:
                try:
                    # Relies on the fact that DictTemplater only accepts primitive
                    # types in it's nested dict structure walk.
                    _config = DictTemplater(config).expand_vars()
                    assert isinstance(_config, dict)
                except ValueError as e:
                    _LOG.error("Non-serializable config: %s", config, exc_info=e)
                    raise e
            return self._new_trial(tunables, ts_start, config)

        @abstractmethod
        def _new_trial(
            self,
            tunables: TunableGroups,
            ts_start: datetime | None = None,
            config: dict[str, Any] | None = None,
        ) -> "Storage.Trial":
            """
            Create a new experiment run in the storage.

            Parameters
            ----------
            tunables : TunableGroups
                Tunable parameters to use for the trial.
            ts_start : datetime.datetime | None
                Timestamp of the trial start (can be in the future).
            config : dict
                Key/value pairs of additional non-tunable parameters of the trial.

            Returns
            -------
            trial : Storage.Trial
                An object that allows to update the storage with
                the results of the experiment trial run.
            """

    class Trial(metaclass=ABCMeta):
        # pylint: disable=too-many-instance-attributes
        """
        Base interface for storing the results of a single run of the experiment.

        This class is instantiated in the `Storage.Experiment.trial()` method.
        """

        def __init__(  # pylint: disable=too-many-arguments
            self,
            *,
            tunables: TunableGroups,
            experiment_id: str,
            trial_id: int,
            tunable_config_id: int,
            opt_targets: dict[str, Literal["min", "max"]],
            config: dict[str, Any] | None = None,
        ):
            self._tunables = tunables
            self._experiment_id = experiment_id
            self._trial_id = trial_id
            self._tunable_config_id = tunable_config_id
            self._opt_targets = opt_targets
            self._config = config or {}
            self._status = Status.UNKNOWN

        def __repr__(self) -> str:
            return f"{self._experiment_id}:{self._trial_id}:{self._tunable_config_id}"

        @property
        def trial_id(self) -> int:
            """ID of the current trial."""
            return self._trial_id

        @property
        def tunable_config_id(self) -> int:
            """ID of the current trial (tunable) configuration."""
            return self._tunable_config_id

        @property
        def opt_targets(self) -> dict[str, Literal["min", "max"]]:
            """Get the Trial's optimization targets and directions."""
            return self._opt_targets

        @property
        def tunables(self) -> TunableGroups:
            """
            Tunable parameters of the current trial.

            (e.g., application Environment's "config")
            """
            return self._tunables

        def config(self, global_config: dict[str, Any] | None = None) -> dict[str, Any]:
            """
            Produce a copy of the global configuration updated with the parameters of
            the current trial.

            Note: this is not the target Environment's "config" (i.e., tunable
            params), but rather the internal "config" which consists of a
            combination of somewhat more static variables defined in the json config
            files.
            """
            config = self._config.copy()
            config.update(global_config or {})
            config["experiment_id"] = self._experiment_id
            config["trial_id"] = self._trial_id
            return config

        @property
        def status(self) -> Status:
            """Get the status of the current trial."""
            return self._status

        @abstractmethod
        def update(
            self,
            status: Status,
            timestamp: datetime,
            metrics: dict[str, Any] | None = None,
        ) -> dict[str, Any] | None:
            """
            Update the storage with the results of the experiment.

            Parameters
            ----------
            status : Status
                Status of the experiment run.
            timestamp: datetime.datetime
                Timestamp of the status and metrics.
            metrics : Optional[dict[str, Any]]
                One or several metrics of the experiment run.
                Must contain the (float) optimization target if the status is SUCCEEDED.

            Returns
            -------
            metrics : Optional[dict[str, Any]]
                Same as `metrics`, but always in the dict format.
            """
            _LOG.info("Store trial: %s :: %s %s", self, status, metrics)
            if status.is_succeeded():
                assert metrics is not None
                opt_targets = set(self._opt_targets.keys())
                if not opt_targets.issubset(metrics.keys()):
                    _LOG.warning(
                        "Trial %s :: opt.targets missing: %s",
                        self,
                        opt_targets.difference(metrics.keys()),
                    )
                    # raise ValueError()
            self._status = status
            return metrics

        @abstractmethod
        def update_telemetry(
            self,
            status: Status,
            timestamp: datetime,
            metrics: list[tuple[datetime, str, Any]],
        ) -> None:
            """
            Save the experiment's telemetry data and intermediate status.

            Parameters
            ----------
            status : Status
                Current status of the trial.
            timestamp: datetime.datetime
                Timestamp of the status (but not the metrics).
            metrics : list[tuple[datetime.datetime, str, Any]]
                Telemetry data.
            """
            _LOG.info("Store telemetry: %s :: %s %d records", self, status, len(metrics))
