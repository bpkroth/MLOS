#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Protocol interface for helper functions to lookup and load configs.
"""

from typing import Dict, List, Iterable, Optional, Union, Protocol, runtime_checkable, TYPE_CHECKING

from mlos_bench.config.schemas import ConfigSchema
from mlos_bench.tunables.tunable import TunableValue


# Avoid's circular import issues.
if TYPE_CHECKING:
    from mlos_bench.tunables.tunable_groups import TunableGroups
    from mlos_bench.services.base_service import Service
    from mlos_bench.environments.base_environment import Environment


@runtime_checkable
class SupportsConfigLoading(Protocol):
    """
    Protocol interface for helper functions to lookup and load configs.
    """

    def resolve_path(self,
                     file_path: str,
                     *,
                     source_file_path: Optional[str] = None,
                     extra_paths: Optional[Iterable[str]] = None) -> str:
        """
        Prepend the suitable `_config_path` to `path` if the latter is not absolute.
        If `_config_path` is `None` or `path` is absolute, return `path` as is.

        Parameters
        ----------
        file_path : str
            Path to the input config file.
        source_file_path : str
            Path to the source config file causing this file_path resolution.
            Used to resolve relative paths.
        extra_paths : Iterable[str]
            Additional directories to prepend to the list of search paths.

        Returns
        -------
        path : str
            An actual path to the config or script.
        """

    def load_config(self,
                    json_file_name: str,
                    schema_type: Optional[ConfigSchema],
                    source_file_path: Optional[str] = None) -> Union[dict, List[dict]]:
        """
        Load JSON config file. Search for a file relative to `_config_path`
        if the input path is not absolute.
        This method is exported to be used as a service.

        Parameters
        ----------
        json_file_name : str
            Path to the input config file.
        schema_type : Optional[ConfigSchema]
            The schema type to validate the config against.
        source_file_path : str
            Path to the source config file causing this load_config.
            Used to resolve relative paths.

        Returns
        -------
        config : Union[dict, List[dict]]
            Free-format dictionary that contains the configuration.
        """

    def build_environment(self,
                          *,
                          config: dict,
                          config_file_path: Optional[str] = None,
                          tunables: "TunableGroups",
                          global_config: Optional[dict] = None,
                          parent_args: Optional[Dict[str, TunableValue]] = None,
                          service: Optional["Service"] = None) -> "Environment":
        """
        Factory method for a new environment with a given config.

        Parameters
        ----------
        config : dict
            A dictionary with three mandatory fields:
                "name": Human-readable string describing the environment;
                "class": FQN of a Python class to instantiate;
                "config": Free-format dictionary to pass to the constructor.
        config_file_path : str
            Path to the config file used to create the config.
            Useful for debugging and to resolve relative paths.
        tunables : TunableGroups
            A (possibly empty) collection of groups of tunable parameters for
            all environments.
        global_config : Optional[dict]
            Global parameters to add to the environment config.
        parent_args : Optional[Dict[str, TunableValue]]
            An optional reference of the parent CompositeEnv's const_args used to
            expand dynamic config parameters from.
        service: Optional[Service]
            An optional service object (e.g., providing methods to
            deploy or reboot a VM, etc.).

        Returns
        -------
        env : Environment
            An instance of the `Environment` class initialized with `config`.
        """

    def load_environment_list(
            self,
            *,
            json_file_name: str,
            source_file_path: Optional[str] = None,
            tunables: "TunableGroups",
            global_config: Optional[dict] = None,
            parent_args: Optional[Dict[str, TunableValue]] = None,
            service: Optional["Service"] = None) -> List["Environment"]:
        """
        Load and build a list of environments from the config file.

        Parameters
        ----------
        json_file_name : str
            The environment JSON configuration file.
            Can contain either one environment or a list of environments.
        source_file_path : str
            Path to the source config file causing this load_environment_list.
            Used to resolve relative paths.
        tunables : TunableGroups
            A (possibly empty) collection of tunables to add to the environment.
        global_config : Optional[dict]
            Global parameters to add to the environment config.
        parent_args : Optional[Dict[str, TunableValue]]
            An optional reference of the parent CompositeEnv's const_args used to
            expand dynamic config parameters from.
        service : Optional[Service]
            An optional reference of the parent service to mix in.

        Returns
        -------
        env : List[Environment]
            A list of new benchmarking environments.
        """
