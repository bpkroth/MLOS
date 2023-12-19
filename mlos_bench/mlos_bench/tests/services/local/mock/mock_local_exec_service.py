#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
A collection Service functions for mocking local exec.
"""

import logging
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
)

from mlos_bench.services.base_service import Service
from mlos_bench.services.local.temp_dir_context import TempDirContextService
from mlos_bench.services.types.local_exec_type import SupportsLocalExec

if TYPE_CHECKING:
    from mlos_bench.tunables.tunable import TunableValue

_LOG = logging.getLogger(__name__)


class MockLocalExecService(TempDirContextService, SupportsLocalExec):
    """
    Mock methods for LocalExecService testing.
    """

    def __init__(self,
                 *,
                 config: Optional[Dict[str, Any]] = None,
                 config_file_path: Optional[str] = None,
                 global_config: Optional[Dict[str, Any]] = None,
                 parent: Optional[Service] = None,
                 methods: Union[Dict[str, Callable], List[Callable], None] = None):
        super().__init__(
            config=config, config_file_path=config_file_path,
            global_config=global_config, parent=parent,
            methods=self.merge_methods(methods, [self.local_exec]),
        )

    def local_exec(self, script_lines: Iterable[str],
                   env: Optional[Mapping[str, "TunableValue"]] = None,
                   cwd: Optional[str] = None) -> Tuple[int, str, str]:
        return (0, "", "")
