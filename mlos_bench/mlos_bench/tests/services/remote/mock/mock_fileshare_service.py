#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""A collection Service functions for mocking file share ops."""

import logging
from collections.abc import Callable
from typing import Any

from mlos_bench.services.base_fileshare import FileShareService
from mlos_bench.services.base_service import Service
from mlos_bench.services.types.fileshare_type import SupportsFileShareOps

_LOG = logging.getLogger(__name__)


class MockFileShareService(FileShareService, SupportsFileShareOps):
    """A collection Service functions for mocking file share ops."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        global_config: dict[str, Any] | None = None,
        parent: Service | None = None,
        methods: dict[str, Callable] | list[Callable] | None = None,
    ):
        super().__init__(
            config,
            global_config,
            parent,
            self.merge_methods(methods, [self.upload, self.download]),
        )
        self._upload: list[tuple[str, str]] = []
        self._download: list[tuple[str, str]] = []

    def upload(
        self,
        params: dict,
        local_path: str,
        remote_path: str,
        recursive: bool = True,
    ) -> None:
        self._upload.append((local_path, remote_path))

    def download(
        self,
        params: dict,
        remote_path: str,
        local_path: str,
        recursive: bool = True,
    ) -> None:
        self._download.append((remote_path, local_path))

    def get_upload(self) -> list[tuple[str, str]]:
        """Get the list of files that were uploaded."""
        return self._upload

    def get_download(self) -> list[tuple[str, str]]:
        """Get the list of files that were downloaded."""
        return self._download
