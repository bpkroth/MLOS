#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Simple platform agnostic abstraction for the OS environment variables. Meant as a
replacement for :external:py:data:`os.environ` vs ``nt.environ``.

Example
-------
>>> # Import the environ object.
>>> from mlos_bench.os_environ import environ
>>> # Set an environment variable.
>>> environ["FOO"] = "bar"
>>> # Get an environment variable.
>>> pwd = environ.get("PWD")
"""

import os
import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if sys.version_info >= (3, 9):
    # pylint: disable=protected-access,disable=unsubscriptable-object
    EnvironType: TypeAlias = os._Environ[str]
else:
    assert False, "Unsupported Python version."

# Handle case sensitivity differences between platforms.
# https://stackoverflow.com/a/19023293
if sys.platform == "win32":
    import nt  # type: ignore[import-not-found]    # pylint: disable=import-error  # (3.8)

    environ: EnvironType = nt.environ
    """A platform agnostic abstraction for the OS environment variables."""
else:
    environ: EnvironType = os.environ
    """A platform agnostic abstraction for the OS environment variables."""

__all__ = ["environ"]
