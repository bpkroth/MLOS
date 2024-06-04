#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""
Setup instructions for the mlos_core package.
"""

# pylint: disable=duplicate-code

from itertools import chain
from typing import Dict, List

import os
import re

from setuptools import setup


# A simple routine to read and adjust the README.md for this module into a format
# suitable for packaging.
# See Also: copy-source-tree-docs.sh
# Unfortunately we can't use that directly due to the way packaging happens inside a
# temp directory.
# Similarly, we can't use a utility script outside this module, so this code has to
# be duplicated for now.
# Also, to avoid caching issues when calculating dependencies for the devcontainer,
# we return nothing when the file is not available.
def _get_long_desc_from_readme(base_url: str) -> dict:
    pkg_dir = os.path.dirname(__file__)
    readme_path = os.path.join(pkg_dir, 'README.md')
    if not os.path.isfile(readme_path):
        return {}
    jsonc_re = re.compile(r'```jsonc')
    link_re = re.compile(r'\]\(([^:#)]+)(#[a-zA-Z0-9_-]+)?\)')
    with open(readme_path, mode='r', encoding='utf-8') as readme_fh:
        lines = readme_fh.readlines()
        # Tweak the lexers for local expansion by pygments instead of github's.
        lines = [link_re.sub(f"]({base_url}" + r'/\1\2)', line) for line in lines]
        # Tweak source source code links.
        lines = [jsonc_re.sub(r'```json', line) for line in lines]
        return {
            'long_description': ''.join(lines),
            'long_description_content_type': 'text/markdown',
        }


extra_requires: Dict[str, List[str]] = {  # pylint: disable=consider-using-namedtuple-or-dataclass
    'flaml': ['flaml[blendsearch]'],
    'smac': ['smac>=2.0.0'],  # NOTE: Major refactoring on SMAC starting from v2.0.0
}

# construct special 'full' extra that adds requirements for all built-in
# backend integrations and additional extra features.
extra_requires['full'] = list(set(chain(*extra_requires.values())))

extra_requires['full-tests'] = extra_requires['full'] + [
    'pytest',
    'pytest-forked',
    'pytest-xdist',
    'pytest-cov',
    'pytest-local-badge',
]

# TODO: Add code to check that "full" and "full-tests" are covered in the config.

setup(
    #extras_require=extra_requires,
    **_get_long_desc_from_readme('https://github.com/microsoft/MLOS/tree/main/mlos_core'),
)
