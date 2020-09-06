#!/usr/bin/env python3
#
# Short script to parse pip install requirements out of a yaml environment file.

import io
import re
import sys
import yaml

if len(sys.argv) != 2:
    raise Exception("usage: {scriptName} <path/to/environment.yml>".format(scriptName = sys.argv[0]))
filePath = sys.argv[1]

data = list()

with open(filePath, 'r') as stream:
    data = yaml.safe_load(stream)

if not data or not data["dependencies"]:
    raise Exception("Failed to load yaml file at '{filePath}'".format(filePath = filePath))

new_dependencies = list()
for item in data["dependencies"]:
    item_type = type(item)
    if item_type == str:
        #  These are not in pip compatible version comparison format.
        # In particular, we need to replace = with ==
        new_item = re.sub(
            pattern=r'([^=])=([^=])',
            repl='\\1==\\2',
            string=item)
        new_dependencies.append(new_item)
    elif item_type == dict:
        for item_key in item:
            if item_key != "pip":
                raise Exception("Unhandled case (non-pip dependency dict): {0}".format(item_key))
            for other_item in item[item_key]:
                # These are already in pip version syntax format.
                new_dependencies.append(other_item)
    else:
        raise Exception("Unhandled type: '{item_type}'".format(item_type = item_type))

excluded_dependencies = ['python']
for item in new_dependencies:
    if item in excluded_dependencies:
        continue
    else:
        print(item)
