#!/bin/bash

set -eu

#set -x

WHL="$1"

whl_metadata=$(unzip -p "$WHL" "*/METADATA")

# Get all non-full extras reported by wheel.
provides_extras=$(echo "$whl_metadata" | grep ^Provides-Extra: | grep -v ': full' | awk '{ print $2 }' | sort)
# Make sure they're all listsed in the "full" extra.
full_requires=$(echo "$whl_metadata" | grep ^Requires-Dist: | grep "extra == 'full'" | sed -r -e "s/^Requires-Dist: mlos[a-z_-]+\[([a-zA-Z0-9,-]+)\] ; extra == 'full'$/\1/" | sed -e 's/,/\n/g' | sort)

if [ "$provides_extras" != "$full_requires" ]; then
    echo -e "ERROR: Mismatched full target requires vs. provides_extras for $WHL.\n" >&2
    echo -e "Provides Extras:\n\n$provides_extras\n" >&2
    echo -e "Full Requires:\n\n$full_requires\n" >&2
    exit 1
fi

# OK
exit 0
