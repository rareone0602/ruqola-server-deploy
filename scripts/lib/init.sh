#!/bin/bash
# Loads every shared library file. Scripts source THIS file only:
#
#     source "$RUQOLA_ADMIN_LIB/init.sh"
#
# Adding a library: create lib/<name>.sh and add <name> to the list below.
# Each file holds one concern and reads its settings from documented globals.
_ruqola_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _ruqola_f in log mail fs; do
    source "$_ruqola_lib_dir/$_ruqola_f.sh" || {
        echo "ruqola-admin: cannot load $_ruqola_lib_dir/$_ruqola_f.sh" >&2; exit 1; }
done
unset _ruqola_f _ruqola_lib_dir
