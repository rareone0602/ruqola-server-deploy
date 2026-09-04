#!/bin/bash
# Generate /scratch/README.txt from the cleaner's own policy, so the two can
# never disagree. Run after any change to the retention numbers.
#
#   ./render-readme.sh            # print to stdout (safe, changes nothing)
#   sudo ./render-readme.sh --write   # install to /scratch/README.txt
set -euo pipefail

CLEANUP="${SCRATCH_CLEANUP_BIN:-/usr/local/bin/scratch-cleanup.sh}"
TARGET="${SCRATCH_README:-/scratch/README.txt}"
WRITE=0; [[ "${1:-}" == "--write" ]] && WRITE=1

[[ -x "$CLEANUP" ]] || { echo "cannot read policy: $CLEANUP not executable" >&2; exit 1; }

DAYS_TO_KEEP=""; DAYS_TO_NOTIFY=""
while IFS='=' read -r k v; do
    case "$k" in
        DAYS_TO_KEEP)   DAYS_TO_KEEP="$v" ;;
        DAYS_TO_NOTIFY) DAYS_TO_NOTIFY="$v" ;;
    esac
done < <("$CLEANUP" --show-config)
[[ -n "$DAYS_TO_KEEP" && -n "$DAYS_TO_NOTIFY" ]] || { echo "could not parse policy" >&2; exit 1; }

render() {
cat <<EOF
SCRATCH FOLDER USAGE GUIDELINES
==============================

This scratch space is for temporary storage of datasets and computational work.

RETENTION POLICY
----------------
A file is deleted once it has gone ${DAYS_TO_KEEP} days with NO READ and NO WRITE.
Both count: opening a file resets its clock, and so does modifying it.
You are emailed a warning after ${DAYS_TO_NOTIFY} days, about $(( DAYS_TO_KEEP - DAYS_TO_NOTIFY )) days before removal.

/scratch/datasets/ is EXEMPT and is never automatically cleaned.

To check what is at risk, run:  scratch-status

DIRECTORY STRUCTURE
-------------------
/scratch/
|-- shared/     - Shared space for all users (group writable)
|-- temp/       - Scratch space, world writable with sticky bit
|-- datasets/   - Shared datasets - EXEMPT from automatic cleanup
\`-- users/      - Individual user directories
    |-- <you>/  - Your personal scratch space
    \`-- ...

Note: shared/, temp/ and users/ are ALL subject to the same ${DAYS_TO_KEEP}-day rule.
Empty directories in them are also removed once they are ${DAYS_TO_KEEP} days old.

IMPORTANT RULES
---------------
1. Files neither read nor modified for ${DAYS_TO_KEEP} days are deleted automatically
2. This is NOT a backup location - keep anything important elsewhere
3. Large datasets belong in /scratch/datasets/ so they are shared and exempt
4. Use appropriate subdirectories for your work
5. Be respectful of shared space

ACCESS PERMISSIONS
------------------
- Your personal directory (/scratch/users/<you>/): full control
- Shared directories: read/write for all scratch-users group members
- Temp directory: world writable with sticky bit (your files are protected)

For questions or issues, contact your system administrator.

This file is generated from the cleanup policy - do not edit it by hand.
Generated: $(date '+%Y-%m-%d %H:%M:%S %z')
EOF
}

if (( WRITE )); then
    tmp=$(mktemp); render > "$tmp"
    install -o root -g root -m 644 "$tmp" "$TARGET"; rm -f "$tmp"
    echo "wrote $TARGET"
else
    render
fi
