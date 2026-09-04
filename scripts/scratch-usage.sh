#!/bin/bash
# Scratch folder usage report. Safe for any user to run; reads only.
#
# The retention numbers are NOT written here. They are read from
# scratch-cleanup.sh --show-config, which is the single source of truth.
# Restating them is what produced three different published numbers
# (25 here, 30 in README.txt, 180 in the actual cleaner).

SCRATCH_BASE="/scratch"
CLEANUP="${SCRATCH_CLEANUP_BIN:-/usr/local/bin/scratch-cleanup.sh}"

# --- read the policy from the SSOT ----------------------------------------
DAYS_TO_KEEP=""; DAYS_TO_NOTIFY=""; SCRATCH_DIRS=""
if [[ -x "$CLEANUP" ]]; then
    while IFS='=' read -r k v; do
        case "$k" in
            DAYS_TO_KEEP)   DAYS_TO_KEEP="$v" ;;
            DAYS_TO_NOTIFY) DAYS_TO_NOTIFY="$v" ;;
            SCRATCH_DIRS)   SCRATCH_DIRS="$v" ;;
        esac
    done < <("$CLEANUP" --show-config 2>/dev/null)
fi

echo "=== Scratch Folder Usage Report ==="
echo "Generated: $(date)"
echo

echo "=== Overall Usage ==="
df -h "$SCRATCH_BASE" 2>/dev/null || echo "Scratch folder not mounted"
echo

echo "=== Usage by Directory ==="
du -sh "$SCRATCH_BASE"/* 2>/dev/null | sort -hr || echo "No directories found"
echo

echo "=== User Directory Usage ==="
if [[ -d "$SCRATCH_BASE/users" ]]; then
    du -sh "$SCRATCH_BASE/users"/* 2>/dev/null | sort -hr || echo "No user directories found"
else
    echo "User directories not found"
fi
echo

if [[ -z "$DAYS_TO_KEEP" || -z "$SCRATCH_DIRS" ]]; then
    echo "=== Files Approaching Deletion ==="
    echo "Cannot read the retention policy from $CLEANUP -- skipping this section"
    echo "rather than printing a number that might be wrong."
    echo
    echo "=== End Report ==="
    exit 0
fi

# Only the directories the cleaner actually processes. Previously this scanned
# all of /scratch, so it warned about ~2 million files in /scratch/datasets --
# a directory the cleaner explicitly skips. Those files are never at risk.
echo "=== Files Approaching Deletion (${DAYS_TO_NOTIFY}+ days unread AND unmodified) ==="
echo "Scanning: $SCRATCH_DIRS"
echo "(/scratch/datasets is exempt from cleanup and is not scanned)"
echo

# Same rule as the cleaner: a file is at risk only when BOTH timestamps are old.
mapfile -t at_risk < <(find $SCRATCH_DIRS -type f \
    \( -atime +"$DAYS_TO_NOTIFY" -a -mtime +"$DAYS_TO_NOTIFY" \) \
    -printf "%p (last read: %AY-%Am-%Ad, last modified: %TY-%Tm-%Td)\n" 2>/dev/null)

if (( ${#at_risk[@]} == 0 )); then
    echo "None. Nothing under those directories is within ${DAYS_TO_NOTIFY} days of removal."
else
    printf '%s\n' "${at_risk[@]:0:20}"
    (( ${#at_risk[@]} > 20 )) && echo "... and $(( ${#at_risk[@]} - 20 )) more file(s)"
    echo
    echo "Files are removed after ${DAYS_TO_KEEP} days with no read and no write."
    echo "Reading or modifying a file resets its clock."
fi

echo
echo "=== End Report ==="
