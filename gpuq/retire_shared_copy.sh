#!/usr/bin/env bash
# One-off (todo E2): stop everyone running gpuq from the group-writable queue dir.
#
# /var/lib/gpu_queue is writable by every gpuqueue member, so any of them could
# replace the shared gpuq.py there, or drop a same-named Python module beside it,
# and their code ran as whoever started the shared copy. This script:
#   1. re-points every ~/.local/bin/gpuq symlink that targets the shared copy at
#      the root-owned /usr/local/bin/gpuq. Each link is changed AS its owner
#      (runuser), so root never follows a path a user controls. The link's own
#      path does not change, so shells that remembered it keep working;
#   2. then, only if every re-point worked, removes the shared gpuq.py and its
#      __pycache__. Anything else in the directory is listed, not touched.
# Private copies (a regular file at ~/.local/bin/gpuq) belong to their owner and
# are listed, not touched.
#
# Usage:
#   sudo ./retire_shared_copy.sh           show what it would do; change nothing
#   sudo ./retire_shared_copy.sh --apply   do it
#
# GPUQ_DESTDIR (a prefix for every file path), GETENT and RUNUSER exist for the tests.

set -euo pipefail

DEST="${GPUQ_DESTDIR:-}"
GETENT="${GETENT:-getent}"
RUNUSER="${RUNUSER:-runuser}"
QUEUE_DIR=/var/lib/gpu_queue
SHARED="$QUEUE_DIR/gpuq.py"
SYSTEM_BIN=/usr/local/bin/gpuq

APPLY=0
case "${1:-}" in
    "") ;;
    --apply) APPLY=1 ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
esac

if [[ -z "$DEST" && $EUID -ne 0 ]]; then
    echo "Needs root (to read every home directory): sudo $0 ${1:-}" >&2
    exit 1
fi
if [[ ! -x "$DEST$SYSTEM_BIN" ]]; then
    echo "$SYSTEM_BIN is missing; run install_system.sh first. Nothing changed." >&2
    exit 1
fi

(( APPLY )) || echo "Dry run: nothing will change. Re-run with --apply to do it."
failed=0
while IFS=: read -r user _ uid _ _ home _; do
    [[ -n "$home" && "$home" != / ]] || continue
    link="$DEST$home/.local/bin/gpuq"
    if [[ -L "$link" ]]; then
        target=$(readlink -- "$link")
        case "$(readlink -m -- "$link")" in
            "$DEST$SHARED"|"$SHARED")
                if (( ! APPLY )); then
                    echo "would re-point  $user: ~/.local/bin/gpuq -> $SYSTEM_BIN (now -> $target)"
                elif "$RUNUSER" -u "$user" -- ln -sfn "$SYSTEM_BIN" "$link" \
                        && [[ "$(readlink -- "$link")" == "$SYSTEM_BIN" ]]; then
                    echo "re-pointed      $user: ~/.local/bin/gpuq -> $SYSTEM_BIN"
                else
                    echo "FAILED          $user: ~/.local/bin/gpuq still -> $(readlink -- "$link" || true)" >&2
                    failed=1
                fi ;;
            "$DEST$SYSTEM_BIN"|"$SYSTEM_BIN")
                echo "already fine    $user: ~/.local/bin/gpuq -> $target" ;;
            *)
                echo "left alone      $user: ~/.local/bin/gpuq -> $target" ;;
        esac
    elif [[ -f "$link" ]]; then
        echo "left alone      $user: ~/.local/bin/gpuq is a private copy (their own file)"
    fi
done < <("$GETENT" passwd)

others=$(find "$DEST$QUEUE_DIR" -maxdepth 1 -name '*.py' ! -name gpuq.py -printf '%f ' 2>/dev/null || true)
[[ -z "$others" ]] || echo "NOTE: other .py files in $QUEUE_DIR (not touched; nothing should run them): $others"

if (( failed )); then
    echo "Some links could not be re-pointed; the shared copy is KEPT so they still work." >&2
    exit 1
fi
if (( ! APPLY )); then
    echo "would remove    $SHARED and $QUEUE_DIR/__pycache__"
    exit 0
fi
rm -f -- "$DEST$SHARED"
rm -rf -- "$DEST$QUEUE_DIR/__pycache__"
echo "removed         $SHARED and $QUEUE_DIR/__pycache__"
echo "Done. Everyone now runs $SYSTEM_BIN; publish new versions with install_system.sh only."
