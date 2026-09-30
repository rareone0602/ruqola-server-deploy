#!/usr/bin/env bash
# Put gpuq on your PATH as ~/.local/bin/gpuq (no root required).
#
# Usage:
#   ./install_user.sh                    # link ~/.local/bin/gpuq -> /usr/local/bin/gpuq
#   ./install_user.sh --copy-from-repo   # private copy of this repo's userspace.py (testing)
#
# /usr/local/bin/gpuq is root-owned; install_system.sh is the only way to publish
# a new version. The old shared copy, /var/lib/gpu_queue/gpuq.py, sat in a
# directory every gpuqueue member can write to, so anyone could replace what
# everyone ran (todo E2). It is retired: --symlink-shared, --copy-shared and
# --publish-shared are still accepted, say so, and link to /usr/local/bin/gpuq.
#
# GPUQ_SYSTEM_BIN exists for the tests.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_REPO="$REPO_ROOT/gpuq/userspace.py"
SYSTEM_BIN="${GPUQ_SYSTEM_BIN:-/usr/local/bin/gpuq}"
TARGET="$HOME/.local/bin/gpuq"

MODE="link-system"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --copy-from-repo) MODE="copy-from-repo"; shift ;;
        --symlink-shared|--copy-shared|--publish-shared)
            echo "$1 is retired: the shared copy is gone (todo E2). Linking to $SYSTEM_BIN instead." >&2
            shift ;;
        --help|-h)
            sed -n '2,14p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$HOME/.local/bin"

case "$MODE" in
    link-system)
        if [[ ! -x "$SYSTEM_BIN" ]]; then
            echo "$SYSTEM_BIN is missing; an admin must run install_system.sh first." >&2
            exit 1
        fi
        ln -snf "$SYSTEM_BIN" "$TARGET"
        echo "Linked $TARGET -> $SYSTEM_BIN"
        ;;
    copy-from-repo)
        if [[ ! -r "$SCRIPT_REPO" ]]; then
            echo "Cannot read $SCRIPT_REPO" >&2
            exit 1
        fi
        cp "$SCRIPT_REPO" "$TARGET"
        chmod +x "$TARGET"
        echo "Copied $SCRIPT_REPO -> $TARGET"
        ;;
esac

# Verify which gpuq the shell will run
if command -v gpuq >/dev/null 2>&1; then
    RESOLVED="$(command -v gpuq)"
    if [[ "$RESOLVED" == "$TARGET" || "$RESOLVED" == "$SYSTEM_BIN" ]]; then
        echo "OK: 'gpuq' resolves to $RESOLVED"
    else
        echo "WARNING: 'gpuq' resolves to $RESOLVED (not $TARGET)" >&2
    fi
else
    echo "WARNING: 'gpuq' is not in PATH. Add ~/.local/bin to PATH." >&2
fi
