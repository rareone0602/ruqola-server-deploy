#!/usr/bin/env bash
# Install the userspace gpuq shadow into ~/.local/bin (no root required).
#
# Usage:
#   ./install_user.sh                       # symlink mode (default)
#   ./install_user.sh --copy-shared         # per-user copy from shared master
#   ./install_user.sh --copy-from-repo      # per-user copy from this repo
#   ./install_user.sh --publish-shared      # (admin-ish) publish master to shared dir
#
# Distribution paths:
#   Repo source:    $REPO/gpuq/userspace.py
#   Shared master:  /var/lib/gpu_queue/gpuq.py  (gpuqueue-writable)
#   User target:    ~/.local/bin/gpuq

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_REPO="$REPO_ROOT/gpuq/userspace.py"
SCRIPT_SHARED="/var/lib/gpu_queue/gpuq.py"
TARGET="$HOME/.local/bin/gpuq"

MODE="symlink-shared"
PUBLISH_SHARED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --symlink-shared) MODE="symlink-shared"; shift ;;
        --copy-shared)    MODE="copy-shared"; shift ;;
        --copy-from-repo) MODE="copy-from-repo"; shift ;;
        --publish-shared) PUBLISH_SHARED=1; shift ;;
        --help|-h)
            sed -n '2,15p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$HOME/.local/bin"

if [[ $PUBLISH_SHARED -eq 1 ]]; then
    if [[ ! -r "$SCRIPT_REPO" ]]; then
        echo "Cannot read repo source $SCRIPT_REPO" >&2
        exit 1
    fi
    # Atomic publish: a user invoking the symlinked master mid-copy must never
    # execute a truncated script. Group-writable so any gpuqueue member can
    # republish later.
    TMP="$(mktemp "${SCRIPT_SHARED}.XXXXXX")"
    trap 'rm -f -- "$TMP"' EXIT
    cp "$SCRIPT_REPO" "$TMP"
    chmod 0775 "$TMP"
    mv -f -- "$TMP" "$SCRIPT_SHARED"
    trap - EXIT
    echo "Published $SCRIPT_REPO -> $SCRIPT_SHARED"
fi

case "$MODE" in
    symlink-shared)
        if [[ ! -e "$SCRIPT_SHARED" ]]; then
            echo "Master copy at $SCRIPT_SHARED does not exist." >&2
            echo "Publish it first:  $0 --publish-shared" >&2
            echo "(requires read access to the repo source)" >&2
            exit 1
        fi
        ln -snf "$SCRIPT_SHARED" "$TARGET"
        echo "Linked $TARGET -> $SCRIPT_SHARED"
        ;;
    copy-shared)
        if [[ ! -r "$SCRIPT_SHARED" ]]; then
            echo "Cannot read $SCRIPT_SHARED" >&2
            exit 1
        fi
        cp "$SCRIPT_SHARED" "$TARGET"
        chmod +x "$TARGET"
        echo "Copied $SCRIPT_SHARED -> $TARGET"
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

# Verify shadowing
if command -v gpuq >/dev/null 2>&1; then
    RESOLVED="$(command -v gpuq)"
    if [[ "$RESOLVED" == "$TARGET" ]]; then
        echo "OK: 'gpuq' resolves to $RESOLVED"
    else
        echo "WARNING: 'gpuq' resolves to $RESOLVED (not $TARGET)" >&2
        echo "Make sure ~/.local/bin precedes /usr/local/bin in PATH." >&2
    fi
else
    echo "WARNING: 'gpuq' is not in PATH. Add ~/.local/bin to PATH." >&2
fi
