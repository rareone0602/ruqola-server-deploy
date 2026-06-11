#!/usr/bin/env bash
# Deploy the userspace gpuq to the system path /usr/local/bin/gpuq (needs root).
#
# Installs $REPO/gpuq/userspace.py as /usr/local/bin/gpuq: validates it compiles
# BEFORE touching the live path, backs up any existing binary, installs
# atomically (write-temp + rename), never clobbers an existing config, and
# ensures the shared coordination dir exists. There is no daemon to restart —
# the userspace gpuq runs in the foreground per submit.
#
# Usage:
#   sudo ./install_system.sh                 # deploy from this repo
#   sudo ./install_system.sh --source PATH   # deploy from an explicit file
#   ./install_system.sh                      # re-execs itself under sudo
#
# Paths:
#   Source:  $REPO/gpuq/userspace.py        (or --source PATH)
#   Target:  /usr/local/bin/gpuq
#   Config:  /usr/local/bin/gpu_queue_config.json   (created only if missing)
#   Shared:  /var/lib/gpu_queue/            (group gpuqueue, SGID 2775)

set -euo pipefail

TARGET="/usr/local/bin/gpuq"
CONFIG="/usr/local/bin/gpu_queue_config.json"
QUEUE_DIR="/var/lib/gpu_queue"
QUEUE_GROUP="gpuqueue"

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$REPO_ROOT/gpuq/userspace.py"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE="$2"; shift 2 ;;
        --help|-h) sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done
SOURCE="$(realpath -- "$SOURCE" 2>/dev/null || echo "$SOURCE")"

# Writing /usr/local/bin needs root — re-exec under sudo if we are not.
if [[ $EUID -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || {
        echo "Not root and 'sudo' not found; re-run this script as root." >&2
        exit 1
    }
    echo "Not root; re-running under sudo ..." >&2
    exec sudo -- "$SELF" --source "$SOURCE"
fi

# 1) Validate the source compiles BEFORE we touch the live binary.
[[ -r "$SOURCE" ]] || { echo "Cannot read source: $SOURCE" >&2; exit 1; }
PYBIN="$(command -v python3 || true)"
[[ -n "$PYBIN" ]] || { echo "python3 not found in PATH" >&2; exit 1; }
"$PYBIN" -m py_compile "$SOURCE" || {
    echo "Source failed to compile; aborting (live $TARGET untouched)." >&2
    exit 1
}

# 2) Install atomically, backing up any existing binary first.
if [[ -e "$TARGET" ]]; then
    BACKUP="${TARGET}.bak-$(date +%Y%m%d-%H%M%S)"
    cp -p -- "$TARGET" "$BACKUP"
    echo "Backed up existing binary -> $BACKUP"
fi
TMP="$(mktemp "${TARGET}.XXXXXX")"          # same dir => mv is an atomic rename
trap 'rm -f -- "$TMP"' EXIT
cat -- "$SOURCE" > "$TMP"
chmod 0755 "$TMP"
chown root:root "$TMP" 2>/dev/null || true
mv -f -- "$TMP" "$TARGET"
trap - EXIT
echo "Installed $SOURCE -> $TARGET"

# 3) Smoke-test the installed binary.
if "$TARGET" --help >/dev/null 2>&1; then
    echo "OK: $TARGET runs"
else
    echo "WARNING: '$TARGET --help' did not exit cleanly; check the install." >&2
fi

# 4) Ensure the shared coordination dir (gpuq won't coordinate without it).
if getent group "$QUEUE_GROUP" >/dev/null 2>&1; then
    if [[ ! -d "$QUEUE_DIR" ]]; then
        install -d -m 2775 -g "$QUEUE_GROUP" "$QUEUE_DIR"
        echo "Created $QUEUE_DIR (group $QUEUE_GROUP, SGID 2775)"
    else
        echo "Coordination dir $QUEUE_DIR present (left as-is)"
    fi
else
    echo "NOTE: group '$QUEUE_GROUP' is missing. Create it, then:" >&2
    echo "      install -d -m 2775 -g $QUEUE_GROUP $QUEUE_DIR" >&2
fi

# 5) Seed a starter config only if none exists — never clobber the live one
#    (it holds the SMTP app password).
if [[ -e "$CONFIG" ]]; then
    echo "Config $CONFIG present (left untouched)"
else
    GPUQ_CONFIG_FILE="$CONFIG" "$TARGET" config >/dev/null 2>&1 || true
    if [[ -e "$CONFIG" ]]; then
        chmod 0644 "$CONFIG"
        echo "Wrote starter config $CONFIG (edit to enable email/Slack/audit)"
    else
        echo "NOTE: could not seed $CONFIG; create it later with 'gpuq config'." >&2
    fi
fi

echo
echo "Done. 'gpuq' deployed to $TARGET."
echo "Schedule the auditor from root's crontab if you want enforcement, e.g.:"
echo "  */15 * * * * $TARGET audit --enforce --quiet"
