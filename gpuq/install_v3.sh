#!/usr/bin/env bash
# Install or update gpuq (needs root).
#
# Usage:
#   sudo ./install_v3.sh
#
# What it does:
#   1. Installs the code. It is compiled beside the live copy first, so a compile
#      error changes nothing.
#   2. Installs gpuqd.service and the client, /usr/local/bin/gpuq.
#   3. Restarts gpuqd. Each job is its own systemd unit, so running jobs carry on.
#
# Paths:
#   Code:    /usr/local/lib/gpuq-v3/{scheduler,gpuqd,gpuqcli}
#   Client:  /usr/local/bin/gpuq
#   Daemon:  /etc/systemd/system/gpuqd.service; socket /run/gpuq/gpuqd.sock
#   State:   /var/lib/gpuq  (state.json root only; usage.jsonl is the ledger)
#   Mail:    /etc/gpuq/mail.json  (root only, written by hand; format in gpuqd/mail.py).
#            Without it gpuq sends no email.
#
# GPUQD_DESTDIR (a prefix for every path) and SYSTEMCTL exist for the tests.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${GPUQD_DESTDIR:-}"
LIB="$DEST/usr/local/lib/gpuq-v3"
BIN="$DEST/usr/local/bin/gpuq"
UNIT="$DEST/etc/systemd/system/gpuqd.service"
STATE="$DEST/var/lib/gpuq"
MAIL="$DEST/etc/gpuq/mail.json"
SOCK="$DEST/run/gpuq/gpuqd.sock"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"
PY=/usr/bin/python3

case "${1:-}" in
    "") ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
esac

if [[ -z "$DEST" && $EUID -ne 0 ]]; then
    echo "Not root; re-running under sudo ..." >&2
    exec sudo -- "$HERE/install_v3.sh" "$@"
fi

put_client() {   # put_client SOURCE: replace $BIN atomically
    mkdir -p "$(dirname "$BIN")"
    local tmp
    tmp="$(mktemp "$BIN.XXXXXX")"
    cat -- "$1" > "$tmp"
    chmod 0755 "$tmp"
    mv -f -- "$tmp" "$BIN"
}

install_code() {
    # Build the new copy beside the old one and compile it there, so a compile
    # error leaves the live copy alone and nothing is written into the source tree.
    mkdir -p "$(dirname "$LIB")"
    local new
    new="$(mktemp -d "$LIB.new.XXXXXX")"
    trap 'rm -rf -- "$new"' EXIT
    for pkg in scheduler gpuqd gpuqcli; do
        mkdir "$new/$pkg"
        cp -- "$HERE/$pkg"/*.py "$new/$pkg/"
    done
    "$PY" -m compileall -q "$new" || {
        echo "Source failed to compile; nothing installed." >&2
        exit 1
    }
    chmod -R a+rX,go-w "$new"
    if [[ -e "$LIB" ]]; then
        rm -rf -- "$LIB.old"
        mv -- "$LIB" "$LIB.old"
    fi
    mv -- "$new" "$LIB"
    rm -rf -- "$LIB.old"
    trap - EXIT
    echo "Installed code -> $LIB"
}

wait_for_socket() {
    [[ -n "$DEST" ]] && return 0
    for _ in $(seq 1 50); do
        [[ -S "$SOCK" ]] && return 0
        sleep 0.2
    done
    echo "gpuqd did not open $SOCK; see: journalctl -u gpuqd -n 50" >&2
    return 1
}

install_code
install -d -m 0755 "$STATE" "$(dirname "$MAIL")"
[[ -e "$MAIL" ]] || echo "No $MAIL: gpuq sends no email until it exists (format: gpuqd/mail.py)."
mkdir -p "$(dirname "$UNIT")"
install -m 0644 -- "$HERE/gpuqd/gpuqd.service" "$UNIT"
"$SYSTEMCTL" daemon-reload
"$SYSTEMCTL" enable gpuqd.service
put_client "$HERE/gpuqcli/gpuq"
echo "Installed client -> $BIN"
"$SYSTEMCTL" restart gpuqd.service
wait_for_socket
echo "gpuq is installed and gpuqd restarted; running jobs carried on. Watch it: journalctl -u gpuqd -f"
