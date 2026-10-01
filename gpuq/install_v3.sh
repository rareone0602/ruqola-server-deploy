#!/usr/bin/env bash
# Cut over to gpuq v3, update it, roll it back, or retire the previous gpuq (needs root).
#
# Usage:
#   sudo ./install_v3.sh              cut over (first run), or update (later runs)
#   sudo ./install_v3.sh --rollback   back to the previous gpuq (refused while gpuqd has jobs)
#   sudo ./install_v3.sh --retire     after the previous gpuq's last jobs have ended: archive and
#                                     remove /var/lib/gpu_queue and its shared config
#
# Cutting over (docs/v3-design.md §10, step 3):
#   1. Refuses while anyone waits in the previous gpuq's queue: a waiting `gpuq submit` cannot
#      move to gpuqd. Try again when `gpuq status` shows nobody queued.
#   2. Installs the code, the mail settings (root only) and gpuqd.service.
#   3. Swaps /usr/local/bin/gpuq for the v3 client, keeping the previous one for --rollback.
#   4. Notes the previous gpuq's running jobs: they finish as they would have and keep their
#      cards until then. Copies the previous gpuq's ledger, so fair-share and `gpuq history`
#      start with the full history.
#   5. Starts gpuqd, and stops shadow mode, whose work is done.
# An update repeats 2, 3 and the restart. Running jobs carry on through it.
#
# Paths:
#   Code:    /usr/local/lib/gpuq-v3/{scheduler,gpuqd,gpuqcli}
#   Client:  /usr/local/bin/gpuq  (the previous one is kept in /usr/local/lib/gpuq-v2/)
#   Daemon:  /etc/systemd/system/gpuqd.service; socket /run/gpuq/gpuqd.sock
#   State:   /var/lib/gpuq  (state.json root only; usage.jsonl is the ledger)
#   Mail:    /etc/gpuq/mail.json  (root only)
#
# GPUQD_DESTDIR (a prefix for every path), SYSTEMCTL and GPUQD_PROC exist for the tests.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${GPUQD_DESTDIR:-}"
LIB="$DEST/usr/local/lib/gpuq-v3"
KEEP="$DEST/usr/local/lib/gpuq-v2"
BIN="$DEST/usr/local/bin/gpuq"
UNIT="$DEST/etc/systemd/system/gpuqd.service"
SHADOW_UNIT="$DEST/etc/systemd/system/gpuqd-shadow.service"
STATE="$DEST/var/lib/gpuq"
LEGACY="$DEST/var/lib/gpu_queue"
OLD_CONFIG="$DEST/usr/local/bin/gpu_queue_config.json"
MAIL="$DEST/etc/gpuq/mail.json"
SOCK="$DEST/run/gpuq/gpuqd.sock"
BACKUPS="$DEST/var/backups"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"
PY=/usr/bin/python3
MARK="gpuq v3 client"

ACTION=install
case "${1:-}" in
    "") ;;
    --rollback) ACTION=rollback ;;
    --retire) ACTION=retire ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
esac

if [[ -z "$DEST" && $EUID -ne 0 ]]; then
    echo "Not root; re-running under sudo ..." >&2
    exec sudo -- "$HERE/install_v3.sh" "$@"
fi

cutover() { PYTHONPATH="$HERE" GPUQD_PROC="${GPUQD_PROC:-/proc}" "$PY" -m gpuqd.cutover "$@"; }
is_v3() { grep -qs "$MARK" "$BIN"; }

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

if [[ $ACTION == install ]]; then
    first=1
    is_v3 && first=0
    if (( first )); then
        echo "Checking the previous gpuq's queue ..."
        if ! cutover waiting "$LEGACY"; then
            echo "These submits are waiting in the previous gpuq. A waiting submit cannot move to" >&2
            echo "gpuqd, so nothing was changed. Try again when nobody is queued." >&2
            exit 1
        fi
    fi
    install_code
    install -d -m 0755 "$STATE" "$(dirname "$MAIL")"
    cutover mail "$OLD_CONFIG" "$MAIL"
    mkdir -p "$(dirname "$UNIT")"
    install -m 0644 -- "$HERE/gpuqd/gpuqd.service" "$UNIT"
    "$SYSTEMCTL" daemon-reload
    "$SYSTEMCTL" enable gpuqd.service
    if (( first )) && [[ -e "$BIN" ]]; then
        install -d -m 0755 "$KEEP"
        cp -p -- "$BIN" "$KEEP/gpuq"
        echo "Kept the previous gpuq's client -> $KEEP/gpuq"
    fi
    put_client "$HERE/gpuqcli/gpuq"
    echo "Installed client -> $BIN"
    if (( first )); then
        cutover snapshot "$LEGACY" "$STATE"
        cutover ledger "$LEGACY" "$STATE"
    fi
    "$SYSTEMCTL" restart gpuqd.service
    wait_for_socket
    if (( first )) && [[ -e "$SHADOW_UNIT" ]]; then
        "$SYSTEMCTL" disable --now gpuqd-shadow.service || true
        rm -f -- "$SHADOW_UNIT"
        "$SYSTEMCTL" daemon-reload
        echo "Stopped shadow mode; its log stays in /var/lib/gpuq-shadow."
    fi
    if (( first )); then
        echo "Cut over: gpuq v3 is live. Watch it: journalctl -u gpuqd -f"
    else
        echo "Updated gpuq v3; running jobs carried on."
    fi
    exit 0
fi

if [[ $ACTION == rollback ]]; then
    n="$(cutover jobs "$STATE")"
    if (( n > 0 )); then
        echo "gpuqd has $n job(s) queued or running. The previous gpuq cannot see them, and its" >&2
        echo "audit would stop them as untracked. Wait for them or kill them first." >&2
        exit 1
    fi
    [[ -e "$KEEP/gpuq" ]] || { echo "No saved copy of the previous gpuq in $KEEP." >&2; exit 1; }
    put_client "$KEEP/gpuq"
    "$SYSTEMCTL" disable --now gpuqd.service || true
    rm -f -- "$UNIT"
    "$SYSTEMCTL" daemon-reload
    cutover unledger "$LEGACY" "$STATE"
    echo "Back on the previous gpuq. gpuqd's state stays in $STATE; cutting over again keeps it."
    echo "Shadow mode is not restarted; for that: sudo ./install_shadow.sh"
    exit 0
fi

# --retire
is_v3 || { echo "gpuq v3 is not installed; nothing to retire." >&2; exit 1; }
n="$(cutover old "$STATE")"
if (( n > 0 )); then
    echo "$n job(s) started by the previous gpuq are still running; retire once they end." >&2
    exit 1
fi
mkdir -p "$BACKUPS"
archive="$BACKUPS/gpuq-v2-$(date +%Y%m%d-%H%M%S).tar.gz"
items=()
[[ -e "$LEGACY" ]] && items+=("${LEGACY#/}")
[[ -e "$OLD_CONFIG" ]] && items+=("${OLD_CONFIG#/}")
[[ -e "$KEEP" ]] && items+=("${KEEP#/}")
if (( ${#items[@]} )); then
    ( umask 077; tar -C / -czf "$archive" -- "${items[@]}" )
    echo "Archived ${items[*]} -> $archive (root only)"
fi
rm -rf -- "$LEGACY" "$KEEP"
rm -f -- "$OLD_CONFIG"
echo "Retired the previous gpuq. Its config held the mail password readable by members (todo B1);"
echo "that file is gone. Root's cron line 'gpuq audit --enforce --quiet' does nothing now;"
echo "remove it when convenient (ruqola-server-deploy MANIFEST, todo E5)."
