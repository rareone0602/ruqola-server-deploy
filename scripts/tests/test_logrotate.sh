#!/bin/bash
# logrotate.d/scratch-cleanup: parses, targets the chatty log, leaves the manifest alone.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
CONF="$ROOT/logrotate.d/scratch-cleanup"

t "logrotate accepts the config (debug mode, sandbox path)"
new_sandbox
sed "s|/var/log/scratch-cleanup.log|$SB/log/cleanup.log|" "$CONF" > "$SB/rot.conf"
: > "$SB/log/cleanup.log"
out=$(logrotate -d -s "$SB/rot.state" "$SB/rot.conf" 2>&1); rc=$?
check "exit 0" "$rc" "0"
check "targets the log" "$(grep -c "rotating pattern: $SB/log/cleanup.log" <<<"$out")" "1"
check "no 'error' in output" "$(grep -ci 'error' <<<"$out")" "0"
truncate -s 51M "$SB/log/cleanup.log"
out=$(logrotate -d -s "$SB/rot.state" "$SB/rot.conf" 2>&1)
check "a 51M log needs rotating (maxsize)" "$(grep -c 'log needs rotating' <<<"$out")" "1"
drop_sandbox

t "The manifest directory is never rotated"
check "config does not mention the manifest dir" "$(grep -c 'scratch-cleanup/' "$CONF" | head -1)" "1"
check "...except in the comment explaining why" "$(grep -v '^#' "$CONF" | grep -c 'scratch-cleanup/')" "0"
check "manifest path is what the reaper reports" "$(bash "$REAPER" --show-config | sed -n 's/^MANIFEST_DIR=//p')" "/var/log/scratch-cleanup"

finish
