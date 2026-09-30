#!/bin/bash
# systemd/: the unit files the installer ships. These run as root on a timer, so
# a sandbox setting that silently breaks a feature is only ever seen in the journal.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
UNIT="$ROOT/systemd/scratch-cleanup.service"

# unit_value <Key>: the last value set for Key (systemd lets later lines win).
unit_value() { grep -E "^$1=" "$UNIT" | tail -1 | cut -d= -f2-; }

t "The reaper's sandbox leaves msmtp somewhere to write"
# ProtectSystem=strict makes every path read-only except ReadWritePaths. msmtp
# writes a temporary file before it sends, so with /tmp read-only every send
# failed ("msmtp: cannot create temporary file: Read-only file system", journal
# 2026-09-30) and the reaper logged it as "No email address".
check "ProtectSystem is strict (the premise of this test)" "$(unit_value ProtectSystem)" "strict"
check "PrivateTmp=true gives the unit its own writable /tmp" "$(unit_value PrivateTmp)" "true"

finish
