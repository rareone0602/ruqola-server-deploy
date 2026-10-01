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

t "systemd reads every line of every unit (E16)"
# A key in the wrong section is only a warning: systemd ignores the line and the
# unit still starts. RequiresMountsFor= sat in [Service] from the start, so the
# reaper never actually required /scratch to be mounted. Any parser message that
# points at a line of our file fails here. Messages without a line number (such
# as "Command ... is not executable" on a machine without the scripts installed)
# are about the machine, not the file, and are ignored.
if command -v systemd-analyze >/dev/null; then
    units=("$ROOT"/systemd/*.service "$ROOT"/systemd/*.timer)
    out=$(systemd-analyze verify --man=no "${units[@]}" 2>&1)
    for u in "${units[@]}"; do
        check "no line of $(basename "$u") is ignored or rejected" \
            "$(grep -F "$u:" <<<"$out" | grep -E "^$u:[0-9]+:" || true)" ""
    done
else
    ok "systemd-analyze is not installed here; unit files not verified"
fi
check "RequiresMountsFor= is in [Unit]" \
    "$(awk '/^\[/{sec=$0} /^RequiresMountsFor=/{print sec}' "$UNIT")" "[Unit]"

t "The reaper can write only to the scratch directories it cleans (E3)"
# Defence in depth behind deleting inside find: even a redirected deletion
# cannot reach /scratch/datasets. The list must match the reaper's own.
rw_scratch=$(unit_value ReadWritePaths | tr ' ' '\n' | sed 's/^-//' | grep '^/scratch' | sort | tr '\n' ' ')
dirs=$(env -u SCRATCH_CLEANUP_DIRS bash "$ROOT/bin/scratch-cleanup.sh" --show-config | sed -n 's/^SCRATCH_DIRS=//p' | tr ' ' '\n' | sort | tr '\n' ' ')
check "writable /scratch paths = the reaper's SCRATCH_DIRS" "$rw_scratch" "$dirs"
check "neither /scratch nor /scratch/datasets is writable" "$(grep -cE '^/scratch(/datasets)?$' <<<"${rw_scratch// /$'\n'}")" "0"
check "ProtectSystem=strict makes everything else read-only" "$(unit_value ProtectSystem)" "strict"

t "A long night is never cut short (E18)"
# A kill mid-deletion skips that night's mail. Written out, not left to the
# default, so the choice is visible in the unit.
check "TimeoutStartSec=infinity" "$(unit_value TimeoutStartSec)" "infinity"

finish
