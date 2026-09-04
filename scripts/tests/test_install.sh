#!/bin/bash
# install.sh: manifest-driven, idempotent, backs up before touching, can roll back.
# Everything happens under a sandbox DESTROOT; nothing here needs root.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
INSTALL="$ROOT/install.sh"

inst() {   # run install.sh against the sandbox with every seam set
    PATH="$STUBS:$PATH" RUQOLA_ADMIN_DESTROOT="$SB/root" RUQOLA_ADMIN_BACKUP_ROOT="$SB/backups" \
    RUQOLA_ADMIN_SKIP_TESTS=1 RUQOLA_ADMIN_NO_CHOWN=1 \
        bash "$INSTALL" "$@" >"$SB/out" 2>"$SB/err"; echo $? > "$SB/exit"
}
out() { sed 's/\x1b\[[0-9;]*m//g' "$SB/out"; }   # colour codes stripped
BIN="$SB/root/usr/local/bin"; LIB="$SB/root/usr/local/lib/ruqola-admin"

t "--check on an empty host: everything missing, exit 1"
new_sandbox; BIN="$SB/root/usr/local/bin"; LIB="$SB/root/usr/local/lib/ruqola-admin"
inst --check
check "exit 1" "$(cat "$SB/exit")" "1"
check "reaper reported MISSING" "$(out | grep -c 'MISSING *\S*/usr/local/bin/scratch-cleanup.sh')" "1"
check "lib reported MISSING" "$(out | grep -c 'MISSING *\S*/usr/local/lib/ruqola-admin/mail.sh')" "1"
check "retire entries reported absent" "$(out | grep -c 'absent')" "3"

t "install --yes: everything lands with the right shape"
inst --yes
check "exit 0" "$(cat "$SB/exit")" "0"
check "reaper installed 0755" "$(stat -c %a "$BIN/scratch-cleanup.sh")" "755"
check "lib installed 0644" "$(stat -c %a "$LIB/mail.sh")" "644"
check "unit installed" "$(present "$SB/root/etc/systemd/system/scratch-cleanup.timer")" "present"
check "logrotate installed" "$(present "$SB/root/etc/logrotate.d/scratch-cleanup")" "present"
check "create_users is a symlink to add_users.sh" "$(readlink "$BIN/create_users")" "add_users.sh"
check "delete_users is a symlink to delete_users.sh" "$(readlink "$BIN/delete_users")" "delete_users.sh"
check "README rendered from the policy" "$(grep -c 'gone 180 days' "$SB/root/scratch/README.txt")" "1"
check "daemon-reload ran once (units were new)" "$(grep -c 'systemctl daemon-reload' "$STUB_CALLS")" "1"
check "backup folder with rollback.sh" "$(ls "$SB/backups"/*/rollback.sh | wc -l)" "1"
check "installed reaper runs against the installed lib" \
      "$(RUQOLA_ADMIN_LIB="$LIB" bash "$BIN/scratch-cleanup.sh" --show-config | grep -c '^DAYS_TO_KEEP=180$')" "1"
check "installed scratch-status finds its sibling" \
      "$(SCRATCH_USAGE_BASE="$SB/scratch" SCRATCH_CLEANUP_DIRS="$SB/scratch/users" RUQOLA_ADMIN_LIB="$LIB" bash "$BIN/scratch-status" 2>&1 | grep -c 'Approaching Deletion (166+')" "1"

t "--check after install: all same, exit 0; a second install changes nothing"
inst --check
check "exit 0" "$(cat "$SB/exit")" "0"
check "no MISSING/DIFFERS/MODE lines" "$(out | grep -cE 'MISSING|DIFFERS|MODE|PRESENT')" "0"
: > "$STUB_CALLS"
inst --yes
check "second install exit 0" "$(cat "$SB/exit")" "0"
check "says nothing to do" "$(out | grep -c 'nothing to do')" "1"
check "no daemon-reload on a no-op run" "$(grep -c 'daemon-reload' "$STUB_CALLS")" "0"

t "Drift: a hand-edited live file is reported, diffed, replaced, and can be rolled back"
echo "# hand edit" >> "$BIN/scratch-cleanup.sh"
inst --check
check "exit 1" "$(cat "$SB/exit")" "1"
check "exactly one DIFFERS, the reaper" "$(out | grep -c 'DIFFERS')/$(out | grep -c 'DIFFERS *\S*/scratch-cleanup.sh')" "1/1"
inst --diff
check "--diff shows the edit" "$(out | grep -c '^ *-# hand edit')" "1"
inst --yes
check "reinstalled cleanly" "$(cat "$SB/exit")" "0"
check "hand edit gone" "$(grep -c '# hand edit' "$BIN/scratch-cleanup.sh")" "0"
latest=$(ls -d "$SB/backups"/*/ | sort | tail -1)
check "backup holds the hand-edited copy" "$(grep -c '# hand edit' "$latest/files/usr/local/bin/scratch-cleanup.sh")" "1"
check "backup is outside PATH" "$(ls "$BIN" | grep -c 'bak')" "0"
bash "$latest/rollback.sh"
check "rollback restores the hand-edited copy" "$(grep -c '# hand edit' "$BIN/scratch-cleanup.sh")" "1"
inst --yes   # put it right again for the next tests
check "after rollback+reinstall, clean" "$(cat "$SB/exit")" "0"

t "Mode drift is a finding too"
chmod 700 "$BIN/scratch-usage.sh"
inst --check
check "MODE reported" "$(out | grep -c 'MODE *\S*/scratch-usage.sh')" "1"
inst --yes
check "mode restored" "$(stat -c %a "$BIN/scratch-usage.sh")" "755"

t "Retire: the landmine and the .bak files are swept into the backup folder"
printf '#!/bin/bash\necho bad\n' > "$BIN/scratch-backup.sh"; chmod 755 "$BIN/scratch-backup.sh"
touch "$BIN/scratch-cleanup.sh.bak-20260904-215943" "$BIN/scratch-usage.sh.bak-20260904-215943" "$SB/root/scratch/README.txt.bak-1"
inst --check
check "PRESENT reported for the landmine" "$(out | grep -c 'PRESENT *\S*/scratch-backup.sh')" "1"
check "PRESENT reported for the .bak glob" "$(out | grep -c 'PRESENT *\S*/usr/local/bin/\*.bak-\*')" "1"
inst --yes
check "exit 0" "$(cat "$SB/exit")" "0"
check "nothing named .bak left in bin" "$(ls "$BIN" | grep -c 'bak')" "0"
check "landmine gone from bin" "$(present "$BIN/scratch-backup.sh")" "gone"
latest=$(ls -d "$SB/backups"/*/ | sort | tail -1)
check "all four retired files kept" "$(ls "$latest/retired" | wc -l)" "4"
check "rollback.sh would put them back" "$(grep -c "^mv '$latest" "$latest/rollback.sh" | sed 's/.*/&/')" "$(grep -c "^mv " "$latest/rollback.sh")"

t "--only limits the work to one entry"
rm -f "$BIN/check_quotas.sh" "$BIN/scratch-usage.sh"
inst --yes --only check_quotas.sh
check "named entry installed" "$(present "$BIN/check_quotas.sh")" "present"
check "other missing entry untouched" "$(present "$BIN/scratch-usage.sh")" "gone"
inst --yes

t "A broken manifest is refused before anything happens"
printf 'bogus  x  /usr/local/bin/x  0755  root:root\n' > "$SB/bad.manifest"
RUQOLA_ADMIN_MANIFEST="$SB/bad.manifest" inst --check
check "exit non-zero" "$(( $(cat "$SB/exit") != 0 ))" "1"
check "names the problem" "$(grep -c "unknown kind 'bogus'" "$SB/err")" "1"
printf 'file  bin/scratch-cleanup.sh  relative/path  0755  root:root\n' > "$SB/bad.manifest"
RUQOLA_ADMIN_MANIFEST="$SB/bad.manifest" inst --check
check "relative destination refused" "$(grep -c 'must be absolute' "$SB/err")" "1"

t "Without root and without a sandbox, installing refuses"
PATH="$STUBS:$PATH" RUQOLA_ADMIN_SKIP_TESTS=1 bash "$INSTALL" --yes >/dev/null 2>"$SB/err"; rc=$?
check "exit 1" "$rc" "1"
check "says it needs root" "$(grep -c 'needs root' "$SB/err")" "1"
drop_sandbox

finish
