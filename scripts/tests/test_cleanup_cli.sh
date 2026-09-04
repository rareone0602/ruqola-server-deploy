#!/bin/bash
# scratch-cleanup.sh: flags, dry run, and the single-source-of-truth output.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

t "Safety: dry-run deletes nothing and contacts nobody"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/old.bin 200 200 alice
DRYRUN=1 run_reaper; unset DRYRUN
check "dry-run leaves the file on disk" "$(present "$SB/scratch/users/alice/old.bin")" "present"
check "dry-run sends NO email" "$(mail_count)" "0"
check "dry-run still logs what it would send" "$(reaper_log | grep -c 'DRY RUN would email')" "1"
check "dry-run logs what it would remove" "$(reaper_log | grep -c 'DRY RUN would remove: .*old.bin')" "1"
drop_sandbox
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/inwindow.bin 170 170 alice
DRYRUN=1 run_reaper; unset DRYRUN
check "dry-run sends no WARNING email either" "$(mail_count)" "0"
drop_sandbox
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/old.bin 200 200 alice
run_reaper --dry-run
check "--dry-run flag works like the env var" "$(present "$SB/scratch/users/alice/old.bin")" "present"
drop_sandbox

t "SSOT: the script reports its own policy, and honours real flags"
new_sandbox
out=$(SCRATCH_CLEANUP_DIRS="$SB/scratch/users" bash "$REAPER" --show-config 2>&1)
check "--show-config reports DAYS_TO_KEEP"   "$(grep -c '^DAYS_TO_KEEP=180$'   <<<"$out")" "1"
check "--show-config reports DAYS_TO_NOTIFY" "$(grep -c '^DAYS_TO_NOTIFY=166$' <<<"$out")" "1"
check "--show-config reports MANIFEST_DIR"   "$(grep -c '^MANIFEST_DIR=/var/log/scratch-cleanup$' <<<"$out")" "1"
drop_sandbox
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/old.bin 200 200 alice
SCRATCH_CLEANUP_DIRS="$SB/scratch/users" bash "$REAPER" --show-config >/dev/null 2>&1
check "--show-config deletes nothing" "$(present "$SB/scratch/users/alice/old.bin")" "present"
drop_sandbox
out=$(SCRATCH_CLEANUP_DAYS_KEEP=90 SCRATCH_CLEANUP_DAYS_NOTIFY=80 bash "$REAPER" --show-config 2>&1)
check "--show-config reflects overrides" "$(grep -c '^DAYS_TO_KEEP=90$' <<<"$out")" "1"
new_sandbox
SCRATCH_CLEANUP_DIRS="$SB/scratch/users" bash "$REAPER" --nonsense >/dev/null 2>&1
check "unknown flag is rejected (exit 2)" "$?" "2"
bash "$REAPER" --help >/dev/null 2>&1
check "--help exits 0" "$?" "0"
drop_sandbox

t "Lock: a second instance does not run while the first is alive"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/old.bin 200 200 alice
echo $$ > "$SB/run/cleanup.lock"      # this shell is alive, so the lock is live
run_reaper
check "file untouched while lock held by a live PID" "$(present "$SB/scratch/users/alice/old.bin")" "present"
check "exit 1" "$(reaper_exit)" "1"
drop_sandbox

t "Library: the script refuses to start without its shared library"
new_sandbox
RUQOLA_ADMIN_LIB="$SB/nolib" bash "$REAPER" --show-config >/dev/null 2>"$SB/err"; rc=$?
check "missing lib -> non-zero exit" "$(( rc != 0 ))" "1"
check "missing lib -> says so" "$(grep -c 'cannot load' "$SB/err")" "1"
drop_sandbox

finish
