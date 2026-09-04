#!/bin/bash
# scratch-cleanup.sh: empty directories age out on the same clock as files.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

t "Directories: the same age rule applies to empty directories as to files"
new_sandbox; mkdir_aged users/alice/fresh_empty 0; run_reaper
check "empty dir created today is KEPT" "$(present "$SB/scratch/users/alice/fresh_empty")" "present"
drop_sandbox
new_sandbox; mkdir_aged users/alice/stale_empty 200; run_reaper
check "empty dir older than DAYS_TO_KEEP is removed" "$(present "$SB/scratch/users/alice/stale_empty")" "gone"
drop_sandbox
new_sandbox; mkdir_aged users/alice/borderline 100; run_reaper
check "empty dir aged 100d (< 180) is KEPT" "$(present "$SB/scratch/users/alice/borderline")" "present"
drop_sandbox
new_sandbox; mkdir_aged users/alice/fresh_empty 0; DRYRUN=1 run_reaper; unset DRYRUN
check "dry-run removes no directories" "$(present "$SB/scratch/users/alice/fresh_empty")" "present"
drop_sandbox

t "Structure: a user's top-level directory is never removed, however old"
new_sandbox; mkdir_aged users/olduser 400; run_reaper
check "empty 400-day-old /users/<name> is KEPT" "$(present "$SB/scratch/users/olduser")" "present"
check "...and logged as structural" "$(reaper_log | grep -c 'Preserving structural directory:.*users/olduser')" "1"
drop_sandbox
new_sandbox; mkdir_aged users/olduser/proj 400; run_reaper
check "but an empty 400-day-old subdirectory of it goes" "$(present "$SB/scratch/users/olduser/proj")" "gone"
drop_sandbox

finish
