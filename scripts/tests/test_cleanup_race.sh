#!/bin/bash
# scratch-cleanup.sh: a user racing the reaper cannot redirect a root deletion
# (todo E3). The old reaper listed paths with find and deleted them later with
# rm / rmdir, which resolve the path again. A user who swapped one of their own
# directories for a symlink in between made root delete a file of the same
# name wherever the symlink pointed. The stub find runs the swap at the first
# moment the reaper can see the path (FAKE_FIND_HOOK).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
month=$(date +%Y-%m)

t "Files: swapping a directory for a symlink mid-run deletes only the user's own file"
new_sandbox; add_user mallory "Mallory,,,,mallory@example.com"
mkdir -p "$SB/scratch/datasets/victim"; echo precious > "$SB/scratch/datasets/victim/model.pt"
mkfile users/mallory/d/model.pt 200 200 mallory
DIRS="$SB/scratch/users" \
FAKE_FIND_HOOK_PATH="$SB/scratch/users/mallory/d/model.pt" \
FAKE_FIND_HOOK="mv '$SB/scratch/users/mallory/d' '$SB/scratch/users/mallory/d.orig' && ln -s '$SB/scratch/datasets/victim' '$SB/scratch/users/mallory/d'" \
    run_reaper
check "the swap really happened" "$(readlink "$SB/scratch/users/mallory/d")" "$SB/scratch/datasets/victim"
check "the file behind the symlink survives" "$(cat "$SB/scratch/datasets/victim/model.pt" 2>/dev/null)" "precious"
check "the user's own expired file is the one deleted" "$(present "$SB/scratch/users/mallory/d.orig/model.pt")" "gone"
check "one file recorded, under the path find saw" \
    "$(grep -c '	file	' "$SB/log/manifest/deleted-$month.tsv") $(grep '	file	' "$SB/log/manifest/deleted-$month.tsv" | cut -f3,7)" \
    "1 mallory	$SB/scratch/users/mallory/d/model.pt"
check "exit 0" "$(reaper_exit)" "0"
drop_sandbox

t "Empty directories: the same swap cannot make root rmdir outside the scan"
new_sandbox
mkdir -p "$SB/run/sshd"                                # e.g. /run/sshd: sshd needs it
mkdir_aged users/mallory/e/sshd 200
DIRS="$SB/scratch/users" \
FAKE_FIND_HOOK_PATH="$SB/scratch/users/mallory/e/sshd" \
FAKE_FIND_HOOK="mv '$SB/scratch/users/mallory/e' '$SB/scratch/users/mallory/e.orig' && ln -s '$SB/run' '$SB/scratch/users/mallory/e'" \
    run_reaper
check "the swap really happened" "$(readlink "$SB/scratch/users/mallory/e")" "$SB/run"
check "the directory behind the symlink survives" "$(present "$SB/run/sshd")" "present"
check "the user's own stale directory is the one removed" "$(present "$SB/scratch/users/mallory/e.orig/sshd")" "gone"
drop_sandbox

t "A deletion find cannot make is an error, and nothing is recorded for it"
new_sandbox; add_user alice "Alice,,,,alice@example.com"
mkfile users/alice/locked/old.bin 200 200 alice
chmod 555 "$SB/scratch/users/alice/locked"
DIRS="$SB/scratch/users" run_reaper
chmod 755 "$SB/scratch/users/alice/locked"
check "the file is still there" "$(present "$SB/scratch/users/alice/locked/old.bin")" "present"
check "logged as a failed removal" "$(reaper_log | grep -c "Failed to remove: '$SB/scratch/users/alice/locked/old.bin'")" "1"
check "no manifest record, no deletion mail" \
    "$(grep -c '	file	' "$SB/log/manifest/deleted-$month.tsv" 2>/dev/null) $(mail_count)" "0 0"
check "exit 1 so the unit shows failed" "$(reaper_exit)" "1"
drop_sandbox

t "An old tree of empty directories goes in one run; a user's top level stays"
new_sandbox
mkdir_aged users/bob/a/b/c 200; touch -d '200 days ago' "$SB/scratch/users/bob/a/b" "$SB/scratch/users/bob/a"
touch -d '400 days ago' "$SB/scratch/users/bob"
DIRS="$SB/scratch/users" run_reaper
check "a, a/b and a/b/c removed" "$(present "$SB/scratch/users/bob/a")" "gone"
check "three dir records" "$(grep -c '	dir	' "$SB/log/manifest/deleted-$month.tsv")" "3"
check "bob's own top-level directory kept" "$(present "$SB/scratch/users/bob")" "present"
drop_sandbox

finish
