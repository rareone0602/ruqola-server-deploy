#!/bin/bash
# scratch-cleanup.sh: the durable deletion record (root cause B).
# The chatty log rotates away; the manifest does not.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
month=$(date +%Y-%m)

t "Every deletion is appended to MANIFEST_DIR/deleted-YYYY-MM.tsv"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/old.bin 200 210 alice; size=$(stat -c %s "$SB/scratch/users/alice/old.bin")
run_reaper
M="$SB/log/manifest/deleted-$month.tsv"
check "manifest for this month exists" "$(present "$M")" "present"
check "header + one record" "$(wc -l < "$M")" "2"
rec=$(sed -n 2p "$M")
check "kind=file owner=alice" "$(cut -f2,3 <<<"$rec")" "file	alice"
check "bytes = the file's size" "$(cut -f4 <<<"$rec")" "$size"
check "last_access = 200 days ago" "$(cut -f5 <<<"$rec")" "$(date -d '200 days ago' +%F)"
check "last_modified = 210 days ago" "$(cut -f6 <<<"$rec")" "$(date -d '210 days ago' +%F)"
check "path is the deleted file" "$(cut -f7 <<<"$rec")" "$SB/scratch/users/alice/old.bin"
check "manifest is not world-readable" "$(stat -c %a "$M")" "640"
drop_sandbox

t "A second run appends; the header is written once"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/a.bin 200 200 alice; run_reaper
mkfile users/alice/b.bin 200 200 alice; run_reaper
M="$SB/log/manifest/deleted-$month.tsv"
check "header + two records after two runs" "$(wc -l < "$M")" "3"
check "exactly one header line" "$(grep -c '^deleted_at' "$M")" "1"
drop_sandbox

t "Removed empty directories are recorded too, as kind=dir"
new_sandbox; mkdir_aged users/alice/stale 200; run_reaper
M="$SB/log/manifest/deleted-$month.tsv"
check "one dir record" "$(grep -c "	dir	" "$M")" "1"
check "path recorded" "$(grep "	dir	" "$M" | cut -f7)" "$SB/scratch/users/alice/stale"
drop_sandbox

t "Dry run records nothing: 'would delete' is not a deletion"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/old.bin 200 200 alice
DRYRUN=1 run_reaper; unset DRYRUN
check "no manifest file created" "$(present "$SB/log/manifest/deleted-$month.tsv")" "gone"
drop_sandbox

t "No record, no deletion: an unwritable manifest aborts before anything is removed"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/old.bin 200 200 alice
mkdir "$SB/ro"; chmod 500 "$SB/ro"
DIRS="$SB/scratch/users" MANIFEST_DIR="$SB/ro/manifest" run_reaper
chmod 700 "$SB/ro"
check "file NOT deleted" "$(present "$SB/scratch/users/alice/old.bin")" "present"
check "no deletion email" "$(mail_count)" "0"
check "ERROR logged naming the manifest" "$(reaper_log | grep -c 'Cannot write deletion manifest')" "1"
check "exit 1 so systemd shows the unit failed" "$(reaper_exit)" "1"
drop_sandbox

t "The chatty log is no longer rotated by the script (logrotate owns that)"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/old.bin 200 200 alice
echo "OLD HISTORY LINE" > "$SB/log/cleanup.log"; truncate -s 11M "$SB/log/cleanup.log"
run_reaper
check "no .old file created" "$(present "$SB/log/cleanup.log.old")" "gone"
check "earlier history still in the log" "$(grep -c 'OLD HISTORY LINE' "$SB/log/cleanup.log")" "1"
check "run appended after it" "$(grep -c 'Scratch folder cleanup started' "$SB/log/cleanup.log")" "1"
drop_sandbox

finish
