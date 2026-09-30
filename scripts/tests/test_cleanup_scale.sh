#!/bin/bash
# scratch-cleanup.sh at the size of a real cohort (todo E18, E7).
# The December 2026 cohort is ~300,000 files for one owner. The reaper used to
# build each owner's list as one bash string, which costs time in the square of
# its length (about half an hour at that size, inside a one-hour unit limit),
# and mailed every path: a 30 MB message Gmail refuses. Now lists live in
# files and a digest shows DIGEST_MAX_FILES (100) paths plus a count.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
month=$(date +%Y-%m)

# bulk <dir-under-scratch> <count> <age_days>: files owned by the test user, fast.
bulk() {
    python3 - "$SB/scratch/$1" "$2" "$3" <<'PY'
import os, sys, time
d, n, days = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(d, exist_ok=True); t = time.time() - days * 86400
for i in range(n):
    p = os.path.join(d, f"f{i:05d}.bin"); open(p, "w").close(); os.utime(p, (t, t))
PY
}

t "Warning digest: at most 100 paths, most urgent first, then a count and a find command"
new_sandbox; add_user alice "Alice,,,,alice@example.com"
for i in $(seq 1 145); do mkfile "users/alice/w/later$i.bin" 170 170 alice; done
for i in $(seq 1 5);   do mkfile "users/alice/w/soon$i.bin"  178 178 alice; done
DIRS="$SB/scratch/users" run_reaper
body=$(mail_bodies)
check "one mail for all 150" "$(mail_count) $(grep -c 'Imminent removal: 150 file(s)' <<<"$body")" "1 1"
check "100 paths listed" "$(grep -c 'day(s) remaining)' <<<"$body")" "100"
check "the 5 most urgent come first" "$(grep 'day(s) remaining)' <<<"$body" | head -5 | grep -c 'soon')" "5"
check_contains "then the count" "$body" "  ... and 50 more. List them all with:"
check_contains "and the command that lists them" "$body" "find $SB/scratch/users -user alice -type f -atime +166 -mtime +166"
drop_sandbox

t "Deletion digest: at most 100 paths; the manifest still records every file"
new_sandbox; add_user alice "Alice,,,,alice@example.com"
for i in $(seq 1 120); do mkfile "users/alice/d/old$i.bin" 200 200 alice; done
DIRS="$SB/scratch/users" run_reaper
body=$(mail_bodies)
check "120 files deleted" "$(find "$SB/scratch/users/alice/d" -type f | wc -l)" "0"
check "100 paths listed" "$(grep -c ' bytes)$' <<<"$body")" "100"
check_contains "then the count" "$body" "  ... and 20 more. The server keeps a record of every deleted file"
check "the manifest has all 120" "$(grep -c '	file	alice	' "$SB/log/manifest/deleted-$month.tsv")" "120"
drop_sandbox

t "A small digest lists everything and adds no count line"
new_sandbox; add_user alice "Alice,,,,alice@example.com"
for i in 1 2 3; do mkfile "users/alice/s/f$i.bin" 170 170 alice; done
DIRS="$SB/scratch/users" run_reaper
check "3 paths, no '... and N more'" "$(mail_bodies | grep -c 'day(s) remaining)') $(mail_bodies | grep -c '\.\.\. and')" "3 0"
drop_sandbox

t "40,000 files for one owner: seconds, not minutes, and a mail of normal size"
# Before E18 this took 220 s here (the old per-file stat plus string appends).
new_sandbox; me=$(id -un); add_user "$me" "Me,,,,me@example.com"
bulk "users/$me/bulk" 40000 170
start=$(date +%s)
DIRS="$SB/scratch/users" run_reaper
took=$(( $(date +%s) - start ))
check "exit 0" "$(reaper_exit)" "0"
check "under 30 s (took ${took}s)" "$(( took < 30 ))" "1"
check "one warning mail naming all 40000" "$(mail_count) $(mail_bodies | grep -c 'Imminent removal: 40000 file(s)')" "1 1"
check "the mail stays small (under 20 KB)" "$(( $(mail_bodies | wc -c) < 20480 ))" "1"
drop_sandbox

finish
