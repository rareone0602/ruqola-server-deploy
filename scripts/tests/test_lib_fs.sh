#!/bin/bash
# lib/fs.sh: byte formatting, file metadata, and the append-only deletion manifest.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

t "format_bytes"
check "1536 -> 1.5K" "$(bash -c 'source "$0/init.sh"; format_bytes 1536' "$ROOT/lib")" "1.5K"
check "0 -> 0"       "$(bash -c 'source "$0/init.sh"; format_bytes 0' "$ROOT/lib")" "0"

t "file_meta: owner size atime mtime in one call"
new_sandbox
mkfile users/alice/f.bin 200 100 alice
read -r owner size atime mtime < <(with_stubs bash -c 'source "$0/init.sh"; file_meta "$1"' "$ROOT/lib" "$SB/scratch/users/alice/f.bin")
check "owner via stub" "$owner" "alice"
check "size is the file size" "$size" "$(stat -c %s "$SB/scratch/users/alice/f.bin")"
check "atime older than mtime as created" "$(( atime < mtime ))" "1"
drop_sandbox

t "manifest_record: header once, one TSV line per record, 7 columns"
new_sandbox
M="$SB/log/deleted-2026-09.tsv"
bash -c 'source "$0/init.sh"; manifest_record "$1" file alice 1234 1700000000 1710000000 /scratch/users/alice/a.bin' "$ROOT/lib" "$M"
bash -c 'source "$0/init.sh"; manifest_record "$1" dir bob 0 1700000000 1710000000 /scratch/users/bob/empty' "$ROOT/lib" "$M"
check "file created with header + 2 records" "$(wc -l < "$M")" "3"
check "header names the columns" "$(head -1 "$M")" "deleted_at	kind	owner	bytes	last_access	last_modified	path"
check "every line has exactly 7 tab-separated fields" "$(awk -F'\t' 'NF!=7{bad++} END{print bad+0}' "$M")" "0"
check "record carries kind/owner/bytes/path" "$(sed -n 2p "$M" | cut -f2-4,7)" "file	alice	1234	/scratch/users/alice/a.bin"
check "timestamps are ISO dates, not epochs" "$(sed -n 2p "$M" | cut -f5)" "$(date -d @1700000000 '+%Y-%m-%d')"
check "deleted_at is a full timestamp" "$(sed -n 2p "$M" | cut -f1 | grep -cE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{4}$')" "1"
drop_sandbox

t "manifest_record: a tab or newline in a path cannot break the table"
new_sandbox
M="$SB/log/deleted.tsv"
bash -c 'source "$0/init.sh"; manifest_record "$1" file alice 1 1 1 "$2"' "$ROOT/lib" "$M" $'/scratch/users/alice/odd\tname\nline2\\end'
check "still one record line" "$(wc -l < "$M")" "2"
check "still 7 fields" "$(awk -F'\t' 'NF!=7{bad++} END{print bad+0}' "$M")" "0"
check "escaped as \\t \\n \\\\" "$(sed -n 2p "$M" | cut -f7)" '/scratch/users/alice/odd\tname\nline2\\end'
drop_sandbox

t "manifest_record: directory is created, mode is private"
new_sandbox
M="$SB/log/deep/er/deleted.tsv"
bash -c 'source "$0/init.sh"; manifest_record "$1" file alice 1 1 1 /x' "$ROOT/lib" "$M"
check "parent dirs created" "$([[ -f "$M" ]] && echo yes || echo no)" "yes"
check "manifest not world-readable" "$(stat -c %a "$M")" "640"
drop_sandbox

finish
