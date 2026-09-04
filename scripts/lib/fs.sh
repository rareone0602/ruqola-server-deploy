#!/bin/bash
# File metadata, byte formatting, and the append-only deletion manifest.

# format_bytes <n>  -> "1.5K", "206G", ...
format_bytes() { numfmt --to=iec "$1" 2>/dev/null || echo "$1 bytes"; }

# file_meta <path>  -> "owner size atime_epoch mtime_epoch" in one stat call
file_meta() { stat -c '%U %s %X %Y' -- "$1" 2>/dev/null; }

# --- deletion manifest ------------------------------------------------------
# One TSV per month, append-only, never rotated by automation. This is the
# durable answer to "what did the reaper delete, and when": the chatty log is
# rotated away, the manifest is not.
#
#   deleted_at  kind  owner  bytes  last_access  last_modified  path
#
# Tabs, newlines and backslashes inside a path are escaped as \t \n \\ so a
# line is always exactly seven fields.

manifest_escape() {
    local p="$1"
    p=${p//\\/\\\\}; p=${p//$'\t'/\\t}; p=${p//$'\n'/\\n}
    printf '%s' "$p"
}

# manifest_open <file>: create parent directory and header if missing; verify
# writable. Call BEFORE deleting anything so a bad manifest path aborts the
# run instead of producing unrecorded deletions.
manifest_open() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        mkdir -p -- "$(dirname -- "$file")" 2>/dev/null || return 1
        ( umask 027; printf 'deleted_at\tkind\towner\tbytes\tlast_access\tlast_modified\tpath\n' > "$file" ) 2>/dev/null || return 1
    fi
    [[ -w "$file" ]]
}

# manifest_record <file> <kind> <owner> <bytes> <atime_epoch> <mtime_epoch> <path>
manifest_record() {
    local file="$1" kind="$2" owner="$3" bytes="$4" atime="$5" mtime="$6" path="$7"
    local now la lm
    manifest_open "$file" || return 1
    printf -v now '%(%Y-%m-%dT%H:%M:%S%z)T' -1
    printf -v la  '%(%Y-%m-%d)T' "$atime"
    printf -v lm  '%(%Y-%m-%d)T' "$mtime"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$kind" "$owner" "$bytes" "$la" "$lm" "$(manifest_escape "$path")" >> "$file"
}
