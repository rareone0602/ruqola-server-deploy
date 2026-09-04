#!/bin/bash
# scratch-usage.sh (what scratch-status runs): reads the policy from the cleaner,
# scans only the cleaned directories, applies the same both-timestamps rule.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
USAGE="$ROOT/bin/scratch-usage.sh"

run_usage() {
    SCRATCH_USAGE_BASE="$SB/scratch" SCRATCH_CLEANUP_BIN="$REAPER" \
    SCRATCH_CLEANUP_DIRS="$SB/scratch/shared:$SB/scratch/temp:$SB/scratch/users" \
        bash "$USAGE" "$@" 2>&1
}

t "Policy comes from the cleaner, not from this script"
new_sandbox
out=$(run_usage)
check "quotes DAYS_TO_NOTIFY from --show-config" "$(grep -c 'Approaching Deletion (166+ days' <<<"$out")" "1"
check "quotes DAYS_TO_KEEP from --show-config" "$(grep -c 'removed after 180 days' <<<"$out"; true)" "0"   # only printed when something is at risk
check "lists the scanned directories" "$(grep -c "Scanning: $SB/scratch/shared $SB/scratch/temp $SB/scratch/users" <<<"$out")" "1"
out=$(SCRATCH_CLEANUP_DAYS_NOTIFY=99 run_usage)
check "follows an override (99)" "$(grep -c 'Approaching Deletion (99+ days' <<<"$out")" "1"
drop_sandbox

t "Only genuinely at-risk files are flagged"
new_sandbox
mkfile users/alice/atrisk.bin 170 170 alice
mkfile users/alice/fresh.bin 10 10 alice
mkfile users/alice/readlately.bin 5 300 alice
mkdir -p "$SB/scratch/datasets"; mkfile datasets/exempt.bin 300 300 alice
out=$(run_usage)
check "170d file flagged" "$(grep -c 'atrisk.bin' <<<"$out")" "1"
check "fresh file not flagged" "$(grep -c 'fresh.bin' <<<"$out")" "0"
check "recently READ file not flagged (both timestamps rule)" "$(grep -c 'readlately.bin' <<<"$out")" "0"
check "exempt datasets file not flagged" "$(grep -c 'exempt.bin' <<<"$out")" "0"
check "the 180-day rule is stated when something is at risk" "$(grep -c 'removed after 180 days' <<<"$out")" "1"
drop_sandbox

t "Without a readable policy it says so instead of guessing"
new_sandbox
out=$(SCRATCH_USAGE_BASE="$SB/scratch" SCRATCH_CLEANUP_BIN="$SB/nonexistent" bash "$USAGE" 2>&1); rc=$?
check "exit 0" "$rc" "0"
check "explains the skipped section" "$(grep -c 'Cannot read the retention policy' <<<"$out")" "1"
check "prints no number" "$(grep -c 'Approaching Deletion (' <<<"$out")" "0"
drop_sandbox

t "scratch-status is a thin wrapper around scratch-usage.sh"
check "wrapper execs the sibling script" "$(grep -c 'exec .*scratch-usage.sh' "$ROOT/bin/scratch-status")" "1"
new_sandbox
out=$(SCRATCH_USAGE_BASE="$SB/scratch" SCRATCH_CLEANUP_BIN="$REAPER" SCRATCH_CLEANUP_DIRS="$SB/scratch/users" bash "$ROOT/bin/scratch-status" 2>&1)
check "runs the report" "$(grep -c 'Scratch Folder Usage Report' <<<"$out")" "1"
drop_sandbox

finish
