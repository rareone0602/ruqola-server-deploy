#!/bin/bash
# Run every tests/test_*.sh and sum the results.
#
#   tests/run_tests.sh              # everything
#   tests/run_tests.sh cleanup      # only files whose name contains "cleanup"
#
# To test a new script, add tests/test_<name>.sh that sources tests/lib.sh and
# ends with `finish`. Nothing else needs registering.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER="${1:-}"
TOTAL_PASS=0; TOTAL_FAIL=0; FILES_FAILED=()

for f in "$HERE"/test_*.sh; do
    [[ -z "$FILTER" || "$(basename "$f")" == *"$FILTER"* ]] || continue
    printf '\n\033[1;34m### %s\033[0m\n' "$(basename "$f")"
    out=$(bash "$f" 2>&1); rc=$?
    printf '%s\n' "$out" | grep -v '^##RESULT'
    res=$(printf '%s\n' "$out" | grep '^##RESULT' | tail -1)
    p=$(sed -n 's/.*pass=\([0-9]*\).*/\1/p' <<<"$res"); fl=$(sed -n 's/.*fail=\([0-9]*\).*/\1/p' <<<"$res")
    TOTAL_PASS=$((TOTAL_PASS + ${p:-0})); TOTAL_FAIL=$((TOTAL_FAIL + ${fl:-0}))
    if (( rc != 0 )) || [[ -z "$res" ]]; then
        FILES_FAILED+=("$(basename "$f")")
        [[ -z "$res" ]] && { TOTAL_FAIL=$((TOTAL_FAIL + 1)); printf '  \033[31mFAIL\033[0m %s did not finish (crashed?)\n' "$(basename "$f")"; }
    fi
done

printf '\n\033[1m=== TOTAL: %d passed, %d failed ===\033[0m\n' "$TOTAL_PASS" "$TOTAL_FAIL"
if (( TOTAL_FAIL )); then printf 'failing files: %s\n' "${FILES_FAILED[*]}"; exit 1; fi
