#!/bin/bash
# The user-facing docs must quote the cleaner's numbers and rule, or say nothing.
# If the policy changes, this test fails until the docs are updated.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
DOCS="$ROOT/../docs"
KEEP=$(bash "$REAPER" --show-config | sed -n 's/^DAYS_TO_KEEP=//p')
NOTIFY=$(bash "$REAPER" --show-config | sed -n 's/^DAYS_TO_NOTIFY=//p')
GAP=$((KEEP - NOTIFY))

for doc in scratch-folder.md notifications-faq.md; do
    t "docs/$doc quotes the live policy ($KEEP/$NOTIFY)"
    f="$DOCS/$doc"
    check "mentions $KEEP days" "$(( $(grep -c "$KEEP days" "$f") > 0 ))" "1"
    check "mentions $NOTIFY days" "$(( $(grep -c "$NOTIFY days" "$f") > 0 ))" "1"
    stray=$(grep -oE '\b[0-9]+ days?\b' "$f" | sort -u | grep -vE "^($KEEP|$NOTIFY|$GAP) days?$" | tr '\n' ' ')
    check "no other 'N days' figure (found: '${stray}')" "$stray" ""
    check "does not claim noatime" "$(grep -ci 'noatime' "$f")" "0"
    check "does not claim mtime-only" "$(grep -ci 'mtime) only\|modification time (mtime)\*\* only' "$f")" "0"
done

t "docs/users-creation.md links resolve to files in this project"
for link in $(grep -oE '\]\(/scripts/[^)]+\)' "$DOCS/users-creation.md" | sed 's/^](\/scripts\///; s/)$//'); do
    check "scripts/$link exists" "$(present "$ROOT/$link")" "present"
done

finish
