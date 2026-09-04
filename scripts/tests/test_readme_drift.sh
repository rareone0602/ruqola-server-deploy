#!/bin/bash
# tools/render-readme.sh must quote the cleaner's numbers, never its own.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RENDER="$ROOT/tools/render-readme.sh"

t "Anti-drift: the README is generated from --show-config"
keep=$(SCRATCH_CLEANUP_DAYS_KEEP=123 SCRATCH_CLEANUP_DAYS_NOTIFY=100 bash "$REAPER" --show-config | sed -n 's/^DAYS_TO_KEEP=//p')
check "cleaner reports the overridden value" "$keep" "123"

new_sandbox
cat > "$SB/fake-cleanup" <<FAKE
#!/bin/bash
[[ "\$1" == "--show-config" ]] && { echo DAYS_TO_KEEP=123; echo DAYS_TO_NOTIFY=100; exit 0; }
FAKE
chmod +x "$SB/fake-cleanup"
readme=$(SCRATCH_CLEANUP_BIN="$SB/fake-cleanup" bash "$RENDER")
check "README quotes 123, not a hardcoded number" "$(grep -c 'gone 123 days' <<<"$readme")" "1"
check "README computes the warning gap from both values" "$(grep -c 'about 23 days before removal' <<<"$readme")" "1"
check "README no longer claims 30 days" "$(grep -c '30 days' <<<"$readme")" "0"
check "README says both reads and writes count" "$(grep -c 'NO READ and NO WRITE' <<<"$readme")" "1"
drop_sandbox

t "Rendering to a file"
new_sandbox
SCRATCH_CLEANUP_BIN="$REAPER" SCRATCH_README="$SB/README.txt" bash "$RENDER" --write >/dev/null 2>&1
check "--write creates the file" "$(present "$SB/README.txt")" "present"
check "with the live policy numbers" "$(grep -c 'gone 180 days' "$SB/README.txt")" "1"
drop_sandbox

finish
