#!/bin/bash
# Hermetic test harness for scratch-cleanup.sh.
# Never touches /scratch, /var/log, /run, or sends real mail.
# Usage: tests/run_tests.sh [path-to-script]

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${1:-$HERE/../scratch-cleanup.sh}"
STUBS="$HERE/stubs"
PASS=0; FAIL=0; FAILED_NAMES=()

# --- SAFETY: refuse to run a script that does not honour the sandbox seams.
# Without these the script would operate on the real /scratch as root-owned paths.
if ! grep -q 'SCRATCH_CLEANUP_DIRS' "$SCRIPT"; then
    echo "REFUSING TO RUN: $SCRIPT has no SCRATCH_CLEANUP_DIRS seam." >&2
    echo "It would target the real /scratch, /var/log and /run. Aborting." >&2
    exit 2
fi

# --- sandbox helpers -------------------------------------------------------
new_sandbox() {
    SB=$(mktemp -d); export SB
    mkdir -p "$SB/scratch/users" "$SB/scratch/temp" "$SB/scratch/shared" "$SB/log" "$SB/run"
    export MAIL_CAPTURE="$SB/mail.txt"; : > "$MAIL_CAPTURE"
    export FAKE_PASSWD="$SB/passwd";    : > "$FAKE_PASSWD"
    export FAKE_OWNER="$SB/owners";     : > "$FAKE_OWNER"
}
# add_user <name> <gecos>
add_user() { printf '%s:x:1000:1000:%s:/home/%s:/bin/bash\n' "$1" "$2" "$1" >> "$FAKE_PASSWD"; }
# mkfile <relpath-under-scratch> <age_days_atime> <age_days_mtime> <owner>
mkfile() {
    local p="$SB/scratch/$1" a="$2" m="$3" o="$4"
    mkdir -p "$(dirname "$p")"; echo "data-$RANDOM" > "$p"
    touch -m -d "$(date -d "$m days ago" '+%Y-%m-%d %H:%M:%S')" "$p"
    touch -a -d "$(date -d "$a days ago" '+%Y-%m-%d %H:%M:%S')" "$p"
    printf '%s\t%s\n' "$p" "$o" >> "$FAKE_OWNER"
}
# mkdir_aged <relpath-under-scratch> <age_days>
mkdir_aged() {
    local d="$SB/scratch/$1"
    mkdir -p "$d"
    touch -d "$(date -d "$2 days ago" '+%Y-%m-%d %H:%M:%S')" "$d"
}
run_reaper() {
    PATH="$STUBS:$PATH" \
    SCRATCH_CLEANUP_DIRS="$SB/scratch/shared:$SB/scratch/temp:$SB/scratch/users" \
    SCRATCH_CLEANUP_LOG="$SB/log/cleanup.log" \
    SCRATCH_CLEANUP_LOCK="$SB/run/cleanup.lock" \
    SCRATCH_CLEANUP_MSMTP="$STUBS/msmtp" \
    SCRATCH_CLEANUP_DAYS_KEEP="${KEEP:-180}" \
    SCRATCH_CLEANUP_DAYS_NOTIFY="${NOTIFY:-166}" \
    SCRATCH_CLEANUP_DRYRUN="${DRYRUN:-}" \
        bash "$SCRIPT" >"$SB/stdout.txt" 2>"$SB/stderr.txt"
    return 0
}
mail_count()      { grep -c '^=== SEND' "$MAIL_CAPTURE" 2>/dev/null || true; }
mail_recipients() { grep '^=== SEND' "$MAIL_CAPTURE" 2>/dev/null | sed 's/.*recipient=\[\(.*\)\]/\1/'; }
mail_bodies()     { cat "$MAIL_CAPTURE" 2>/dev/null; }

# --- assertions ------------------------------------------------------------
ok()   { PASS=$((PASS+1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); FAILED_NAMES+=("$1"); printf '  \033[31mFAIL\033[0m %s\n         %s\n' "$1" "$2"; }
check(){ [[ "$2" == "$3" ]] && ok "$1" || bad "$1" "expected [$3], got [$2]"; }

t() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# ===========================================================================
t "Parser: address must be found wherever it sits in the GECOS field"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/old.bin 200 200 alice
run_reaper
check "five-field GECOS -> address extracted" "$(mail_recipients | sort -u | head -1)" "alice@ntu.edu.sg"
rm -rf "$SB"

new_sandbox
add_user helen "helenfan27@gmail.com"
mkfile users/helen/old.bin 200 200 helen
run_reaper
check "bare-email GECOS -> address extracted" "$(mail_recipients | sort -u | head -1)" "helenfan27@gmail.com"
rm -rf "$SB"

new_sandbox
add_user nobody "No Email Person"
mkfile users/nobody/old.bin 200 200 nobody
run_reaper
check "GECOS with no address -> zero mails sent" "$(mail_count)" "0"
rm -rf "$SB"

# ===========================================================================
t "Volume: one digest per user per run, not one mail per file"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
for i in 1 2 3 4 5; do mkfile "users/alice/old$i.bin" 200 200 alice; done
run_reaper
check "5 deletable files -> 1 email" "$(mail_count)" "1"
rm -rf "$SB"

# ===========================================================================
t "Ordering: a file must not be warned and deleted in the same run"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/doomed.bin 200 200 alice
run_reaper
check "deletable file produces no 'Imminent' warning" \
      "$(mail_bodies | grep -ci 'imminent')" "0"
rm -rf "$SB"

# ===========================================================================
t "Warning window: only files between NOTIFY and KEEP get warned"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/inwindow.bin 170 170 alice
run_reaper
check "file aged 170d (166<x<=180) -> warned" "$(mail_bodies | grep -ci 'imminent')" "1"
rm -rf "$SB"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/fresh.bin 10 10 alice
run_reaper
check "file aged 10d -> no mail at all" "$(mail_count)" "0"
rm -rf "$SB"

# ===========================================================================
t "Warning text: countdown must never be negative, no hardcoded 23"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/inwindow.bin 179 179 alice
run_reaper
check "no negative countdown in warning" "$(mail_bodies | grep -c 'within -')" "0"
check "notification period not hardcoded to 23" \
      "$(mail_bodies | grep -c 'notification period of 23 days')" "0"
rm -rf "$SB"

# ===========================================================================
t "Safety: dry-run deletes nothing"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/old.bin 200 200 alice
DRYRUN=1 run_reaper
check "dry-run leaves the file on disk" "$([[ -f $SB/scratch/users/alice/old.bin ]] && echo present || echo gone)" "present"
check "dry-run sends NO email" "$(mail_count)" "0"
check "dry-run still logs what it would send" \
      "$(grep -c 'DRY RUN would email' "$SB/log/cleanup.log" 2>/dev/null || true)" "1"
unset DRYRUN
rm -rf "$SB"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/inwindow.bin 170 170 alice
DRYRUN=1 run_reaper
check "dry-run sends no WARNING email either" "$(mail_count)" "0"
unset DRYRUN
rm -rf "$SB"

# ===========================================================================
t "Regression: a normal deletion still happens and still notifies"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/old.bin 200 200 alice
run_reaper
check "deletable file is actually removed" "$([[ -f $SB/scratch/users/alice/old.bin ]] && echo present || echo gone)" "gone"
check "and the owner is told once" "$(mail_count)" "1"
rm -rf "$SB"

# ===========================================================================
t "Directories: the same age rule applies to empty directories as to files"

new_sandbox
mkdir_aged users/alice/fresh_empty 0
run_reaper
check "empty dir created today is KEPT" \
      "$([[ -d $SB/scratch/users/alice/fresh_empty ]] && echo present || echo gone)" "present"
rm -rf "$SB"

new_sandbox
mkdir_aged users/alice/stale_empty 200
run_reaper
check "empty dir older than DAYS_TO_KEEP is removed" \
      "$([[ -d $SB/scratch/users/alice/stale_empty ]] && echo present || echo gone)" "gone"
rm -rf "$SB"

new_sandbox
mkdir_aged users/alice/borderline 100
run_reaper
check "empty dir aged 100d (< 180) is KEPT" \
      "$([[ -d $SB/scratch/users/alice/borderline ]] && echo present || echo gone)" "present"
rm -rf "$SB"

new_sandbox
mkdir_aged users/alice/fresh_empty 0
DRYRUN=1 run_reaper
check "dry-run removes no directories" \
      "$([[ -d $SB/scratch/users/alice/fresh_empty ]] && echo present || echo gone)" "present"
unset DRYRUN
rm -rf "$SB"

# ===========================================================================
t "SSOT: the script can report its own policy, and honours real flags"

new_sandbox
out=$(SCRATCH_CLEANUP_DIRS="$SB/scratch/users" bash "$SCRIPT" --show-config 2>&1)
check "--show-config reports DAYS_TO_KEEP"   "$(grep -c '^DAYS_TO_KEEP=180$'   <<<"$out")" "1"
check "--show-config reports DAYS_TO_NOTIFY" "$(grep -c '^DAYS_TO_NOTIFY=166$' <<<"$out")" "1"
rm -rf "$SB"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/old.bin 200 200 alice
out=$(SCRATCH_CLEANUP_DIRS="$SB/scratch/users" bash "$SCRIPT" --show-config 2>&1)
check "--show-config deletes nothing" "$([[ -f $SB/scratch/users/alice/old.bin ]] && echo present || echo gone)" "present"
rm -rf "$SB"

new_sandbox
KEEP=90 NOTIFY=80
out=$(SCRATCH_CLEANUP_DIRS="$SB/scratch/users" SCRATCH_CLEANUP_DAYS_KEEP=90 \
      SCRATCH_CLEANUP_DAYS_NOTIFY=80 bash "$SCRIPT" --show-config 2>&1)
check "--show-config reflects overrides" "$(grep -c '^DAYS_TO_KEEP=90$' <<<"$out")" "1"
unset KEEP NOTIFY
rm -rf "$SB"

new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
mkfile users/alice/old.bin 200 200 alice
PATH="$STUBS:$PATH" SCRATCH_CLEANUP_DIRS="$SB/scratch/users" \
  SCRATCH_CLEANUP_LOG="$SB/log/c.log" SCRATCH_CLEANUP_LOCK="$SB/run/c.lock" \
  SCRATCH_CLEANUP_MSMTP="$STUBS/msmtp" bash "$SCRIPT" --dry-run >/dev/null 2>&1
check "--dry-run flag works like the env var" "$([[ -f $SB/scratch/users/alice/old.bin ]] && echo present || echo gone)" "present"
rm -rf "$SB"

new_sandbox
SCRATCH_CLEANUP_DIRS="$SB/scratch/users" bash "$SCRIPT" --nonsense >/dev/null 2>&1
check "unknown flag is rejected (non-zero exit)" "$?" "2"
rm -rf "$SB"

# ===========================================================================
t "Anti-drift: the README and the usage report quote the cleaner's own numbers"

keep=$(SCRATCH_CLEANUP_DAYS_KEEP=123 SCRATCH_CLEANUP_DAYS_NOTIFY=100 \
       bash "$SCRIPT" --show-config | sed -n 's/^DAYS_TO_KEEP=//p')
check "cleaner reports the overridden value" "$keep" "123"

cat > "$HERE/../.fake-cleanup-$$" <<FAKE
#!/bin/bash
[[ "\$1" == "--show-config" ]] && { echo DAYS_TO_KEEP=123; echo DAYS_TO_NOTIFY=100; exit 0; }
FAKE
chmod +x "$HERE/../.fake-cleanup-$$"
readme=$(SCRATCH_CLEANUP_BIN="$HERE/../.fake-cleanup-$$" bash "$HERE/../render-readme.sh")
check "README quotes 123, not a hardcoded number" "$(grep -c 'gone 123 days' <<<"$readme")" "1"
check "README computes the warning gap from both values" "$(grep -c 'about 23 days before removal' <<<"$readme")" "1"
check "README no longer claims 30 days" "$(grep -c '30 days' <<<"$readme")" "0"
check "README no longer says 'not accessed' only" "$(grep -c 'NO READ and NO WRITE' <<<"$readme")" "1"
rm -f "$HERE/../.fake-cleanup-$$"

# ===========================================================================
printf '\n\033[1m%d passed, %d failed\033[0m\n' "$PASS" "$FAIL"
if (( FAIL )); then printf 'failing: %s\n' "${FAILED_NAMES[*]}"; exit 1; fi
