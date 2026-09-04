#!/bin/bash
# Shared helpers for every tests/test_*.sh. Source this file; do not run it.
#
# Every test builds a throwaway tree under mktemp -d, points the script under test
# at it through its *_DIRS / *_LOG / *_LOCK environment seams, and captures mail
# in a file. Nothing here touches /scratch, /var/log, /run, or a mail server.

# Never as root. Root ignores the file permissions several tests depend on,
# and one installer test that expects "refuses without root" would instead
# install onto the real host. install.sh drops to $SUDO_USER for the tests.
if (( EUID == 0 )); then
    echo "REFUSING TO RUN TESTS AS ROOT. Run them as a normal user: tests/run_tests.sh" >&2
    exit 2
fi

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$TESTS_DIR/.." && pwd)"          # the scripts/ project root
STUBS="$TESTS_DIR/stubs"
PASS=0; FAIL=0; FAILED_NAMES=()

# Scripts under test must use the REPO copy of the shared library, never an
# installed one, so a change here is what gets tested.
export RUQOLA_ADMIN_LIB="$ROOT/lib"

# --- assertions ------------------------------------------------------------
ok()   { PASS=$((PASS+1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); FAILED_NAMES+=("$1"); printf '  \033[31mFAIL\033[0m %s\n         %s\n' "$1" "$2"; }
# check <name> <actual> <expected>
check(){ [[ "$2" == "$3" ]] && ok "$1" || bad "$1" "expected [$3], got [$2]"; }
# check_contains <name> <haystack> <needle>
check_contains(){ [[ "$2" == *"$3"* ]] && ok "$1" || bad "$1" "expected to contain [$3], got [$2]"; }
t() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# Print the summary and exit non-zero on any failure. Call last.
finish() {
    printf '\n\033[1m%d passed, %d failed\033[0m\n' "$PASS" "$FAIL"
    if (( FAIL )); then printf 'failing: %s\n' "${FAILED_NAMES[*]}"; fi
    printf '##RESULT pass=%d fail=%d\n' "$PASS" "$FAIL"
    (( FAIL == 0 ))
}

# --- sandbox ----------------------------------------------------------------
new_sandbox() {
    SB=$(mktemp -d); export SB
    mkdir -p "$SB/scratch/users" "$SB/scratch/temp" "$SB/scratch/shared" "$SB/log" "$SB/run"
    export MAIL_CAPTURE="$SB/mail.txt"; : > "$MAIL_CAPTURE"
    export FAKE_PASSWD="$SB/passwd";    : > "$FAKE_PASSWD"
    export FAKE_OWNER="$SB/owners";     : > "$FAKE_OWNER"
    export STUB_CALLS="$SB/calls.txt";  : > "$STUB_CALLS"   # what stubbed commands were asked to do
}
drop_sandbox() { [[ -n "${SB:-}" && -d "$SB" ]] && rm -rf "$SB"; unset SB; }

# add_user <name> <gecos>      -- an account the fake getent will report
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

# --- mail capture (from stubs/msmtp) ---------------------------------------
mail_count()      { grep -c '^=== SEND' "$MAIL_CAPTURE" 2>/dev/null || true; }
mail_recipients() { grep '^=== SEND' "$MAIL_CAPTURE" 2>/dev/null | sed 's/.*recipient=\[\(.*\)\]/\1/'; }
mail_bodies()     { cat "$MAIL_CAPTURE" 2>/dev/null; }

# --- running things with the stubs first on PATH ---------------------------
# with_stubs <cmd...>   -- run a command with tests/stubs ahead of everything
with_stubs() { PATH="$STUBS:$PATH" "$@"; }

# --- the reaper under test --------------------------------------------------
REAPER="${REAPER:-$ROOT/bin/scratch-cleanup.sh}"

# assert_has_seam <script> <seam-name>: refuse to run a script that would
# operate on the real system because it lacks the sandbox seam.
assert_has_seam() {
    if ! grep -q "$2" "$1"; then
        echo "REFUSING TO RUN: $1 has no $2 seam; it would target the real system." >&2
        exit 2
    fi
}

# run_reaper: run bin/scratch-cleanup.sh against the sandbox with all seams set.
# Honour KEEP, NOTIFY, DRYRUN, and EXTRA_ARGS from the caller's environment.
run_reaper() {
    assert_has_seam "$REAPER" SCRATCH_CLEANUP_DIRS
    PATH="$STUBS:$PATH" \
    SCRATCH_CLEANUP_DIRS="${DIRS:-$SB/scratch/shared:$SB/scratch/temp:$SB/scratch/users}" \
    SCRATCH_CLEANUP_LOG="$SB/log/cleanup.log" \
    SCRATCH_CLEANUP_MANIFEST_DIR="${MANIFEST_DIR:-$SB/log/manifest}" \
    SCRATCH_CLEANUP_LOCK="$SB/run/cleanup.lock" \
    SCRATCH_CLEANUP_MSMTP="$STUBS/msmtp" \
    SCRATCH_CLEANUP_DAYS_KEEP="${KEEP:-180}" \
    SCRATCH_CLEANUP_DAYS_NOTIFY="${NOTIFY:-166}" \
    SCRATCH_CLEANUP_DRYRUN="${DRYRUN:-}" \
        bash "$REAPER" "$@" >"$SB/stdout.txt" 2>"$SB/stderr.txt"
    echo $? > "$SB/exit.txt"
    return 0
}
reaper_exit() { cat "$SB/exit.txt"; }
reaper_log()  { cat "$SB/log/cleanup.log" 2>/dev/null; }
present() { [[ -e "$1" ]] && echo present || echo gone; }
