#!/bin/bash
# install.sh -- put this project onto the host, or tell you how far the host has drifted.
#
#   ./install.sh --check          compare every MANIFEST entry with the live system (no root)
#   ./install.sh --diff           the same, plus unified diffs of what differs
#   sudo ./install.sh             tests -> check -> confirm -> install -> verify
#   sudo ./install.sh --yes       no confirmation prompt
#   sudo ./install.sh --only X    limit to entries whose destination basename is X
#
# Every replaced or retired file is copied to /var/backups/ruqola-admin/<stamp>/
# first (never next to the target), and that folder gets a rollback.sh.
#
# Seams used by the tests: RUQOLA_ADMIN_DESTROOT (prefix for every destination),
# RUQOLA_ADMIN_BACKUP_ROOT, RUQOLA_ADMIN_MANIFEST, RUQOLA_ADMIN_SKIP_TESTS=1,
# RUQOLA_ADMIN_NO_CHOWN=1; systemctl is found on PATH.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${RUQOLA_ADMIN_MANIFEST:-$HERE/MANIFEST}"
DESTROOT="${RUQOLA_ADMIN_DESTROOT:-}"
BACKUP_ROOT="${RUQOLA_ADMIN_BACKUP_ROOT:-$DESTROOT/var/backups/ruqola-admin}"
STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$STAMP"
MODE=install; ASSUME_YES=0; ONLY=""
CHANGED=0; UNITS_CHANGED=0; REAPER_CHANGED=0
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

say(){ printf '%s\n' "$*"; }
step(){ printf '\n\033[1m== %s\033[0m\n' "$*"; }
die(){ printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) MODE=check; shift ;;
        --diff)  MODE=diff; shift ;;
        --yes)   ASSUME_YES=1; shift ;;
        --only)  ONLY="${2:-}"; [[ -n "$ONLY" ]] || die "--only needs a name"; shift 2 ;;
        -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------
# read_manifest -> one "kind|src|dst|mode|owner" per line, validated.
read_manifest() {
    local kind src dst mode owner _rest n=0
    [[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST"
    while read -r kind src dst mode owner _rest; do
        [[ -z "$kind" || "$kind" == \#* ]] && continue
        n=$((n + 1))
        case "$kind" in
            file|unit|render)
                [[ -f "$HERE/$src" ]] || die "MANIFEST entry $n: source missing: $src"
                [[ "$mode" =~ ^0[0-7]{3}$ ]] || die "MANIFEST entry $n: mode must be 4 octal digits, got '$mode'"
                [[ "$owner" == ?*:?* ]] || die "MANIFEST entry $n: owner must be user:group, got '$owner'" ;;
            link|retire) ;;
            *) die "MANIFEST entry $n: unknown kind '$kind'" ;;
        esac
        [[ "$dst" == /* ]] || die "MANIFEST entry $n: destination must be absolute: '$dst'"
        [[ -z "$ONLY" && -n "$dst" ]] || [[ "$(basename "$dst")" == "$ONLY" ]] || continue
        printf '%s|%s|%s|%s|%s\n' "$kind" "$src" "$dst" "$mode" "$owner"
    done < "$MANIFEST"
}

ENTRIES="$TMP/entries"
read_manifest > "$ENTRIES" || exit 1

# ---------------------------------------------------------------------------
# Comparing one entry with the host
# ---------------------------------------------------------------------------
is_bash_script() { [[ "$(head -c 20 "$1" 2>/dev/null)" == '#!/bin/bash'* || "$(head -c 20 "$1" 2>/dev/null)" == '#!/usr/bin/env bash'* ]]; }
strip_generated() { grep -v '^Generated:' "$1"; }
# expected_content <kind> <src> -> path of a file holding what dst should contain
expected_content() {
    case "$1" in
        render) bash "$HERE/$2" > "$TMP/render.$$" 2>/dev/null || return 1; printf '%s' "$TMP/render.$$" ;;
        *)      printf '%s' "$HERE/$2" ;;
    esac
}
same_content() {   # <kind> <expected-file> <target>
    if [[ "$1" == render ]]; then diff -q <(strip_generated "$2") <(strip_generated "$3") >/dev/null
    else cmp -s "$2" "$3"; fi
}
same_mode()  { [[ $(( 8#$(stat -c %a "$2") )) -eq $(( 8#$1 )) ]]; }
same_owner() { [[ -n "${RUQOLA_ADMIN_NO_CHOWN:-}" ]] || [[ "$(stat -c '%U:%G' "$2")" == "$1" ]]; }

# entry_status <kind> <src> <dst> <mode> <owner> -> same|missing|differs|mode|owner|present|absent
entry_status() {
    local kind="$1" src="$2" dst="$3" mode="$4" owner="$5" target="$DESTROOT$3" exp f
    case "$kind" in
        file|unit|render)
            [[ -e "$target" ]] || { echo missing; return; }
            exp=$(expected_content "$kind" "$src") || { echo render-failed; return; }
            same_content "$kind" "$exp" "$target" || { echo differs; return; }
            same_mode "$mode" "$target"   || { echo mode; return; }
            same_owner "$owner" "$target" || { echo owner; return; }
            echo same ;;
        link)
            if [[ -L "$target" ]]; then
                [[ "$(readlink "$target")" == "$src" ]] && echo same || echo differs
            elif [[ -e "$target" ]]; then echo differs
            else echo missing; fi ;;
        retire)
            for f in $DESTROOT$dst; do [[ -e "$f" || -L "$f" ]] && { echo present; return; }; done
            echo absent ;;
    esac
}

# ---------------------------------------------------------------------------
# Backups and rollback
# ---------------------------------------------------------------------------
backup_init() {
    [[ -d "$BACKUP_DIR" ]] && return 0
    mkdir -p "$BACKUP_DIR/retired" || die "cannot create backup folder $BACKUP_DIR"
    cp "$MANIFEST" "$BACKUP_DIR/MANIFEST"
    { echo "installed: $STAMP"; echo "from: $HERE"; echo "git: $(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"; } > "$BACKUP_DIR/PROVENANCE"
    printf '#!/bin/bash\n# Undo the ruqola-admin install of %s. Run as root.\nset -e\n' "$STAMP" > "$BACKUP_DIR/rollback.sh"
    chmod 700 "$BACKUP_DIR/rollback.sh"
}
rollback_add() { printf '%s\n' "$*" >> "$BACKUP_DIR/rollback.sh"; }
# backup_path <target>: copy it (or the symlink itself) under the backup folder
backup_path() {
    local target="$1" dest="$BACKUP_DIR/files${1#"$DESTROOT"}"
    backup_init
    mkdir -p "$(dirname "$dest")"
    cp -a "$target" "$dest" || die "backup failed for $target"
    rollback_add "rm -rf '$target' && cp -a '$dest' '$target'"
    say "     backed up -> $dest"
}

# ---------------------------------------------------------------------------
# Applying one entry
# ---------------------------------------------------------------------------
install_atomic() {   # <src> <target> <mode> <owner>
    local tmp="$2.ruqola-admin.$$"
    mkdir -p "$(dirname "$2")" || return 1
    if [[ -n "${RUQOLA_ADMIN_NO_CHOWN:-}" ]]; then install -m "$3" "$1" "$tmp" || return 1
    else install -m "$3" -o "${4%%:*}" -g "${4##*:}" "$1" "$tmp" || return 1; fi
    mv -f "$tmp" "$2"
}

apply_entry() {
    local kind="$1" src="$2" dst="$3" mode="$4" owner="$5" target="$DESTROOT$3" st exp f
    st=$(entry_status "$@")
    case "$kind" in
        file|unit|render)
            [[ "$st" == same ]] && { say "   same       $dst"; return 0; }
            [[ "$st" == render-failed ]] && die "generator failed: $src"
            exp=$(expected_content "$kind" "$src") || die "generator failed: $src"
            if [[ -e "$target" ]]; then backup_path "$target"; else backup_init; rollback_add "rm -f '$target'"; fi
            install_atomic "$exp" "$target" "$mode" "$owner" || die "install failed: $dst"
            same_content "$kind" "$exp" "$target" || die "verify failed after install: $dst"
            if is_bash_script "$target"; then bash -n "$target" || die "syntax check failed: $dst"; fi
            [[ "$kind" == unit ]] && UNITS_CHANGED=1
            [[ "$(basename "$dst")" == scratch-cleanup.sh ]] && REAPER_CHANGED=1
            CHANGED=$((CHANGED + 1)); say "   $st -> installed  $dst" ;;
        link)
            [[ "$st" == same ]] && { say "   same       $dst"; return 0; }
            if [[ -e "$target" || -L "$target" ]]; then backup_path "$target"; else backup_init; rollback_add "rm -f '$target'"; fi
            mkdir -p "$(dirname "$target")"
            ln -sfn "$src" "$target" || die "link failed: $dst"
            CHANGED=$((CHANGED + 1)); say "   $st -> linked     $dst -> $src" ;;
        retire)
            [[ "$st" == absent ]] && { say "   absent     $dst"; return 0; }
            backup_init
            for f in $DESTROOT$dst; do
                [[ -e "$f" || -L "$f" ]] || continue
                mv "$f" "$BACKUP_DIR/retired/$(basename "$f")" || die "could not retire $f"
                rollback_add "mv '$BACKUP_DIR/retired/$(basename "$f")' '$f'"
                CHANGED=$((CHANGED + 1)); say "   retired    $f -> $BACKUP_DIR/retired/"
            done ;;
    esac
}

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
# check_all [diff] -> prints a status line per entry; returns 1 if anything is off
check_all() {
    local line kind src dst mode owner st bad=0 exp
    while IFS='|' read -r kind src dst mode owner; do
        st=$(entry_status "$kind" "$src" "$dst" "$mode" "$owner")
        case "$st" in
            same|absent) printf '   %-10s %s\n' "$st" "$dst" ;;
            present)     printf '   \033[33m%-10s\033[0m %s   (to retire)\n' "PRESENT" "$dst"; bad=1 ;;
            *)           printf '   \033[31m%-10s\033[0m %s\n' "${st^^}" "$dst"; bad=1
                         if [[ "${1:-}" == diff && "$st" == differs && "$kind" != link ]]; then
                             exp=$(expected_content "$kind" "$src") && diff -u --label "live:$dst" --label "repo:$src" "$DESTROOT$dst" "$exp" | sed 's/^/       /'
                         fi ;;
        esac
    done < "$ENTRIES"
    return $bad
}

run_tests() {
    [[ -n "${RUQOLA_ADMIN_SKIP_TESTS:-}" ]] && { say "   (tests skipped by request)"; return 0; }
    local log="$TMP/tests.log" -a runner=()
    # The suite refuses to run as root (see tests/lib.sh). Under sudo, run it as
    # the person who typed sudo; as a bare root login there is nobody to drop to.
    if (( EUID == 0 )); then
        [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != root ]] \
            || die "cannot run the tests as root and no SUDO_USER to drop to; run tests/run_tests.sh as a user, then re-run with RUQOLA_ADMIN_SKIP_TESTS=1"
        if command -v runuser >/dev/null; then runner=(runuser -u "$SUDO_USER" --); else runner=(sudo -u "$SUDO_USER" --); fi
        say "   running as $SUDO_USER"
    fi
    if "${runner[@]}" "$HERE/tests/run_tests.sh" >"$log" 2>&1; then tail -1 "$log" | sed 's/^/   /'
    else cat "$log"; die "tests failed; nothing installed"; fi
}

case "$MODE" in
    check) step "Live system vs $MANIFEST"; check_all && { say ""; say "   everything matches"; exit 0; } || { say ""; say "   drift found; 'sudo $0' installs, '$0 --diff' shows details"; exit 1; } ;;
    diff)  step "Live system vs $MANIFEST"; check_all diff; exit $? ;;
esac

# --- install ---------------------------------------------------------------
[[ $EUID -eq 0 || -n "$DESTROOT" ]] || die "installing needs root:  sudo $0"

step "1. Tests"
run_tests

step "2. What would change"
if check_all; then say ""; say "   nothing to do: the host already matches the manifest"; exit 0; fi

if (( REAPER_CHANGED == 0 )); then
    # Decide whether the reaper (or its library) is among the changes; if so,
    # show what tonight's run would do before touching anything.
    while IFS='|' read -r kind src dst mode owner; do
        st=$(entry_status "$kind" "$src" "$dst" "$mode" "$owner")
        [[ "$st" != same && ( "$src" == bin/scratch-cleanup.sh || "$src" == lib/* ) ]] && REAPER_CHANGED=1
    done < "$ENTRIES"
fi
if (( REAPER_CHANGED )) && [[ -z "$DESTROOT" ]]; then
    step "3. The reaper is changing: DRY RUN against the real /scratch (deletes nothing, emails nobody)"
    DRYLOG="$TMP/dry.log"
    SCRATCH_CLEANUP_DRYRUN=1 SCRATCH_CLEANUP_LOG="$DRYLOG" SCRATCH_CLEANUP_LOCK="$TMP/dry.lock" bash "$HERE/bin/scratch-cleanup.sh" >/dev/null 2>&1 || true
    say "   would remove : $(grep -c 'DRY RUN would remove:' "$DRYLOG" || true) file(s)"
    say "   would rmdir  : $(grep -c 'DRY RUN would remove empty directory' "$DRYLOG" || true) directory(ies)"
    say "   would email  : $(grep -c 'DRY RUN would email' "$DRYLOG" || true) message(s)"
    grep 'DRY RUN would email' "$DRYLOG" | sed 's/.*would email /     /' | sort -u
    grep -o "No email address on account '[^']*'" "$DRYLOG" | sort -u | sed 's/^/     /'
else
    step "3. Reaper unchanged; skipping the live dry run"
fi

if (( ! ASSUME_YES )); then
    printf '\n   Install? type yes: '; read -r reply
    [[ "$reply" == "yes" ]] || { say "   aborted, nothing changed"; exit 1; }
fi

step "4. Installing (backups -> $BACKUP_DIR)"
while IFS='|' read -r kind src dst mode owner; do
    apply_entry "$kind" "$src" "$dst" "$mode" "$owner"
done < "$ENTRIES"

step "5. After-install"
if (( UNITS_CHANGED )); then
    systemctl daemon-reload && say "   systemctl daemon-reload done (a unit changed)" || die "daemon-reload failed"
else
    say "   no unit changed; no daemon-reload needed"
fi
if [[ -z "$DESTROOT" ]] && command -v logrotate >/dev/null; then
    logrotate -d /etc/logrotate.d/scratch-cleanup >/dev/null 2>&1 && say "   logrotate config parses" || say "   WARNING: logrotate -d rejected /etc/logrotate.d/scratch-cleanup"
fi

step "6. Verify"
if check_all; then say ""; say "   verified: host matches the manifest ($CHANGED change(s))"
else die "verification failed after install; see above. Rollback: $BACKUP_DIR/rollback.sh"; fi

if [[ -d "$BACKUP_DIR" ]]; then
    say ""
    say "   backups + rollback: $BACKUP_DIR"
    say "   to undo:            sudo $BACKUP_DIR/rollback.sh"
fi
