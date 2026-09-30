#!/bin/bash
# scratch-cleanup.sh -- nightly reaper for /scratch
# Runs as root from scratch-cleanup.timer (02:00, see ../systemd/).
#
# A file is deleted once it has gone DAYS_TO_KEEP days with no read AND no write.
# Its owner is emailed once per run: a warning digest when files enter the
# window after DAYS_TO_NOTIFY days, and a deletion digest when they go. Every
# deletion is appended to a monthly manifest under MANIFEST_DIR; that record is
# never rotated. The chatty log at LOG_FILE is rotated by logrotate
# (../logrotate.d/scratch-cleanup), not by this script.
#
# THIS FILE IS THE SINGLE SOURCE OF TRUTH for the retention numbers. Anything
# that quotes them (scratch-usage.sh, /scratch/README.txt, the docs) must read
# them from `--show-config`. Tests enforce this.
#
# Layout: settings -> shared library -> policy -> command line.
set -uo pipefail

# ---------------------------------------------------------------------------
# Settings. Each can be overridden by a SCRATCH_CLEANUP_* variable. Production
# leaves them unset; the tests point them at a throwaway tree.
# ---------------------------------------------------------------------------
if [[ -n "${SCRATCH_CLEANUP_DIRS:-}" ]]; then
    IFS=':' read -r -a SCRATCH_DIRS <<< "$SCRATCH_CLEANUP_DIRS"
else
    SCRATCH_DIRS=(
        /scratch/shared
        /scratch/temp
        # /scratch/datasets is exempt on purpose. Never list it here.
        /scratch/users
    )
fi
DAYS_TO_KEEP="${SCRATCH_CLEANUP_DAYS_KEEP:-180}"
DAYS_TO_NOTIFY="${SCRATCH_CLEANUP_DAYS_NOTIFY:-166}"
LOG_FILE="${SCRATCH_CLEANUP_LOG:-/var/log/scratch-cleanup.log}"
MANIFEST_DIR="${SCRATCH_CLEANUP_MANIFEST_DIR:-/var/log/scratch-cleanup}"
LOCK_FILE="${SCRATCH_CLEANUP_LOCK:-/run/scratch-cleanup.lock}"
MSMTP="${SCRATCH_CLEANUP_MSMTP:-/usr/bin/msmtp}"
ADMIN_EMAIL="${SCRATCH_CLEANUP_ADMIN_EMAIL:-mjolnirruqola@gmail.com}"
DRY_RUN="${SCRATCH_CLEANUP_DRYRUN:-}"

# Structural directories: never removed, even when empty.
PRESERVE_DIRS=(/scratch/shared /scratch/temp /scratch/datasets /scratch/users)

# --- shared library: lib/log.sh lib/mail.sh lib/fs.sh -----------------------
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${RUQOLA_ADMIN_LIB:-}" ]]; then
    if [[ -f "$_here/../lib/init.sh" ]]; then RUQOLA_ADMIN_LIB="$_here/../lib"   # running from the repo
    else RUQOLA_ADMIN_LIB=/usr/local/lib/ruqola-admin; fi                        # installed
fi
source "$RUQOLA_ADMIN_LIB/init.sh" || { echo "scratch-cleanup.sh: cannot load shared library from $RUQOLA_ADMIN_LIB" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Policy: the age tests, as find expressions. Nothing else in this file, and
# nothing outside it, may restate these.
# ---------------------------------------------------------------------------
EXPIRED=(         -type f \( -atime +"$DAYS_TO_KEEP" -a -mtime +"$DAYS_TO_KEEP" \) )
IN_WINDOW=(       -type f \( -atime +"$DAYS_TO_NOTIFY" -a -mtime +"$DAYS_TO_NOTIFY" \)
                  ! \( -atime +"$DAYS_TO_KEEP" -a -mtime +"$DAYS_TO_KEEP" \) )
STALE_EMPTY_DIR=( -type d -empty -mtime +"$DAYS_TO_KEEP" )

# Deletion happens INSIDE find (-delete), never by handing a path to rm or
# rmdir later. find removes each entry relative to the directory it is already
# in, so a user who swaps one of their directories for a symlink mid-run cannot
# steer a root deletion outside the scan (todo E3). -printf after -delete runs
# only when the delete succeeded; a failure is one "cannot delete" line on stderr.
# find_act <start> <find expression...>: run find on one scratch directory,
# printing a FIND_RECORD per match; LC_ALL=C keeps the stderr text parseable.
find_act() { LC_ALL=C find "$1" -ignore_readdir_race "${@:2}" -printf "$FIND_RECORD"; }

manifest_file() { local m; printf -v m '%(%Y-%m)T' -1; printf '%s/deleted-%s.tsv' "$MANIFEST_DIR" "$m"; }

validate_directory() {
    local dir="$1"
    [[ -d "$dir" ]] || { log_message WARN  "Directory does not exist: $dir"; return 1; }
    [[ -r "$dir" ]] || { log_message ERROR "No read permission for directory: $dir"; return 1; }
}

# find arguments that keep structural directories. Each user's top-level
# directory (a child of a scratch dir named "users") is kept too: callers pass
# -mindepth 2 there.
preserve_args() {
    local p
    for p in "${PRESERVE_DIRS[@]}"; do printf '%s\0' '!' -path "$p"; done
}

# ---------------------------------------------------------------------------
# Pass 1: warn owners about files inside the window (past NOTIFY, not yet KEEP)
# ---------------------------------------------------------------------------
notify_imminent_deletion() {
    local scratch_dir="$1"
    local -A warn_list warn_count warn_min
    validate_directory "$scratch_dir" || return 1

    local now owner size atime mtime path newest elapsed_days days_remaining
    printf -v now '%(%s)T' -1
    while read_record; do
        [[ -n "$owner" ]] || continue
        # Deletion needs BOTH timestamps past the limit, so the countdown is
        # governed by the NEWER of the two.
        if (( atime > mtime )); then newest=$atime; else newest=$mtime; fi
        elapsed_days=$(( (now - newest) / 86400 ))
        days_remaining=$(( DAYS_TO_KEEP - elapsed_days ))
        (( days_remaining < 0 )) && days_remaining=0
        warn_list[$owner]+="  ${path} (${days_remaining} day(s) remaining)"$'\n'
        warn_count[$owner]=$(( ${warn_count[$owner]:-0} + 1 ))
        if [[ -z "${warn_min[$owner]:-}" ]] || (( days_remaining < warn_min[$owner] )); then
            warn_min[$owner]=$days_remaining
        fi
    done < <(find_act "$scratch_dir" "${IN_WINDOW[@]}" 2>/dev/null)

    local u to name mail_failed=0
    for u in "${!warn_list[@]}"; do
        to=$(email_for_user "$u"); name=$(full_name_for_user "$u")
        if [[ -z "$to" ]]; then
            log_message WARN "No email address on account '$u'; ${warn_count[$u]} warning(s) undelivered"
        elif send_mail "$to" "Imminent removal: ${warn_count[$u]} file(s) under ${scratch_dir}" "Hello ${name},

This is an automated notification from the server.

The following ${warn_count[$u]} file(s) have gone more than ${DAYS_TO_NOTIFY} days
without being modified or accessed. Files under /scratch/ (except
/scratch/datasets/) are removed after ${DAYS_TO_KEEP} days.

${warn_list[$u]}
To keep them, modify them or copy them somewhere permanent within
${warn_min[$u]} day(s).

Thank you,
System Administrator"
        then log_message INFO "Warning digest sent to $u <$to> (${warn_count[$u]} file(s))"
        else log_message ERROR "Mail to $u <$to> failed; ${warn_count[$u]} warning(s) undelivered"; mail_failed=1
        fi
    done
    return "$mail_failed"
}

# ---------------------------------------------------------------------------
# Pass 2: delete expired files and stale empty directories; record; report
# ---------------------------------------------------------------------------
clean_directory() {
    local scratch_dir="$1"
    local files_removed=0 dirs_removed=0 total_size=0 errors=0
    local -A user_list user_bytes user_count
    validate_directory "$scratch_dir" || return 1

    log_message INFO "Cleaning directory: $scratch_dir (older than $DAYS_TO_KEEP days)"
    [[ -n "$DRY_RUN" ]] && log_message WARN "DRY RUN active: nothing will be deleted"

    # No record, no deletion: a manifest we cannot write aborts this directory
    # before anything is removed.
    local manifest; manifest=$(manifest_file)
    if [[ -z "$DRY_RUN" ]] && ! manifest_open "$manifest"; then
        log_message ERROR "Cannot write deletion manifest $manifest; skipping $scratch_dir"
        return 1
    fi

    # find's stderr, kept to count the deletions that failed.
    local errlog; errlog=$(mktemp) || { log_message ERROR "mktemp failed; skipping $scratch_dir"; return 1; }
    local delete=(-delete); [[ -n "$DRY_RUN" ]] && delete=()

    local owner size atime mtime path line
    while read_record; do
        if [[ -n "$DRY_RUN" ]]; then
            log_message INFO "DRY RUN would remove: $path (${size} bytes)"
        else
            log_message INFO "Removed: $path (${size} bytes)"
            manifest_record "$manifest" file "$owner" "$size" "$atime" "$mtime" "$path" \
                || log_message ERROR "Deleted but could not record in $manifest: $path"
        fi
        files_removed=$((files_removed + 1)); total_size=$((total_size + size))
        if [[ -n "$owner" ]]; then
            user_list[$owner]+="  ${path} (${size} bytes)"$'\n'
            user_bytes[$owner]=$(( ${user_bytes[$owner]:-0} + size ))
            user_count[$owner]=$(( ${user_count[$owner]:-0} + 1 ))
        fi
    done < <(find_act "$scratch_dir" "${EXPIRED[@]}" "${delete[@]}" 2>"$errlog")
    wait "$!" 2>/dev/null
    while IFS= read -r line; do
        if [[ "$line" == *"cannot delete"* ]]; then
            log_message ERROR "Failed to remove: ${line#*: cannot delete }"; errors=$((errors + 1))
        else
            log_message WARN "find: ${line#*: }"
        fi
    done < "$errlog"

    # One digest per owner per run, never one message per file.
    local u to name
    for u in "${!user_list[@]}"; do
        to=$(email_for_user "$u"); name=$(full_name_for_user "$u")
        if [[ -z "$to" ]]; then
            log_message WARN "No email address on account '$u'; ${user_count[$u]} deletion(s) unreported"
        elif send_mail "$to" "Scratch cleanup: ${user_count[$u]} file(s) removed" "Hello ${name},

This is an automated notification from the server.

The following ${user_count[$u]} file(s) under ${scratch_dir} were removed because
they were neither modified nor accessed in the last ${DAYS_TO_KEEP} days
(total $(format_bytes "${user_bytes[$u]}")):

${user_list[$u]}
To keep files permanently, store them under /scratch/datasets/ or in your home
directory.

Thank you,
System Administrator"
        then log_message INFO "Deletion digest sent to $u <$to> (${user_count[$u]} file(s))"
        else log_message ERROR "Mail to $u <$to> failed; ${user_count[$u]} deletion(s) unreported"; errors=$((errors + 1))
        fi
    done

    # Empty directories age out on the same clock as files. Without the age
    # gate, a job that pre-created its output tree and was still running at
    # 02:00 lost it. find works bottom-up, so an old tree of empty directories
    # goes in one run (each one's age is read before its children are removed).
    log_message INFO "Removing empty directories in $scratch_dir older than $DAYS_TO_KEEP days"
    local mindepth=1 keep dir
    if [[ "${scratch_dir##*/}" == users ]]; then
        mindepth=2
        while IFS= read -r -d '' dir; do
            log_message DEBUG "Preserving structural directory: $dir"
        done < <(find "$scratch_dir" -mindepth 1 -maxdepth 1 "${STALE_EMPTY_DIR[@]}" -print0 2>/dev/null)
    fi
    mapfile -d '' -t keep < <(preserve_args)
    : > "$errlog"
    while read_record; do
        if [[ -n "$DRY_RUN" ]]; then
            log_message INFO "DRY RUN would remove empty directory: $path"
        else
            log_message INFO "Removed empty directory: $path"
            manifest_record "$manifest" dir "$owner" 0 "$atime" "$mtime" "$path" \
                || log_message ERROR "Removed but could not record in $manifest: $path"
            dirs_removed=$((dirs_removed + 1))
        fi
    done < <(find_act "$scratch_dir" -mindepth "$mindepth" "${STALE_EMPTY_DIR[@]}" "${keep[@]}" "${delete[@]}" 2>"$errlog")
    wait "$!" 2>/dev/null
    while IFS= read -r line; do
        log_message DEBUG "Could not remove directory: ${line#*: cannot delete }"
    done < "$errlog"
    rm -f -- "$errlog"

    log_message INFO "Cleanup summary for $scratch_dir: $files_removed file(s) removed ($(format_bytes "$total_size")), $dirs_removed directory(ies) removed, $errors error(s)"
    (( errors == 0 ))
}

# ---------------------------------------------------------------------------
main() {
    local start_time end_time total_errors=0 scratch_dir
    printf -v start_time '%(%s)T' -1
    [[ $EUID -eq 0 ]] || log_message WARN "Not running as root/administrator. Some operations may fail."

    acquire_lock
    trap 'release_lock; log_message INFO "Cleanup process completed"' EXIT

    log_message INFO "=== Scratch folder cleanup started ==="
    log_message INFO "PID: $$"
    log_message INFO "Configuration: DAYS_TO_KEEP=$DAYS_TO_KEEP DAYS_TO_NOTIFY=$DAYS_TO_NOTIFY MANIFEST_DIR=$MANIFEST_DIR SCRATCH_DIRS=(${SCRATCH_DIRS[*]})"

    for scratch_dir in "${SCRATCH_DIRS[@]}"; do
        if notify_imminent_deletion "$scratch_dir"; then
            log_message INFO "Successfully sent notifications for: $scratch_dir"
        else
            log_message ERROR "Errors occurred while notification processing of folder: $scratch_dir"
            total_errors=$((total_errors + 1))
        fi
        if clean_directory "$scratch_dir"; then
            log_message INFO "Successfully processed: $scratch_dir"
        else
            log_message ERROR "Errors occurred while processing: $scratch_dir"
            total_errors=$((total_errors + 1))
        fi
    done

    printf -v end_time '%(%s)T' -1
    log_message INFO "=== Cleanup completed ==="
    log_message INFO "Total execution time: $((end_time - start_time)) seconds"
    log_message INFO "Directories processed: ${#SCRATCH_DIRS[@]}"
    if (( total_errors > 0 )); then
        log_message WARN "Completed with $total_errors error(s)"; exit 1
    fi
    log_message INFO "All operations completed successfully"; exit 0
}

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
usage() {
    cat <<USAGE
Usage: scratch-cleanup.sh [OPTION]

Removes files under the scratch directories that have been neither read nor
modified for DAYS_TO_KEEP days, warning their owners DAYS_TO_NOTIFY days first.
Every deletion is appended to MANIFEST_DIR/deleted-YYYY-MM.tsv.

  --dry-run       report what would be removed and emailed; change nothing,
                  contact nobody, record nothing
  --show-config   print the retention policy and exit
  -h, --help      this text

THIS FILE IS THE SINGLE SOURCE OF TRUTH for the retention numbers. Anything
else that quotes them must read them from '--show-config'.

Every setting can be overridden with a SCRATCH_CLEANUP_* environment variable;
see the settings block at the top of this file.
USAGE
}

show_config() {
    cat <<CONF
DAYS_TO_KEEP=$DAYS_TO_KEEP
DAYS_TO_NOTIFY=$DAYS_TO_NOTIFY
SCRATCH_DIRS=${SCRATCH_DIRS[*]}
LOG_FILE=$LOG_FILE
MANIFEST_DIR=$MANIFEST_DIR
CONF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=1; shift ;;
        --show-config) show_config; exit 0 ;;
        -h|--help)     usage; exit 0 ;;
        *)             printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

main
