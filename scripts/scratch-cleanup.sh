#!/bin/bash

# Scratch Folder Cleanup Script
# Author: System Administrator
# Purpose: Clean up unused files from scratch folders with comprehensive logging
# Usage: Run as root/administrator via cron

# Configuration
# All settings below may be overridden by SCRATCH_CLEANUP_* environment variables.
# Production leaves them unset and gets the defaults; the test suite overrides them
# so it never touches the real /scratch, /var/log or /run. Same pattern as gpuq's
# GPUQ_QUEUE_DIR / GPUQ_CONFIG_FILE seams.
if [[ -n "${SCRATCH_CLEANUP_DIRS:-}" ]]; then
    IFS=':' read -r -a SCRATCH_DIRS <<< "$SCRATCH_CLEANUP_DIRS"
else
    SCRATCH_DIRS=(
        "/scratch/shared"
        "/scratch/temp"
        # "/scratch/datasets" # this we won't check so that files here don't get deleted
        "/scratch/users"
        # Add more directories as needed
    )
fi
DAYS_TO_KEEP="${SCRATCH_CLEANUP_DAYS_KEEP:-180}"
DAYS_TO_NOTIFY="${SCRATCH_CLEANUP_DAYS_NOTIFY:-166}"
LOG_FILE="${SCRATCH_CLEANUP_LOG:-/var/log/scratch-cleanup.log}"
LOG_DIR="$(dirname "$LOG_FILE")"
MAX_LOG_SIZE=$((10 * 1024 * 1024))  # 10MB in bytes
LOCK_FILE="${SCRATCH_CLEANUP_LOCK:-/run/scratch-cleanup.lock}"
MSMTP="${SCRATCH_CLEANUP_MSMTP:-/usr/bin/msmtp}"
DRY_RUN="${SCRATCH_CLEANUP_DRYRUN:-}"
ADMIN_EMAIL="${SCRATCH_CLEANUP_ADMIN_EMAIL:-mjolnirruqola@gmail.com}"

# Colors for output (when run interactively)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Write to log file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Also output to console if running interactively
    if [[ -t 1 ]]; then
        case "$level" in
            "INFO")  echo -e "${GREEN}[$timestamp] [INFO]${NC} $message" ;;
            "WARN")  echo -e "${YELLOW}[$timestamp] [WARN]${NC} $message" ;;
            "ERROR") echo -e "${RED}[$timestamp] [ERROR]${NC} $message" ;;
            "DEBUG") echo -e "${BLUE}[$timestamp] [DEBUG]${NC} $message" ;;
            *)       echo "[$timestamp] [$level] $message" ;;
        esac
    fi
}

# Function to rotate log file if it gets too large
rotate_log() {
    if [[ -f "$LOG_FILE" ]] && [[ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null) -gt $MAX_LOG_SIZE ]]; then
        log_message "INFO" "Rotating log file (size exceeded ${MAX_LOG_SIZE} bytes)"
        mv "$LOG_FILE" "${LOG_FILE}.old"
        touch "$LOG_FILE"
        chmod 640 "$LOG_FILE"
    fi
}

# Function to check if process is already running
check_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        local pid=$(cat "$LOCK_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log_message "WARN" "Another instance is already running (PID: $pid). Exiting."
            exit 1
        else
            log_message "INFO" "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    # Create lock file
    echo $$ > "$LOCK_FILE"
}

# Function to clean up on exit
cleanup_exit() {
    rm -f "$LOCK_FILE"
    log_message "INFO" "Cleanup process completed"
}

# Function to validate directory
validate_directory() {
    local dir="$1"
    
    if [[ ! -d "$dir" ]]; then
        log_message "WARN" "Directory does not exist: $dir"
        return 1
    fi
    
    if [[ ! -r "$dir" ]]; then
        log_message "ERROR" "No read permission for directory: $dir"
        return 1
    fi
    
    return 0
}

# Function to check filesystem atime support
check_atime_support() {
    local dir="$1"
    local test_file="$dir/.atime_test_$"
    
    # Create test file
    if ! touch "$test_file" 2>/dev/null; then
        log_message "DEBUG" "Cannot create test file in $dir for atime check"
        return 1
    fi
    
    # Wait a moment to ensure different timestamps
    sleep 1
    
    # Get initial atime
    local initial_atime
    if command -v stat >/dev/null 2>&1; then
        # Try Linux stat first, then BSD stat
        initial_atime=$(stat -c %X "$test_file" 2>/dev/null || stat -f %a "$test_file" 2>/dev/null)
    fi
    
    if [[ -z "$initial_atime" ]]; then
        log_message "DEBUG" "Cannot get initial atime for $test_file"
        rm -f "$test_file"
        return 1
    fi
    
    # Wait another moment to ensure time difference
    sleep 1
    
    # Access the file
    cat "$test_file" >/dev/null 2>&1
    
    # Wait a moment to ensure atime update is written
    sleep 1
    
    # Get new atime
    local new_atime
    if command -v stat >/dev/null 2>&1; then
        new_atime=$(stat -c %X "$test_file" 2>/dev/null || stat -f %a "$test_file" 2>/dev/null)
    fi
    
    log_message "DEBUG" "Atime test for $dir: initial=$initial_atime, new=$new_atime"
    
    # Cleanup test file
    rm -f "$test_file"
    
    # Check if atime changed (allow for small differences due to timing)
    if [[ -n "$new_atime" ]] && [[ "$initial_atime" != "$new_atime" ]]; then
        log_message "DEBUG" "Atime working properly in $dir"
        return 0  # atime is working
    else
        log_message "DEBUG" "Atime not updating properly in $dir (initial: $initial_atime, new: $new_atime)"
        return 1  # atime is not updating
    fi
}

# Function to clean directory

# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------

# Notification address for an account, read from its GECOS field.
# GECOS shapes on this host vary: some accounts are "Full Name,,,,addr@x" and
# some are a bare "addr@x", so match an address ANYWHERE in the field instead of
# assuming a fixed comma position. This is the same rule gpuq uses
# (see email_for_user() in gpuq), so both tools agree on every account.
email_for_user() {
    getent passwd "$1" 2>/dev/null | awk -F: '{print $5}' \
        | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' | head -1
}

# Human name for an account, falling back to the username when GECOS holds only
# an address (otherwise we would greet people by their own email address).
full_name_for_user() {
    local gecos name
    gecos=$(getent passwd "$1" 2>/dev/null | awk -F: '{print $5}')
    name="${gecos%%,*}"
    if [[ -z "$name" || "$name" == *"@"* ]]; then printf '%s' "$1"; else printf '%s' "$name"; fi
}

# send_mail <recipient> <subject> <body>
# Never invokes the mailer with an empty recipient.
send_mail() {
    local to="$1" subject="$2" body="$3"
    if [[ -z "$to" ]]; then
        return 1
    fi
    # A preview must not contact anyone. Under DRY_RUN, record the intent and stop.
    if [[ -n "$DRY_RUN" ]]; then
        log_message "INFO" "DRY RUN would email $to: $subject"
        return 0
    fi
    printf 'To: %s\nFrom: %s\nSubject: %s\n\n%s\n' \
        "$to" "$ADMIN_EMAIL" "$subject" "$body" | "$MSMTP" "$to"
}

# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------
clean_directory() {
    local scratch_dir="$1"
    local files_removed=0 total_size=0 errors=0 dirs_removed=0
    local -A user_list user_bytes user_count

    if ! validate_directory "$scratch_dir"; then
        return 1
    fi

    log_message "INFO" "Cleaning directory: $scratch_dir (older than $DAYS_TO_KEEP days)"
    [[ -n "$DRY_RUN" ]] && log_message "WARN" "DRY RUN active: nothing will be deleted"

    while IFS= read -r -d '' file; do
        [[ -f "$file" ]] || continue

        local file_size username
        file_size=$(stat -c %s "$file" 2>/dev/null || echo 0)
        username=$(stat -c %U "$file" 2>/dev/null || echo "")

        if [[ -n "$DRY_RUN" ]]; then
            log_message "INFO" "DRY RUN would remove: $file (${file_size} bytes)"
        elif rm -f "$file"; then
            log_message "INFO" "Removed: $file (${file_size} bytes)"
        else
            log_message "ERROR" "Failed to remove: $file"
            ((errors++))
            continue
        fi

        files_removed=$((files_removed + 1))
        total_size=$((total_size + file_size))
        if [[ -n "$username" ]]; then
            user_list[$username]+="  ${file} (${file_size} bytes)"$'\n'
            user_bytes[$username]=$(( ${user_bytes[$username]:-0} + file_size ))
            user_count[$username]=$(( ${user_count[$username]:-0} + 1 ))
        fi
    done < <(find "$scratch_dir" -type f \( -atime +$DAYS_TO_KEEP -a -mtime +$DAYS_TO_KEEP \) -print0 2>/dev/null)

    # One digest per owner per run. Previously this sent one message per FILE,
    # which on a large cohort would exhaust the shared Gmail quota and take the
    # queue notifications down with it.
    local u to name body subject
    for u in "${!user_list[@]}"; do
        to=$(email_for_user "$u")
        name=$(full_name_for_user "$u")
        subject="Scratch cleanup: ${user_count[$u]} file(s) removed"
        body="Hello ${name},

This is an automated notification from the server.

The following ${user_count[$u]} file(s) under ${scratch_dir} were removed because
they were neither modified nor accessed in the last ${DAYS_TO_KEEP} days
(total $(format_bytes ${user_bytes[$u]})):

${user_list[$u]}
To keep files permanently, store them under /scratch/datasets/ or in your home
directory.

Thank you,
System Administrator"
        if send_mail "$to" "$subject" "$body"; then
            log_message "INFO" "Deletion digest sent to $u <$to> (${user_count[$u]} file(s))"
        else
            log_message "WARN" "No email address on account '$u'; ${user_count[$u]} deletion(s) unreported"
        fi
    done

    # Clean up empty directories (but preserve important structure).
    # The age gate matters: without -mtime this removed ANY empty directory,
    # including one created seconds earlier, so a job that pre-created its
    # output tree and was still running at 02:30 lost it. Directories now age
    # out on the same DAYS_TO_KEEP clock as files.
    log_message "INFO" "Removing empty directories in $scratch_dir older than $DAYS_TO_KEEP days"

    # Define directories to preserve (never delete these even if empty)
    local preserve_dirs=(
        "/scratch/shared"
        "/scratch/temp"
        "/scratch/datasets"
        "/scratch/users"
    )

    # Add all user directories to preserve list
    if [[ -d "/scratch/users" ]]; then
        while IFS= read -r -d '' user_dir; do
            preserve_dirs+=("$user_dir")
        done < <(find "/scratch/users" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
    fi

    while IFS= read -r -d '' dir; do
        if [[ -d "$dir" ]] && [[ "$dir" != "$scratch_dir" ]]; then
            local preserve=false
            for preserve_dir in "${preserve_dirs[@]}"; do
                if [[ "$dir" == "$preserve_dir" ]]; then
                    preserve=true
                    log_message "DEBUG" "Preserving structural directory: $dir"
                    break
                fi
            done

            if [[ "$preserve" == false ]]; then
                if [[ -n "$DRY_RUN" ]]; then
                    log_message "INFO" "DRY RUN would remove empty directory: $dir"
                elif rmdir "$dir" 2>/dev/null; then
                    log_message "INFO" "Removed empty directory: $dir"
                    ((dirs_removed++))
                else
                    log_message "DEBUG" "Could not remove directory (not empty or permission denied): $dir"
                fi
            fi
        fi
    done < <(find "$scratch_dir" -mindepth 1 -type d -empty -mtime +$DAYS_TO_KEEP -print0 2>/dev/null)

    log_message "INFO" "Cleanup summary for $scratch_dir: $files_removed file(s) removed ($(numfmt --to=iec $total_size 2>/dev/null || echo "${total_size} bytes")), $dirs_removed directory(ies) removed, $errors error(s)"

    return $errors
}

# ---------------------------------------------------------------------------
# Advance warning
# ---------------------------------------------------------------------------
notify_imminent_deletion(){
    local scratch_dir="$1"
    local -A warn_list warn_count warn_min

    if ! validate_directory "$scratch_dir"; then
        return 1
    fi

    local now
    now=$(date +%s)

    # Warn only about files INSIDE the window: past the notification age but not
    # yet past the deletion age. The old query selected a superset that included
    # everything about to be deleted in this same run, so users received a "you
    # have time to save this" mail seconds before the "this was deleted" mail.
    while IFS= read -r -d '' file; do
        [[ -f "$file" ]] || continue

        local username atime_s mtime_s newest elapsed_days days_remaining
        username=$(stat -c %U "$file" 2>/dev/null || echo "")
        [[ -n "$username" ]] || continue
        atime_s=$(stat -c %X "$file" 2>/dev/null || echo 0)
        mtime_s=$(stat -c %Y "$file" 2>/dev/null || echo 0)

        # Deletion needs BOTH timestamps past the limit, so the countdown is
        # governed by the NEWER of the two, not by atime alone.
        if (( atime_s > mtime_s )); then newest=$atime_s; else newest=$mtime_s; fi
        elapsed_days=$(( (now - newest) / 86400 ))
        days_remaining=$(( DAYS_TO_KEEP - elapsed_days ))
        (( days_remaining < 0 )) && days_remaining=0

        warn_list[$username]+="  ${file} (${days_remaining} day(s) remaining)"$'\n'
        warn_count[$username]=$(( ${warn_count[$username]:-0} + 1 ))
        if [[ -z "${warn_min[$username]:-}" ]] || (( days_remaining < warn_min[$username] )); then
            warn_min[$username]=$days_remaining
        fi
    done < <(find "$scratch_dir" -type f \
                   \( -atime +$DAYS_TO_NOTIFY -a -mtime +$DAYS_TO_NOTIFY \) \
                 ! \( -atime +$DAYS_TO_KEEP   -a -mtime +$DAYS_TO_KEEP   \) -print0 2>/dev/null)

    local u to name body subject
    for u in "${!warn_list[@]}"; do
        to=$(email_for_user "$u")
        name=$(full_name_for_user "$u")
        subject="Imminent removal: ${warn_count[$u]} file(s) under ${scratch_dir}"
        body="Hello ${name},

This is an automated notification from the server.

The following ${warn_count[$u]} file(s) have gone more than ${DAYS_TO_NOTIFY} days
without being modified or accessed. Files under /scratch/ (except
/scratch/datasets/) are removed after ${DAYS_TO_KEEP} days.

${warn_list[$u]}
To keep them, modify them or copy them somewhere permanent within
${warn_min[$u]} day(s).

Thank you,
System Administrator"
        if send_mail "$to" "$subject" "$body"; then
            log_message "INFO" "Warning digest sent to $u <$to> (${warn_count[$u]} file(s))"
        else
            log_message "WARN" "No email address on account '$u'; ${warn_count[$u]} warning(s) undelivered"
        fi
    done

    return 0
}

# Function to format bytes
format_bytes() {
    local bytes=$1
    if command -v numfmt >/dev/null 2>&1; then
        numfmt --to=iec "$bytes"
    else
        echo "${bytes} bytes"
    fi
}

# Main function
main() {
    local start_time=$(date '+%s')
    local total_files=0
    local total_dirs=0
    local total_errors=0
    
    # Check if running as root/administrator
    if [[ $EUID -ne 0 ]]; then
        log_message "WARN" "Not running as root/administrator. Some operations may fail."
    fi
    
    # Setup
    rotate_log
    check_lock
    trap cleanup_exit EXIT
    
    log_message "INFO" "=== Scratch folder cleanup started ==="
    log_message "INFO" "PID: $$"
    log_message "INFO" "Configuration: DAYS_TO_KEEP=$DAYS_TO_KEEP, SCRATCH_DIRS=(${SCRATCH_DIRS[*]})"

    # Process each scratch directory
    for scratch_dir in "${SCRATCH_DIRS[@]}"; do
        if notify_imminent_deletion "$scratch_dir"; then
            log_message "INFO" "Successfully sent notifications for: $scratch_dir"
        else
            log_message "ERROR" "Errors occurred while notification processing of folder: $scratch_dir"
            ((total_errors++))
        fi
        if clean_directory "$scratch_dir"; then
            log_message "INFO" "Successfully processed: $scratch_dir"
        else
            log_message "ERROR" "Errors occurred while processing: $scratch_dir"
            ((total_errors++))
        fi
    done
    
    # Final summary
    local end_time=$(date '+%s')
    local duration=$((end_time - start_time))
    
    log_message "INFO" "=== Cleanup completed ==="
    log_message "INFO" "Total execution time: ${duration} seconds"
    log_message "INFO" "Directories processed: ${#SCRATCH_DIRS[@]}"
    
    if [[ $total_errors -gt 0 ]]; then
        log_message "WARN" "Completed with $total_errors error(s)"
        exit 1
    else
        log_message "INFO" "All operations completed successfully"
        exit 0
    fi
}

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: scratch-cleanup.sh [OPTION]

Removes files under the scratch directories that have been neither read nor
modified for DAYS_TO_KEEP days, warning their owners DAYS_TO_NOTIFY days first.

  --dry-run       report what would be removed and emailed; change nothing,
                  contact nobody
  --show-config   print the retention policy and exit

  -h, --help      this text

THIS FILE IS THE SINGLE SOURCE OF TRUTH for the retention numbers. Anything
else that quotes them (scratch-usage.sh, /scratch/README.txt, the docs) must
read them from '--show-config' rather than restating them, or they drift.

Every setting can be overridden with a SCRATCH_CLEANUP_* environment variable;
see the configuration block at the top of this file.
EOF
}

show_config() {
    cat <<EOF
DAYS_TO_KEEP=$DAYS_TO_KEEP
DAYS_TO_NOTIFY=$DAYS_TO_NOTIFY
SCRATCH_DIRS=${SCRATCH_DIRS[*]}
LOG_FILE=$LOG_FILE
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=1; shift ;;
        --show-config) show_config; exit 0 ;;
        -h|--help)     usage; exit 0 ;;
        *)             printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

# Run main function
main
