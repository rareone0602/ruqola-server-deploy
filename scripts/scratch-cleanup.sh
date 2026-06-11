#!/bin/bash

# Scratch Folder Cleanup Script
# Author: System Administrator
# Purpose: Clean up unused files from scratch folders with comprehensive logging
# Usage: Run as root/administrator via cron

# Configuration
SCRATCH_DIRS=(
    "/scratch/shared"
    "/scratch/temp"
    # "/scratch/datasets" # this we won't check so that files here don't get deleted
    "/scratch/users"
    # Add more directories as needed
)
DAYS_TO_KEEP=30
DAYS_TO_NOTIFY=23
LOG_DIR="/var/log"
LOG_FILE="$LOG_DIR/scratch-cleanup.log"
MAX_LOG_SIZE=$((10 * 1024 * 1024))  # 10MB in bytes
LOCK_FILE="/var/run/scratch-cleanup.lock"

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
    # Use $$ (the PID) so concurrent runs get unique test files; a lone "$"
    # is a literal dollar sign in a double-quoted string, not the PID.
    local test_file="$dir/.atime_test_$$"
    
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
clean_directory() {
    local scratch_dir="$1"
    local files_removed=0
    local dirs_removed=0
    local total_size=0
    local errors=0
    
    log_message "INFO" "Starting cleanup of directory: $scratch_dir"
    
    # Validate directory
    if ! validate_directory "$scratch_dir"; then
        return 1
    fi
    
    # Check atime support
    if ! check_atime_support "$scratch_dir"; then
        log_message "WARN" "Access time tracking may not be working properly in $scratch_dir (filesystem mounted with noatime/relatime?)"
    fi
    
    # Find and process old files
    log_message "INFO" "Searching for files not accessed in $DAYS_TO_KEEP days in $scratch_dir"
    
    # Use find with -atime for access time
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            # Get file size before deletion
            local file_size
            if command -v stat >/dev/null 2>&1; then
                file_size=$(stat -c %s "$file" 2>/dev/null || stat -f %z "$file" 2>/dev/null || echo 0)
            else
                file_size=0
            fi
            
            # Get file access time for logging
            local access_time
            local username
            if command -v stat >/dev/null 2>&1; then
                access_time=$(stat -c %x "$file" 2>/dev/null || stat -f %Sa "$file" 2>/dev/null || echo "unknown")
                # Leave the owner empty if it cannot be resolved; the
                # notification below then falls back to the admin address
                # instead of mis-attributing the file to a specific user.
                username=$(stat -c %U "$file" 2>/dev/null || echo "")
            else
                access_time="unknown"
            fi

            # Attempt to remove file
            if rm -f "$file"; then
                log_message "INFO" "Removed: $file (size: ${file_size} bytes, last access: ${access_time})"
                ((files_removed++))
                ((total_size += file_size))

                # Form the user's email address
                ADMIN_EMAIL=mjolnirruqola@gmail.com
                USER_EMAIL=$(getent passwd ${username} | awk -F ':' '{print $5}' | awk -F ',' '{print $5}')

                USER_FULL_NAME=$(getent passwd ${username} | awk -F ':' '{print $5}' | awk -F ',' '{print $1}')

                # Fall back to the admin if the owner / their email is unknown,
                # so notifications about orphaned files are not misdirected.
                if [[ -z "$USER_EMAIL" ]]; then
                    USER_EMAIL="$ADMIN_EMAIL"
                fi



                DELETION_MAIL=$(cat <<EOF
To: ${USER_EMAIL}
From: ${ADMIN_EMAIL}
Subject: Removed ${file}

Hello ${USER_FULL_NAME},

This is an automated notification from the server.
The file ${file} has now been automatically deleted due to it not being accessed or modified in the last 30 days.

If the file was not to be deleted: sincere apologies. Do make sure next time to use either the "/scratch/datasets" folder for permanent files or your own home directory for smaller permanent files.
Any file in any other folder in the "/scratch/" directory will be deleted after 30 days of it being unaccessed or unmodified.

Thank you,
System Administrator
EOF
)

            # Send the email using msmtp
            echo "$DELETION_MAIL" | /usr/bin/msmtp "$USER_EMAIL"
            echo "Sent deletion email to $USER_EMAIL"

            else
                log_message "ERROR" "Failed to remove: $file"
                ((errors++))
            fi
        fi
    # /scratch is mounted `noatime`, so atime is frozen at creation and never
    # updates on read. The old `-atime +N -o -mtime +N` (OR) therefore deleted
    # any file *created* over N days ago even if it was modified today. Key the
    # cleanup off modification time only — the sole reliable staleness signal here.
    done < <(find "$scratch_dir" -type f -mtime +$DAYS_TO_KEEP -print0 2>/dev/null)
    
    # Clean up empty directories (but preserve important structure)
    log_message "INFO" "Removing empty directories in $scratch_dir (preserving structure)"
    
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
            # Check if this directory should be preserved
            local preserve=false
            for preserve_dir in "${preserve_dirs[@]}"; do
                if [[ "$dir" == "$preserve_dir" ]]; then
                    preserve=true
                    log_message "DEBUG" "Preserving structural directory: $dir"
                    break
                fi
            done
            
            # Only remove if not in preserve list
            if [[ "$preserve" == false ]]; then
                if rmdir "$dir" 2>/dev/null; then
                    log_message "INFO" "Removed empty directory: $dir"
                    ((dirs_removed++))
                else
                    log_message "DEBUG" "Could not remove directory (not empty or permission denied): $dir"
                fi
            fi
        fi
    done < <(find "$scratch_dir" -mindepth 1 -type d -empty -print0 2>/dev/null)
    
    # Summary for this directory
    log_message "INFO" "Cleanup summary for $scratch_dir: $files_removed files removed ($(numfmt --to=iec $total_size 2>/dev/null || echo "${total_size} bytes")), $dirs_removed directories removed, $errors errors"
    
    return $errors
}

notify_imminent_deletion(){
    local scratch_dir="$1"

    # Use find with -atime for access time
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            
            # Get file access time for logging
            local access_time
            local username
            if command -v stat >/dev/null 2>&1; then
                access_time=$(stat -c %x "$file" 2>/dev/null || stat -f %Sa "$file" 2>/dev/null || echo "unknown")
                # Leave the owner empty if it cannot be resolved; the
                # notification below then falls back to the admin address
                # instead of mis-attributing the file to a specific user.
                username=$(stat -c %U "$file" 2>/dev/null || echo "")
                # Last *modification* time in seconds since the epoch. (atime is
                # unreliable here: /scratch is mounted noatime, so %X never moves.)
                last_modify_seconds=$(stat -c %Y "$file")

                # Get the current time in seconds
                current_seconds=$(date +%s)

                # Calculate the elapsed time in seconds
                elapsed_seconds=$((current_seconds - last_modify_seconds))

                # Calculate the elapsed time in days
                elapsed_days=$((elapsed_seconds / 86400))

                # Calculate the days remaining until 30 days
                days_remaining=$((30 - elapsed_days))

            else
                access_time="unknown"
            fi

            log_message "INFO" "Notification sent for imminent deletion: $file (last access: ${access_time})"

            # Form the user's email address
            ADMIN_EMAIL=mjolnirruqola@gmail.com
            USER_EMAIL=$(getent passwd ${username} | awk -F ':' '{print $5}' | awk -F ',' '{print $5}')

            USER_FULL_NAME=$(getent passwd ${username} | awk -F ':' '{print $5}' | awk -F ',' '{print $1}')

            # Fall back to the admin if the owner / their email is unknown,
            # so notifications about orphaned files are not misdirected.
            if [[ -z "$USER_EMAIL" ]]; then
                USER_EMAIL="$ADMIN_EMAIL"
            fi

            NOTIFICATION_MAIL=$(cat <<EOF
To: ${USER_EMAIL}
From: ${ADMIN_EMAIL}
Subject: Imminent Removal ${file}

Hello ${USER_FULL_NAME},

This is an automated notification from the server.
The file ${file} has passed the notification period of 23 days without being accessed or modified.
After 30 days without being accessed or modified files in "/scratch/" sub-directories except the ones in "/scratch/datasets" will be automatically deleted.

If the file is not to be deleted, please modify it (re-save it, or run `touch` on it) or copy it to a permanent directory within ${days_remaining} days. Note: simply opening or reading the file does NOT reset the timer.

Thank you,
System Administrator
EOF
)

            # Send the email using msmtp
            echo "$NOTIFICATION_MAIL" | /usr/bin/msmtp "$USER_EMAIL"
            echo "Sent notification email to $USER_EMAIL"
        fi
    # Only notify for files in the warning window (>= DAYS_TO_NOTIFY but not yet
    # eligible for deletion). Excluding files already past DAYS_TO_KEEP stops
    # users getting a contradictory "imminent removal" email in the same run
    # that also deletes the file.
    done < <(find "$scratch_dir" -type f -mtime +$DAYS_TO_NOTIFY ! -mtime +$DAYS_TO_KEEP -print0 2>/dev/null)
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

# Run main function
main "$@"