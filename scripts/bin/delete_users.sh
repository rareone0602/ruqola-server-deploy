#!/bin/bash
# Research Group User Deletion Script
# Usage: ./delete_research_users.sh users.csv

# Configuration
LOG_FILE="/var/log/user_deletion.log"
BACKUP_DIR="/var/backups/deleted_users"
HOME_BASE="/home"
SCRATCH_BASE="/scratch/users"
USER_GROUPS="users,scratch-users,gpuqueue"  # Standard research user groups

# Accounts this script never deletes: uids outside the range useradd gives people
# (UID_MIN..UID_MAX in /etc/login.defs; nobody is 65534), and admins. To delete a
# former admin, take them out of the admin group first.
PEOPLE_UID_MIN=1000
PEOPLE_UID_MAX=60000
ADMIN_GROUPS="sudo admin"

# BACKUP_DIR is on /. A backup must leave at least this much free there.
BACKUP_RESERVE_KB=$((10 * 1024 * 1024))

# Function to log messages
# /var/log is root:syslog 775, so the log is only writable through sudo.
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" | sudo tee -a "$LOG_FILE"
}

# Why an account must never be deleted by this script; prints nothing if it may be.
protected_reason() {
    local username="$1" uid group
    uid=$(id -u "$username" 2>/dev/null) || return 0
    if (( uid < PEOPLE_UID_MIN || uid > PEOPLE_UID_MAX )); then
        echo "uid $uid is a system account, not a person"
        return 0
    fi
    for group in $(id -nG "$username" 2>/dev/null); do
        if [[ " $ADMIN_GROUPS " == *" $group "* ]]; then
            echo "member of '$group' (an admin); remove them from it first"
            return 0
        fi
    done
}

# Refuse the whole run, before anything is deleted, if any name is malformed or
# protected: a list that names root or an admin is the wrong list.
check_targets() {
    local username reason bad=0
    for username in "$@"; do
        if [[ ! "$username" =~ ^[a-z][a-z0-9_-]*$ ]]; then
            echo "Error: invalid username '$username'" >&2
            bad=1
            continue
        fi
        reason=$(protected_reason "$username")
        if [[ -n "$reason" ]]; then
            echo "Error: refusing to delete '$username': $reason" >&2
            bad=1
        fi
    done
    return "$bad"
}

# Function to check if quota tools are available
check_quota_support() {
    if ! command -v setquota &> /dev/null; then
        log_message "WARNING: quota tools not available. Quota cleanup will be skipped."
        return 1
    fi
    return 0
}

# Function to create backup of user data
# Any failure returns 1, and the caller then deletes nothing.
backup_user_data() {
    local username="$1"
    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local user_backup_dir="$BACKUP_DIR/${username}_${backup_timestamp}"
    local sources=() need_kb free_kb
    
    [[ -d "$HOME_BASE/$username" ]] && sources+=("$HOME_BASE/$username")
    [[ -d "$SCRATCH_BASE/$username" ]] && sources+=("$SCRATCH_BASE/$username")
    
    # A large scratch directory copied onto / could fill it for everyone.
    if (( ${#sources[@]} )); then
        need_kb=$(sudo du -sk "${sources[@]}" | awk '{s += $1} END {print s}')
        free_kb=$(df -Pk "$BACKUP_DIR" | awk 'NR == 2 {print $4}')
        if [[ ! "$need_kb" =~ ^[0-9]+$ || ! "$free_kb" =~ ^[0-9]+$ ]]; then
            log_message "ERROR: Could not measure the backup of $username"
            return 1
        fi
        if (( need_kb + BACKUP_RESERVE_KB > free_kb )); then
            log_message "ERROR: Backup of $username needs $need_kb KB; $BACKUP_DIR has $free_kb KB free and must keep $BACKUP_RESERVE_KB KB"
            return 1
        fi
    fi
    
    # Create backup directory
    if ! sudo mkdir -p "$user_backup_dir"; then
        log_message "ERROR: Failed to create backup directory for $username"
        return 1
    fi
    # Root-only at the top; the copies inside keep their owners and modes, so a
    # restore is a plain copy back.
    sudo chmod 700 "$user_backup_dir"
    
    log_message "Creating backup of user data for $username"
    
    # Backup home directory
    if [[ -d "$HOME_BASE/$username" ]]; then
        if sudo cp -a "$HOME_BASE/$username" "$user_backup_dir/home"; then
            log_message "SUCCESS: Backed up home directory for $username"
        else
            log_message "ERROR: Failed to back up home directory for $username"
            return 1
        fi
    fi
    
    # Backup scratch directory
    if [[ -d "$SCRATCH_BASE/$username" ]]; then
        if sudo cp -a "$SCRATCH_BASE/$username" "$user_backup_dir/scratch"; then
            log_message "SUCCESS: Backed up scratch directory for $username"
        else
            log_message "ERROR: Failed to back up scratch directory for $username"
            return 1
        fi
    fi
    
    # Create user info file
    if ! {
        echo "User: $username"
        echo "Deletion Date: $(date)"
        echo "UID: $(id -u $username 2>/dev/null || echo 'N/A')"
        echo "GID: $(id -g $username 2>/dev/null || echo 'N/A')"
        echo "Groups: $(id -nG $username 2>/dev/null || echo 'N/A')"
        echo "Shell: $(getent passwd $username | cut -d: -f7 2>/dev/null || echo 'N/A')"
        echo "Home: $(getent passwd $username | cut -d: -f6 2>/dev/null || echo 'N/A')"
    } | sudo tee "$user_backup_dir/user_info.txt" > /dev/null; then
        log_message "ERROR: Failed to write user_info.txt for $username"
        return 1
    fi
    
    log_message "SUCCESS: User data backed up to $user_backup_dir"
    return 0
}

# Function to kill user processes
kill_user_processes() {
    local username="$1"
    
    # Check if user has running processes
    if pgrep -u "$username" > /dev/null; then
        log_message "WARNING: User $username has running processes. Attempting to terminate..."
        
        # First try graceful termination
        if sudo pkill -TERM -u "$username"; then
            sleep 5
            
            # Force kill if processes still running
            if pgrep -u "$username" > /dev/null; then
                log_message "WARNING: Forcefully killing remaining processes for $username"
                sudo pkill -KILL -u "$username"
                sleep 2
            fi
        fi
        
        # Final check
        if pgrep -u "$username" > /dev/null; then
            log_message "ERROR: Unable to kill all processes for $username"
            return 1
        else
            log_message "SUCCESS: Terminated all processes for $username"
        fi
    fi
    
    return 0
}

# Function to remove user quota
remove_user_quota() {
    local username="$1"
    
    if check_quota_support; then
        if sudo setquota -u "$username" 0 0 0 0 /; then
            log_message "SUCCESS: Removed disk quota for $username"
        else
            log_message "WARNING: Failed to remove quota for $username"
        fi
    fi
}

# Function to remove user directories
remove_user_directories() {
    local username="$1"
    
    # Remove scratch directory
    if [[ -d "$SCRATCH_BASE/$username" ]]; then
        if sudo rm -rf "$SCRATCH_BASE/$username"; then
            log_message "SUCCESS: Removed scratch directory for $username"
        else
            log_message "ERROR: Failed to remove scratch directory for $username"
            return 1
        fi
    fi
    
    # Home directory will be removed by userdel -r
    return 0
}

# Function to delete a single user
delete_user() {
    local username="$1"
    local skip_backup="$2"
    local reason
    
    # Validate username
    if [[ ! "$username" =~ ^[a-z][a-z0-9_-]*$ ]]; then
        log_message "ERROR: Invalid username '$username'"
        return 1
    fi
    
    # Check if user exists
    if ! id -u "$username" &>/dev/null; then
        log_message "ERROR: User '$username' does not exist"
        return 1
    fi
    
    # main checks the whole list first; this repeats it for callers that don't.
    reason=$(protected_reason "$username")
    if [[ -n "$reason" ]]; then
        log_message "REFUSED: Not deleting $username: $reason"
        return 1
    fi
    
    log_message "Starting deletion process for user: $username"
    
    # Create backup unless skipped. No backup, no deletion.
    if [[ "$skip_backup" != "true" ]] && ! backup_user_data "$username"; then
        log_message "ABORTED: Backup failed, so $username was not deleted. Fix the cause, or re-run with --no-backup."
        return 1
    fi
    
    # Kill user processes
    if ! kill_user_processes "$username"; then
        log_message "WARNING: Could not kill all user processes. Continuing..."
    fi
    
    # Remove user quota
    remove_user_quota "$username"
    
    # Remove additional directories
    remove_user_directories "$username"
    
    # Remove user account and home directory
    if sudo userdel -r "$username" 2>/dev/null; then
        log_message "SUCCESS: Removed user account and home directory for $username"
    else
        log_message "ERROR: Failed to remove user account for $username"
        # Try without -r flag if home directory removal failed
        if sudo userdel "$username" 2>/dev/null; then
            log_message "SUCCESS: Removed user account for $username (home directory may remain)"
        else
            log_message "ERROR: Complete failure to remove user account for $username"
            return 1
        fi
    fi
    
    # Clean up any remaining group memberships (shouldn't be necessary, but just in case)
    for group in $(echo "$USER_GROUPS" | tr ',' ' '); do
        if getent group "$group" | grep -q "$username"; then
            sudo gpasswd -d "$username" "$group" 2>/dev/null
        fi
    done
    
    log_message "COMPLETED: User $username deletion finished"
    return 0
}

# Usernames from a CSV, one per line: the first column, header line skipped.
csv_usernames() {
    local username rest
    tail -n +2 "$1" | while IFS=',' read -r username rest || [[ -n "$username" ]]; do
        # Remove quotes, whitespace and Excel's CR
        username="${username//[$'"\r\t ']/}"
        [[ -n "$username" ]] && echo "$username"
    done
}

# Delete each user in turn; returns 1 if any failed.
delete_all() {
    local skip_backup="$1"; shift
    local username success_count=0 error_count=0
    
    for username in "$@"; do
        if delete_user "$username" "$skip_backup"; then
            success_count=$((success_count + 1))
        else
            error_count=$((error_count + 1))
        fi
    done
    
    log_message "Deletion completed: $success_count successful, $error_count errors"
    (( error_count == 0 ))
}

# Function to confirm deletion
confirm_deletion() {
    local skip_backup="$1"; shift
    
    echo "WARNING: This will permanently delete $# user account(s) and all associated data:"
    printf '    %s\n' "$@"
    if [[ "$skip_backup" == "true" ]]; then
        echo "Backup: none (--no-backup)"
    else
        echo "Backup directory: $BACKUP_DIR"
    fi
    echo ""
    read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 0
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Research Group User Deletion Script

Usage:
    $0 users.csv                           # Delete users from CSV
    $0 --single username                   # Delete single user
    $0 --single username --no-backup       # Delete single user without backup
    $0 --csv users.csv --no-backup         # Delete users from CSV without backup
    $0 --help                             # Show this help

CSV Format (users.csv):
    username,password,fullname
    jsmith,mypassword123,John Smith
    agarcia,securepass456,Ana Garcia
    (Only username column is used for deletion)

Safety Features:
    - Refuses the whole run if any name is malformed, a system account
      (uid outside $PEOPLE_UID_MIN-$PEOPLE_UID_MAX), or an admin (groups: $ADMIN_GROUPS)
    - Lists the accounts and asks once before deleting
    - Creates backup of user data in $BACKUP_DIR; if the backup fails or
      would not fit, that user is not deleted
    - Kills user processes before deletion
    - Removes quotas and custom directories
    - Logs to $LOG_FILE; exits non-zero if any user was not deleted

What gets deleted:
    - User account
    - Home directory (/home/username)
    - Scratch directory (/scratch/users/username)
    - User quotas
    - Group memberships

Prerequisites:
    - Run as user with sudo privileges
    - Ensure no critical processes are running as target users

EOF
}

# Main script logic
main() {
    local skip_backup="false" csv="" targets=()
    
    # Check if running as root (don't allow this)
    if [[ $EUID -eq 0 ]]; then
        echo "Error: Don't run this script as root. Run as user with sudo privileges."
        exit 1
    fi
    
    # Create backup directory
    sudo mkdir -p "$BACKUP_DIR"
    
    # Parse arguments
    case "$1" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --single)
            if [[ $# -lt 2 ]]; then
                echo "Error: --single requires username"
                show_usage
                exit 1
            fi
            if [[ "$3" == "--no-backup" ]]; then
                skip_backup="true"
            fi
            targets=("$2")
            ;;
        --csv)
            if [[ $# -lt 2 ]]; then
                echo "Error: --csv requires CSV filename"
                show_usage
                exit 1
            fi
            if [[ "$3" == "--no-backup" ]]; then
                skip_backup="true"
            fi
            csv="$2"
            ;;
        *.csv)
            # BUGFIX: honor --no-backup in the bare-CSV form too
            # (e.g. "delete_users.sh users.csv --no-backup"), mirroring the
            # --csv and --single cases. Previously $2 was ignored here, so the
            # documented --no-backup option silently still created backups.
            if [[ "$2" == "--no-backup" ]]; then
                skip_backup="true"
            fi
            csv="$1"
            ;;
        "")
            echo "Error: No input provided"
            show_usage
            exit 1
            ;;
        *)
            echo "Error: Invalid argument '$1'"
            show_usage
            exit 1
            ;;
    esac
    
    if [[ -n "$csv" ]]; then
        if [[ ! -f "$csv" ]]; then
            echo "Error: CSV file '$csv' not found"
            exit 1
        fi
        mapfile -t targets < <(csv_usernames "$csv")
        if (( ${#targets[@]} == 0 )); then
            echo "Error: no usernames in '$csv'"
            exit 1
        fi
    fi
    
    if ! check_targets "${targets[@]}"; then
        echo "Nothing was deleted." >&2
        exit 1
    fi
    confirm_deletion "$skip_backup" "${targets[@]}"
    log_message "Deleting ${#targets[@]} user(s): ${targets[*]}"
    delete_all "$skip_backup" "${targets[@]}"
}

# Run main only when executed, so tests can source the functions.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi