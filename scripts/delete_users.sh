#!/bin/bash
# Research Group User Deletion Script
# Usage: ./delete_research_users.sh users.csv

# Configuration
LOG_FILE="/var/log/user_deletion.log"
BACKUP_DIR="/var/backups/deleted_users"
USER_GROUPS="users,scratch-users,gpuqueue"  # Standard research user groups

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" | tee -a "$LOG_FILE"
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
backup_user_data() {
    local username="$1"
    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local user_backup_dir="$BACKUP_DIR/${username}_${backup_timestamp}"
    
    # Create backup directory
    if ! sudo mkdir -p "$user_backup_dir"; then
        log_message "ERROR: Failed to create backup directory for $username"
        return 1
    fi
    
    log_message "Creating backup of user data for $username"
    
    # Backup home directory
    if [[ -d "/home/$username" ]]; then
        if sudo cp -r "/home/$username" "$user_backup_dir/home"; then
            log_message "SUCCESS: Backed up home directory for $username"
        else
            log_message "WARNING: Failed to backup home directory for $username"
        fi
    fi
    
    # Backup scratch directory
    if [[ -d "/scratch/users/$username" ]]; then
        if sudo cp -r "/scratch/users/$username" "$user_backup_dir/scratch"; then
            log_message "SUCCESS: Backed up scratch directory for $username"
        else
            log_message "WARNING: Failed to backup scratch directory for $username"
        fi
    fi
    
    # Create user info file
    {
        echo "User: $username"
        echo "Deletion Date: $(date)"
        echo "UID: $(id -u $username 2>/dev/null || echo 'N/A')"
        echo "GID: $(id -g $username 2>/dev/null || echo 'N/A')"
        echo "Groups: $(groups $username 2>/dev/null || echo 'N/A')"
        echo "Shell: $(getent passwd $username | cut -d: -f7 2>/dev/null || echo 'N/A')"
        echo "Home: $(getent passwd $username | cut -d: -f6 2>/dev/null || echo 'N/A')"
    } | sudo tee "$user_backup_dir/user_info.txt" > /dev/null
    
    # Set proper permissions for backup
    sudo chmod -R 600 "$user_backup_dir"
    sudo chown -R root:root "$user_backup_dir"
    
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
    if [[ -d "/scratch/users/$username" ]]; then
        if sudo rm -rf "/scratch/users/$username"; then
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
    
    # Validate username
    if [[ ! "$username" =~ ^[a-z][a-z0-9_-]*$ ]]; then
        log_message "ERROR: Invalid username '$username'"
        return 1
    fi
    
    # Check if user exists
    if ! id "$username" &>/dev/null; then
        log_message "ERROR: User '$username' does not exist"
        return 1
    fi
    
    log_message "Starting deletion process for user: $username"
    
    # Create backup unless skipped
    if [[ "$skip_backup" != "true" ]]; then
        if ! backup_user_data "$username"; then
            read -p "Backup failed. Continue with deletion? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_message "ABORTED: User deletion cancelled for $username"
                return 1
            fi
        fi
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

# Function to process CSV file
process_csv() {
    local csv_file="$1"
    local skip_backup="$2"
    
    if [[ ! -f "$csv_file" ]]; then
        echo "Error: CSV file '$csv_file' not found"
        exit 1
    fi
    
    log_message "Starting bulk user deletion from $csv_file"
    
    local success_count=0
    local error_count=0
    
    # Read CSV (skip header line)
    tail -n +2 "$csv_file" | while IFS=',' read -r username password fullname email || [[ -n "$username" ]]; do
        # Skip empty lines
        [[ -z "$username" ]] && continue
        
        # Remove quotes and whitespace
        username=$(echo "$username" | tr -d '"' | xargs)
        
        if delete_user "$username" "$skip_backup"; then
            ((success_count++))
        else
            ((error_count++))
        fi
    done
    
    log_message "Bulk deletion completed: $success_count successful, $error_count errors"
}

# Function to confirm deletion
confirm_deletion() {
    local target="$1"
    
    echo "WARNING: This will permanently delete user account(s) and all associated data!"
    echo "Target: $target"
    echo "Backup directory: $BACKUP_DIR"
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
    - Creates backup of user data in $BACKUP_DIR
    - Kills user processes before deletion
    - Removes quotas and custom directories
    - Confirmation prompt before proceeding
    - Comprehensive logging

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
    local skip_backup="false"
    
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
            confirm_deletion "user $2"
            delete_user "$2" "$skip_backup"
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
            confirm_deletion "users from $2"
            process_csv "$2" "$skip_backup"
            ;;
        *.csv)
            confirm_deletion "users from $1"
            process_csv "$1" "$skip_backup"
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
}

# Run main function with all arguments
main "$@"