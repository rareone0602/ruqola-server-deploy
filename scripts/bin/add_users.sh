#!/bin/bash
# Research Group User Management Script
# Usage: ./create_research_users.sh users.csv

# Configuration
QUOTA_SOFT="90G"     # Soft limit (warning)
QUOTA_HARD="100G"     # Hard limit (enforced)
DEFAULT_SHELL="/bin/bash"
USER_GROUPS="users,scratch-users,gpuqueue"  # Standard research user groups
LOG_FILE="/var/log/user_creation.log"
ADMIN_EMAIL="mjolnirruqola@gmail.com"
DOCS_URL="https://ighina.github.io/ruqola-server-deploy/"

# --- shared library: lib/log.sh lib/mail.sh lib/fs.sh -----------------------
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${RUQOLA_ADMIN_LIB:-}" ]]; then
    if [[ -f "$_here/../lib/init.sh" ]]; then RUQOLA_ADMIN_LIB="$_here/../lib"   # running from the repo
    else RUQOLA_ADMIN_LIB=/usr/local/lib/ruqola-admin; fi                        # installed
fi
source "$RUQOLA_ADMIN_LIB/init.sh" || { echo "add_users.sh: cannot load shared library from $RUQOLA_ADMIN_LIB" >&2; exit 1; }

# This script's own one-argument logger (defined AFTER the library so it wins).
# The library's send_mail logs through _mail_log, which tolerates this form.
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" | tee -a "$LOG_FILE"
}


# Function to check if quota is enabled
check_quota_support() {
    if ! command -v setquota &> /dev/null; then
        echo "Error: quota tools not installed. Install with:"
        echo "sudo apt install quota quotatool"
        exit 1
    fi
    
    # Check if quotas are enabled on filesystem
    if ! quotaon -p / 2>/dev/null | grep -q "is on"; then
        echo "Warning: Quotas may not be enabled on this filesystem"
        echo "See setup instructions in comments"
    fi
}

# Function to create a single user
create_user() {
    local username="$1"
    local password="$2"
    local fullname="$3"
    local email="$4"
    
    # Validate input
    if [[ ! "$username" =~ ^[a-z][a-z0-9_-]*$ ]]; then
        log_message "ERROR: Invalid username '$username'. Must start with letter, contain only lowercase letters, numbers, underscore, hyphen"
        return 1
    fi
    
    # Check if user already exists
    if id "$username" &>/dev/null; then
        log_message "ERROR: User '$username' already exists"
        return 1
    fi
    
    log_message "Creating user: $username"
    
    # Create user account
    if sudo useradd -m -s "$DEFAULT_SHELL" -c "$fullname" -G "$USER_GROUPS" "$username"; then
        log_message "SUCCESS: Created user account for $username"
    else
        log_message "ERROR: Failed to create user account for $username"
        return 1
    fi
    
    # Set password
    if echo "$username:$password" | sudo chpasswd; then
        log_message "SUCCESS: Set password for $username"
    else
        log_message "ERROR: Failed to set password for $username"
        return 1
    fi
    
    # Set up home directory permissions
    sudo chmod 750 "/home/$username"
    sudo chown "$username:$username" "/home/$username"
    
    # Create common directories for research work
    sudo -u "$username" mkdir -p "/home/$username"/{projects,data,scripts,venvs}
    
    
    sudo mkdir -p "/scratch/users/${username}"
    # Set up scratch user directory permissions
    sudo chmod 750 "/scratch/users/$username"
    sudo chown "$username:$username" "/scratch/users/$username"
    
    # Set disk quota
    if command -v setquota &> /dev/null; then
        # Set user quota (0 = unlimited inodes, soft/hard disk limits)
        if sudo setquota -u "$username" "$QUOTA_SOFT" "$QUOTA_HARD" 0 0 /; then
            log_message "SUCCESS: Set disk quota for $username ($QUOTA_SOFT soft, $QUOTA_HARD hard)"
        else
            log_message "WARNING: Failed to set quota for $username (quotas may not be enabled)"
        fi
    fi
    
    # Force password change on first login
    if sudo passwd -e "$username"; then
        log_message "SUCCESS: Set password expiration for $username (must change on first login)"
    else
        log_message "WARNING: Failed to set password expiration for $username"
    fi

    sudo chfn -o "$email" "$username";
    
    # Welcome mail (HTML). send_mail runs msmtp under sudo when we are not root.
    if send_mail "$email" "User Account Created on Mjolnir" "Hello ${fullname},
<br>
This is an automated notification from the server.<br>
A user account has been created for you with the username:
<br><b>${username}</b>
<br><br>
Please log in with the following password:<br>
<b>$password</b>
<br>
and change it immediately when prompted.<br><br>
Once done it you can start using the server following the documentation at:<br>
<a href=\"${DOCS_URL}\">Mjolnir Documentation</a>
<br><br>
Thank you,<br>
System Administrator" 'text/html; charset="UTF-8"'
    then
        echo "Sent account creation notification to $email"
    else
        log_message "WARNING: could not send the welcome email to '$email' for $username"
    fi

    log_message "COMPLETED: User $username setup finished"
    return 0
}

# Function to process CSV file
process_csv() {
    local csv_file="$1"
    
    if [[ ! -f "$csv_file" ]]; then
        echo "Error: CSV file '$csv_file' not found"
        exit 1
    fi
    
    log_message "Starting bulk user creation from $csv_file"
    
    local success_count=0
    local error_count=0
    
    # Read CSV (skip header line)
    tail -n +2 "$csv_file" | while IFS=',' read -r username password fullname email || [[ -n "$username" ]]; do
        # Skip empty lines
        [[ -z "$username" ]] && continue
        
        # Remove quotes and whitespace
        username=$(echo "$username" | tr -d '"' | xargs)
        password=$(echo "$password" | tr -d '"' | xargs)
        fullname=$(echo "$fullname" | tr -d '"' | xargs)
        # BUGFIX: clean the 4th CSV column (email), not the full-name field.
        # Previously this read "$fullname", overwriting the email with the
        # user's name so notifications and the GECOS email subfield were wrong.
        email=$(echo "$email" | tr -d '"' | xargs)
        
        if create_user "$username" "$password" "$fullname" "$email"; then
            ((success_count++))
        else
            ((error_count++))
        fi
    done
    
    log_message "Bulk creation completed: $success_count successful, $error_count errors"
}

# Function to show usage
show_usage() {
    cat << EOF
Research Group User Management Script

Usage:
    $0 users.csv                                       # Create users from CSV
    $0 --single username password "full name" email    # Create single user
    $0 --help                                          # Show this help

CSV Format (users.csv):
    username,password,fullname,email
    jsmith,mypassword123,John Smith,jsmith@example.com
    agarcia,securepass456,Ana Garcia,agarcia@example.com

Prerequisites:
    - Run as user with sudo privileges
    - For quotas, install: sudo apt install quota quotatool
    - Enable quotas in /etc/fstab (see script comments)

Groups assigned: $USER_GROUPS
Disk quota: $QUOTA_SOFT soft limit, $QUOTA_HARD hard limit
Home directory: /home/username with projects/, data/, scripts/, venvs/ subdirs

EOF
}

# Main script logic
main() {
    # Check if running as root (don't allow this)
    if [[ $EUID -eq 0 ]]; then
        echo "Error: Don't run this script as root. Run as user with sudo privileges."
        exit 1
    fi
    
    # Check quota support
    check_quota_support
    
    case "$1" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --single)
            if [[ $# -ne 5 ]]; then
                echo "Error: --single requires username password \"full name\" email"
                show_usage
                exit 1
            fi
            create_user "$2" "$3" "$4" "$5"
            ;;
        *.csv)
            process_csv "$1"
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

# Quota setup instructions (for reference)
: '
TO ENABLE DISK QUOTAS:

1. Edit /etc/fstab and add usrquota option:
   /dev/sda1 / ext4 defaults,usrquota 0 1

2. Remount filesystem:
   sudo mount -o remount /

3. Initialize quota files:
   sudo quotacheck -cum /
   sudo quotaon /

4. Verify quotas are working:
   sudo repquota -a
'

# Run main only when executed, so tests can source the functions.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
