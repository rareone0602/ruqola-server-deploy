#!/bin/bash

# --- Configuration ---
# Email address for the system administrator
ADMIN_EMAIL="mjolnirruqola@gmail.com"

# Subject line for the notification email
SUBJECT="Disk Quota Warning"

# --- Script Logic ---
# Get a clean report from repquota, skipping the header and filesystem totals
# We are looking for users whose block usage has a '+' (over soft limit)
repquota -as | grep -E '\w+\s+\+' | while read -r line; do
    # Extract the username from the line
    USERNAME=$(echo "$line" | awk '{print $1}')

    # Extract usage details
    USED_BLOCKS=$(echo "$line" | awk '{print $3}')
    SOFT_LIMIT=$(echo "$line" | awk '{print $4}')
    HARD_LIMIT=$(echo "$line" | awk '{print $5}')

    # Form the user's email address
    USER_EMAIL=$(getent passwd ${USERNAME} | awk -F ':' '{print $5}' | awk -F ',' '{print $5}') 

    USER_FULL_NAME=$(getent passwd ${USERNAME} | awk -F ':' '{print $5}' | awk -F ',' '{print $1}')

    # Create the email message body
    MAIL=$(cat <<EOF
To: ${USER_EMAIL}
From: ${ADMIN_EMAIL}
Subject: ${SUBJECT}

Hello ${USER_FULL_NAME},

This is an automated notification from the server.
Your home directory is over its allocated disk quota.

Your current usage: ${USED_BLOCKS}B
Your soft limit:   ${SOFT_LIMIT}B
Your hard limit:   ${HARD_LIMIT}B

Please remove unnecessary files to get back under your quota. If you fail to do so, you may be unable to save new files.

Thank you,
System Administrator
EOF
)

    # Send the email using msmtp
    echo "$MAIL" | /usr/bin/msmtp "$USER_EMAIL"
    echo "Sent quota warning to $USER_EMAIL"
done