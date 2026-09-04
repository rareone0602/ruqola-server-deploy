#!/bin/bash
# Who to email, and the one way every script sends mail.
#
# Reads:  MSMTP        mailer binary            (default /usr/bin/msmtp)
#         ADMIN_EMAIL  From: address            (default mjolnirruqola@gmail.com)
#         DRY_RUN      non-empty -> log instead of sending
#
# /etc/msmtprc is root-only (it holds the Gmail app password), so when the
# caller is not root the mailer runs under sudo, as add_users.sh always did.

# email_for_user <account>
# The notification address, read from the account's GECOS field. GECOS shapes on
# this host vary ("Full Name,,,,addr@x" and bare "addr@x" both occur), so match an
# address ANYWHERE in the field. Same rule as gpuq's email_for_user(), so every
# tool agrees on every account.
email_for_user() {
    getent passwd "$1" 2>/dev/null | awk -F: '{print $5}' \
        | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' | head -1
}

# full_name_for_user <account>
# Falls back to the username when GECOS holds only an address, so nobody is
# greeted by their own email address.
full_name_for_user() {
    local gecos name
    gecos=$(getent passwd "$1" 2>/dev/null | awk -F: '{print $5}')
    name="${gecos%%,*}"
    if [[ -z "$name" || "$name" == *"@"* ]]; then printf '%s' "$1"; else printf '%s' "$name"; fi
}

_mail_log() {
    if declare -F log_message >/dev/null; then log_message "$1" "$2"; else printf '[%s] %s\n' "$1" "$2" >&2; fi
}

# send_mail <to> <subject> <body> [content-type]
# Returns 1 and sends nothing for an empty recipient. Under DRY_RUN, logs the
# intent and returns 0 without contacting anyone.
send_mail() {
    local to="$1" subject="$2" body="$3" ctype="${4:-}"
    [[ -n "$to" ]] || return 1
    if [[ -n "${DRY_RUN:-}" ]]; then
        _mail_log INFO "DRY RUN would email $to: $subject"
        return 0
    fi
    local -a mailer=("${MSMTP:-/usr/bin/msmtp}")
    (( EUID == 0 )) || mailer=(sudo "${mailer[@]}")
    {
        printf 'To: %s\nFrom: %s\nSubject: %s\n' "$to" "${ADMIN_EMAIL:-mjolnirruqola@gmail.com}" "$subject"
        [[ -n "$ctype" ]] && printf 'Content-Type: %s\n' "$ctype"
        printf '\n%s\n' "$body"
    } | "${mailer[@]}" "$to"
}
