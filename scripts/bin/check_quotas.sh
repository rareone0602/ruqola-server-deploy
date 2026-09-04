#!/bin/bash
# check_quotas.sh -- email users whose home directory is over its soft disk quota.
# Run by an administrator; repquota needs root. Not scheduled on this host.
#
#   sudo check_quotas.sh             # notify
#   sudo check_quotas.sh --dry-run   # show who would be notified, send nothing
set -uo pipefail

ADMIN_EMAIL="${CHECK_QUOTAS_ADMIN_EMAIL:-mjolnirruqola@gmail.com}"
DRY_RUN="${CHECK_QUOTAS_DRYRUN:-}"
SUBJECT="Disk Quota Warning"

# --- shared library: lib/log.sh lib/mail.sh lib/fs.sh -----------------------
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${RUQOLA_ADMIN_LIB:-}" ]]; then
    if [[ -f "$_here/../lib/init.sh" ]]; then RUQOLA_ADMIN_LIB="$_here/../lib"   # running from the repo
    else RUQOLA_ADMIN_LIB=/usr/local/lib/ruqola-admin; fi                        # installed
fi
source "$RUQOLA_ADMIN_LIB/init.sh" || { echo "check_quotas.sh: cannot load shared library from $RUQOLA_ADMIN_LIB" >&2; exit 1; }

# repquota -as prints one line per account:
#   name  status  used  soft  hard  [grace]  files  soft  hard  [grace]
# status is two characters, block flag then inode flag; '+' means over the
# soft limit. Only the BLOCK flag is acted on, as this script always did.
over_block_soft_limit() {
    local out
    out=$(repquota -as 2>/dev/null) || return 1
    awk '$2 ~ /^\+/ {print $1, $3, $4, $5}' <<<"$out"
}

main() {
    local user used soft hard to name notified=0 skipped=0
    local rows
    if ! rows=$(over_block_soft_limit); then
        echo "check_quotas.sh: repquota failed (run as root?)" >&2; exit 1
    fi
    while read -r user used soft hard; do
        [[ -n "$user" ]] || continue
        to=$(email_for_user "$user"); name=$(full_name_for_user "$user")
        if send_mail "$to" "$SUBJECT" "Hello ${name},

This is an automated notification from the server.
Your home directory is over its allocated disk quota.

Your current usage: ${used}B
Your soft limit:   ${soft}B
Your hard limit:   ${hard}B

Please remove unnecessary files to get back under your quota. If you fail to do so, you may be unable to save new files.

Thank you,
System Administrator"
        then
            echo "Quota warning: $user <$to>"; notified=$((notified + 1))
        else
            echo "WARNING: no email address on account '$user'; not notified" >&2; skipped=$((skipped + 1))
        fi
    done <<<"$rows"
    echo "$notified notified, $skipped without an address"
}

usage() {
    cat <<USAGE
Usage: check_quotas.sh [--dry-run]

Emails every account whose home-directory BLOCK usage is over its soft quota
(the '+' in the first status column of 'repquota -as'). Addresses come from
the GECOS field, the same way scratch-cleanup.sh and gpuq find them.

  --dry-run    list who would be emailed; send nothing
  -h, --help   this text
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
main
