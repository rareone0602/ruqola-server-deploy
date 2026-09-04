#!/bin/bash
# lib/mail.sh: recipient lookup and the one mail-sending path every script uses.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Load the library the way a script would, inside a subshell per test so
# settings do not leak between cases.
load() { source "$ROOT/lib/init.sh"; }

t "email_for_user: address found wherever it sits in GECOS (same rule as gpuq)"
new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
add_user helen "helenfan27@gmail.com"
add_user bob   "Bob Nobody"
check "five-field GECOS" "$(with_stubs bash -c 'source "$0/init.sh"; email_for_user alice' "$ROOT/lib")" "alice@ntu.edu.sg"
check "bare-email GECOS" "$(with_stubs bash -c 'source "$0/init.sh"; email_for_user helen' "$ROOT/lib")" "helenfan27@gmail.com"
check "no address -> empty" "$(with_stubs bash -c 'source "$0/init.sh"; email_for_user bob' "$ROOT/lib")" ""
check "unknown account -> empty" "$(with_stubs bash -c 'source "$0/init.sh"; email_for_user nosuch' "$ROOT/lib")" ""
drop_sandbox

t "full_name_for_user: never greets someone by their own email address"
new_sandbox
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
add_user helen "helenfan27@gmail.com"
check "name from GECOS" "$(with_stubs bash -c 'source "$0/init.sh"; full_name_for_user alice' "$ROOT/lib")" "Alice Smith"
check "bare-email GECOS -> username" "$(with_stubs bash -c 'source "$0/init.sh"; full_name_for_user helen' "$ROOT/lib")" "helen"
drop_sandbox

t "send_mail: refuses an empty recipient, honours DRY_RUN, sends otherwise"
new_sandbox
with_stubs bash -c 'source "$0/init.sh"; MSMTP="$1"; send_mail "" "subj" "body"' "$ROOT/lib" "$STUBS/msmtp"
check "empty recipient -> non-zero, nothing sent" "$?/$(mail_count)" "1/0"
with_stubs bash -c 'source "$0/init.sh"; MSMTP="$1"; DRY_RUN=1; send_mail a@b.co "subj" "body"' "$ROOT/lib" "$STUBS/msmtp" 2>"$SB/err"
check "dry run -> zero, nothing sent" "$?/$(mail_count)" "0/0"
check "dry run says what it would have sent" "$(grep -c 'would email a@b.co' "$SB/err")" "1"
with_stubs bash -c 'source "$0/init.sh"; MSMTP="$1"; send_mail a@b.co "Hello subj" "the body"' "$ROOT/lib" "$STUBS/msmtp"
check "real send -> one message" "$(mail_count)" "1"
check "headers and body present" "$(mail_bodies | grep -c -e '^To: a@b.co' -e '^Subject: Hello subj' -e '^the body')" "3"
check "From is the admin address" "$(mail_bodies | grep -c '^From: mjolnirruqola@gmail.com')" "1"
drop_sandbox

t "send_mail: optional content type for HTML mail, and sudo when not root"
new_sandbox
with_stubs bash -c 'source "$0/init.sh"; MSMTP="$1"; send_mail a@b.co "s" "<b>hi</b>" "text/html; charset=UTF-8"' "$ROOT/lib" "$STUBS/msmtp"
check "Content-Type header emitted when asked" "$(mail_bodies | grep -c '^Content-Type: text/html')" "1"
check "msmtp was run through sudo (config is root-only)" "$(grep -c "^sudo .*msmtp a@b.co" "$STUB_CALLS")" "1"
drop_sandbox

finish
