#!/bin/bash
# scratch-cleanup.sh: who gets told, and how many messages.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

t "Recipient: address must be found wherever it sits in the GECOS field"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/old.bin 200 200 alice; run_reaper
check "five-field GECOS -> address extracted" "$(mail_recipients | sort -u | head -1)" "alice@ntu.edu.sg"
drop_sandbox
new_sandbox; add_user helen "helenfan27@gmail.com"; mkfile users/helen/old.bin 200 200 helen; run_reaper
check "bare-email GECOS -> address extracted" "$(mail_recipients | sort -u | head -1)" "helenfan27@gmail.com"
drop_sandbox
new_sandbox; add_user nobody "No Email Person"; mkfile users/nobody/old.bin 200 200 nobody; run_reaper
check "GECOS with no address -> zero mails sent" "$(mail_count)" "0"
check "...and the log names the account" "$(reaper_log | grep -c "No email address on account 'nobody'")" "1"
drop_sandbox

t "Volume: one digest per user per run, not one mail per file"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
for i in 1 2 3 4 5; do mkfile "users/alice/old$i.bin" 200 200 alice; done
run_reaper
check "5 deletable files -> 1 email" "$(mail_count)" "1"
check "the one email lists all 5" "$(mail_bodies | grep -c 'old[1-5].bin')" "5"
drop_sandbox

t "Regression: a normal deletion still happens and still notifies"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/old.bin 200 200 alice; run_reaper
check "deletable file is actually removed" "$(present "$SB/scratch/users/alice/old.bin")" "gone"
check "and the owner is told once" "$(mail_count)" "1"
check "greeting uses the name, not the address" "$(mail_bodies | grep -c '^Hello Alice Smith,')" "1"
check "exit status 0" "$(reaper_exit)" "0"
drop_sandbox

finish
