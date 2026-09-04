#!/bin/bash
# scratch-cleanup.sh: the warning window and the countdown text.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

t "Ordering: a file must not be warned and deleted in the same run"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/doomed.bin 200 200 alice; run_reaper
check "deletable file produces no 'Imminent' warning" "$(mail_bodies | grep -ci 'imminent')" "0"
drop_sandbox

t "Window: only files between NOTIFY and KEEP get warned"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/inwindow.bin 170 170 alice; run_reaper
check "file aged 170d (166<x<=180) -> warned" "$(mail_bodies | grep -ci 'imminent')" "1"
check "...and NOT deleted" "$(present "$SB/scratch/users/alice/inwindow.bin")" "present"
drop_sandbox
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/fresh.bin 10 10 alice; run_reaper
check "file aged 10d -> no mail at all" "$(mail_count)" "0"
drop_sandbox

t "Both timestamps count: a recent READ keeps a file out of the window"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/readlately.bin 5 300 alice; run_reaper
check "mtime 300d but atime 5d -> kept, no mail" "$(present "$SB/scratch/users/alice/readlately.bin")/$(mail_count)" "present/0"
drop_sandbox

t "Countdown: never negative, tracks the NEWER timestamp, no hardcoded 23"
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/inwindow.bin 179 179 alice; run_reaper
check "no negative countdown in warning" "$(mail_bodies | grep -c 'within -')" "0"
check "notification period not hardcoded to 23" "$(mail_bodies | grep -c 'notification period of 23 days')" "0"
drop_sandbox
new_sandbox; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"; mkfile users/alice/mixed.bin 170 179 alice; run_reaper
check "atime 170d, mtime 179d -> 10 days remaining (from the newer, atime)" "$(mail_bodies | grep -c 'mixed.bin (10 day(s) remaining)')" "1"
drop_sandbox

finish
