#!/bin/bash
# check_quotas.sh: same recipient rule as the reaper, one mail per over-quota user.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
QUOTAS="$ROOT/bin/check_quotas.sh"

report() {   # writes a repquota -as report into the sandbox
cat > "$SB/repquota.txt" <<'R'
*** Report for user quotas on device /dev/nvme0n1p2
Block grace time: 7days; Inode grace time: 7days
                        Block limits                File limits
User            used    soft    hard  grace    used  soft  hard  grace
----------------------------------------------------------------------
root      --   1234M       0       0           1234     0     0
alice     +-     95G     90G    100G  6days   12345     0     0
helen     ++     99G     90G    100G  6days   99999  1000  2000  6days
bob       +-     91G     90G    100G  6days     100     0     0
carol     --     10G     90G    100G            100     0     0
dave      -+      1G     90G    100G           5000  1000  2000  6days
R
export FAKE_REPQUOTA="$SB/repquota.txt"
}
run_quotas() { PATH="$STUBS:$PATH" MSMTP="$STUBS/msmtp" bash "$QUOTAS" "$@" >"$SB/out" 2>"$SB/err"; echo $? > "$SB/exit"; }

t "Over the block soft limit -> one mail each, to the GECOS address"
new_sandbox; report
add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
add_user helen "helenfan27@gmail.com"
add_user bob   "Bob Nobody"
add_user carol "Carol,,,,carol@x.org"
add_user dave  "Dave,,,,dave@x.org"
run_quotas
check "exactly two mails (alice, helen); bob has no address" "$(mail_count)" "2"
check "alice via five-field GECOS" "$(mail_recipients | grep -c '^alice@ntu.edu.sg$')" "1"
check "helen via bare-email GECOS" "$(mail_recipients | grep -c '^helenfan27@gmail.com$')" "1"
check "carol (under quota) not mailed" "$(mail_recipients | grep -c carol)" "0"
check "dave (inode-only '-+') not mailed" "$(mail_recipients | grep -c dave)" "0"
check "bob reported on stderr" "$(grep -c "no email address on account 'bob'" "$SB/err")" "1"
check "alice's body quotes her usage" "$(mail_bodies | grep -c 'Your current usage: 95GB')" "1"
check "both bodies quote the limits" "$(mail_bodies | grep -c 'Your hard limit:   100GB')" "2"
check "greets by name" "$(mail_bodies | grep -c '^Hello Alice Smith,')" "1"
check "summary line" "$(grep -c '^2 notified, 1 without an address$' "$SB/out")" "1"
check "exit 0" "$(cat "$SB/exit")" "0"
drop_sandbox

t "Dry run sends nothing and says who it would have mailed"
new_sandbox; report; add_user alice "Alice Smith,,,,alice@ntu.edu.sg"
run_quotas --dry-run
check "no mail" "$(mail_count)" "0"
check "names alice" "$(grep -c 'would email alice@ntu.edu.sg' "$SB/err")" "1"
drop_sandbox

t "Failure modes"
new_sandbox   # no FAKE_REPQUOTA -> the stub fails like repquota without root
run_quotas
check "repquota failure -> exit 1" "$(cat "$SB/exit")" "1"
check "...with a hint" "$(grep -c 'run as root' "$SB/err")" "1"
run_quotas --bogus
check "unknown flag -> exit 2" "$(cat "$SB/exit")" "2"
run_quotas --help
check "--help -> exit 0" "$(cat "$SB/exit")" "0"
drop_sandbox

finish
