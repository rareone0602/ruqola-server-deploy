#!/bin/bash
# add_users.sh / delete_users.sh: argument handling the docs promise.
# These scripts call useradd/userdel; the tests only exercise parsing by
# sourcing them and replacing the functions that touch the system.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ADD="$ROOT/bin/add_users.sh"; DEL="$ROOT/bin/delete_users.sh"

t "Both scripts parse and answer --help"
bash -n "$ADD"; check "add_users.sh syntax" "$?" "0"
bash -n "$DEL"; check "delete_users.sh syntax" "$?" "0"
with_stubs bash "$DEL" --help >/dev/null 2>&1; check "delete_users.sh --help exits 0" "$?" "0"
check "add_users.sh --help documents the email column" "$(with_stubs bash "$ADD" --help 2>/dev/null | grep -c 'username,password,fullname,email')" "1"

t "add_users.sh: the 4th CSV column is the email (the field the old script overwrote)"
new_sandbox
printf 'username,password,fullname,email\n"jsmith","pw1","John Smith","jsmith@example.com"\nagarcia,pw2,Ana Garcia,agarcia@example.com\n' > "$SB/users.csv"
( source "$ADD"; LOG_FILE="$SB/creation.log"
  create_user() { printf '%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" >> "$SB/calls"; return 0; }
  process_csv "$SB/users.csv" >/dev/null 2>&1 )
check "two users processed" "$(wc -l < "$SB/calls")" "2"
check "quoted row: email column reaches create_user as email" "$(sed -n 1p "$SB/calls")" "jsmith|pw1|John Smith|jsmith@example.com"
check "plain row too" "$(sed -n 2p "$SB/calls")" "agarcia|pw2|Ana Garcia|agarcia@example.com"
drop_sandbox

t "delete_users.sh: --no-backup is honoured in every form the docs list"
new_sandbox
printf 'username\nalice\nbob\n' > "$SB/users.csv"
probe() {   # run main with the system-touching functions replaced; print skip_backup per call
    ( source "$DEL"; BACKUP_DIR="$SB/backups"; LOG_FILE="$SB/del.log"
      confirm_deletion() { :; }
      delete_user() { printf '%s:%s\n' "$1" "$2" >> "$SB/calls"; return 0; }
      PATH="$STUBS:$PATH" main "$@" >/dev/null 2>&1 )
    cat "$SB/calls" 2>/dev/null; : > "$SB/calls"
}
check "bare CSV + --no-backup"  "$(probe "$SB/users.csv" --no-backup | tr '\n' ' ')" "alice:true bob:true "
check "--csv CSV --no-backup"   "$(probe --csv "$SB/users.csv" --no-backup | tr '\n' ' ')" "alice:true bob:true "
check "--single u --no-backup"  "$(probe --single carol --no-backup)" "carol:true"
check "--single u (default keeps backup)" "$(probe --single carol)" "carol:false"
check "bare CSV (default keeps backup)" "$(probe "$SB/users.csv" | tr '\n' ' ')" "alice:false bob:false "
drop_sandbox

finish
