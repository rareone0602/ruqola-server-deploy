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

# fake_host: call inside a subshell after sourcing delete_users.sh. Accounts come
# from $SB/accounts ("name uid group..."). sudo records every call to $SB/sudo and
# runs only tee/mkdir/chmod/du/cp, all inside the sandbox; FAIL_CP makes cp fail,
# FREE_KB sets what df reports free. Everything else "succeeds" without running.
fake_host() {
    id() {   # id NAME | id -u NAME | id -g NAME | id -nG NAME
        local line; line=$(grep "^${2:-$1} " "$SB/accounts" 2>/dev/null) || return 1
        set -- "$1" $line
        case "$1" in -u|-g) echo "$3" ;; -nG) shift 3; echo "$*" ;; *) echo "uid=$3($2)" ;; esac
    }
    sudo() {
        echo "$*" >> "$SB/sudo"
        case "$1" in
            cp) [[ -z "${FAIL_CP:-}" ]] && "$@" ;;
            tee|mkdir|chmod|du) "$@" ;;
            *) return 0 ;;
        esac
    }
    df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\nfake 0 0 %s 0%% /\n' "${FREE_KB:-999999999}"; }
    pgrep() { return 1; }
    getent() { return 2; }
    LOG_FILE="$SB/del.log"; BACKUP_DIR="$SB/backups"
    HOME_BASE="$SB/home"; SCRATCH_BASE="$SB/scratch/users"
}
# del_run <stdin> <args...>: run main on the fake host; output in $SB/out, $SB/err, $SB/rc.
del_run() {
    : > "$SB/sudo"
    ( source "$DEL"; fake_host; main "${@:2}" ) <<< "$1" > "$SB/out" 2> "$SB/err"
    echo $? > "$SB/rc"
}
sudo_calls() { grep -c "^$1 " "$SB/sudo"; }

t "delete_users.sh: --no-backup is honoured in every form the docs list"
new_sandbox
printf 'username\nalice\nbob\n' > "$SB/users.csv"
probe() {   # run main with the system-touching functions replaced; print skip_backup per call
    ( source "$DEL"; fake_host
      confirm_deletion() { :; }
      delete_user() { printf '%s:%s\n' "$1" "$2" >> "$SB/calls"; return 0; }
      main "$@" >/dev/null 2>&1 )
    cat "$SB/calls" 2>/dev/null; : > "$SB/calls"
}
check "bare CSV + --no-backup"  "$(probe "$SB/users.csv" --no-backup | tr '\n' ' ')" "alice:true bob:true "
check "--csv CSV --no-backup"   "$(probe --csv "$SB/users.csv" --no-backup | tr '\n' ' ')" "alice:true bob:true "
check "--single u --no-backup"  "$(probe --single carol --no-backup)" "carol:true"
check "--single u (default keeps backup)" "$(probe --single carol)" "carol:false"
check "bare CSV (default keeps backup)" "$(probe "$SB/users.csv" | tr '\n' ' ')" "alice:false bob:false "
drop_sandbox

t "delete_users.sh: root, system accounts, nobody and admins are refused before the prompt"
new_sandbox
printf '%s\n' "root 0 root" "ollama 996 ollama" "nobody 65534 nogroup" \
    "boss 1002 boss sudo" "alice 1010 alice users" > "$SB/accounts"
for who in root ollama nobody boss; do
    del_run y --single "$who"
    check "$who: exit 1" "$(cat "$SB/rc")" "1"
    check_contains "$who: says why" "$(cat "$SB/err")" "refusing to delete '$who'"
    check "$who: never asked, never killed, never deleted" \
        "$(grep -c 'will permanently delete' "$SB/out")$(sudo_calls pkill)$(sudo_calls userdel)" "000"
done
check_contains "an admin is told how to proceed" "$(del_run y --single boss; cat "$SB/err")" "remove them from it first"
printf 'username\nalice\nboss\n' > "$SB/users.csv"
del_run y "$SB/users.csv"
check "a CSV naming an admin deletes nobody, not even alice" \
    "$(cat "$SB/rc") $(sudo_calls userdel)" "1 0"
check_contains "and says so" "$(cat "$SB/err")" "Nothing was deleted."
printf 'username\nalice\nAlice Smith\n' > "$SB/users.csv"
del_run y "$SB/users.csv"
check "a malformed name (a shifted column) deletes nobody either" "$(cat "$SB/rc") $(sudo_calls userdel)" "1 0"
: > "$SB/sudo"
( source "$DEL"; fake_host; delete_user root false ) > /dev/null 2>&1
check "delete_user alone still refuses root" "$? $(sudo_calls userdel)" "1 0"
check_contains "and logs it" "$(cat "$SB/del.log")" "REFUSED: Not deleting root: uid 0 is a system account"
drop_sandbox

t "delete_users.sh: an ordinary account is backed up, then deleted"
new_sandbox
echo "alice 1010 alice users" > "$SB/accounts"
mkdir -p "$SB/home/alice" "$SB/scratch/users/alice/run1"
echo weights > "$SB/scratch/users/alice/run1/model.pt"; chmod 640 "$SB/scratch/users/alice/run1/model.pt"
del_run y --single alice
check "exit 0" "$(cat "$SB/rc")" "0"
check_contains "the prompt names the account" "$(cat "$SB/out")" $'\n    alice\n'
check "account removed once" "$(sudo_calls userdel)" "1"
bk=$(ls -d "$SB"/backups/alice_* 2>/dev/null)
check "scratch copied with its content" "$(cat "$bk/scratch/run1/model.pt" 2>/dev/null)" "weights"
check "copy keeps the file's mode, for a plain restore" "$(stat -c %a "$bk/scratch/run1/model.pt" 2>/dev/null)" "640"
check "the backup folder itself is root-only" "$(stat -c %a "$bk" 2>/dev/null)" "700"
check_contains "every line reaches the log through sudo" "$(cat "$SB/sudo")" "tee -a $SB/del.log"
check_contains "the log records the run" "$(cat "$SB/del.log")" "Deletion completed: 1 successful, 0 errors"
del_run n --single alice
check "answering n deletes nothing" "$(cat "$SB/rc") $(sudo_calls userdel)" "0 0"
drop_sandbox

t "delete_users.sh: no backup, no deletion"
new_sandbox
echo "alice 1010 alice users" > "$SB/accounts"
mkdir -p "$SB/home/alice" "$SB/scratch/users/alice"; echo x > "$SB/scratch/users/alice/f"
FAIL_CP=1 del_run y --single alice
check "a failed copy: exit 1" "$(cat "$SB/rc")" "1"
check "nothing killed or deleted" "$(sudo_calls pkill)$(sudo_calls rm)$(sudo_calls userdel)" "000"
check_contains "and the log says what to do" "$(cat "$SB/del.log")" "ABORTED: Backup failed, so alice was not deleted"
FREE_KB=100 del_run y --single alice
check "a backup that would fill / is not attempted" "$(cat "$SB/rc") $(sudo_calls cp) $(sudo_calls userdel)" "1 0 0"
check_contains "and the log gives the sizes" "$(cat "$SB/del.log")" "KB free and must keep"
del_run y --single alice --no-backup
check "--no-backup still deletes without copying" "$(cat "$SB/rc") $(sudo_calls cp) $(sudo_calls userdel)" "0 0 1"
drop_sandbox

t "delete_users.sh: the CSV summary counts every user"
new_sandbox
printf '%s\n' "alice 1010 alice users" "carol 1011 carol users" > "$SB/accounts"
printf 'username,password\r\n"alice",x\r\nbob,x\r\n carol ,x\r\n' > "$SB/users.csv"
del_run y --csv "$SB/users.csv" --no-backup
check "exit 1 because bob does not exist" "$(cat "$SB/rc")" "1"
check "alice and carol deleted" "$(sudo_calls userdel)" "2"
check_contains "summary is 2 and 1, not 0 and 0" "$(cat "$SB/del.log")" "Deletion completed: 2 successful, 1 errors"
drop_sandbox

finish
