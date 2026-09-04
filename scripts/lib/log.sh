#!/bin/bash
# Logging, single-instance locking, and root checks.
#
# Reads:  LOG_FILE   log lines are appended here; unset -> stderr
#         LOCK_FILE  used by acquire_lock / release_lock

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

# log_message <LEVEL> <message>
log_message() {
    local level="$1" message="$2" ts
    printf -v ts '%(%Y-%m-%d %H:%M:%S)T' -1
    if [[ -n "${LOG_FILE:-}" ]]; then
        printf '[%s] [%s] %s\n' "$ts" "$level" "$message" >> "$LOG_FILE"
    else
        printf '[%s] [%s] %s\n' "$ts" "$level" "$message" >&2
    fi
    if [[ -t 1 ]]; then
        case "$level" in
            INFO)  echo -e "${GREEN}[$ts] [INFO]${NC} $message" ;;
            WARN)  echo -e "${YELLOW}[$ts] [WARN]${NC} $message" ;;
            ERROR) echo -e "${RED}[$ts] [ERROR]${NC} $message" ;;
            DEBUG) echo -e "${BLUE}[$ts] [DEBUG]${NC} $message" ;;
            *)     echo "[$ts] [$level] $message" ;;
        esac
    fi
}

# acquire_lock: exit 1 if another live instance holds LOCK_FILE; else take it.
acquire_lock() {
    local pid
    if [[ -f "$LOCK_FILE" ]]; then
        pid=$(cat "$LOCK_FILE" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log_message WARN "Another instance is already running (PID: $pid). Exiting."
            exit 1
        fi
        log_message INFO "Removing stale lock file"
        rm -f "$LOCK_FILE"
    fi
    echo $$ > "$LOCK_FILE"
}
release_lock() { rm -f "$LOCK_FILE"; }

# require_root <program-name> / refuse_root <program-name>
require_root() { [[ $EUID -eq 0 ]] || { echo "$1 must run as root:  sudo $1" >&2; exit 1; }; }
refuse_root()  { [[ $EUID -ne 0 ]] || { echo "$1: do not run as root; run as a user with sudo rights" >&2; exit 1; }; }
