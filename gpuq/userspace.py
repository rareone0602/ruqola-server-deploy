#!/usr/bin/env python3
"""
gpuq — cooperative GPU job queue for a shared host.

Coordinates GPU usage among users in the `gpuqueue` group on a single host.
Each `gpuq submit` claims free GPU(s), sets CUDA_VISIBLE_DEVICES, and runs the
command in the foreground — that process supervises the job, enforces its time
limit, and records usage on exit. There is no daemon. Shared state lives in
/var/lib/gpu_queue/ (SGID-writable by the gpuqueue group).

Design:
  - Foreground execution: each submit is its own executor; no daemon to run.
  - Owned-GPU allocation: a GPU you hold is yours to stack more jobs on; GPUs
    held by other users are off-limits until freed.
  - Opportunistic scheduling: free slots go to whichever submit asks first
    (waiters poll; the queue is informational). Low-priority (over-quota)
    submits always yield to normal-priority waiters.
  - Per-user rolling 7-day GPU-hour quotas; every job is logged to a usage
    ledger (`gpuq history`, `gpuq quota`) so budgets can be set from data.
  - Opt-in notifications; a user's email is read from their account (GECOS).
  - Coordination is cooperative (/dev/nvidia* is world-accessible). `gpuq audit`
    detects resource hogs, over-quota users, and GPU jobs not launched through
    gpuq; `gpuq audit --enforce` kills untracked jobs past their grace deadline.
"""
import argparse
import fcntl
import json
import os
import random
import re
import shlex
import signal
import smtplib
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ---------------------------------------------------------------------------
# Constants (paths overridable via env for tests / dev)
#   GPUQ_QUEUE_DIR    -> shared coordination dir (default /var/lib/gpu_queue)
#   GPUQ_CONFIG_FILE  -> admin config json     (default /usr/local/bin/gpu_queue_config.json)
#   GPUQ_NVSMI        -> nvidia-smi binary     (default `nvidia-smi` on PATH)
# ---------------------------------------------------------------------------
QUEUE_DIR = Path(os.environ.get("GPUQ_QUEUE_DIR", "/var/lib/gpu_queue"))
JOBS_FILE = QUEUE_DIR / "jobs.json"
RUNNING_FILE = QUEUE_DIR / "running.json"
USAGE_FILE = QUEUE_DIR / "usage.jsonl"
LOCK_FILE = QUEUE_DIR / ".lock"
UNTRACKED_STATE_FILE = QUEUE_DIR / "untracked_state.json"
REBIND_STATE_FILE = QUEUE_DIR / "rebind_state.json"
CONFIG_FILE = Path(os.environ.get("GPUQ_CONFIG_FILE", "/usr/local/bin/gpu_queue_config.json"))
NVSMI_BIN = os.environ.get("GPUQ_NVSMI", "nvidia-smi")

DEFAULT_MAX_TIME_HOURS = 24
DEFAULT_MAX_TIME_HOURS_CAP = 48    # hard wall-time ceiling = 2 days (config: max_job_time_hours_cap)
DEFAULT_MAX_MEMORY_GB = 70
GPU_UTIL_AVAILABLE_THRESHOLD = 10  # percent; below this, GPU counts as idle
GPU_OWN_MIN_FREE_GB = 2            # owned-GPU stacking still needs this much free VRAM
QUEUE_POLL_INTERVAL_SEC = 30
KILL_GRACE_SEC = 10
QUOTA_WINDOW_HOURS = 24 * 7        # rolling 7 days
NVSMI_TIMEOUT_SEC = 15             # a wedged driver must not hang gpuq under the lock
COMMAND_LOG_MAX_CHARS = 300        # command text stored per ledger record
DEPRIORITIZED_POLL_INTERVAL_SEC = int(
    os.environ.get("GPUQ_DEPRIORITIZED_POLL_SEC", "120")
)

HOST = socket.gethostname()
USER = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"

# UNIX accounts that legitimately hold GPU memory (display managers, drivers,
# system services). These are never flagged as "untracked" users by `gpuq
# audit`, regardless of the admin-configured allowlist.
SYSTEM_GPU_ACCOUNTS = frozenset({
    "root", "nobody", "gdm", "lightdm", "sddm", "systemd+",
    "_apt", "nvidia-persistenced",
})


# ---------------------------------------------------------------------------
# Shared-file lock
# ---------------------------------------------------------------------------
@contextmanager
def file_lock(path: Path):
    """Exclusive flock on `path`; creates the file group-rw if absent."""
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o664)
    try:
        try:
            os.fchmod(fd, 0o664)
        except OSError:
            pass
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        os.close(fd)


# ---------------------------------------------------------------------------
# JSON I/O on shared state
# ---------------------------------------------------------------------------
def _load(path: Path):
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"warning: could not parse {path}: {e}", file=sys.stderr)
    return []


def _save(path: Path, data):
    # Unique tmp name + group-rw mode from creation: a writer killed mid-save
    # must never leave a fixed-name tmp file another user cannot reopen.
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        try:
            os.fchmod(fd, 0o664)
        except OSError:
            pass
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_running():
    return _load(RUNNING_FILE)


def save_running(jobs):
    _save(RUNNING_FILE, jobs)


def load_queued():
    return _load(JOBS_FILE)


def save_queued(jobs):
    _save(JOBS_FILE, jobs)


def load_config():
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except PermissionError:
        # Falling back to built-in defaults here would silently disable
        # quotas/limits for this user — make the misconfiguration visible.
        print(f"warning: cannot read {CONFIG_FILE} (permission denied); "
              "using built-in defaults. Ask the admin to make it readable "
              "(e.g. root:gpuqueue 640).", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"warning: invalid JSON in {CONFIG_FILE}: {e}; "
              "using built-in defaults.", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# nvidia-smi wrappers (read-only)
# ---------------------------------------------------------------------------
def _nvsmi(query_kind, fields):
    """Run one nvidia-smi query. Returns its stdout, or None on failure
    (missing binary, error exit, or a wedged driver hitting the timeout).
    Callers that must tell 'nvidia-smi failed' from 'nothing running' check
    for None; the parsers below treat None like empty output."""
    try:
        r = subprocess.run(
            [NVSMI_BIN, f"--query-{query_kind}={fields}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
            timeout=NVSMI_TIMEOUT_SEC,
        )
        return r.stdout
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return None


def get_gpu_info():
    out = _nvsmi("gpu", "index,name,memory.used,memory.total,utilization.gpu") or ""
    gpus = []
    for line in out.strip().split("\n"):
        if not line:
            continue
        parts = [s.strip() for s in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            mem_used = int(parts[2])
            mem_total = int(parts[3])
            util = int(parts[4])
        except ValueError:
            continue
        gpus.append({
            "index": idx,
            "name": parts[1],
            "memory_used_mb": mem_used,
            "memory_total_mb": mem_total,
            "memory_free_mb": mem_total - mem_used,
            "utilization": util,
        })
    return gpus


def get_gpu_processes():
    """GPU compute processes, or None if nvidia-smi itself failed (callers in
    the audit path must not mistake a driver blip for 'all processes gone')."""
    out = _nvsmi("compute-apps", "pid,process_name,gpu_uuid,used_memory")
    if out is None:
        return None
    procs = []
    for line in out.strip().split("\n"):
        if not line or "No running processes" in line:
            continue
        parts = [s.strip() for s in line.split(",")]
        if len(parts) >= 4:
            try:
                procs.append({
                    "pid": int(parts[0]),
                    "name": parts[1],
                    "gpu_uuid": parts[2],
                    "memory_mb": int(parts[3]),
                })
            except ValueError:
                pass
    return procs


def get_gpu_uuid_to_index():
    out = _nvsmi("gpu", "index,uuid")
    if out is None:
        return None
    m = {}
    for line in out.strip().split("\n"):
        if not line:
            continue
        parts = [s.strip() for s in line.split(",")]
        if len(parts) >= 2:
            try:
                m[parts[1]] = int(parts[0])
            except ValueError:
                pass
    return m


def get_process_user(pid):
    # `user:32` (not bare `user=`) so logins longer than 8 chars are not
    # truncated to an 8-char "+"-suffixed form; the trailing `=` keeps the
    # column header suppressed. Matching tracked vs. untracked jobs by owner
    # name depends on this not being truncated.
    try:
        r = subprocess.run(["ps", "-o", "user:32=", "-p", str(pid)],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip() or None
    except subprocess.CalledProcessError:
        return None


def get_process_pgid(pid):
    """Process-group id of `pid`, or None if it is gone. No privilege needed."""
    try:
        return os.getpgid(int(pid))
    except (ProcessLookupError, OSError, TypeError, ValueError):
        return None


def get_parent_pid(pid):
    """Parent pid from /proc/<pid>/stat, or None. Linux-only; world-readable."""
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            data = f.read()
    except (OSError, TypeError, ValueError):
        return None
    # comm (field 2) is parenthesised and may contain spaces/parens, so parse
    # the fields AFTER the last ')':  state, ppid, pgrp, ...
    try:
        after = data[data.rindex(")") + 1:].split()
        return int(after[1])
    except (ValueError, IndexError):
        return None


def list_gpu_processes_with_owner():
    """Every GPU compute process joined to its owner, GPU index and pgid.

    Returns None if nvidia-smi failed (so audit callers can skip the run
    instead of treating it as 'no processes'). `gpu_idx` is None for a uuid
    not in the full-GPU map (e.g. MIG instances); `owner`/`pgid` are None if
    the process exited between the nvidia-smi and ps/getpgid calls. Callers
    decide how to treat the None cases.
    """
    uuid_to_idx = get_gpu_uuid_to_index()
    procs = get_gpu_processes()
    if uuid_to_idx is None or procs is None:
        return None
    out = []
    for p in procs:
        out.append({
            **p,
            "gpu_idx": uuid_to_idx.get(p["gpu_uuid"]),
            "owner": get_process_user(p["pid"]),
            "pgid": get_process_pgid(p["pid"]),
        })
    return out


# ---------------------------------------------------------------------------
# Virtual environment detection
# ---------------------------------------------------------------------------
def detect_virtual_environment():
    if os.environ.get("CONDA_DEFAULT_ENV"):
        path = os.environ.get("CONDA_PREFIX", "")
        return {
            "type": "conda",
            "name": os.environ["CONDA_DEFAULT_ENV"],
            "path": path,
            "python": os.path.join(path, "bin", "python") if path else sys.executable,
        }
    if os.environ.get("VIRTUAL_ENV"):
        path = os.environ["VIRTUAL_ENV"]
        return {
            "type": "venv",
            "name": os.path.basename(path),
            "path": path,
            "python": os.path.join(path, "bin", "python"),
        }
    if os.environ.get("PYENV_VERSION"):
        return {
            "type": "pyenv",
            "name": os.environ["PYENV_VERSION"],
            "path": "",
            "python": sys.executable,
        }
    return None


# ---------------------------------------------------------------------------
# Reaping stale entries
# ---------------------------------------------------------------------------
def is_pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, owned by another user
    except (OSError, TypeError, ValueError):
        return False


def pid_start_time(pid):
    """Kernel start time of `pid` (field 22 of /proc/<pid>/stat), or None.

    Recorded at claim time and compared at reap time so a recycled PID (same
    number, different process — e.g. after a reboot) is not mistaken for a
    still-running supervisor."""
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            data = f.read()
        return int(data[data.rindex(")") + 1:].split()[19])
    except (OSError, ValueError, IndexError, TypeError):
        return None


def entry_pid_alive(entry):
    """Is the entry's recorded supervisor process still the same live process?"""
    pid = entry.get("pid")
    if not is_pid_alive(pid):
        return False
    recorded = entry.get("pid_start")
    if recorded is not None:
        current = pid_start_time(pid)
        if current is not None and current != recorded:
            return False  # PID was recycled by an unrelated process
    return True


def is_pgid_alive(pgid):
    try:
        os.killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # group exists, owned by another user
    except (OSError, TypeError, ValueError):
        return False


def child_group_alive(entry):
    """Is the entry's recorded child process group still THIS job's group?

    pgid numbers recycle (and running.json persists across reboots), so a bare
    is_pgid_alive would let an unrelated group pin a dead job's entry forever.
    When the entry recorded the group leader's start time, require it to still
    match; if the leader exited while its workers live on, /proc/<pgid> is
    gone and we conservatively treat the group as alive."""
    pgid = entry.get("child_pgid")
    if pgid is None or not is_pgid_alive(pgid):
        return False
    recorded = entry.get("child_pgid_start")
    if recorded is not None:
        current = pid_start_time(pgid)
        if current is not None and current != recorded:
            return False  # pgid number was recycled by an unrelated group
    return True


def reap_running(jobs):
    """Split running entries into (kept, lost). Pure function, no I/O.

    An entry from this host is kept while its supervisor is alive — or, with
    the supervisor gone, while its child process group still runs (an orphaned
    job: the GPU genuinely stays in use, so it must stay allocated and keep
    accruing live usage). Only when both are gone is the entry `lost`; callers
    that persist the kept list must also call record_lost_running(lost) so the
    job's GPU-hours land in the ledger instead of vanishing.
    Foreign-host entries are not ours to reap.
    """
    kept, lost = [], []
    for j in jobs:
        if j.get("host") != HOST:
            kept.append(j)
            continue
        if entry_pid_alive(j):
            kept.append(j)
        elif child_group_alive(j):
            kept.append(j)  # orphaned but still running on the GPU
        else:
            lost.append(j)
    return kept, lost


def reap_queued(jobs):
    """Split queued entries into (kept, lost): drop this-host entries whose
    waiting submit process is gone (SIGKILL, crash, reboot). Without this, one
    dead normal-priority waiter would starve every deprioritized submitter
    forever via the yield check in _wait_for_slot. Pure function, no I/O."""
    kept, lost = [], []
    for j in jobs:
        if j.get("host") != HOST:
            kept.append(j)
        elif entry_pid_alive(j):
            kept.append(j)
        else:
            lost.append(j)
    return kept, lost


# ---------------------------------------------------------------------------
# Job ledger: one JSON line per job event in usage.jsonl
#
# Record types (the "event" key; absent = "end", which absorbs legacy lines):
#   end       — a job finished; carries the full job context plus exit_code and
#               end_reason (completed|failed|timed_out|killed|lost). Synthetic
#               "lost" records are written by the reaper when a job's
#               supervisor AND child died without reaching run_and_wait's
#               accounting (e.g. SIGKILL of the supervisor).
#   cancelled — a queued submit was abandoned before it ran (Ctrl-C / SIGTERM /
#               waiter process died).
#   rejected  — a submit was refused outright (no free slot without --queue,
#               or pinned --devices unavailable). Records unmet demand.
#
# Compatibility contract (lets old and new gpuq versions share the file):
# every record that should be quota-charged carries user + ended_at + numeric
# gpu_hours; every record that should NOT be charged uses "at" instead of
# "ended_at". Old readers skip records without a parseable ended_at; new
# readers default a missing "event" to "end".
# ---------------------------------------------------------------------------
def append_usage(record):
    """Append one event line to USAGE_FILE. Caller holds the lock."""
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with open(USAGE_FILE, "a") as f:
        f.write(line)
    try:
        os.chmod(USAGE_FILE, 0o664)
    except OSError:
        pass


def _parse_iso(s):
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def iter_usage_records():
    """Yield every parseable ledger record, oldest file first, with `event`
    defaulted to "end". Reads any manually rotated usage-*.jsonl files (sorted,
    so usage-2026-01.jsonl precedes usage-2026-02.jsonl) before the live file.
    One malformed line never aborts the scan."""
    paths = sorted(QUEUE_DIR.glob("usage-*.jsonl")) + [USAGE_FILE]
    for path in paths:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    rec.setdefault("event", "end")
                    yield rec
        except OSError:
            continue


def _record_window_hours(rec, cutoff, now):
    """GPU-hours of one end record that fall inside [cutoff, now], or 0.0.

    Records whose span is known (started_at + ended_at) are clamped to the
    window so a job that straddles the cutoff is only charged for the
    in-window portion; records without a span fall back to full gpu_hours.
    """
    ended = _parse_iso(rec.get("ended_at", ""))
    if ended is None or ended < cutoff:
        return 0.0
    try:
        gpu_hours = float(rec.get("gpu_hours", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    if gpu_hours <= 0:
        return 0.0
    started = _parse_iso(rec.get("started_at", ""))
    if started is None or started >= ended:
        return gpu_hours
    span_h = (ended - started).total_seconds() / 3600.0
    overlap_h = ((min(ended, now) - max(started, cutoff)).total_seconds()
                 / 3600.0)
    if overlap_h <= 0:
        return 0.0
    return gpu_hours * min(1.0, overlap_h / span_h) if span_h > 0 else gpu_hours


def usage_in_window(user, window_hours=QUOTA_WINDOW_HOURS, now=None):
    """Sum gpu_hours for `user` in the trailing window. Reads ledger + running.

    Counts running jobs at their current elapsed runtime so a long-running job
    is reflected in the budget before it ends, and clamps both finished and
    running jobs to the window so hours burned before the cutoff don't count.
    """
    now = now or datetime.now()
    cutoff = now - timedelta(hours=window_hours)
    total = 0.0
    for rec in iter_usage_records():
        if rec.get("event") != "end" or rec.get("user") != user:
            continue
        total += _record_window_hours(rec, cutoff, now)
    for j in load_running():
        if j.get("user") != user:
            continue
        started = _parse_iso(j.get("started_at", ""))
        if started is None:
            continue
        elapsed = max(0.0, (now - max(started, cutoff)).total_seconds() / 3600.0)
        count = len(j.get("gpus") or []) or j.get("gpu_count", 1)
        total += elapsed * count
    return total


def _end_record(job, started, ended, exit_code, reason, synthetic=False):
    """Build a ledger end record from a running-list entry / job dict."""
    elapsed_h = max(0.0, (ended - started).total_seconds() / 3600.0)
    gpus = list(job.get("gpus") or [])
    rec = {
        "v": 2,
        "event": "end",
        "id": job.get("id"),
        "user": job.get("user", USER),
        "host": job.get("host", HOST),
        "command": (job.get("command") or "")[:COMMAND_LOG_MAX_CHARS],
        "name": job.get("name"),
        "gpus_requested": job.get("gpu_count"),
        "gpus": gpus,
        "devices": job.get("devices"),
        "memory_gb": job.get("memory_gb"),
        "max_time_hours": job.get("max_time_hours"),
        "priority": job.get("priority", "normal"),
        "over_quota_at_submit": job.get("over_quota_at_submit", False),
        "submitted_at": job.get("submitted_at"),
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "queue_wait_sec": job.get("queue_wait_sec"),
        "elapsed_hours": round(elapsed_h, 4),
        "gpu_hours": round(elapsed_h * len(gpus), 4),
        "exit_code": exit_code,
        "end_reason": reason,
    }
    if synthetic:
        rec["synthetic"] = True
    return rec


def record_lost_running(lost, now=None):
    """Write a synthetic end record for each reaped running entry whose
    supervisor and child died without reaching run_and_wait's accounting.
    Caller holds the lock and has already persisted the kept list. Charging
    runs to the reap time but is capped at the job's own time limit (the
    timer would have killed it there anyway)."""
    now = now or datetime.now()
    for j in lost:
        started = _parse_iso(j.get("started_at", "")) or now
        ended = now
        max_h = j.get("max_time_hours")
        try:
            if max_h and float(max_h) > 0:
                cap = started + timedelta(hours=float(max_h))
                ended = min(now, cap)
        except (TypeError, ValueError):
            pass
        if ended < started:
            ended = started
        append_usage(_end_record(j, started, ended, None, "lost",
                                 synthetic=True))


def _cancelled_record(j, reason, now):
    """Ledger event for a queued submit that never ran. Uses "at" (not
    ended_at) so quota readers — old and new — never charge these."""
    submitted = _parse_iso(j.get("submitted_at", ""))
    wait_sec = int((now - submitted).total_seconds()) if submitted else None
    return {
        "v": 2,
        "event": "cancelled",
        "id": j.get("id"),
        "user": j.get("user", USER),
        "host": j.get("host", HOST),
        "command": (j.get("command") or "")[:COMMAND_LOG_MAX_CHARS],
        "name": j.get("name"),
        "gpus_requested": j.get("gpu_count"),
        "devices": j.get("devices"),
        "memory_gb": j.get("memory_gb"),
        "max_time_hours": j.get("max_time_hours"),
        "priority": j.get("priority", "normal"),
        "submitted_at": j.get("submitted_at"),
        "at": now.isoformat(timespec="seconds"),
        "wait_sec": wait_sec,
        "reason": reason,
    }


def record_lost_queued(lost, now=None):
    """Write a cancelled event for each queued entry whose waiting submit
    process died. Caller holds the lock."""
    now = now or datetime.now()
    for j in lost:
        append_usage(_cancelled_record(j, "lost", now))


def record_rejected_submit(args, memory_gb, max_time_hours, reason,
                           devices=None, now=None):
    """Log a submit that was refused outright (unmet demand). Takes the lock
    itself — the rejection paths run outside it."""
    now = now or datetime.now()
    rec = {
        "v": 2,
        "event": "rejected",
        "user": USER,
        "host": HOST,
        "at": now.isoformat(timespec="seconds"),
        "reason": reason,
        "gpus_requested": args.gpus,
        "devices": devices,
        "memory_gb": memory_gb,
        "max_time_hours": max_time_hours,
        "name": args.name,
    }
    try:
        with file_lock(LOCK_FILE):
            append_usage(rec)
    except OSError:
        pass  # never let logging break the user-facing error path


def reap_all_locked():
    """Reap running + queued state and persist the survivors; synthesize
    ledger records for what died. Caller holds the lock. Returns the kept
    (running, queued) lists."""
    running, lost_r = reap_running(load_running())
    if lost_r:
        save_running(running)
        record_lost_running(lost_r)
    queued, lost_q = reap_queued(load_queued())
    if lost_q:
        save_queued(queued)
        record_lost_queued(lost_q)
    return running, queued


def quota_for_user(user, config):
    """Return budget in GPU-hours per QUOTA_WINDOW_HOURS, or None for unlimited."""
    q = config.get("quotas", {}) or {}
    users = q.get("users", {}) or {}
    if user in users:
        v = users[user]
    else:
        v = q.get("default_gpu_hours_per_week")
    if v is None or v == 0:
        return None
    try:
        v = float(v)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def would_exceed_quota(user, requested_gpu_hours, config):
    """Returns (over: bool, used: float, budget: float|None)."""
    budget = quota_for_user(user, config)
    if budget is None:
        return False, 0.0, None
    used = usage_in_window(user)
    return (used + requested_gpu_hours) > budget, used, budget


def quota_delay_hours(config):
    """Hours an over-quota submit is HELD before it may claim any slot (the
    deprioritized low-priority queue applies after the hold). 0 = no hold.
    Misconfigured values fail OFF, loudly: a hand-edited live config must
    never silently change policy (JSON true would otherwise become a 1h hold).
    """
    q = config.get("quotas", {}) or {}
    raw = q.get("delay_hours")
    if isinstance(raw, bool):
        print(f"warning: quotas.delay_hours must be a number, got {raw!r}; "
              "treating the hold as off.", file=sys.stderr)
        return 0.0
    try:
        v = float(raw or 0)
    except (TypeError, ValueError):
        print(f"warning: quotas.delay_hours is not a number ({raw!r}); "
              "treating the hold as off.", file=sys.stderr)
        return 0.0
    return v if v > 0 else 0.0


def user_card_cap(config):
    """Per-user concurrent-card hard cap (max_gpus_per_user_hard); 0 = off.
    Misconfigured values fail OFF, loudly: JSON true would otherwise become
    int(True) == a silent cap of ONE card for every user on the host."""
    raw = config.get("max_gpus_per_user_hard", 0)
    if isinstance(raw, bool):
        print(f"warning: max_gpus_per_user_hard must be a number, got {raw!r}; "
              "treating the cap as off.", file=sys.stderr)
        return 0
    try:
        v = int(raw or 0)
    except (TypeError, ValueError):
        print(f"warning: max_gpus_per_user_hard is not a number ({raw!r}); "
              "treating the cap as off.", file=sys.stderr)
        return 0
    return v if v > 0 else 0


# ---------------------------------------------------------------------------
# GPU slot picking (ownership-aware)
# ---------------------------------------------------------------------------
def _gpu_owner_sets(running_jobs, user):
    """Split GPUs held by running jobs into those `user` owns and those held by
    other users. A GPU co-tenanted by both counts as other-held — we never hand
    a user a card someone else is on. A job with no recorded user is treated as
    other-held (safe: such a card is never auto-claimed). Only this host's jobs
    count: GPU indices from records written under another hostname (e.g. before
    a rename) are meaningless here and must not block local cards."""
    owned, others = set(), set()
    for j in running_jobs:
        if j.get("host", HOST) != HOST:
            continue
        target = owned if j.get("user") == user else others
        for gi in j.get("gpus", []):
            target.add(int(gi))
    owned -= others
    return owned, others


def free_gpus(want_memory_gb, gpus, running_jobs, user):
    """Sorted indices of GPUs selectable for `user`.

    Two kinds qualify: GPUs in no running job that have >= want_memory_gb free
    VRAM and utilization below the idle threshold, PLUS GPUs `user` already owns
    ("you own your allocated GPU" — stacking). An owned GPU skips the utilization
    gate (the user's own job legitimately drives util up) but must still clear the
    SAME free-VRAM admission filter the submitter asked for — at least
    max(GPU_OWN_MIN_FREE_GB, want_memory_gb) free — so an explicit -m is honored
    when stacking too, and a small floor always guards against a doomed-OOM stack
    when -m is tiny. GPUs held by other users are never selectable."""
    want_mem_mb = want_memory_gb * 1024
    own_min_mb = GPU_OWN_MIN_FREE_GB * 1024
    owned, others = _gpu_owner_sets(running_jobs, user)
    out = []
    for g in gpus:
        idx = g["index"]
        if idx in others:
            continue
        if idx in owned:
            if g["memory_free_mb"] >= max(own_min_mb, want_mem_mb):
                out.append(idx)
        elif (g["memory_free_mb"] >= want_mem_mb
                and g["utilization"] < GPU_UTIL_AVAILABLE_THRESHOLD):
            out.append(idx)
    return sorted(out)


def _user_held_gpus(running_jobs, user):
    """Distinct GPU indices `user`'s running jobs occupy on this host. Includes
    cards co-tenanted with another user: they count toward the user's
    concurrent-card cap even though they are not stackable."""
    held = set()
    for j in running_jobs:
        if j.get("host", HOST) != HOST or j.get("user") != user:
            continue
        for gi in j.get("gpus", []):
            held.add(int(gi))
    return held


def pick_gpus(want_count, want_memory_gb, gpus, running_jobs, user, devices=None,
              hard_cap=0):
    """Choose GPU indices to run on, or None if the request can't be met now.

    With `devices`, pin exactly those indices — all must be selectable for `user`
    (free, or already owned by them), else None. Otherwise choose `want_count`
    among the selectable GPUs, preferring FREE GPUs over ones the user already
    owns so load spreads across the box before stacking; the free choice is
    random (as before) to avoid always starting at GPU 0.

    With hard_cap > 0, the pick may never leave `user` holding more than
    hard_cap DISTINCT cards on this host: new-card claims are limited to the
    remaining headroom, the rest of the request redirects onto cards they
    already own (stacking), and a request that cannot fit returns None.
    """
    free = free_gpus(want_memory_gb, gpus, running_jobs, user)
    held = _user_held_gpus(running_jobs, user) if hard_cap > 0 else set()
    if devices is not None:
        want = sorted(set(devices))
        # Charge only NEW cards against the headroom, like the default path:
        # a user already over the cap (cards claimed before it was enabled or
        # lowered) may still pin cards they hold — that is stacking, the exact
        # behavior the cap redirects everyone toward.
        if hard_cap > 0 and (len(set(want) - held)
                             > max(0, hard_cap - len(held))):
            return None
        return want if all(d in free for d in want) else None
    owned, _ = _gpu_owner_sets(running_jobs, user)
    free_only = [i for i in free if i not in owned]
    owned_sel = [i for i in free if i in owned]
    take_free = min(want_count, len(free_only))
    if hard_cap > 0:
        take_free = min(take_free, max(0, hard_cap - len(held)))
    topup = want_count - take_free
    if topup > len(owned_sel):
        return None
    picked = random.sample(free_only, take_free) if take_free else []
    if topup:
        picked += random.sample(owned_sel, topup)
    return sorted(picked)


def gpu_unavailability(devices, gpus, running_jobs, want_memory_gb, user,
                       hard_cap=0):
    """Human-readable reason each pinned GPU can't be used by `user`; [] means all
    usable. Mirrors free_gpus/pick_gpus exactly so a reason never contradicts the
    picker: a GPU `user` owns is usable while it keeps max(GPU_OWN_MIN_FREE_GB,
    want_memory_gb) free (util ignored, so an explicit -m is honored when
    stacking); a GPU held by another user is reported as held; a free GPU uses the
    normal util/VRAM gates; with hard_cap set, a pin whose NEW cards exceed the
    user's remaining card-cap headroom gets a request-level cap reason."""
    want_mem_mb = want_memory_gb * 1024
    own_min_mb = GPU_OWN_MIN_FREE_GB * 1024
    by_idx = {g["index"]: g for g in gpus}
    owned, others = _gpu_owner_sets(running_jobs, user)
    held = {}
    for j in running_jobs:
        if j.get("user") == user or j.get("host", HOST) != HOST:
            continue
        for gi in j.get("gpus", []):
            held[int(gi)] = j
    reasons = []
    for d in sorted(set(devices)):
        g = by_idx.get(d)
        if g is None:
            reasons.append(f"GPU {d}: no such GPU on {HOST}")
        elif d in others:
            j = held.get(d, {})
            reasons.append(f"GPU {d}: held by gpuq job {j.get('id')} ({j.get('user')})")
        elif d in owned:
            need_mb = max(own_min_mb, want_mem_mb)
            if g["memory_free_mb"] < need_mb:
                reasons.append(f"GPU {d}: only {g['memory_free_mb'] // 1024} GB free, "
                               f"need {need_mb // 1024} GB to stack on your own card")
        elif g["utilization"] >= GPU_UTIL_AVAILABLE_THRESHOLD:
            reasons.append(f"GPU {d}: in use ({g['utilization']}% util)")
        elif g["memory_free_mb"] < want_mem_mb:
            reasons.append(f"GPU {d}: only {g['memory_free_mb'] // 1024} GB free "
                           f"(need {want_memory_gb} GB)")
    if hard_cap > 0:
        held = _user_held_gpus(running_jobs, user)
        new = sorted(set(devices) - held)
        if len(new) > max(0, hard_cap - len(held)):
            reasons.append(
                f"per-user card cap: you hold {len(held)} card(s) "
                f"{sorted(held)} of the {hard_cap}-card cap; pinning "
                f"{len(new)} new card(s) {new} exceeds it")
    return reasons


def _normal_waiter_can_claim(queued, running_jobs, gpus, hard_cap):
    """Could any normal-priority waiter queued on this host claim a slot right
    now? Deprioritized submitters yield only to waiters who actually can: a
    normal waiter wedged by the per-user card cap (or VRAM/pinning) must not
    starve low-priority jobs forever while a card sits idle."""
    for j in queued:
        if j.get("priority", "normal") != "normal" or j.get("host") != HOST:
            continue
        try:
            want = int(j.get("gpu_count") or 1)
            mem = float(j.get("memory_gb") or 0)
        except (TypeError, ValueError):
            return True   # unparseable entry: be conservative and yield
        if pick_gpus(want, mem, gpus, running_jobs, j.get("user"),
                     devices=j.get("devices"), hard_cap=hard_cap) is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# Notifications (opt-in)
# ---------------------------------------------------------------------------
def send_email(to_email, subject, body, config):
    nec = config.get("notification_email", {})
    if not nec.get("enabled") or not to_email:
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = nec.get("username", "")
        msg["To"] = to_email
        with smtplib.SMTP(nec["smtp_server"], nec.get("smtp_port", 587), timeout=10) as s:
            s.starttls()
            s.login(nec["username"], nec["password"])
            s.sendmail(nec["username"], [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"warning: email notification failed: {e}", file=sys.stderr)
        return False


def send_slack(text, config):
    sc = config.get("slack", {})
    if not sc.get("enabled") or not _HAS_REQUESTS:
        return False
    try:
        requests.post(sc["webhook_url"], json={"text": text}, timeout=10)
        return True
    except Exception as e:
        print(f"warning: slack notification failed: {e}", file=sys.stderr)
        return False


def email_for_user(user):
    """Notification address for `user`, read from their account's GECOS field.

    Accounts here are provisioned (create_users) with the email in GECOS, so the
    account is the single source of truth — there is no user_emails config map.
    GECOS is field 5 of `getent passwd`; the address may be the 5th comma-subfield
    (",,,,addr") or the whole field. Returns None if the account or its email is
    absent (the caller then skips the personal email; admins still see the breach).
    """
    if not user:
        return None
    try:
        out = subprocess.run(["getent", "passwd", str(user)],
                             capture_output=True, text=True, check=True).stdout
        gecos = out.split(":")[4]
    except (subprocess.CalledProcessError, IndexError, OSError):
        return None
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", gecos)
    return m.group(0) if m else None


def notify_quota_exceeded(user, notify_email, used_hours, requested_hours, budget,
                          config, hold_until=None):
    """Email the user that their submit was deprioritized for going over quota
    (and held until `hold_until`, when the admin configured a quota hold)."""
    if not notify_email:
        notify_email = email_for_user(user)
    if not notify_email:
        return
    subject = f"[gpuq] {user}: GPU-hour quota exceeded - job deprioritized"
    hold_note = ""
    if hold_until is not None:
        hold_note = (f"Held until:                      "
                     f"{hold_until.isoformat(timespec='minutes')}\n")
    body = (
        f"Host: {HOST}\n"
        f"User: {user}\n"
        f"Used in last {QUOTA_WINDOW_HOURS}h: {used_hours:.1f} GPU-hours\n"
        f"Requested by this job:           {requested_hours:.1f} GPU-hours\n"
        f"Rolling 7-day budget:            {budget:.1f} GPU-hours\n"
        f"{hold_note}\n"
        "Your job was queued at low priority"
        + (", and is held until the time\nabove before it may start at all"
           if hold_until is not None else "")
        + ". It will only run once on-quota\n"
        "submitters on this host have had a chance to grab the next free slot.\n"
        "If this is unexpected, ask the admin to adjust your quota.\n"
    )
    send_email(notify_email, subject, body, config)


def notify_completion(job, exit_code, reason, config, notify_email):
    if not notify_email:
        notify_email = email_for_user(job.get("user", ""))
    if not notify_email:
        return
    subject = f"[gpuq] job {job['id']} {reason}"
    body = (
        f"Host: {job.get('host', HOST)}\n"
        f"User: {job['user']}\n"
        f"Command: {job.get('command', '')}\n"
        f"GPUs: {job.get('gpus', [])}\n"
        f"Started: {job.get('started_at', '')}\n"
        f"Reason: {reason} (exit code {exit_code})\n"
    )
    send_email(notify_email, subject, body, config)


def notify_untracked(owner, sample, deadline, config, kind="warn"):
    """Email an offender about a GPU process not launched via `gpuq submit`.

    `kind` is one of: warn (first detection), remind (still running, before the
    deadline), overdue (past the deadline, pending/failed enforcement), killed
    (terminated by --enforce). Returns the send_email result; silently no-ops
    (returns False) if the owner has no email on their account (GECOS).
    """
    to = email_for_user(owner)
    if not to:
        return False
    gpus = sample.get("gpus", [])
    gb = sample.get("memory_mb", 0) / 1024
    name = sample.get("name", "?")
    pid = sample.get("pid", "?")
    deadline_str = deadline.isoformat(timespec="minutes") if deadline else "?"
    details = (
        f"Host:    {HOST}\n"
        f"User:    {owner}\n"
        f"GPU(s):  {gpus}\n"
        f"Process: pid {pid} ({name}), ~{gb:.1f} GB GPU memory\n"
    )
    move = ("Please stop it and relaunch under `gpuq submit` so GPU scheduling\n"
            "stays fair for everyone. (gpuq is coordination, not a hard lock —\n"
            "/dev/nvidia* is world-accessible — so this relies on cooperation.)\n")
    if kind == "killed":
        subject = f"[gpuq] {owner}: untracked GPU process on {HOST} was KILLED"
        body = (f"Your GPU process below was not tracked by gpuq and had passed its\n"
                f"grace deadline ({deadline_str}), so it was terminated.\n\n"
                f"{details}\n{move}")
    elif kind == "overdue":
        subject = f"[gpuq] {owner}: untracked GPU process on {HOST} PAST DEADLINE"
        body = (f"Your untracked GPU process below has passed its grace deadline\n"
                f"({deadline_str}) and is now subject to termination.\n\n"
                f"{details}\n{move}")
    elif kind == "remind":
        subject = f"[gpuq] reminder — {owner}: untracked GPU process on {HOST}"
        body = (f"Reminder: the GPU process below is not tracked by gpuq and will be\n"
                f"killed after its grace deadline ({deadline_str}).\n\n"
                f"{details}\n{move}")
    else:  # warn
        subject = f"[gpuq] {owner}: untracked GPU process on {HOST}"
        body = (f"A GPU process you are running was not launched via `gpuq submit`,\n"
                f"so gpuq is not tracking it. It will be killed if still untracked\n"
                f"after its grace deadline ({deadline_str}).\n\n"
                f"{details}\n{move}")
    return send_email(to, subject, body, config)


def notify_rebind(owner, sample, deadline, config, kind="warn"):
    """Email an offender whose gpuq job is running on a GPU it was not allocated.

    `kind` is one of: warn (first detection), remind (before the deadline),
    overdue (past it, pending/failed enforcement), killed (terminated by
    --enforce). Returns the send_email result; no-ops (False) if the owner has no
    email on their account (GECOS).
    """
    to = email_for_user(owner)
    if not to:
        return False
    job_id = sample.get("job_id", "?")
    allocated = sample.get("allocated", [])
    gpus = sample.get("gpus", [])
    gb = sample.get("memory_mb", 0) / 1024
    name = sample.get("name", "?")
    pid = sample.get("pid", "?")
    deadline_str = deadline.isoformat(timespec="minutes") if deadline else "?"
    details = (
        f"Host:       {HOST}\n"
        f"User:       {owner}\n"
        f"Job:        {job_id}\n"
        f"Allocated:  GPU {allocated}\n"
        f"Running on: GPU {gpus}\n"
        f"Process:    pid {pid} ({name}), ~{gb:.1f} GB GPU memory\n"
    )
    fix = ("gpuq allocated your job the GPU(s) above and set CUDA_VISIBLE_DEVICES to\n"
           "match, but it is running on a different GPU — so the allocated card is\n"
           "reserved but idle. Please don't override the device gpuq picked (e.g. no\n"
           "`--gpu N` / `device=N` and don't reset CUDA_VISIBLE_DEVICES); relaunch via\n"
           "`gpuq submit`, or pin the GPU you want with `gpuq submit --devices`.\n"
           "(gpuq is coordination, not a hard lock — /dev/nvidia* is world-accessible —\n"
           "so this relies on cooperation.)\n")
    if kind == "killed":
        subject = f"[gpuq] {owner}: GPU rebind on {HOST} was KILLED"
        body = (f"Your gpuq job below was running on a GPU it was not allocated and had\n"
                f"passed its grace deadline ({deadline_str}), so it was terminated.\n\n"
                f"{details}\n{fix}")
    elif kind == "overdue":
        subject = f"[gpuq] {owner}: GPU rebind on {HOST} PAST DEADLINE"
        body = (f"Your gpuq job below is running on a GPU it was not allocated and has\n"
                f"passed its grace deadline ({deadline_str}); it is now subject to\n"
                f"termination.\n\n{details}\n{fix}")
    elif kind == "remind":
        subject = f"[gpuq] reminder — {owner}: GPU rebind on {HOST}"
        body = (f"Reminder: the gpuq job below is running on a GPU it was not allocated\n"
                f"and will be killed after its grace deadline ({deadline_str}).\n\n"
                f"{details}\n{fix}")
    else:  # warn
        subject = f"[gpuq] {owner}: GPU rebind on {HOST} (job {job_id})"
        body = (f"One of your gpuq jobs is running on a GPU it was not allocated. It\n"
                f"will be killed if not corrected after its grace deadline "
                f"({deadline_str}).\n\n{details}\n{fix}")
    return send_email(to, subject, body, config)


def enforce_kill_pgid(pgid, expect_owner=None, sample_pid=None):
    """SIGTERM a process group, grace, then SIGKILL. Privilege-aware.

    Returns True if the group is gone (or was already gone), False if we lack
    the privilege to signal it (another user's process and we are not root).

    The audit data this acts on can be minutes old and pgids recycle, so when
    the caller provides the recorded owner and a sampled member pid, the
    target is re-verified immediately before signalling; on mismatch nothing
    is killed (the next audit run re-flags the real offender).
    """
    if expect_owner is not None and sample_pid is not None:
        cur_owner = get_process_user(sample_pid)
        cur_pgid = get_process_pgid(sample_pid)
        if (cur_owner != expect_owner or cur_pgid is None
                or int(cur_pgid) != int(pgid)):
            print(f"[gpuq audit] not killing process group {pgid}: it no "
                  f"longer matches the recorded offender ({expect_owner}); "
                  "deferring to the next audit run.", file=sys.stderr)
            return False
    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except ProcessLookupError:
        return True  # already gone
    except PermissionError:
        print(f"[gpuq audit] cannot signal process group {pgid} "
              f"(needs sudo/root); leaving it for the admin.", file=sys.stderr)
        return False
    except (OSError, TypeError, ValueError) as e:
        print(f"[gpuq audit] failed to signal process group {pgid}: {e}",
              file=sys.stderr)
        return False
    for _ in range(KILL_GRACE_SEC):
        time.sleep(1)
        try:
            os.killpg(int(pgid), 0)
        except ProcessLookupError:
            return True
        except OSError:
            return True
    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except OSError:
        pass
    return True


# ---------------------------------------------------------------------------
# Building job records
# ---------------------------------------------------------------------------
def _new_job_id():
    # Mix in the PID so two submits landing in the same millisecond (e.g. a
    # sweep loop) don't mint the same id; removals also match on (host, pid).
    return (int(time.time() * 1000) ^ (os.getpid() << 12)) % 2**31


def build_running_job(cmd, args, gpus, memory_gb, max_time_hours, job_id=None,
                      submitted_at=None, devices=None, over_quota=False):
    now = datetime.now()
    submitted = submitted_at or now.isoformat(timespec="seconds")
    sub_dt = _parse_iso(submitted)
    wait_sec = int((now - sub_dt).total_seconds()) if sub_dt else None
    return {
        "id": job_id or _new_job_id(),
        "user": USER,
        "host": HOST,
        "command": " ".join(cmd) if isinstance(cmd, list) else cmd,
        "working_directory": os.getcwd(),
        "virtual_env": detect_virtual_environment(),
        "gpu_count": args.gpus,
        "gpus": gpus,
        "devices": devices,
        "memory_gb": memory_gb,
        "max_time_hours": max_time_hours,
        "over_quota_at_submit": over_quota,
        "submitted_at": submitted,
        "started_at": now.isoformat(timespec="seconds"),
        "queue_wait_sec": max(0, wait_sec) if wait_sec is not None else None,
        "pid": os.getpid(),
        "pid_start": pid_start_time(os.getpid()),
        "name": args.name,
        "notify_email": args.notify,
        "status": "running",
    }


def build_queued_job(cmd, args, memory_gb, max_time_hours, priority="normal",
                     submitted_at=None, devices=None):
    return {
        "id": _new_job_id(),
        "user": USER,
        "host": HOST,
        "command": " ".join(cmd) if isinstance(cmd, list) else cmd,
        "working_directory": os.getcwd(),
        "virtual_env": detect_virtual_environment(),
        "gpu_count": args.gpus,
        "devices": devices,
        "memory_gb": memory_gb,
        "max_time_hours": max_time_hours,
        "submitted_at": submitted_at or datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "pid_start": pid_start_time(os.getpid()),
        "name": args.name,
        "notify_email": args.notify,
        "priority": priority,
        "status": "queued",
    }


# ---------------------------------------------------------------------------
# Subcommand: submit
# ---------------------------------------------------------------------------
def _drop_my_queued_entry(queued):
    """Return `queued` without this process's own entry on this host."""
    return [j for j in queued
            if not (j.get("host") == HOST and j.get("pid") == os.getpid())]


def _try_claim(cmd, args, memory_gb, max_time_hours, devices=None,
               deprioritized=False, submitted_at=None, hard_cap=0):
    """One attempt, under the lock, to grab a slot. Returns (job, picked) on
    success or (None, None). Reaps stale jobs first and drops our own queued
    entry when we win, so a waiter that finally claims also leaves the queue."""
    with file_lock(LOCK_FILE):
        running, queued = reap_all_locked()
        gpus = get_gpu_info()
        picked = pick_gpus(args.gpus, memory_gb, gpus, running, USER,
                           devices=devices, hard_cap=hard_cap)
        if picked is None:
            return None, None
        job = build_running_job(cmd, args, picked, memory_gb, max_time_hours,
                                submitted_at=submitted_at, devices=devices,
                                over_quota=deprioritized)
        if deprioritized:
            job["priority"] = "low"
        running.append(job)
        save_running(running)
        new_q = _drop_my_queued_entry(queued)
        if len(new_q) != len(queued):
            save_queued(new_q)
        return job, picked


def _wait_for_slot(cmd, args, memory_gb, max_time_hours, config, devices=None,
                   deprioritized=False, submitted_at=None, hard_cap=0,
                   hold_until=None):
    """Poll until a slot can be claimed, returning (job, picked). Registers a
    one-time queued entry so other submitters see us and yield as configured;
    the job keeps the queued entry's id when it finally runs. Ctrl-C, SIGTERM
    and SIGHUP (closed terminal) all drop that entry, log a cancelled event,
    and exit with the conventional code. Used by --queue submits (with
    `devices` pinned or not) and by deprioritized over-quota submits, which
    additionally may not claim before `hold_until` (the over-quota hold)."""
    poll_interval = (DEPRIORITIZED_POLL_INTERVAL_SEC if deprioritized
                     else QUEUE_POLL_INTERVAL_SEC)
    qjob = None
    first_iter = True
    prev_handlers = {}

    def _restore_handlers():
        for s, h in prev_handlers.items():
            try:
                signal.signal(s, h)
            except (ValueError, OSError):
                pass

    def _cancel(reason, exit_code, label):
        # Ignore further termination signals while cleaning up: a second
        # signal re-entering this function mid-flock would self-deadlock.
        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            try:
                signal.signal(s, signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        with file_lock(LOCK_FILE):
            save_queued(_drop_my_queued_entry(load_queued()))
            if qjob is not None:
                append_usage(_cancelled_record(qjob, reason, datetime.now()))
        print(f"\n[gpuq] {label} while queued.", file=sys.stderr)
        sys.exit(exit_code)

    class _SignalCancel(Exception):
        pass

    def _on_term(sig, _frame):
        # Only raise — NEVER do locked work inside a signal handler. flock is
        # not reentrant for a second fd in the same process, so a handler that
        # locks while the interrupted frame holds the lock wedges this waiter
        # (and, since it HOLDS the host-wide lock, every other gpuq command).
        # The raise unwinds the with-block, releasing the lock, and the except
        # below runs _cancel outside the locked region — exactly the mechanism
        # that makes the Ctrl-C/KeyboardInterrupt path safe.
        raise _SignalCancel(sig)

    try:
        for s in (signal.SIGTERM, signal.SIGHUP):
            try:
                prev_handlers[s] = signal.signal(s, _on_term)
            except (ValueError, OSError):
                pass
        while True:
            with file_lock(LOCK_FILE):
                running, queued = reap_all_locked()
                # Deprioritized submitters always queue on the first iteration
                # (giving any concurrent normal-priority submit time to land),
                # never claim before their over-quota hold expires, and yield
                # while a normal-priority submitter is queued — but only to
                # waiters who could actually claim a slot right now: a normal
                # waiter wedged by the card cap (or VRAM/pinning) must not
                # starve low-priority jobs while a card sits idle.
                skip_pick = deprioritized and (
                    first_iter
                    or (hold_until is not None and datetime.now() < hold_until))
                if not skip_pick:
                    gpus = get_gpu_info()
                    if deprioritized and _normal_waiter_can_claim(
                            queued, running, gpus, hard_cap):
                        skip_pick = True
                if not skip_pick:
                    picked = pick_gpus(args.gpus, memory_gb, gpus, running, USER,
                                       devices=devices, hard_cap=hard_cap)
                    if picked is not None:
                        job = build_running_job(
                            cmd, args, picked, memory_gb, max_time_hours,
                            job_id=qjob["id"] if qjob else None,
                            submitted_at=submitted_at, devices=devices,
                            over_quota=deprioritized)
                        if deprioritized:
                            job["priority"] = "low"
                        running.append(job)
                        save_running(running)
                        new_q = _drop_my_queued_entry(queued)
                        if len(new_q) != len(queued):
                            save_queued(new_q)
                        # Back to the previous disposition: from here the job
                        # is RUNNING, and a stale _on_term firing before
                        # run_and_wait installs its forwarder would log a
                        # bogus 'cancelled' event for a job that started.
                        _restore_handlers()
                        return job, picked
                if qjob is None:
                    priority = "low" if deprioritized else "normal"
                    qjob = build_queued_job(cmd, args, memory_gb, max_time_hours,
                                            priority=priority,
                                            submitted_at=submitted_at,
                                            devices=devices)
                    if hold_until is not None:
                        qjob["hold_until"] = hold_until.isoformat(
                            timespec="seconds")
                    queued.append(qjob)
                    save_queued(queued)
                    tag = " (deprioritized)" if deprioritized else ""
                    if hold_until is not None and datetime.now() < hold_until:
                        tag = (f" (deprioritized, held until "
                               f"{hold_until.isoformat(timespec='minutes')})")
                    pin = (f" [devices {','.join(map(str, devices))}]"
                           if devices else "")
                    print(f"[gpuq] no slot; queued as job {qjob['id']}{tag}{pin}, "
                          f"polling every {poll_interval}s. Ctrl-C to cancel "
                          f"(or `gpuq kill {qjob['id']}` from another terminal).",
                          file=sys.stderr)
            first_iter = False
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        _cancel("user", 130, "cancelled")
    except _SignalCancel as e:
        sig = int(e.args[0])
        _cancel("signal", 128 + sig, f"cancelled (signal {sig})")


def cmd_submit(args, config):
    submitted_at = datetime.now().isoformat(timespec="seconds")
    cmd = list(args.cmd_args or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if args.command:
        if cmd:
            die("give the command either after `--` or via --command, not both.")
        cmd = shlex.split(args.command)
    if not cmd:
        die("no command given. Use:  gpuq submit [opts] -- COMMAND ARGS...")

    # -m default: the admission filter (min free VRAM a candidate GPU needs).
    # Falls back to max_memory_per_gpu_gb for configs that predate the
    # dedicated default_min_free_gb key.
    memory_default = config.get(
        "default_min_free_gb",
        config.get("max_memory_per_gpu_gb", DEFAULT_MAX_MEMORY_GB))
    memory_gb = args.memory if args.memory is not None else memory_default
    max_time_hours = (args.time if args.time is not None
                      else config.get("max_job_time_hours", DEFAULT_MAX_TIME_HOURS))
    try:
        max_time_hours = float(max_time_hours)
    except (TypeError, ValueError):
        max_time_hours = DEFAULT_MAX_TIME_HOURS
    if max_time_hours <= 0:
        die("-t/--time must be > 0 hours; there is no unlimited mode "
            "(ask the admin to raise max_job_time_hours if you need longer).")
    # Hard wall-time ceiling. An explicit -t over the cap is rejected so the
    # submitter knows; a config default that exceeds the cap is silently clamped
    # (an admin misconfig must never break no-flag submits).
    time_cap = config.get("max_job_time_hours_cap", DEFAULT_MAX_TIME_HOURS_CAP)
    try:
        time_cap = float(time_cap)
    except (TypeError, ValueError):
        time_cap = DEFAULT_MAX_TIME_HOURS_CAP
    if time_cap > 0:
        if args.time is not None and max_time_hours > time_cap:
            die(f"-t/--time {max_time_hours:g}h exceeds the {time_cap:g}h "
                f"({time_cap / 24:g}-day) wall-time cap on {HOST}.")
        max_time_hours = min(max_time_hours, time_cap)

    # --devices: pin exact GPU(s). Each must be free or already owned by you.
    # Without --queue, reject immediately with a per-GPU reason; with --queue,
    # wait until all of them are available. Pinning bypasses the random picker
    # but NOT the quota gate.
    devices = None
    if args.devices:
        try:
            devices = sorted({int(x) for x in args.devices.split(",") if x.strip()})
        except ValueError:
            die(f"--devices must be comma-separated GPU indices, got {args.devices!r}")
        if not devices:
            die("--devices was given but empty")
        if args.gpus is not None and args.gpus != len(devices):
            die(f"--devices {args.devices} implies -g {len(devices)}; "
                "drop -g or make them match.")
        args.gpus = len(devices)   # records + accounting use the real count
    elif args.gpus is None:
        args.gpus = 1

    # Sanity-check the request against the actual hardware so --queue can
    # never poll forever for something this host cannot satisfy.
    if args.gpus < 1:
        die("-g/--gpus must be at least 1.")
    n_total = len(get_gpu_info())
    if n_total:
        if args.gpus > n_total:
            die(f"this host has {n_total} GPU(s); you asked for {args.gpus}.")
        if devices:
            bad = [d for d in devices if d < 0 or d >= n_total]
            if bad:
                die(f"no such GPU index(es) {bad} on {HOST} "
                    f"(valid: 0..{n_total - 1}).")

    # Per-user concurrent-card hard cap: a single job asking for more cards
    # than any user may ever hold can never run, --queue or not. Requests
    # within the cap are enforced at claim time (pick_gpus), where jobs the
    # user already holds count too and stacking onto owned cards stays free.
    hard_cap = user_card_cap(config)
    if hard_cap and args.gpus > hard_cap:
        die(f"this host caps each user at {hard_cap} concurrent GPU(s) "
            f"(per-user hard limit); you asked for {args.gpus} in one job.")

    # Quota: rolling 7-day GPU-hour budget, per user. Over-quota submitters are
    # deprioritized — forced into queue mode with a longer poll interval — and
    # emailed once. Applies to pinned (--devices) submits too: pinning chooses
    # WHICH card you get, not WHETHER you skip the line. Charging is by actual
    # runtime at job end (see run_and_wait).
    requested_gpu_hours = args.gpus * max_time_hours
    over, used, budget = would_exceed_quota(USER, requested_gpu_hours, config)
    deprioritized = bool(over)
    hold_until = None
    if over:
        delay_h = quota_delay_hours(config)
        if delay_h > 0:
            hold_until = datetime.now() + timedelta(hours=delay_h)
        msg = (f"[gpuq] over quota: used {used:.1f} + requested {requested_gpu_hours:.1f} "
               f"GPU-hours > budget {budget:.1f} (rolling {QUOTA_WINDOW_HOURS}h). "
               "Job DEPRIORITIZED and queued; will only start once on-quota submitters "
               "have a chance to grab slots."
               + (f" It is also held until "
                  f"{hold_until.isoformat(timespec='minutes')} "
                  f"({delay_h:g}h over-quota hold) before it may claim at all."
                  if hold_until is not None else ""))
        print(msg, file=sys.stderr)
        notify_quota_exceeded(USER, args.notify, used, requested_gpu_hours,
                              budget, config, hold_until=hold_until)

    # On-quota: try once, then reject (no --queue) or wait. Over-quota always
    # waits and yields to normal-priority submitters on its first poll.
    job = picked = None
    if not deprioritized:
        job, picked = _try_claim(cmd, args, memory_gb, max_time_hours,
                                 devices=devices, submitted_at=submitted_at,
                                 hard_cap=hard_cap)
        if job is None and not args.queue:
            with file_lock(LOCK_FILE):
                running, _ = reap_running(load_running())
                gpus = get_gpu_info()
            # Was the per-user card cap the actual blocker? (The same request
            # with the cap lifted would have been granted.) Label the message
            # and the demand ledger accordingly, so a cap block is never
            # misdiagnosed as a VRAM shortage or unavailable devices.
            cap_blocked = (hard_cap > 0
                           and pick_gpus(args.gpus, memory_gb, gpus, running,
                                         USER, devices=devices,
                                         hard_cap=0) is not None)
            if devices:
                reasons = gpu_unavailability(devices, gpus, running, memory_gb,
                                             USER, hard_cap=hard_cap)
                record_rejected_submit(
                    args, memory_gb, max_time_hours,
                    "user_card_cap" if cap_blocked else "devices_unavailable",
                    devices=devices)
                die("requested GPU(s) not available — not submitting:\n  "
                    + "\n  ".join(reasons) + "\nPass --queue to wait for them.")
            hint = ("" if args.memory is not None else
                    f"\n(The {memory_gb} GB free-VRAM filter is the default; "
                    "pass -m <smaller> if your job needs less.)")
            cap_hint = ""
            if cap_blocked:
                held = sorted(_user_held_gpus(running, USER))
                cap_hint = (f"\nPer-user card cap: you already hold GPU(s) "
                            f"{held} of the {hard_cap}-card cap, so this "
                            "submit may only stack onto those cards (needs "
                            "enough free VRAM there) until one of your jobs "
                            "ends.")
            record_rejected_submit(
                args, memory_gb, max_time_hours,
                "user_card_cap" if cap_blocked else "no_free_gpu")
            die(f"no free GPU matches your request "
                f"(need {args.gpus} GPU(s) with >= {memory_gb} GB free; free cards "
                f"also need util < {GPU_UTIL_AVAILABLE_THRESHOLD}%, a card you "
                f"already run on does not but still needs that much free)."
                f"{hint}{cap_hint} Pass --queue to wait.")
    if job is None:
        job, picked = _wait_for_slot(cmd, args, memory_gb, max_time_hours, config,
                                     devices=devices, deprioritized=deprioritized,
                                     submitted_at=submitted_at, hard_cap=hard_cap,
                                     hold_until=hold_until)

    print(f"[gpuq] job {job['id']} starting on GPU(s) {','.join(map(str, picked))}",
          file=sys.stderr)
    run_and_wait(job, cmd, picked, max_time_hours, args, config)


_SCOPE_SUPPORTED = None


def scope_supported():
    """Whether jobs can run in their own systemd --user scope (cgroup isolation).

    Cached. GPUQ_SCOPE=off forces the plain process-group fallback (used by the
    test suite and as an escape hatch on hosts without a user systemd manager).
    """
    global _SCOPE_SUPPORTED
    if _SCOPE_SUPPORTED is None:
        mode = os.environ.get("GPUQ_SCOPE", "auto").lower()
        if mode == "off":
            _SCOPE_SUPPORTED = False
        else:
            try:
                have_run = subprocess.run(
                    ["systemd-run", "--version"], capture_output=True, timeout=5
                ).returncode == 0
                mgr_ok = subprocess.run(
                    ["systemctl", "--user", "is-active", "default.target"],
                    capture_output=True, timeout=5
                ).returncode == 0
                # In 'auto' (default) only scope when the user LINGERS — otherwise
                # the scope (and the job) would be killed when their last login
                # session ends. With linger off we fall back to a plain process
                # group, which survives logout as before. 'on' forces scoping.
                linger = (mode == "on") or subprocess.run(
                    ["loginctl", "show-user", USER, "-p", "Linger", "--value"],
                    capture_output=True, text=True, timeout=5
                ).stdout.strip() == "yes"
                _SCOPE_SUPPORTED = have_run and mgr_ok and linger
            except (OSError, subprocess.SubprocessError):
                _SCOPE_SUPPORTED = False
    return _SCOPE_SUPPORTED


def run_and_wait(job, cmd, gpus, max_time_hours, args, config):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["GPUQ_JOB_ID"] = str(job["id"])

    # Launch inside the job's own systemd --user scope (a dedicated cgroup) so
    # EVERY descendant — including framework workers that setsid/double-fork into
    # their own session (vLLM, Ray, torchrun) — stays in the job's cgroup and is
    # recognised as tracked. Falls back to a plain setsid process group when no
    # user manager is available; pgid + ancestry tracking then applies.
    scope = None
    launch = cmd
    if scope_supported():
        scope = f"gpuq-{job['id']}.scope"
        launch = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                  f"--unit=gpuq-{job['id']}", "--"] + list(cmd)

    started_wall = datetime.now()
    child = subprocess.Popen(
        launch,
        env=env,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid,
    )

    def kill_job(sig):
        """Kill the whole job: the scope (all descendants) if scoped, else the
        child's process group."""
        if scope:
            subprocess.run(["systemctl", "--user", "kill",
                            f"--signal={int(sig)}", scope], capture_output=True)
        else:
            try:
                os.killpg(os.getpgid(child.pid), sig)
            except (ProcessLookupError, OSError):
                pass

    def forward(sig, _frame):
        kill_job(sig)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(s, forward)
        except (ValueError, OSError):
            pass

    # Record how `gpuq audit` recognises this job's processes: its cgroup scope
    # (covers any descendant, however it forks) and, for the fallback path, the
    # child process group (== child.pid via setsid). The submit-race grace
    # window covers the moment until this lands.
    with file_lock(LOCK_FILE):
        running = load_running()
        for j in running:
            if j.get("id") == job["id"] and j.get("host") == HOST:
                j["child_pid"] = child.pid
                j["child_pgid"] = child.pid
                # Group-leader identity, so a recycled pgid number can never
                # pin this entry after the job actually died (see
                # child_group_alive).
                j["child_pgid_start"] = pid_start_time(child.pid)
                if scope:
                    j["cgroup_scope"] = scope
        save_running(running)

    timeout_fired = {"v": False}

    def timeout():
        timeout_fired["v"] = True
        print(f"\n[gpuq] job {job['id']} reached its {max_time_hours}h time "
              f"limit — sending SIGTERM (SIGKILL in {KILL_GRACE_SEC}s). "
              "Resubmit with a larger -t to run longer.", file=sys.stderr)
        kill_job(signal.SIGTERM)
        time.sleep(KILL_GRACE_SEC)
        kill_job(signal.SIGKILL)

    timer = None
    if max_time_hours and max_time_hours > 0:
        timer = threading.Timer(max_time_hours * 3600, timeout)
        timer.daemon = True
        timer.start()

    rc = None
    try:
        rc = child.wait()
    finally:
        if timer is not None:
            timer.cancel()
        ended_wall = datetime.now()
        elapsed_h = max(0.0, (ended_wall - started_wall).total_seconds() / 3600.0)
        gpu_hours = elapsed_h * len(gpus)
        reason = ("timed_out" if timeout_fired["v"]
                  else "completed" if rc == 0
                  else "killed" if (rc is not None and rc < 0)
                  else "failed")
        with file_lock(LOCK_FILE):
            running = load_running()
            running = [j for j in running
                       if not (j.get("id") == job["id"] and j.get("host") == HOST
                               and j.get("pid") == os.getpid())]
            save_running(running)
            append_usage(_end_record(job, started_wall, ended_wall, rc, reason))
        hms = str(ended_wall - started_wall).split(".")[0]
        print(f"[gpuq] job {job['id']} {reason}: ran {hms} on GPU(s) "
              f"{','.join(map(str, gpus))}, {gpu_hours:.2f} GPU-hours recorded "
              f"(exit {rc}).", file=sys.stderr)
        notify_completion(job, rc, reason, config, args.notify)
    # Shell convention for a signal-killed child: 128 + signum (Popen.wait
    # reports -signum). sys.exit(-15) would otherwise become status 241.
    sys.exit(128 - rc if rc is not None and rc < 0 else rc)


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------
def _fmt_duration(start_iso):
    try:
        t0 = datetime.fromisoformat(start_iso)
        return str(datetime.now() - t0).split(".")[0]
    except Exception:
        return "?"


def _truncate(text, width=72):
    text = (text or "").replace("\n", " ")
    return text if len(text) <= width else text[:width - 1] + "…"


def cmd_status(args, config):
    with file_lock(LOCK_FILE):
        running, queued = reap_all_locked()
    gpus = get_gpu_info()

    print(f"=== GPU Queue Status (host: {HOST}) ===\n")
    holders = {}
    for j in running:
        if j.get("host", HOST) != HOST:
            continue
        label = f"{j.get('user', '?')} (job {j.get('id')}"
        if j.get("name"):
            label += f", {j['name']}"
        label += ")"
        for gi in j.get("gpus", []):
            holders.setdefault(int(gi), []).append(label)
    print(f"GPUs ({len(gpus)} total):")
    for g in gpus:
        state = "BUSY" if g["index"] in holders else "FREE"
        pct = 100 * g["memory_used_mb"] / g["memory_total_mb"] if g["memory_total_mb"] else 0
        used_gb = g["memory_used_mb"] / 1024
        total_gb = g["memory_total_mb"] / 1024
        print(f"  GPU {g['index']}: {g['name']} - {state}")
        print(f"    Memory: {used_gb:.0f}/{total_gb:.0f} GB ({pct:.0f}%) - "
              f"Utilization: {g['utilization']}%")
        if g["index"] in holders:
            print(f"    Held by: {'; '.join(holders[g['index']])}")
    print()

    print(f"Running Jobs ({len(running)}):")
    if not running:
        print("  (none)")
    for j in running:
        limit = _safe_float(j.get("max_time_hours"), default=None)
        line = (f"  Job {j['id']} by {j['user']} - GPUs: {j.get('gpus', [])} - "
                f"Runtime: {_fmt_duration(j.get('started_at', ''))}"
                + (f" / limit {limit:g}h" if limit else ""))
        if j.get("name"):
            line += f" - Name: {j['name']}"
        print(line)
        if j.get("command"):
            print(f"    Command: {_truncate(j['command'])}")
    print()

    queued = sorted(queued, key=lambda j: j.get("submitted_at") or "")
    print(f"Queued Jobs ({len(queued)}):")
    if not queued:
        print("  (none)")
    for j in queued:
        prio = j.get("priority", "normal")
        tag = f" [{prio}]" if prio != "normal" else ""
        hold = _parse_iso(j.get("hold_until") or "")
        if hold and hold > datetime.now():
            tag += f" (held until {hold.strftime('%m-%d %H:%M')})"
        want = f"{j.get('gpu_count', 1)} GPU(s)"
        if j.get("devices"):
            want = f"GPU {','.join(map(str, j['devices']))} (pinned)"
        line = (f"  Job {j['id']} by {j['user']}{tag} - Wants: {want} - Waiting: "
                f"{_fmt_duration(j.get('submitted_at', ''))}")
        if j.get("name"):
            line += f" - Name: {j['name']}"
        print(line)
        if j.get("command"):
            print(f"    Command: {_truncate(j['command'])}")
    if queued:
        print("  (queue order is informational: free slots go to whichever "
              "waiter polls first; low priority always yields)")
    print()

    print("All GPU compute processes (from nvidia-smi):")
    live = list_gpu_processes_with_owner()
    if live is None:
        print("  (nvidia-smi failed — GPU process list unavailable)")
        live = []
    elif not live:
        print("  (none)")
    seen = set()
    for p in live:
        gpu_idx = p["gpu_idx"] if p["gpu_idx"] is not None else "?"
        owner = p["owner"] or "unknown"
        key = (owner, gpu_idx, p["pid"])
        if key in seen:
            continue
        seen.add(key)
        gb = p["memory_mb"] / 1024
        print(f"  pid {p['pid']} ({owner}) on GPU {gpu_idx}: {gb:.1f} GB ({p['name']})")

    used = usage_in_window(USER)
    budget = quota_for_user(USER, config)
    print()
    if budget is None:
        print(f"Your rolling 7-day usage: {used:.1f} GPU-hours "
              "(no quota set — usage is recorded to calibrate future budgets; "
              "see `gpuq quota`)")
    else:
        print(f"Your rolling 7-day usage: {used:.1f} / {budget:.1f} GPU-hours "
              f"({100 * used / budget:.0f}% of budget; see `gpuq quota`)")


# ---------------------------------------------------------------------------
# Subcommand: history
# ---------------------------------------------------------------------------
def _safe_float(value, default=0.0):
    """The ledger is group-appendable; a wrong-typed field in one line must
    render as a placeholder, never crash the whole listing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_secs(sec):
    try:
        sec = int(sec)
    except (TypeError, ValueError):
        return "-"
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def _fmt_hours(hours):
    try:
        return str(timedelta(seconds=int(float(hours) * 3600)))
    except (TypeError, ValueError):
        return "-"


def _fmt_when(iso):
    dt = _parse_iso(iso or "")
    return dt.strftime("%m-%d %H:%M") if dt else "-"


def cmd_history(args, config):
    """Show recent ledger records (default: your own finished jobs)."""
    user_filter = None if args.all else (args.user or USER)
    records = [r for r in iter_usage_records()
               if user_filter is None or r.get("user") == user_filter]
    records.sort(key=lambda r: str(r.get("ended_at") or r.get("at") or ""))
    if not args.events:
        records = [r for r in records if r.get("event") == "end"]
    if args.limit > 0:
        records = records[-args.limit:]

    if args.json:
        for rec in records:
            print(json.dumps(rec, separators=(",", ":")))
        return
    if not records:
        who = f" for {user_filter}" if user_filter else ""
        print(f"no ledger records{who} yet (the ledger only has jobs that "
              "ended after the job-log deploy).")
        return

    scope = "all users" if user_filter is None else user_filter
    print(f"Job history on {HOST} — {scope}, oldest first "
          f"(see `gpuq history -h` for filters):")
    cols = (f"{'JOB':<11} " + (f"{'USER':<10} " if args.all else "")
            + f"{'ENDED':<12} {'WAIT':>7} {'RUNTIME':>9} {'GPUS':<5} "
              f"{'GPU-H':>7} {'EXIT':>4} {'RESULT':<10} NAME/COMMAND")
    print(cols)
    for rec in records:
        event = rec.get("event", "end")
        name = rec.get("name") or ""
        command = (rec.get("command") or "").replace("\n", " ")
        label = f"{name}: {command}" if name else (command or "-")
        user_col = f"{(rec.get('user') or '?'):<10} " if args.all else ""
        if event != "end":
            when = _fmt_when(rec.get("at"))
            wait = _fmt_secs(rec.get("wait_sec"))
            result = f"{event} ({rec.get('reason', '?')})"
            want = rec.get("gpus_requested")
            print(f"{str(rec.get('id') or '-'):<11} {user_col}{when:<12} "
                  f"{wait:>7} {'-':>9} {('?' if want is None else want):<5} "
                  f"{'-':>7} {'-':>4} {result:<10} {_truncate(label, 48)}")
            continue
        gpus = ",".join(map(str, rec.get("gpus") or [])) or "-"
        exit_code = rec.get("exit_code")
        result = rec.get("end_reason") or "-"
        if rec.get("synthetic"):
            result += "*"
        print(f"{str(rec.get('id') or '-'):<11} {user_col}"
              f"{_fmt_when(rec.get('ended_at')):<12} "
              f"{_fmt_secs(rec.get('queue_wait_sec')):>7} "
              f"{_fmt_hours(rec.get('elapsed_hours')):>9} {gpus:<5} "
              f"{_safe_float(rec.get('gpu_hours')):>7.2f} "
              f"{('-' if exit_code is None else exit_code):>4} "
              f"{result:<10} {_truncate(label, 48)}")
    if any(r.get("synthetic") for r in records):
        print("(*synthetic record: the job's supervisor died before normal "
              "accounting; charged up to its reap, capped at its time limit)")


# ---------------------------------------------------------------------------
# Subcommand: quota
# ---------------------------------------------------------------------------
def _window_user_stats(now=None, window_hours=QUOTA_WINDOW_HOURS):
    """Per-user ledger + live-running stats for the trailing window."""
    now = now or datetime.now()
    cutoff = now - timedelta(hours=window_hours)
    stats = {}

    def bucket(user):
        return stats.setdefault(user, {
            "finished_hours": 0.0, "jobs": 0, "lost": 0,
            "running_jobs": 0, "running_gpus": 0, "live_hours": 0.0,
        })

    for rec in iter_usage_records():
        if rec.get("event") != "end" or not rec.get("user"):
            continue
        hours = _record_window_hours(rec, cutoff, now)
        ended = _parse_iso(rec.get("ended_at", ""))
        if ended is None or ended < cutoff:
            continue
        b = bucket(rec["user"])
        b["finished_hours"] += hours
        b["jobs"] += 1
        if rec.get("end_reason") == "lost":
            b["lost"] += 1
    for j in load_running():
        user = j.get("user")
        started = _parse_iso(j.get("started_at", ""))
        if not user or started is None:
            continue
        gpus = len(j.get("gpus") or []) or j.get("gpu_count", 1)
        elapsed = max(0.0, (now - max(started, cutoff)).total_seconds() / 3600.0)
        b = bucket(user)
        b["running_jobs"] += 1
        b["running_gpus"] += gpus
        b["live_hours"] += elapsed * gpus
    return stats


def cmd_quota(args, config):
    if getattr(args, "report", False):
        return _quota_report(args, config)
    now = datetime.now()
    stats = _window_user_stats(now)
    if getattr(args, "all", False):
        q = config.get("quotas") or {}
        users = sorted(set(stats) | set((q.get("users") or {}).keys()))
        enforced = any(quota_for_user(u, config) is not None for u in users)
        mode = ("budgets enforced (over-budget submits are deprioritized)"
                if enforced else
                "budgets NOT enforced (usage recorded to calibrate them)")
        print(f"Rolling {QUOTA_WINDOW_HOURS // 24}-day GPU-hours on {HOST} — {mode}")
        print(f"{'USER':<12} {'USED':>8} {'LIVE':>8} {'JOBS':>5} {'LOST':>5} "
              f"{'BUDGET':>8} {'USED%':>6}")
        total_used = total_live = 0.0
        for u in users:
            b = stats.get(u, {"finished_hours": 0, "jobs": 0, "lost": 0,
                              "running_gpus": 0, "live_hours": 0})
            used = b["finished_hours"] + b["live_hours"]
            total_used += b["finished_hours"]
            total_live += b["live_hours"]
            budget = quota_for_user(u, config)
            budget_s = "unlim" if budget is None else f"{budget:.0f}"
            pct = "-" if budget is None else f"{100 * used / budget:.0f}%"
            live_s = (f"+{b['live_hours']:.1f}" if b["live_hours"] else "-")
            print(f"{u:<12} {b['finished_hours']:>8.1f} {live_s:>8} "
                  f"{b['jobs']:>5} {b['lost']:>5} {budget_s:>8} {pct:>6}")
        n_gpus = len(get_gpu_info())
        if n_gpus:
            capacity = n_gpus * QUOTA_WINDOW_HOURS
            pct = 100 * (total_used + total_live) / capacity
            print(f"{'TOTAL':<12} {total_used:>8.1f} "
                  f"{('+%.1f' % total_live):>8}")
            print(f"(host capacity {capacity:.0f} GPU-h per window -> "
                  f"{pct:.0f}% used)")
        return

    user = getattr(args, "user", None) or USER
    b = stats.get(user, {"finished_hours": 0.0, "jobs": 0, "lost": 0,
                         "running_jobs": 0, "running_gpus": 0, "live_hours": 0.0})
    used = b["finished_hours"] + b["live_hours"]
    budget = quota_for_user(user, config)
    print(f"{user} @ {HOST} — rolling {QUOTA_WINDOW_HOURS // 24}-day GPU-hours "
          f"(window ends {now.strftime('%Y-%m-%d %H:%M')})")
    lost = f"  ({b['lost']} lost)" if b["lost"] else ""
    print(f"  finished:  {b['jobs']} job(s), {b['finished_hours']:.1f} GPU-hours{lost}")
    if b["running_jobs"]:
        print(f"  running:   {b['running_jobs']} job(s) on {b['running_gpus']} "
              f"GPU(s), +{b['live_hours']:.1f} GPU-hours and counting")
    print(f"  total:     {used:.1f} GPU-hours")
    if budget is None:
        print("  budget:    unlimited — quotas are not enforced yet; usage is "
              "recorded so budgets can be set from real data")
    else:
        pct = 100 * used / budget
        print(f"  budget:    {budget:.1f} GPU-hours per {QUOTA_WINDOW_HOURS // 24} "
              f"days ({pct:.0f}% used)")
        if used > budget:
            delay_h = quota_delay_hours(config)
            hold_txt = (f"held {delay_h:g}h, then " if delay_h > 0 else "")
            print(f"  status:    OVER BUDGET — new submits are {hold_txt}"
                  "DEPRIORITIZED (queued behind on-quota users), never rejected")
        else:
            print(f"  status:    OK — {budget - used:.1f} GPU-hours of headroom; "
                  "submits run at normal priority")


def _percentile(values, p):
    """Linear-interpolation percentile; values need not be sorted."""
    if not values:
        return 0.0
    vs = sorted(values)
    if len(vs) == 1:
        return float(vs[0])
    k = (len(vs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(vs) - 1)
    return float(vs[lo] + (vs[hi] - vs[lo]) * (k - lo))


def _quota_report(args, config):
    """Weekly per-user statistics for setting budgets, from the full ledger."""
    weeks = max(1, int(getattr(args, "weeks", 8) or 8))
    now = datetime.now()
    cutoff = now - timedelta(weeks=weeks)
    per_user = {}
    for rec in iter_usage_records():
        if rec.get("event") != "end" or not rec.get("user"):
            continue
        ended = _parse_iso(rec.get("ended_at", ""))
        if ended is None or ended < cutoff:
            continue
        u = per_user.setdefault(rec["user"], {
            "weekly": {}, "waits": [], "jobs": 0, "timeout": 0, "lost": 0,
            "pinned_hours": 0.0, "hours": 0.0,
        })
        gpu_hours = _safe_float(rec.get("gpu_hours"))
        week = ended.strftime("%G-W%V")  # multi-day jobs charge to their end week
        u["weekly"][week] = u["weekly"].get(week, 0.0) + gpu_hours
        u["hours"] += gpu_hours
        u["jobs"] += 1
        wait = rec.get("queue_wait_sec")
        if wait is not None:
            try:
                u["waits"].append(int(wait))
            except (TypeError, ValueError):
                pass
        if rec.get("end_reason") == "timed_out":
            u["timeout"] += 1
        if rec.get("end_reason") == "lost":
            u["lost"] += 1
        if rec.get("devices"):
            u["pinned_hours"] += gpu_hours
    if not per_user:
        print(f"no finished jobs in the last {weeks} week(s); nothing to report.")
        return

    print(f"Weekly GPU-hours per user, last {weeks} ISO week(s) on {HOST}")
    print(f"{'USER':<12} {'WKS':>3} {'P50':>7} {'P95':>7} {'MAX':>7} {'MEAN':>7} "
          f"{'WAIT-P50':>8} {'WAIT-P95':>8} {'TIMEOUT%':>8} {'LOST%':>6} {'PINNED%':>7}")
    host_weekly = {}
    for u in sorted(per_user):
        d = per_user[u]
        totals = list(d["weekly"].values())
        for wk, h in d["weekly"].items():
            host_weekly[wk] = host_weekly.get(wk, 0.0) + h
        print(f"{u:<12} {len(totals):>3} {_percentile(totals, 50):>7.1f} "
              f"{_percentile(totals, 95):>7.1f} {max(totals):>7.1f} "
              f"{sum(totals) / len(totals):>7.1f} "
              f"{_fmt_secs(_percentile(d['waits'], 50)) if d['waits'] else '-':>8} "
              f"{_fmt_secs(_percentile(d['waits'], 95)) if d['waits'] else '-':>8} "
              f"{100 * d['timeout'] / d['jobs']:>8.0f} "
              f"{100 * d['lost'] / d['jobs']:>6.0f} "
              f"{(100 * d['pinned_hours'] / d['hours']) if d['hours'] else 0:>7.0f}")
    n_gpus = len(get_gpu_info())
    if host_weekly and n_gpus:
        totals = list(host_weekly.values())
        capacity = n_gpus * 24 * 7
        print(f"{'HOST':<12} {len(totals):>3} {_percentile(totals, 50):>7.1f} "
              f"{_percentile(totals, 95):>7.1f} {max(totals):>7.1f} "
              f"{sum(totals) / len(totals):>7.1f}   "
              f"-> {100 * _percentile(totals, 50) / capacity:.0f}% / "
              f"{100 * _percentile(totals, 95) / capacity:.0f}% / "
              f"{100 * max(totals) / capacity:.0f}% of capacity "
              f"({capacity:.0f} GPU-h/wk) at p50/p95/max")
    print("\nhint: a per-user budget near their P95 changes nothing in normal "
          "weeks and only\ndeprioritizes outlier weeks; sanity-check that "
          "sum(P95) stays under capacity x ~1.5.\nSet budgets in "
          f"{CONFIG_FILE} (quotas.default_gpu_hours_per_week / quotas.users).")


# ---------------------------------------------------------------------------
# Subcommand: kill
# ---------------------------------------------------------------------------
def _kill_running_job(target):
    """Terminate one of the caller's running jobs. Returns True on success."""
    job_id = target["id"]
    scope = target.get("cgroup_scope")
    if scope:
        print(f"stopping job {job_id} (scope {scope})...")
        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGTERM", scope],
                       capture_output=True)
        for _ in range(KILL_GRACE_SEC):
            time.sleep(1)
            r = subprocess.run(["systemctl", "--user", "is-active", scope],
                               capture_output=True, text=True)
            if r.stdout.strip() != "active":
                print(f"job {job_id} terminated.")
                return True
        print("still alive; escalating to SIGKILL.")
        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", scope],
                       capture_output=True)
        return True

    pid = int(target["pid"])
    child_pgid = target.get("child_pgid")

    # Route on the same identity checks the reaper uses — never signal a raw
    # PID that may have been recycled by an unrelated process.
    if not entry_pid_alive(target):
        # Supervisor gone (or its PID recycled). If the workload survived it
        # (orphan), kill the child's process group — but only while it is
        # verifiably still THIS job's group; the reap below then writes the
        # synthetic ledger record and frees the GPUs.
        if child_group_alive(target):
            print(f"job {job_id}'s supervisor is gone but the job is still "
                  f"running; killing its process group {child_pgid}...")
            try:
                os.killpg(int(child_pgid), signal.SIGTERM)
            except OSError:
                pass
            for _ in range(KILL_GRACE_SEC):
                time.sleep(1)
                if not child_group_alive(target):
                    break
            if child_group_alive(target):
                try:
                    os.killpg(int(child_pgid), signal.SIGKILL)
                except OSError:
                    pass
        with file_lock(LOCK_FILE):
            reap_all_locked()
        print(f"job {job_id} cleaned up.")
        return True

    try:
        os.kill(pid, signal.SIGTERM)  # supervisor forwards to the child group
    except ProcessLookupError:
        with file_lock(LOCK_FILE):
            reap_all_locked()
        print(f"job {job_id} was already gone; cleaned up.")
        return True
    except PermissionError:
        print(f"cannot signal job {job_id}'s supervisor (PID {pid}): "
              "permission denied.", file=sys.stderr)
        return False

    print(f"sent SIGTERM to job {job_id} (PID {pid}); waiting up to "
          f"{KILL_GRACE_SEC}s...")
    for _ in range(KILL_GRACE_SEC):
        time.sleep(1)
        if not entry_pid_alive(target):
            print(f"job {job_id} terminated.")
            return True
    print("still alive; escalating to SIGKILL.")
    # SIGKILL the JOB (the child's process group), never the supervisor: a
    # SIGKILLed supervisor cannot forward anything, so the workload would
    # survive on the GPU while its record is reaped — a stranded double
    # allocation. The supervisor then sees the child die and cleans up.
    if child_pgid is not None:
        try:
            os.killpg(int(child_pgid), signal.SIGKILL)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        os.kill(pid, signal.SIGKILL)  # legacy record without child_pgid
    except (ProcessLookupError, PermissionError, OSError):
        pass
    return True


def _cancel_queued_job(target):
    """Cancel one of the caller's queued submits. Returns True on success."""
    job_id = target["id"]
    if entry_pid_alive(target):
        # Signal the waiting submit process; its own SIGINT handler drops the
        # queue entry and writes the cancelled ledger event.
        try:
            os.kill(int(target["pid"]), signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError) as e:
            print(f"could not signal queued job {job_id}'s waiting process "
                  f"(PID {target.get('pid')}): {e}", file=sys.stderr)
            return False
        for _ in range(5):
            time.sleep(1)
            with file_lock(LOCK_FILE):
                still = any(j.get("id") == job_id and j.get("host") == HOST
                            for j in load_queued())
            if not still:
                print(f"cancelled queued job {job_id} (signalled waiting "
                      f"submit, PID {target['pid']}).")
                return True
        print(f"queued job {job_id} was signalled but its entry has not "
              "cleared yet; check `gpuq status`.", file=sys.stderr)
        return False
    # Waiter process is already dead: drop the stale entry ourselves.
    with file_lock(LOCK_FILE):
        queued = [j for j in load_queued()
                  if not (j.get("id") == job_id and j.get("host") == HOST)]
        save_queued(queued)
        append_usage(_cancelled_record(target, "stale", datetime.now()))
    print(f"removed stale queue entry {job_id} (its submit process was gone).")
    return True


def cmd_kill(args, config):
    with file_lock(LOCK_FILE):
        running, queued = reap_all_locked()
    running = [j for j in running if j.get("host", HOST) == HOST]
    queued = [j for j in queued if j.get("host", HOST) == HOST]

    if getattr(args, "mine", False):
        # Cancel queued waiters FIRST: killing a running job frees its GPUs,
        # and a still-alive waiter polling during the kill's grace window
        # would claim them and dodge the sweep.
        my_queued = [j for j in queued if j.get("user") == USER]
        my_running = [j for j in running if j.get("user") == USER]
        if not my_queued and not my_running:
            print(f"no running or queued jobs for {USER} on {HOST}.")
            return
        print(f"killing {len(my_running) + len(my_queued)} job(s) of {USER}: "
              f"{[j['id'] for j in my_running + my_queued]}")
        failures = 0
        for j in my_queued:
            if not _cancel_queued_job(j):
                failures += 1
        # Re-snapshot: a waiter may have claimed a slot between our first
        # snapshot and its cancellation — its job is now in running.
        with file_lock(LOCK_FILE):
            running, _ = reap_all_locked()
        for j in running:
            if j.get("host", HOST) == HOST and j.get("user") == USER:
                if not _kill_running_job(j):
                    failures += 1
        if failures:
            sys.exit(1)
        return

    ids = list(args.job_id or [])
    if args.job_id_flag is not None:
        ids.append(args.job_id_flag)
    if not ids:
        die("kill: provide a job ID (see `gpuq status`), e.g. "
            "`gpuq kill 12345`, or use --mine.")

    by_id_running = {j["id"]: j for j in running}
    by_id_queued = {j["id"]: j for j in queued}
    failures = 0
    for job_id in ids:
        target = by_id_running.get(job_id)
        if target is not None:
            if target.get("user") != USER:
                print(f"job {job_id} belongs to {target.get('user')}; "
                      "you can only kill your own jobs.", file=sys.stderr)
                failures += 1
            elif not _kill_running_job(target):
                failures += 1
            continue
        target = by_id_queued.get(job_id)
        if target is not None:
            if target.get("user") != USER:
                print(f"queued job {job_id} belongs to {target.get('user')}; "
                      "you can only cancel your own jobs.", file=sys.stderr)
                failures += 1
            elif not _cancel_queued_job(target):
                failures += 1
            continue
        print(f"no running or queued job {job_id} on this host "
              "(or it already finished).", file=sys.stderr)
        failures += 1
    if failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_TEMPLATE = {
    "max_job_time_hours": 24,         # default -t when the submitter passes none
    "max_job_time_hours_cap": 48,     # hard wall-time ceiling = 2 days; -t may not exceed it
    "max_memory_per_gpu_gb": 70,
    # Default -m filter: min free VRAM (GB) a candidate GPU needs when the
    # submitter passes no -m. Distinct from max_memory_per_gpu_gb, which is a
    # legacy fallback for this default on configs that lack this key.
    "default_min_free_gb": 16,
    # Hard cap on DISTINCT cards a user may hold at once (0 = off). Enforced at
    # claim time: at the cap, new submits stack onto cards the user already
    # owns instead of taking free ones. The audit max_gpus_per_user below is
    # the softer warn-only threshold.
    "max_gpus_per_user_hard": 3,
    "notification_email": {
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "",
        "password": "",
        "admin_email": "admin@yourlab.com",
    },
    "slack": {"enabled": False, "webhook_url": "", "channel": "#gpu-alerts"},
    "quotas": {
        "default_gpu_hours_per_week": 0,  # 0 = unlimited
        "delay_hours": 0.25,               # 15 min: over-quota submits are HELD this long before they may claim
        "users": {},                       # e.g. {"alice": 168}
    },
    "audit": {
        "max_gpus_per_user": 2,
        "max_total_memory_gb": 50,
        # Untracked-GPU detector (off by default). When enabled, `gpuq audit`
        # emails users running GPU processes not launched via `gpuq submit`,
        # and `gpuq audit --enforce` kills the ones past their grace deadline.
        "notify_untracked": False,
        "untracked_min_memory_mb": 512,   # ignore procs smaller than this
        "untracked_grace_seconds": 120,   # suppress flags right after a submit
        "untracked_grace_hours": 0.25,    # 15 min: offender's time to react before kill
        "untracked_reminder_hours": 2,    # reminder cadence within the window
        "untracked_allowlist": [],        # extra FULL login names never flagged
        # Rebind detector (off by default). When enabled, `gpuq audit` emails
        # users whose job runs on a GPU other than the one it was allocated, and
        # `gpuq audit --enforce` kills the ones past their grace deadline.
        "notify_rebind": False,
        "rebind_min_memory_mb": 512,      # ignore procs smaller than this
        "rebind_grace_seconds": 120,      # suppress flags right after a submit
        "rebind_grace_hours": 0.25,       # 15 min: offender's time to react before kill
        "rebind_reminder_hours": 2,       # reminder cadence within the window
    },
}


def cmd_config_show(config):
    print(f"Config file: {CONFIG_FILE}")
    print(f"  Exists:    {CONFIG_FILE.exists()}")
    print(f"  Readable:  {os.access(CONFIG_FILE, os.R_OK)}")
    print(f"Queue dir:   {QUEUE_DIR}")
    print(f"  Writable:  {os.access(QUEUE_DIR, os.W_OK)}")
    print(f"Host:        {HOST}")
    print(f"User:        {USER}")
    mem_default = config.get(
        "default_min_free_gb",
        config.get("max_memory_per_gpu_gb", DEFAULT_MAX_MEMORY_GB))
    print(f"Defaults:    "
          f"max_time={config.get('max_job_time_hours', DEFAULT_MAX_TIME_HOURS)}h, "
          f"min_free_vram_filter={mem_default}GB")
    print("Notifications:")
    print(f"  Email enabled: {bool(config.get('notification_email', {}).get('enabled'))}")
    print(f"  Slack enabled: {bool(config.get('slack', {}).get('enabled'))} "
          f"(requests installed: {_HAS_REQUESTS})")
    q = config.get("quotas") or {}
    default_q = q.get("default_gpu_hours_per_week", 0)
    print(f"Quotas:      default={default_q} GPU-h/week, "
          f"per-user={len((q.get('users') or {}))} entries "
          "(0 = unlimited; see `gpuq quota`)")
    if not CONFIG_FILE.exists():
        print("\nNo config file yet; an admin can create one with "
              "`gpuq config init`.", file=sys.stderr)


def cmd_config(args, config):
    """Show active settings (default), or write a starter config (`init`)."""
    if getattr(args, "config_action", None) != "init":
        if getattr(args, "force", False):
            # Pre-init `gpuq config --force` used to overwrite the config;
            # silently showing settings instead would let old automation
            # "succeed" while writing nothing.
            die("--force only applies to `gpuq config init`; "
                "use `gpuq config init --force` to overwrite the config.")
        cmd_config_show(config)
        return
    if CONFIG_FILE.exists() and not getattr(args, "force", False):
        die(f"config already exists at {CONFIG_FILE}; "
            "use `gpuq config` to view, or `gpuq config init --force` to "
            "overwrite.")
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG_TEMPLATE, f, indent=2)
        try:
            os.chmod(CONFIG_FILE, 0o644)
        except OSError:
            pass
    except OSError as e:
        die(f"could not write config to {CONFIG_FILE}: {e}")
    print(f"Wrote default config to {CONFIG_FILE}", file=sys.stderr)
    print("Edit it to enable notifications, quotas, and audit thresholds.",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Untracked-GPU detection (warn -> remind -> enforce)
# ---------------------------------------------------------------------------
def _traces_to_tracked(pid, tracked_pids, cache, max_depth=64):
    """True if `pid` or any ancestor up to init is a tracked job's child pid.

    Robust to a descendant that re-ran setsid() (its pgid/sid change but its
    parent chain does not), so a legitimate worker is never flagged as untracked.
    """
    cur = int(pid)
    for _ in range(max_depth):
        if cur in tracked_pids:
            return True
        if cur <= 1:
            return False
        if cur not in cache:
            cache[cur] = get_parent_pid(cur)
        parent = cache[cur]
        if parent is None or parent == cur:
            return False
        cur = parent
    return False


def _proc_in_scope(pid, scopes):
    """True if the process's cgroup is inside any of the given scope names.

    cgroup membership is inherited by every descendant and survives setsid,
    double-fork and reparenting — so a job's workers match even when a framework
    detaches them into their own session."""
    if not scopes:
        return False
    try:
        with open(f"/proc/{int(pid)}/cgroup") as f:
            cg = f.read()
    except (OSError, ValueError):
        return False
    return any(s in cg for s in scopes)


def _ancestor_chain(pid, ppid_cache, max_depth=64):
    """The pid's ancestry, nearest first: [pid, parent, grandparent, ...]."""
    chain = []
    cur = int(pid)
    for _ in range(max_depth):
        chain.append(cur)
        if cur <= 1:
            break
        if cur not in ppid_cache:
            ppid_cache[cur] = get_parent_pid(cur)
        parent = ppid_cache[cur]
        if parent is None or parent == cur:
            break
        cur = parent
    return chain


def _attribute_to_job(p, jobs, ppid_cache):
    """Return the tracked job (on this host) that owns GPU process `p`, or None.

    Match order mirrors check_untracked's clearing tests, most-specific first:
      1. cgroup scope — p's cgroup is inside the job's cgroup_scope
      2. child_pgid   — p's process group is the job's child pgid
      3. ancestry     — p chains up to a job's child pid (a re-setsid'd
         worker). When several jobs' child pids sit on the chain (a job that
         itself ran `gpuq submit`), the NEAREST ancestor wins — attributing a
         nested job's worker to the outer job would flag it as a rebind.
    A job's cgroup_scope and child_pgid/child_pid are unique per job id, so the
    match is unambiguous even when a user stacks several jobs on one GPU. Jobs
    lacking all three (legacy records) are unmatchable here and yield None — the
    caller then skips them rather than risk pinning the wrong job's allocation.
    """
    for j in jobs:
        sc = j.get("cgroup_scope")
        if sc and _proc_in_scope(p["pid"], {sc}):
            return j
    pgid = p.get("pgid")
    if pgid is not None:
        for j in jobs:
            cpg = j.get("child_pgid")
            if cpg is not None and int(pgid) == int(cpg):
                return j
    chain = _ancestor_chain(p["pid"], ppid_cache)
    depth = {pid: i for i, pid in enumerate(chain)}
    best, best_depth = None, None
    for j in jobs:
        cpid = j.get("child_pid")
        if cpid is None:
            continue
        d = depth.get(int(cpid))
        if d is not None and (best_depth is None or d < best_depth):
            best, best_depth = j, d
    return best


def _state_group_alive(ent):
    """Is a detector state entry's offender group still the SAME group?

    Mirrors child_group_alive: pgid numbers recycle, and a stale entry riding
    a recycled pgid would hand its weeks-old first_seen (= an already-expired
    deadline) to whatever lands on that pgid next — an instant --enforce kill
    with zero grace. Entries that predate pgid_start fall back to bare pgid
    liveness."""
    pgid = ent.get("pgid")
    if pgid is None or not is_pgid_alive(pgid):
        return False
    recorded = ent.get("pgid_start")
    if recorded is not None:
        current = pid_start_time(pgid)
        if current is not None and current != recorded:
            return False
    return True


def _fresh_group_entry(ent, pgid):
    """None-out a matched state entry whose pgid number was recycled by a
    DIFFERENT group, so the caller treats the sighting as first-seen (fresh
    grace deadline) instead of inheriting the dead offender's clock."""
    if ent is None:
        return None
    recorded = ent.get("pgid_start")
    if recorded is not None:
        current = pid_start_time(pgid)
        if current is not None and current != recorded:
            return None
    return ent


def check_untracked(running, audit_cfg, args, config, now=None):
    """Detect GPU processes not launched via gpuq, drive the warn->kill state
    machine, and return breach strings for the audit summary.

    Side effects (under the shared lock): updates UNTRACKED_STATE_FILE and emails
    offenders; with args.enforce, kills process groups past their grace deadline.
    `now` is injectable so tests can cross the deadline without waiting.
    """
    now = now or datetime.now()
    min_mem = int(audit_cfg.get("untracked_min_memory_mb", 512))
    grace_sec = float(audit_cfg.get("untracked_grace_seconds", 120))
    grace_h = float(audit_cfg.get("untracked_grace_hours", 0.25))
    reminder_h = float(audit_cfg.get("untracked_reminder_hours", 6))
    allowlist = SYSTEM_GPU_ACCOUNTS | set(audit_cfg.get("untracked_allowlist", []) or [])
    enforce = bool(getattr(args, "enforce", False))

    # Tracked sets for THIS host. New jobs carry child_pgid (precise match);
    # jobs that predate this feature fall back to (user, gpu) so they are never
    # wrongly killed during the transition.
    tracked_pgids = set()
    tracked_child_pids = set()
    tracked_scopes = set()
    legacy_user_gpus = set()
    youngest = {}
    for j in running:
        if j.get("host") != HOST:
            continue
        cpg = j.get("child_pgid")
        cpid = j.get("child_pid")
        sc = j.get("cgroup_scope")
        if sc:
            tracked_scopes.add(sc)
        if cpg is not None:
            try:
                tracked_pgids.add(int(cpg))
            except (TypeError, ValueError):
                pass
        else:
            for gi in j.get("gpus", []):
                legacy_user_gpus.add((j.get("user"), int(gi)))
        if cpid is not None:
            try:
                tracked_child_pids.add(int(cpid))
            except (TypeError, ValueError):
                pass
        ts = _parse_iso(j.get("started_at", ""))
        u = j.get("user")
        if ts and (u not in youngest or ts > youngest[u]):
            youngest[u] = ts

    # nvidia-smi failing is NOT "no GPU processes": skip the whole check so
    # offender state (first_seen/deadlines) survives a driver blip untouched.
    procs = list_gpu_processes_with_owner()
    if procs is None:
        print("[gpuq audit] nvidia-smi failed; untracked check skipped this "
              "run.", file=sys.stderr)
        return []

    # Group offending processes by (owner, pgid). A job's workers normally share
    # the child's pgid; a worker that re-ran setsid is still protected by the
    # ancestry check below, so a multi-worker job is one offender at most.
    groups = {}
    ppid_cache = {}
    for p in procs:
        owner, idx, pgid = p["owner"], p["gpu_idx"], p["pgid"]
        if owner is None or idx is None or pgid is None:
            continue                        # ps race / MIG / unknown uuid
        if owner in allowlist:
            continue
        if p["memory_mb"] < min_mem:
            continue
        if _proc_in_scope(p["pid"], tracked_scopes):
            continue                        # inside a tracked job's cgroup scope
        if int(pgid) in tracked_pgids:
            continue                        # a tracked gpuq job (fallback path)
        if tracked_child_pids and _traces_to_tracked(p["pid"], tracked_child_pids, ppid_cache):
            continue                        # a descendant of a tracked job (re-setsid'd worker)
        if (owner, idx) in legacy_user_gpus:
            continue                        # pre-feature job: conservative skip
        yt = youngest.get(owner)
        if yt and (now - yt).total_seconds() < grace_sec:
            continue                        # submit -> child-spawn race window
        groups.setdefault((owner, int(pgid)), []).append(p)

    breaches = []
    pending = []   # deferred to after the lock: kills (slow) + emails
    with file_lock(LOCK_FILE):
        state = _load(UNTRACKED_STATE_FILE)
        state = state if isinstance(state, dict) else {}
        active = set()
        for (owner, pgid), procs in sorted(groups.items()):
            sk = f"{HOST}|{owner}|{pgid}"
            active.add(sk)
            gidxs = sorted({p["gpu_idx"] for p in procs})
            total_mb = sum(p["memory_mb"] for p in procs)
            sample = {"pid": procs[0]["pid"], "gpus": gidxs,
                      "memory_mb": total_mb, "name": procs[0]["name"]}
            ent = _fresh_group_entry(state.get(sk), pgid)
            if ent is None:
                deadline = now + timedelta(hours=grace_h)
                state[sk] = {
                    "host": HOST, "owner": owner, "pgid": pgid,
                    "pgid_start": pid_start_time(pgid),
                    "first_seen": now.isoformat(timespec="seconds"),
                    "last_email": now.isoformat(timespec="seconds"),
                    "reminders": 0, "sample": sample,
                }
                breaches.append(
                    f"{owner}: UNTRACKED GPU process group {pgid} on GPU {gidxs} "
                    f"({total_mb / 1024:.1f} GB, {sample['name']}) - not via gpuq; "
                    f"deadline {deadline.isoformat(timespec='minutes')}")
                pending.append({"owner": owner, "sample": sample,
                                "deadline": deadline, "kind": "warn", "kill": False})
                continue

            ent["sample"] = sample
            first = _parse_iso(ent.get("first_seen")) or now
            deadline = first + timedelta(hours=grace_h)
            last = _parse_iso(ent.get("last_email"))
            due = last is None or (now - last).total_seconds() / 3600.0 >= reminder_h
            if now >= deadline:
                breaches.append(
                    f"{owner}: UNTRACKED GPU process group {pgid} on GPU {gidxs} "
                    f"PAST DEADLINE ({deadline.isoformat(timespec='minutes')})"
                    + (" - enforcing" if enforce else " - escalated to admin"))
                if due:
                    ent["last_email"] = now.isoformat(timespec="seconds")
                pending.append({"owner": owner, "sample": sample, "deadline": deadline,
                                "kind": "overdue" if due else None,
                                "kill": enforce, "pgid": pgid})
            else:
                if due:
                    ent["last_email"] = now.isoformat(timespec="seconds")
                    ent["reminders"] = int(ent.get("reminders", 0)) + 1
                    pending.append({"owner": owner, "sample": sample,
                                    "deadline": deadline, "kind": "remind", "kill": False})
                breaches.append(
                    f"{owner}: UNTRACKED GPU process group {pgid} on GPU {gidxs} "
                    f"({total_mb / 1024:.1f} GB) - deadline "
                    f"{deadline.isoformat(timespec='minutes')}")

        # Self-heal: forget entries whose process group is verifiably gone on
        # this host (moved to gpuq or stopped); leave other hosts' entries
        # untouched. An entry whose group was merely skipped this run for a
        # transient reason (VRAM dip below the threshold, the owner's fresh
        # submit opening the grace window) keeps its first_seen/deadline as
        # long as the process group itself is still alive — otherwise the
        # 15-minute enforcement clock restarts on every such blip and never
        # fires.
        state = {k: v for k, v in state.items()
                 if k in active or v.get("host") != HOST
                 or _state_group_alive(v)}
        _save(UNTRACKED_STATE_FILE, state)

    # Outside the lock: kill (slow + privilege-aware), then email. A successful
    # kill ALWAYS notifies the user, even if the reminder throttle would have
    # suppressed an "overdue" email this run (a kill is a one-time terminal
    # event, so there is no spam risk and the user must be told).
    for item in pending:
        killed = (enforce_kill_pgid(item["pgid"],
                                    expect_owner=item["owner"],
                                    sample_pid=item["sample"].get("pid"))
                  if item.get("kill") else False)
        kind = "killed" if killed else item["kind"]
        if kind is None:
            continue              # past deadline, not yet due, and nothing killed
        notify_untracked(item["owner"], item["sample"], item["deadline"], config, kind)
    return breaches


def check_rebind(running, audit_cfg, args, config, now=None):
    """Detect GPU processes attributable to a tracked job but running on a GPU
    OUTSIDE that job's allocation (a "rebind"), drive the same warn->remind->kill
    state machine as check_untracked, and return breach strings for the summary.

    A process is a rebind iff it belongs to a specific tracked job J (matched by
    J's cgroup scope / child pgid / ancestry) and its GPU index is not in
    J["gpus"]. The per-JOB comparison keeps this correct under the "you own your
    allocated GPU" policy: two of a user's jobs stacked on one card each sit
    inside their own job's allocation, so neither is flagged. Processes that
    match no job are the untracked detector's concern, not this one.

    Side effects (under the shared lock): updates REBIND_STATE_FILE and emails
    offenders; with args.enforce, kills process groups past their grace deadline.
    `now` is injectable so tests can cross the deadline without waiting.
    """
    now = now or datetime.now()
    min_mem = int(audit_cfg.get("rebind_min_memory_mb", 512))
    grace_sec = float(audit_cfg.get("rebind_grace_seconds", 120))
    grace_h = float(audit_cfg.get("rebind_grace_hours", 0.25))
    reminder_h = float(audit_cfg.get("rebind_reminder_hours", 6))
    enforce = bool(getattr(args, "enforce", False))

    jobs = [j for j in running if j.get("host") == HOST]

    # nvidia-smi failing is NOT "no GPU processes": skip the whole check so
    # offender state (first_seen/deadlines) survives a driver blip untouched.
    procs = list_gpu_processes_with_owner()
    if procs is None:
        print("[gpuq audit] nvidia-smi failed; rebind check skipped this "
              "run.", file=sys.stderr)
        return []

    # Group offending processes by (job id, pgid): a job's workers share the
    # child's pgid, so a multi-worker rebind is reported as one offender.
    groups = {}
    ppid_cache = {}
    for p in procs:
        owner, idx, pgid = p["owner"], p["gpu_idx"], p["pgid"]
        if owner is None or idx is None or pgid is None:
            continue                        # ps race / MIG / unknown uuid
        if p["memory_mb"] < min_mem:
            continue
        job = _attribute_to_job(p, jobs, ppid_cache)
        if job is None:
            continue                        # not a tracked job -> untracked's path
        if int(idx) in {int(g) for g in job.get("gpus", [])}:
            continue                        # on an allocated GPU (incl. stacking)
        ts = _parse_iso(job.get("started_at", ""))
        if ts and (now - ts).total_seconds() < grace_sec:
            continue                        # submit -> child-spawn race window
        grp = groups.setdefault((job["id"], int(pgid)), {"job": job, "procs": []})
        grp["procs"].append(p)

    breaches = []
    pending = []   # deferred to after the lock: kills (slow) + emails
    with file_lock(LOCK_FILE):
        state = _load(REBIND_STATE_FILE)
        state = state if isinstance(state, dict) else {}
        active = set()
        for (job_id, pgid), grp in sorted(groups.items()):
            owner = grp["job"].get("user")
            allocated = sorted(int(g) for g in grp["job"].get("gpus", []))
            actual = sorted({p["gpu_idx"] for p in grp["procs"]})
            total_mb = sum(p["memory_mb"] for p in grp["procs"])
            sk = f"{HOST}|rebind|{job_id}|{pgid}"
            active.add(sk)
            sample = {"pid": grp["procs"][0]["pid"], "gpus": actual,
                      "allocated": allocated, "job_id": job_id,
                      "memory_mb": total_mb, "name": grp["procs"][0]["name"]}
            ent = _fresh_group_entry(state.get(sk), pgid)
            if ent is None:
                deadline = now + timedelta(hours=grace_h)
                state[sk] = {
                    "host": HOST, "owner": owner, "job_id": job_id, "pgid": pgid,
                    "pgid_start": pid_start_time(pgid),
                    "first_seen": now.isoformat(timespec="seconds"),
                    "last_email": now.isoformat(timespec="seconds"),
                    "reminders": 0, "sample": sample,
                }
                breaches.append(
                    f"{owner}: REBIND job {job_id} allocated GPU {allocated} but "
                    f"process group {pgid} is on GPU {actual} "
                    f"({total_mb / 1024:.1f} GB, {sample['name']}) - "
                    f"deadline {deadline.isoformat(timespec='minutes')}")
                pending.append({"owner": owner, "sample": sample,
                                "deadline": deadline, "kind": "warn", "kill": False})
                continue

            ent["sample"] = sample
            first = _parse_iso(ent.get("first_seen")) or now
            deadline = first + timedelta(hours=grace_h)
            last = _parse_iso(ent.get("last_email"))
            due = last is None or (now - last).total_seconds() / 3600.0 >= reminder_h
            if now >= deadline:
                breaches.append(
                    f"{owner}: REBIND job {job_id} (allocated GPU {allocated}, on GPU "
                    f"{actual}) PAST DEADLINE ({deadline.isoformat(timespec='minutes')})"
                    + (" - enforcing" if enforce else " - escalated to admin"))
                if due:
                    ent["last_email"] = now.isoformat(timespec="seconds")
                pending.append({"owner": owner, "sample": sample, "deadline": deadline,
                                "kind": "overdue" if due else None,
                                "kill": enforce, "pgid": pgid})
            else:
                if due:
                    ent["last_email"] = now.isoformat(timespec="seconds")
                    ent["reminders"] = int(ent.get("reminders", 0)) + 1
                    pending.append({"owner": owner, "sample": sample,
                                    "deadline": deadline, "kind": "remind", "kill": False})
                breaches.append(
                    f"{owner}: REBIND job {job_id} allocated GPU {allocated} but on "
                    f"GPU {actual} ({total_mb / 1024:.1f} GB) - deadline "
                    f"{deadline.isoformat(timespec='minutes')}")

        # Self-heal: forget entries whose offending group is verifiably gone on
        # this host; a group merely skipped this run for a transient reason
        # keeps its deadline while its process group is still alive (see the
        # matching prune in check_untracked).
        state = {k: v for k, v in state.items()
                 if k in active or v.get("host") != HOST
                 or _state_group_alive(v)}
        _save(REBIND_STATE_FILE, state)

    # Outside the lock: kill (slow + privilege-aware), then email. A successful
    # kill ALWAYS notifies, even if the reminder throttle would have suppressed an
    # "overdue" email this run (a kill is a one-time event; the user must be told).
    for item in pending:
        killed = (enforce_kill_pgid(item["pgid"],
                                    expect_owner=item["owner"],
                                    sample_pid=item["sample"].get("pid"))
                  if item.get("kill") else False)
        kind = "killed" if killed else item["kind"]
        if kind is None:
            continue              # past deadline, not yet due, and nothing killed
        notify_rebind(item["owner"], item["sample"], item["deadline"], config, kind)
    return breaches


# ---------------------------------------------------------------------------
# Subcommand: audit  (cron-friendly resource-hog + quota check)
# ---------------------------------------------------------------------------
def cmd_audit(args, config):
    """Detect resource hogs and over-quota users; alert via Slack/email.

    Designed to run from a user-cron, e.g.   */15 * * * * gpuq audit
    Exit code: 0 if clean, 1 if any breach was reported.
    """
    audit_cfg = config.get("audit", {}) or {}
    max_gpus = int(audit_cfg.get("max_gpus_per_user", 2))
    max_total_gb = float(audit_cfg.get("max_total_memory_gb", 50))

    with file_lock(LOCK_FILE):
        running, _queued = reap_all_locked()

    by_user = {}
    for j in running:
        u = j.get("user", "?")
        bucket = by_user.setdefault(u, {"gpus": set(), "memory_gb": 0.0, "jobs": []})
        bucket["gpus"].update(j.get("gpus", []))
        bucket["memory_gb"] += float(j.get("memory_gb", 0)) * len(j.get("gpus", []))
        bucket["jobs"].append(j["id"])

    breaches = []
    for u, data in sorted(by_user.items()):
        if len(data["gpus"]) > max_gpus:
            breaches.append(
                f"{u}: holds {len(data['gpus'])} GPUs (limit {max_gpus}); "
                f"jobs={data['jobs']}"
            )
        if data["memory_gb"] > max_total_gb:
            breaches.append(
                f"{u}: holds {data['memory_gb']:.0f} GB total (limit {max_total_gb:.0f}); "
                f"jobs={data['jobs']}"
            )

    # Quota breaches: anyone whose used hours in the window > their budget.
    # Candidates: users with running jobs, users with an explicit quota entry,
    # AND users seen in the ledger window — a hit-and-run user who is idle at
    # audit time but burned through the default budget must still be flagged.
    now = datetime.now()
    cutoff = now - timedelta(hours=QUOTA_WINDOW_HOURS)
    ledger_users = set()
    for rec in iter_usage_records():
        if rec.get("event") != "end":
            continue
        ended = _parse_iso(rec.get("ended_at", ""))
        if ended is not None and ended >= cutoff and rec.get("user"):
            ledger_users.add(rec["user"])
    quotas_users = ((config.get("quotas") or {}).get("users") or {})
    over_quota_users = set(by_user.keys()) | set(quotas_users.keys()) | ledger_users
    for u in sorted(over_quota_users):
        budget = quota_for_user(u, config)
        if budget is None:
            continue
        used = usage_in_window(u)
        if used > budget:
            breaches.append(
                f"{u}: used {used:.1f} GPU-hours in last {QUOTA_WINDOW_HOURS}h "
                f"(budget {budget:.1f})"
            )

    # Untracked-GPU detector (opt-in): warn/kill users running GPU jobs that
    # were not launched via `gpuq submit`.
    if bool(audit_cfg.get("notify_untracked", False)):
        breaches.extend(check_untracked(running, audit_cfg, args, config))

    # Rebind detector (opt-in): warn/kill jobs running on a GPU other than the
    # one gpuq allocated them.
    if bool(audit_cfg.get("notify_rebind", False)):
        breaches.extend(check_rebind(running, audit_cfg, args, config))

    if not breaches:
        if not getattr(args, "quiet", False):
            print("[gpuq audit] clean.")
        sys.exit(0)

    summary = "\n".join(breaches)
    print(f"[gpuq audit] breaches on {HOST}:\n{summary}")

    # Alert admins. Email goes to admin_email; Slack to its configured webhook.
    admin_email = (config.get("notification_email") or {}).get("admin_email")
    if admin_email:
        send_email(admin_email,
                   f"[gpuq] resource breaches on {HOST}",
                   summary, config)
    send_slack(f"*gpuq audit on {HOST}*\n```{summary}```", config)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def die(msg):
    print(f"gpuq: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="gpuq",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "gpuq — cooperative GPU job queue for a shared host.\n\n"
            "`gpuq submit` claims free GPU(s), sets CUDA_VISIBLE_DEVICES, and runs\n"
            "your command in the foreground (that process supervises the job and\n"
            "enforces its time limit). There is no daemon. `gpuq audit`, run from\n"
            "cron, reports resource hogs, over-quota users, and GPU jobs not started\n"
            "through gpuq, and can kill untracked jobs past a grace deadline."
        ),
        epilog=(
            "examples:\n"
            "  gpuq submit -- python train.py             run on one random free GPU\n"
            "  gpuq submit -g 2 -t 12 -- python x.py       2 random free GPUs, 12h cap\n"
            "  gpuq submit --devices 1,3 -- python x.py    pin exact GPUs (else rejected)\n"
            "  gpuq submit --queue -- python train.py      wait if none are free\n"
            "  gpuq status                                 GPUs + running/queued jobs\n"
            "  gpuq history                                your recent jobs from the ledger\n"
            "  gpuq quota                                  your 7-day GPU-hours vs budget\n"
            "  gpuq kill 12345                             stop or cancel your job\n"
            "  gpuq audit                                  hogs/quota/untracked (cron)\n"
            "\n"
            "Run `gpuq <command> -h` for a command's full options, "
            "e.g. `gpuq submit -h`.\n"
        ),
    )
    sub = parser.add_subparsers(
        dest="action", metavar="{submit,status,history,quota,kill,config,audit}",
        help="run `gpuq <command> -h` for that command's options")

    p_submit = sub.add_parser(
        "submit", help="Claim free GPU(s) and run a command on them.")
    p_submit.add_argument("-g", "--gpus", type=int, default=None,
                          help="Number of whole GPUs to claim (default: 1). gpuq "
                               "picks them at random among the free GPUs.")
    p_submit.add_argument("--devices", default=None,
                          help="Pin to specific GPU index(es), comma-separated "
                               "(e.g. --devices 1,3); count is taken from the list. "
                               "Each must be free or already yours. Rejected if any "
                               "is held by someone else, UNLESS --queue is given, in "
                               "which case it waits until all of them are available. "
                               "Subject to the same quota gate as the picker.")
    p_submit.add_argument("-m", "--memory", type=int, default=None,
                          help="Only pick a GPU with at least this much free VRAM "
                               "in GB (an admission filter on BOTH free GPUs and "
                               "stacking onto cards you already own, not a "
                               "reservation; default: config default_min_free_gb). "
                               "Tip: pass a value your own busy card can't meet to "
                               "make --queue wait for a genuinely free GPU instead "
                               "of stacking.")
    p_submit.add_argument("-t", "--time", type=float, default=None,
                          help="Max runtime in hours, > 0, capped at 48h (2 days); the "
                               "job is killed if it runs longer "
                               "(default: config max_job_time_hours).")
    p_submit.add_argument("--queue", action="store_true",
                          help="If no slot is free now, wait and poll instead of "
                               "exiting. Applies to both the default picker and "
                               "--devices (wait for those exact GPUs).")
    p_submit.add_argument("--notify", default=None,
                          help="Email for the completion notice (overrides the "
                               "address read from your account).")
    p_submit.add_argument("--name", default=None,
                          help="Optional label shown in `gpuq status`.")
    p_submit.add_argument("--command", default=None,
                          help="Command as one shell string, instead of passing it "
                               "after `--`.")
    p_submit.add_argument("cmd_args", nargs=argparse.REMAINDER,
                          help="The command and its arguments, after `--`.")

    sub.add_parser(
        "status", help="Show each GPU's state and the running/queued jobs.")

    p_hist = sub.add_parser(
        "history",
        help="Show your recent jobs from the usage ledger (finished, "
             "timed-out, lost, ...).")
    p_hist.add_argument("-n", "--limit", type=int, default=20,
                        help="Show at most N records (default: 20; 0 = all).")
    p_hist.add_argument("--all", action="store_true",
                        help="All users' jobs, not just yours.")
    p_hist.add_argument("--user", default=None,
                        help="Another user's jobs (the ledger is group-readable).")
    p_hist.add_argument("--events", action="store_true",
                        help="Also show non-job events (cancelled/rejected "
                             "submits).")
    p_hist.add_argument("--json", action="store_true",
                        help="Print raw ledger records as JSON lines.")

    p_quota = sub.add_parser(
        "quota",
        help="Show rolling 7-day GPU-hour usage vs budget (yours by default).")
    p_quota.add_argument("--user", default=None,
                         help="Another user's usage instead of yours.")
    p_quota.add_argument("--all", action="store_true",
                         help="One row per user (plus host capacity).")
    p_quota.add_argument("--report", action="store_true",
                         help="Weekly per-user statistics (p50/p95/max, queue "
                              "waits, timeout/lost rates) for setting budgets.")
    p_quota.add_argument("--weeks", type=int, default=8,
                         help="Window for --report, in ISO weeks (default: 8).")

    p_kill = sub.add_parser(
        "kill", help="Stop your running jobs or cancel your queued ones.")
    p_kill.add_argument("job_id", type=int, nargs="*",
                        help="ID(s) of your job(s) to stop or cancel "
                             "(see `gpuq status`).")
    p_kill.add_argument("--job-id", dest="job_id_flag", type=int,
                        help="Job ID as a flag, instead of the positional argument.")
    p_kill.add_argument("--mine", action="store_true",
                        help="Stop ALL of your running jobs and cancel all "
                             "your queued ones on this host.")

    p_config = sub.add_parser(
        "config",
        help="Show active settings; `gpuq config init` writes a starter "
             "config file (admin).")
    p_config.add_argument("config_action", nargs="?", choices=["init"],
                          help="`init` writes the default config template "
                               "(refuses if one exists; see --force).")
    p_config.add_argument("--show", action="store_true",
                          help="(default) Print the loaded config, paths, and "
                               "host/user.")
    p_config.add_argument("--force", action="store_true",
                          help="With `init`: overwrite an existing config file.")

    p_audit = sub.add_parser(
        "audit",
        help="Report resource hogs, over-quota users, and untracked GPU jobs (cron).")
    p_audit.add_argument("--quiet", "-q", action="store_true",
                         help="Print only on a breach (suppress the 'clean' line).")
    p_audit.add_argument("--enforce", action="store_true",
                         help="Also kill untracked GPU jobs past their grace deadline "
                              "(needs privilege to signal other users' processes).")

    args = parser.parse_args()
    if not args.action:
        parser.print_help(sys.stderr)
        sys.exit(2)

    config = load_config()

    handlers = {
        "submit": cmd_submit,
        "status": cmd_status,
        "history": cmd_history,
        "quota": cmd_quota,
        "kill": cmd_kill,
        "config": cmd_config,
        "audit": cmd_audit,
    }
    handlers[args.action](args, config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
