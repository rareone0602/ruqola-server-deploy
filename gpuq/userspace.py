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
  - First-come-first-serve, with per-user rolling 7-day GPU-hour quotas.
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
DEFAULT_MAX_MEMORY_GB = 70
GPU_UTIL_AVAILABLE_THRESHOLD = 10  # percent; below this, GPU counts as idle
GPU_OWN_MIN_FREE_GB = 2            # owned-GPU stacking still needs this much free VRAM
QUEUE_POLL_INTERVAL_SEC = 30
KILL_GRACE_SEC = 10
QUOTA_WINDOW_HOURS = 24 * 7        # rolling 7 days
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    try:
        os.chmod(tmp, 0o664)
    except OSError:
        pass
    os.replace(tmp, path)


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
    except (PermissionError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# nvidia-smi wrappers (read-only)
# ---------------------------------------------------------------------------
def _nvsmi(query_kind, fields):
    try:
        r = subprocess.run(
            [NVSMI_BIN, f"--query-{query_kind}={fields}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def get_gpu_info():
    out = _nvsmi("gpu", "index,name,memory.used,memory.total,utilization.gpu")
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
    out = _nvsmi("compute-apps", "pid,process_name,gpu_uuid,used_memory")
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

    `gpu_idx` is None for a uuid not in the full-GPU map (e.g. MIG instances);
    `owner`/`pgid` are None if the process exited between the nvidia-smi and
    ps/getpgid calls. Callers decide how to treat the None cases.
    """
    uuid_to_idx = get_gpu_uuid_to_index()
    out = []
    for p in get_gpu_processes():
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


def reap_running(jobs):
    """Drop entries from this host whose PID is gone."""
    out = []
    for j in jobs:
        if j.get("host") != HOST:
            out.append(j)  # foreign-host entries are not ours to reap
            continue
        if is_pid_alive(j.get("pid")):
            out.append(j)
    return out


# ---------------------------------------------------------------------------
# Quota: rolling 7-day GPU-hour ledger
# ---------------------------------------------------------------------------
def append_usage(record):
    """Append one job's usage line to USAGE_FILE. Caller holds the lock."""
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


def usage_in_window(user, window_hours=QUOTA_WINDOW_HOURS, now=None):
    """Sum gpu_hours for `user` in the trailing window. Reads ledger + running.

    Counts running jobs at their current elapsed runtime so a long-running job
    is reflected in the budget before it ends.
    """
    now = now or datetime.now()
    cutoff = now - timedelta(hours=window_hours)
    total = 0.0
    if USAGE_FILE.exists():
        try:
            with open(USAGE_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("user") != user:
                        continue
                    ended = _parse_iso(rec.get("ended_at", ""))
                    if ended is None or ended < cutoff:
                        continue
                    total += float(rec.get("gpu_hours", 0))
        except OSError:
            pass
    for j in load_running():
        if j.get("user") != user:
            continue
        started = _parse_iso(j.get("started_at", ""))
        if started is None:
            continue
        elapsed = max(0.0, (now - started).total_seconds() / 3600.0)
        count = len(j.get("gpus") or []) or j.get("gpu_count", 1)
        total += elapsed * count
    return total


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


# ---------------------------------------------------------------------------
# GPU slot picking (ownership-aware)
# ---------------------------------------------------------------------------
def _gpu_owner_sets(running_jobs, user):
    """Split GPUs held by running jobs into those `user` owns and those held by
    other users. A GPU co-tenanted by both counts as other-held — we never hand
    a user a card someone else is on. A job with no recorded user is treated as
    other-held (safe: such a card is never auto-claimed)."""
    owned, others = set(), set()
    for j in running_jobs:
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
    gate (the user's own job legitimately drives util up) but still needs a small
    VRAM headroom (GPU_OWN_MIN_FREE_GB) so we don't green-light a doomed-OOM
    stack. GPUs held by other users are never selectable."""
    want_mem_mb = want_memory_gb * 1024
    own_min_mb = GPU_OWN_MIN_FREE_GB * 1024
    owned, others = _gpu_owner_sets(running_jobs, user)
    out = []
    for g in gpus:
        idx = g["index"]
        if idx in others:
            continue
        if idx in owned:
            if g["memory_free_mb"] >= own_min_mb:
                out.append(idx)
        elif (g["memory_free_mb"] >= want_mem_mb
                and g["utilization"] < GPU_UTIL_AVAILABLE_THRESHOLD):
            out.append(idx)
    return sorted(out)


def pick_gpus(want_count, want_memory_gb, gpus, running_jobs, user, devices=None):
    """Choose GPU indices to run on, or None if the request can't be met now.

    With `devices`, pin exactly those indices — all must be selectable for `user`
    (free, or already owned by them), else None. Otherwise choose `want_count`
    among the selectable GPUs, preferring FREE GPUs over ones the user already
    owns so load spreads across the box before stacking; the free choice is
    random (as before) to avoid always starting at GPU 0.
    """
    free = free_gpus(want_memory_gb, gpus, running_jobs, user)
    if devices is not None:
        want = sorted(set(devices))
        return want if all(d in free for d in want) else None
    if len(free) < want_count:
        return None
    owned, _ = _gpu_owner_sets(running_jobs, user)
    free_only = [i for i in free if i not in owned]
    if len(free_only) >= want_count:
        return sorted(random.sample(free_only, want_count))
    owned_sel = [i for i in free if i in owned]
    topup = random.sample(owned_sel, want_count - len(free_only))
    return sorted(free_only + topup)


def gpu_unavailability(devices, gpus, running_jobs, want_memory_gb, user):
    """Human-readable reason each pinned GPU can't be used by `user`; [] means all
    usable. Mirrors free_gpus/pick_gpus exactly so a reason never contradicts the
    picker: a GPU `user` owns is usable while it keeps GPU_OWN_MIN_FREE_GB of
    headroom (util ignored); a GPU held by another user is reported as held; a
    free GPU uses the normal util/VRAM gates."""
    want_mem_mb = want_memory_gb * 1024
    own_min_mb = GPU_OWN_MIN_FREE_GB * 1024
    by_idx = {g["index"]: g for g in gpus}
    owned, others = _gpu_owner_sets(running_jobs, user)
    held = {}
    for j in running_jobs:
        if j.get("user") == user:
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
            if g["memory_free_mb"] < own_min_mb:
                reasons.append(f"GPU {d}: only {g['memory_free_mb'] // 1024} GB free, "
                               f"can't safely stack (need >= {GPU_OWN_MIN_FREE_GB} GB "
                               f"headroom)")
        elif g["utilization"] >= GPU_UTIL_AVAILABLE_THRESHOLD:
            reasons.append(f"GPU {d}: in use ({g['utilization']}% util)")
        elif g["memory_free_mb"] < want_mem_mb:
            reasons.append(f"GPU {d}: only {g['memory_free_mb'] // 1024} GB free "
                           f"(need {want_memory_gb} GB)")
    return reasons


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
                          config):
    """Email the user that their submit was deprioritized for going over quota."""
    if not notify_email:
        notify_email = email_for_user(user)
    if not notify_email:
        return
    subject = f"[gpuq] {user}: GPU-hour quota exceeded - job deprioritized"
    body = (
        f"Host: {HOST}\n"
        f"User: {user}\n"
        f"Used in last {QUOTA_WINDOW_HOURS}h: {used_hours:.1f} GPU-hours\n"
        f"Requested by this job:           {requested_hours:.1f} GPU-hours\n"
        f"Rolling 7-day budget:            {budget:.1f} GPU-hours\n\n"
        "Your job was queued at low priority. It will only run once on-quota\n"
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


def enforce_kill_pgid(pgid):
    """SIGTERM a process group, grace, then SIGKILL. Privilege-aware.

    Returns True if the group is gone (or was already gone), False if we lack
    the privilege to signal it (another user's process and we are not root).
    """
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
    return int(time.time() * 1000) % 2**31


def build_running_job(cmd, args, gpus, memory_gb, max_time_hours, job_id=None):
    return {
        "id": job_id or _new_job_id(),
        "user": USER,
        "host": HOST,
        "command": " ".join(cmd) if isinstance(cmd, list) else cmd,
        "working_directory": os.getcwd(),
        "virtual_env": detect_virtual_environment(),
        "gpu_count": args.gpus,
        "gpus": gpus,
        "memory_gb": memory_gb,
        "max_time_hours": max_time_hours,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "name": args.name,
        "notify_email": args.notify,
        "status": "running",
    }


def build_queued_job(cmd, args, memory_gb, max_time_hours, priority="normal"):
    return {
        "id": _new_job_id(),
        "user": USER,
        "host": HOST,
        "command": " ".join(cmd) if isinstance(cmd, list) else cmd,
        "working_directory": os.getcwd(),
        "virtual_env": detect_virtual_environment(),
        "gpu_count": args.gpus,
        "memory_gb": memory_gb,
        "max_time_hours": max_time_hours,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
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
               deprioritized=False):
    """One attempt, under the lock, to grab a slot. Returns (job, picked) on
    success or (None, None). Reaps stale jobs first and drops our own queued
    entry when we win, so a waiter that finally claims also leaves the queue."""
    with file_lock(LOCK_FILE):
        running = reap_running(load_running())
        save_running(running)
        gpus = get_gpu_info()
        picked = pick_gpus(args.gpus, memory_gb, gpus, running, USER, devices=devices)
        if picked is None:
            return None, None
        job = build_running_job(cmd, args, picked, memory_gb, max_time_hours)
        if deprioritized:
            job["priority"] = "low"
        running.append(job)
        save_running(running)
        queued = load_queued()
        new_q = _drop_my_queued_entry(queued)
        if len(new_q) != len(queued):
            save_queued(new_q)
        return job, picked


def _wait_for_slot(cmd, args, memory_gb, max_time_hours, config, devices=None,
                   deprioritized=False):
    """Poll until a slot can be claimed, returning (job, picked). Registers a
    one-time queued entry so other submitters see us and yield as configured;
    Ctrl-C drops that entry and exits 130. Used by --queue submits (with
    `devices` pinned or not) and by deprioritized over-quota submits."""
    poll_interval = (DEPRIORITIZED_POLL_INTERVAL_SEC if deprioritized
                     else QUEUE_POLL_INTERVAL_SEC)
    queued_announced = False
    first_iter = True
    try:
        while True:
            with file_lock(LOCK_FILE):
                running = reap_running(load_running())
                save_running(running)
                queued = load_queued()
                normal_waiting = any(
                    j.get("priority", "normal") == "normal"
                    and j.get("host") == HOST
                    for j in queued
                )
                # Deprioritized submitters always queue on the first iteration
                # (giving any concurrent normal-priority submit time to land),
                # and keep yielding while a normal-priority submitter is queued.
                skip_pick = deprioritized and (first_iter or normal_waiting)
                if not skip_pick:
                    gpus = get_gpu_info()
                    picked = pick_gpus(args.gpus, memory_gb, gpus, running, USER,
                                       devices=devices)
                    if picked is not None:
                        job = build_running_job(cmd, args, picked, memory_gb,
                                                max_time_hours)
                        if deprioritized:
                            job["priority"] = "low"
                        running.append(job)
                        save_running(running)
                        new_q = _drop_my_queued_entry(queued)
                        if len(new_q) != len(queued):
                            save_queued(new_q)
                        return job, picked
                if not queued_announced:
                    priority = "low" if deprioritized else "normal"
                    qjob = build_queued_job(cmd, args, memory_gb, max_time_hours,
                                            priority=priority)
                    queued.append(qjob)
                    save_queued(queued)
                    tag = " (deprioritized)" if deprioritized else ""
                    pin = (f" [devices {','.join(map(str, devices))}]"
                           if devices else "")
                    print(f"[gpuq] no slot; queued as job {qjob['id']}{tag}{pin}, "
                          f"polling every {poll_interval}s. Ctrl-C to cancel.",
                          file=sys.stderr)
                    queued_announced = True
            first_iter = False
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        with file_lock(LOCK_FILE):
            save_queued(_drop_my_queued_entry(load_queued()))
        print("\n[gpuq] cancelled while queued.", file=sys.stderr)
        sys.exit(130)


def cmd_submit(args, config):
    cmd = list(args.cmd_args or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if args.command:
        cmd = shlex.split(args.command)
    if not cmd:
        die("no command given. Use:  gpuq submit [opts] -- COMMAND ARGS...")

    memory_gb = (args.memory if args.memory is not None
                 else config.get("max_memory_per_gpu_gb", DEFAULT_MAX_MEMORY_GB))
    max_time_hours = (args.time if args.time is not None
                      else config.get("max_job_time_hours", DEFAULT_MAX_TIME_HOURS))

    # --devices: pin exact GPU(s). Each must be free or already owned by you.
    # Without --queue, reject immediately with a per-GPU reason; with --queue,
    # wait until all of them are available. Bypasses the random picker and quota.
    if args.devices:
        try:
            devices = sorted({int(x) for x in args.devices.split(",") if x.strip()})
        except ValueError:
            die(f"--devices must be comma-separated GPU indices, got {args.devices!r}")
        if not devices:
            die("--devices was given but empty")
        args.gpus = len(devices)   # records + accounting use the real count
        job, picked = _try_claim(cmd, args, memory_gb, max_time_hours, devices=devices)
        if job is None:
            if not args.queue:
                with file_lock(LOCK_FILE):
                    running = reap_running(load_running())
                    gpus = get_gpu_info()
                reasons = gpu_unavailability(devices, gpus, running, memory_gb, USER)
                die("requested GPU(s) not available — not submitting:\n  "
                    + "\n  ".join(reasons) + "\nPass --queue to wait for them.")
            job, picked = _wait_for_slot(cmd, args, memory_gb, max_time_hours,
                                         config, devices=devices)
        print(f"[gpuq] job {job['id']} starting on GPU(s) "
              f"{','.join(map(str, picked))}", file=sys.stderr)
        run_and_wait(job, cmd, picked, max_time_hours, args, config)
        return

    # Quota: rolling 7-day GPU-hour budget, per user. Over-quota submitters are
    # deprioritized — forced into queue mode with a longer poll interval — and
    # emailed once. Charging is by actual runtime at job end (see run_and_wait).
    requested_gpu_hours = args.gpus * (max_time_hours or 0)
    over, used, budget = would_exceed_quota(USER, requested_gpu_hours, config)
    deprioritized = False
    if over:
        deprioritized = True
        msg = (f"[gpuq] over quota: used {used:.1f} + requested {requested_gpu_hours:.1f} "
               f"GPU-hours > budget {budget:.1f} (rolling {QUOTA_WINDOW_HOURS}h). "
               "Job DEPRIORITIZED and queued; will only start once on-quota submitters "
               "have a chance to grab slots.")
        print(msg, file=sys.stderr)
        notify_quota_exceeded(USER, args.notify, used, requested_gpu_hours, budget, config)

    # On-quota: try once, then reject (no --queue) or wait. Over-quota always
    # waits and yields to normal-priority submitters on its first poll.
    job = picked = None
    if not deprioritized:
        job, picked = _try_claim(cmd, args, memory_gb, max_time_hours)
        if job is None and not args.queue:
            die(f"no free GPU matches your request "
                f"(need {args.gpus} GPU(s) with >= {memory_gb} GB free, util < "
                f"{GPU_UTIL_AVAILABLE_THRESHOLD}%). Pass --queue to wait.")
    if job is None:
        job, picked = _wait_for_slot(cmd, args, memory_gb, max_time_hours, config,
                                     deprioritized=deprioritized)

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
                if scope:
                    j["cgroup_scope"] = scope
        save_running(running)

    timeout_fired = {"v": False}

    def timeout():
        timeout_fired["v"] = True
        kill_job(signal.SIGTERM)
        time.sleep(KILL_GRACE_SEC)
        kill_job(signal.SIGKILL)

    timer = None
    if max_time_hours and max_time_hours > 0:
        timer = threading.Timer(max_time_hours * 3600, timeout)
        timer.daemon = True
        timer.start()

    try:
        rc = child.wait()
    finally:
        if timer is not None:
            timer.cancel()
        ended_wall = datetime.now()
        elapsed_h = max(0.0, (ended_wall - started_wall).total_seconds() / 3600.0)
        gpu_hours = elapsed_h * len(gpus)
        with file_lock(LOCK_FILE):
            running = load_running()
            running = [j for j in running
                       if not (j.get("id") == job["id"] and j.get("host") == HOST)]
            save_running(running)
            append_usage({
                "id": job["id"],
                "user": job.get("user", USER),
                "host": HOST,
                "gpus": list(gpus),
                "started_at": started_wall.isoformat(timespec="seconds"),
                "ended_at": ended_wall.isoformat(timespec="seconds"),
                "elapsed_hours": round(elapsed_h, 4),
                "gpu_hours": round(gpu_hours, 4),
                "priority": job.get("priority", "normal"),
            })
        reason = ("timed_out" if timeout_fired["v"]
                  else "completed" if rc == 0
                  else "failed")
        notify_completion(job, rc, reason, config, args.notify)
    sys.exit(rc)


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------
def _fmt_duration(start_iso):
    try:
        t0 = datetime.fromisoformat(start_iso)
        return str(datetime.now() - t0).split(".")[0]
    except Exception:
        return "?"


def cmd_status(args, config):
    with file_lock(LOCK_FILE):
        running = reap_running(load_running())
        save_running(running)
        queued = load_queued()
    gpus = get_gpu_info()

    print(f"=== GPU Queue Status (host: {HOST}) ===\n")
    print(f"GPUs ({len(gpus)} total):")
    busy_set = {int(gi) for j in running for gi in j.get("gpus", [])}
    for g in gpus:
        state = "BUSY" if g["index"] in busy_set else "FREE"
        pct = 100 * g["memory_used_mb"] / g["memory_total_mb"] if g["memory_total_mb"] else 0
        print(f"  GPU {g['index']}: {g['name']} - {state}")
        print(f"    Memory: {g['memory_used_mb']}/{g['memory_total_mb']} MB ({pct:.0f}%)")
        print(f"    Utilization: {g['utilization']}%")
    print()

    print(f"Running Jobs ({len(running)}):")
    if not running:
        print("  (none)")
    for j in running:
        line = (f"  Job {j['id']} by {j['user']} - GPUs: {j.get('gpus', [])} - "
                f"Runtime: {_fmt_duration(j.get('started_at', ''))}")
        if j.get("name"):
            line += f" - Name: {j['name']}"
        print(line)
    print()

    print(f"Queued Jobs ({len(queued)}):")
    if not queued:
        print("  (none)")
    for j in queued:
        prio = j.get("priority", "normal")
        tag = f" [{prio}]" if prio != "normal" else ""
        print(f"  Job {j['id']} by {j['user']}{tag} - Waiting: "
              f"{_fmt_duration(j.get('submitted_at', ''))}")
    print()

    print("All GPU compute processes (from nvidia-smi):")
    live = list_gpu_processes_with_owner()
    if not live:
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


# ---------------------------------------------------------------------------
# Subcommand: kill
# ---------------------------------------------------------------------------
def cmd_kill(args, config):
    job_id = args.job_id
    target = None
    with file_lock(LOCK_FILE):
        for j in load_running():
            if j.get("id") == job_id and j.get("host") == HOST:
                target = j
                break
    if target is None:
        die(f"no running job {job_id} on this host (or it already finished).")
    if target.get("user") != USER:
        die(f"job {job_id} belongs to {target.get('user')}; "
            "you can only kill your own jobs.")

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
                return
        print("still alive; escalating to SIGKILL.")
        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", scope],
                       capture_output=True)
        return

    pid = int(target["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        with file_lock(LOCK_FILE):
            running = [j for j in load_running() if j.get("id") != job_id]
            save_running(running)
        print(f"job {job_id} was already gone; cleaned up.")
        return

    print(f"sent SIGTERM to job {job_id} (PID {pid}); waiting up to "
          f"{KILL_GRACE_SEC}s...")
    for _ in range(KILL_GRACE_SEC):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"job {job_id} terminated.")
            return
    print("still alive; escalating to SIGKILL.")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# ---------------------------------------------------------------------------
# Subcommand: config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_TEMPLATE = {
    "max_job_time_hours": 24,
    "max_memory_per_gpu_gb": 70,
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
        "untracked_grace_hours": 24,      # offender's time to react before kill
        "untracked_reminder_hours": 6,    # reminder cadence within the window
        "untracked_allowlist": [],        # extra FULL login names never flagged
        # Rebind detector (off by default). When enabled, `gpuq audit` emails
        # users whose job runs on a GPU other than the one it was allocated, and
        # `gpuq audit --enforce` kills the ones past their grace deadline.
        "notify_rebind": False,
        "rebind_min_memory_mb": 512,      # ignore procs smaller than this
        "rebind_grace_seconds": 120,      # suppress flags right after a submit
        "rebind_grace_hours": 24,         # offender's time to react before kill
        "rebind_reminder_hours": 6,       # reminder cadence within the window
    },
}


def cmd_config_show(config):
    print(f"Config file: {CONFIG_FILE}")
    print(f"  Readable:  {os.access(CONFIG_FILE, os.R_OK)}")
    print(f"Queue dir:   {QUEUE_DIR}")
    print(f"  Writable:  {os.access(QUEUE_DIR, os.W_OK)}")
    print(f"Host:        {HOST}")
    print(f"User:        {USER}")
    print(f"Defaults:    "
          f"max_time={config.get('max_job_time_hours', DEFAULT_MAX_TIME_HOURS)}h, "
          f"max_memory={config.get('max_memory_per_gpu_gb', DEFAULT_MAX_MEMORY_GB)}GB")
    print("Notifications:")
    print(f"  Email enabled: {bool(config.get('notification_email', {}).get('enabled'))}")
    print(f"  Slack enabled: {bool(config.get('slack', {}).get('enabled'))} "
          f"(requests installed: {_HAS_REQUESTS})")
    q = config.get("quotas") or {}
    default_q = q.get("default_gpu_hours_per_week", 0)
    print(f"Quotas:      default={default_q} GPU-h/week, "
          f"per-user={len((q.get('users') or {}))} entries")


def cmd_config(args, config):
    """Write a starter config if none exists, or show active settings (--show)."""
    if getattr(args, "show", False):
        cmd_config_show(config)
        return
    if CONFIG_FILE.exists() and not getattr(args, "force", False):
        die(f"config already exists at {CONFIG_FILE}; "
            "use `gpuq config --show` to view, or --force to overwrite.")
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


def _attribute_to_job(p, jobs, ppid_cache):
    """Return the tracked job (on this host) that owns GPU process `p`, or None.

    Match order mirrors check_untracked's clearing tests, most-specific first:
      1. cgroup scope — p's cgroup is inside the job's cgroup_scope
      2. child_pgid   — p's process group is the job's child pgid
      3. ancestry     — p chains up to the job's child pid (a re-setsid'd worker)
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
    for j in jobs:
        cpid = j.get("child_pid")
        if cpid is not None and _traces_to_tracked(p["pid"], {int(cpid)}, ppid_cache):
            return j
    return None


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
    grace_h = float(audit_cfg.get("untracked_grace_hours", 24))
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

    # Group offending processes by (owner, pgid). A job's workers normally share
    # the child's pgid; a worker that re-ran setsid is still protected by the
    # ancestry check below, so a multi-worker job is one offender at most.
    groups = {}
    ppid_cache = {}
    for p in list_gpu_processes_with_owner():
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
            ent = state.get(sk)
            if ent is None:
                deadline = now + timedelta(hours=grace_h)
                state[sk] = {
                    "host": HOST, "owner": owner, "pgid": pgid,
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

        # Self-heal: forget entries whose process group is gone on this host
        # (moved to gpuq or stopped); leave other hosts' entries untouched.
        state = {k: v for k, v in state.items()
                 if k in active or v.get("host") != HOST}
        _save(UNTRACKED_STATE_FILE, state)

    # Outside the lock: kill (slow + privilege-aware), then email. A successful
    # kill ALWAYS notifies the user, even if the reminder throttle would have
    # suppressed an "overdue" email this run (a kill is a one-time terminal
    # event, so there is no spam risk and the user must be told).
    for item in pending:
        killed = enforce_kill_pgid(item["pgid"]) if item.get("kill") else False
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
    grace_h = float(audit_cfg.get("rebind_grace_hours", 24))
    reminder_h = float(audit_cfg.get("rebind_reminder_hours", 6))
    enforce = bool(getattr(args, "enforce", False))

    jobs = [j for j in running if j.get("host") == HOST]

    # Group offending processes by (job id, pgid): a job's workers share the
    # child's pgid, so a multi-worker rebind is reported as one offender.
    groups = {}
    ppid_cache = {}
    for p in list_gpu_processes_with_owner():
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
            ent = state.get(sk)
            if ent is None:
                deadline = now + timedelta(hours=grace_h)
                state[sk] = {
                    "host": HOST, "owner": owner, "job_id": job_id, "pgid": pgid,
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

        # Self-heal: forget entries whose offending group is gone on this host.
        state = {k: v for k, v in state.items()
                 if k in active or v.get("host") != HOST}
        _save(REBIND_STATE_FILE, state)

    # Outside the lock: kill (slow + privilege-aware), then email. A successful
    # kill ALWAYS notifies, even if the reminder throttle would have suppressed an
    # "overdue" email this run (a kill is a one-time event; the user must be told).
    for item in pending:
        killed = enforce_kill_pgid(item["pgid"]) if item.get("kill") else False
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
        running = reap_running(load_running())
        save_running(running)

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
    quotas_users = ((config.get("quotas") or {}).get("users") or {})
    over_quota_users = set(by_user.keys()) | set(quotas_users.keys())
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
            "  gpuq kill 12345                             stop your job\n"
            "  gpuq audit                                  hogs/quota/untracked (cron)\n"
            "\n"
            "Run `gpuq <command> -h` for a command's full options, "
            "e.g. `gpuq submit -h`.\n"
        ),
    )
    sub = parser.add_subparsers(
        dest="action", metavar="{submit,status,kill,config,audit}",
        help="run `gpuq <command> -h` for that command's options")

    p_submit = sub.add_parser(
        "submit", help="Claim free GPU(s) and run a command on them.")
    p_submit.add_argument("-g", "--gpus", type=int, default=1,
                          help="Number of whole GPUs to claim (default: 1). gpuq "
                               "picks them at random among the free GPUs.")
    p_submit.add_argument("--devices", default=None,
                          help="Pin to specific GPU index(es), comma-separated "
                               "(e.g. --devices 1,3); count is taken from the list. "
                               "Each must be free or already yours. Rejected if any "
                               "is held by someone else, UNLESS --queue is given, in "
                               "which case it waits until all of them are available.")
    p_submit.add_argument("-m", "--memory", type=int, default=None,
                          help="Minimum free VRAM (GB) a GPU must have to be chosen "
                               "(default: config max_memory_per_gpu_gb).")
    p_submit.add_argument("-t", "--time", type=float, default=None,
                          help="Max runtime in hours; the job is killed if it runs "
                               "longer (default: config max_job_time_hours).")
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

    p_kill = sub.add_parser("kill", help="Stop one of your own running jobs.")
    p_kill.add_argument("job_id", type=int, nargs="?",
                        help="ID of your job to stop (see `gpuq status`).")
    p_kill.add_argument("--job-id", dest="job_id_flag", type=int,
                        help="Job ID as a flag, instead of the positional argument.")

    p_config = sub.add_parser(
        "config",
        help="Write a starter config file, or show active settings (--show).")
    p_config.add_argument("--show", action="store_true",
                          help="Print the loaded config, paths, and host/user.")
    p_config.add_argument("--force", action="store_true",
                          help="Overwrite an existing config file.")

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

    if args.action == "kill":
        if args.job_id is None and args.job_id_flag is not None:
            args.job_id = args.job_id_flag
        if args.job_id is None:
            die("kill: provide a job ID, e.g. `gpuq kill 12345`.")

    config = load_config()

    handlers = {
        "submit": cmd_submit,
        "status": cmd_status,
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
