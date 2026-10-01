"""The previous gpuq (before the 2026-10-01 cutover), read-only: its state files,
and its rule for which job owns a GPU process. Shadow mode used it; since the
cutover, transition.py and holders.py use it for that gpuq's last jobs.

Nothing here opens a file for writing. The service unit also mounts the whole
filesystem read-only (ProtectSystem=strict), so a bug here cannot change that
gpuq's state either.
"""
import json
from datetime import datetime
from pathlib import Path

from scheduler.model import MAX_RUNTIME_H, Job, Running

from .procinfo import read_proc
from .safeio import read_text

# How far up the parent chain to look for a job's child (the previous gpuq's bound).
ANCESTRY_DEPTH = 64


def _read_list(path):
    # The folder is writable by every member: no symlinks, no FIFOs (safeio).
    try:
        data = json.loads(read_text(path))
    except FileNotFoundError:
        return []
    return data if isinstance(data, list) else None


def read_state(queue_dir, host):
    """(running entries, queued entries) for this host, or None if a file is
    torn or unreadable. The previous gpuq's writers replaced files atomically,
    so a torn read means something else is wrong: skip the pass rather than guess."""
    # jobs.json first: the previous gpuq saved running.json first when a job
    # started, so this order sees a starting job in both files, never in neither.
    try:
        queued = _read_list(Path(queue_dir) / "jobs.json")
        running = _read_list(Path(queue_dir) / "running.json")
    except (OSError, json.JSONDecodeError):
        return None
    if running is None or queued is None:
        return None
    mine = lambda e: isinstance(e, dict) and e.get("host", host) == host
    return [e for e in running if mine(e)], [e for e in queued if mine(e)]


def _epoch(iso):
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return None


def _job(e, gpus, whole_card_gb):
    submit = _epoch(e.get("submitted_at")) or _epoch(e.get("started_at"))
    if submit is None:
        return None
    mem = e.get("memory_gb")
    # v3 has no -t: every job may run MAX_RUNTIME_H, whatever its entry says.
    return Job(id=str(e.get("id")), user=str(e.get("user")), gpus=gpus,
               mem_gb=whole_card_gb if mem is None else float(mem),
               limit_h=MAX_RUNTIME_H, submit_t=submit)


def as_running(e, whole_card_gb):
    """A running.json entry as a Running, or None if its times are unreadable."""
    start = _epoch(e.get("started_at"))
    cards = tuple(int(c) for c in e.get("gpus") or ())
    job = _job(e, len(cards), whole_card_gb)
    if start is None or job is None or not cards:
        return None
    return Running(job=job, cards=cards, start_t=start)


def as_job(e, whole_card_gb):
    """A jobs.json entry as a Job, or None if its times are unreadable.
    The previous gpuq's --devices pin is read as a request for that many
    cards: v3's --devices only adds a job to cards its user already holds
    (§4.5), and shadow mode does not model it."""
    gpus = e.get("gpu_count") or len(e.get("devices") or ()) or 1
    return _job(e, int(gpus), whole_card_gb)


def owner_of(info, entries, proc="/proc"):
    """The running entry whose job `info` is part of, or None.

    The previous gpuq's audit order, most specific first: the job's systemd
    scope, then its child's process group, then the nearest ancestor that is a job's child (a
    worker that started its own session)."""
    parts = set(info.cgroup.split("/"))
    for e in entries:
        if e.get("cgroup_scope") and e["cgroup_scope"] in parts:
            return e
    for e in entries:
        if e.get("child_pgid") is not None and int(e["child_pgid"]) == info.pgid:
            return e
    children = {int(e["child_pid"]): e for e in entries if e.get("child_pid") is not None}
    cur = info
    for _ in range(ANCESTRY_DEPTH):
        if cur.pid in children:
            return children[cur.pid]
        if cur.ppid <= 1:
            return None
        cur = read_proc(cur.ppid, proc)
        if cur is None:
            return None
    return None
