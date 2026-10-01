"""A job as gpuqd keeps it, and what it turns into: a scheduler Job, the
environment it runs with, and its ledger record.

A job is a plain dict, saved as JSON in gpuqd's state file:

  id, user, uid, home          who asked (uid from the kernel, never from the client)
  argv, cwd, env               what to run, where, with what (the submitter's own)
  umask, limits                copied from the submitting process by the daemon
  gpus, mem_gb                 how many cards; mem_gb None = a card to yourself
  devices                      --devices: cards the user held at submit, which the
                               job joins (docs/v3-design.md §4.5); else None
  mode                         "attached" | "detached" | "shell"
  name, notify, time_h         as given; time_h is only recorded
  submitted                    epoch seconds
  state                        "queued" | "running"
  cards, started, deadline     once running (host card numbers, epoch seconds)
  log                          detached jobs: where the output goes
  stop_reason                  set when gpuqd itself stops the job
  warned                       the 1-hour warning has gone out
"""
import shlex
import signal
from datetime import datetime

from scheduler.model import HOUR, MAX_RUNTIME_H, Job, Running

# The ledger keeps this much of each command line.
COMMAND_LOG_MAX_CHARS = 300

MODES = ("attached", "detached", "shell")

# A job joining your own card (--devices) needs at least this much VRAM measured
# free there, whatever its -m: the previous gpuq's GPU_OWN_MIN_FREE_GB, so a join never lands
# on a full card.
JOIN_MIN_FREE_GB = 2.0


def iso(t):
    return datetime.fromtimestamp(t).isoformat(timespec="seconds")


def command_text(job):
    return shlex.join(job["argv"])


def join_gb(job):
    """The VRAM a job joining your own card needs measured free there."""
    return max(JOIN_MIN_FREE_GB, float(job.get("mem_gb") or 0.0))


def as_job(job, whole_card_gb, limit_h=MAX_RUNTIME_H):
    """The scheduler's view. No -m means the whole card: nothing else of yours
    may share it. A job joining your cards (--devices) is as big as join_gb."""
    join = tuple(job.get("devices") or ())
    if join:
        mem = join_gb(job)
    else:
        mem = whole_card_gb if job.get("mem_gb") is None else float(job["mem_gb"])
    return Job(str(job["id"]), job["user"], int(job["gpus"]), mem, limit_h,
               job["submitted"], join)


def as_running(job, whole_card_gb):
    """A running job holds its cards until its deadline: 48 h after it started,
    or earlier for a job that joined your cards."""
    limit_h = (job["deadline"] - job["started"]) / HOUR
    return Running(as_job(job, whole_card_gb, limit_h), tuple(job["cards"]), job["started"])


def job_env(job):
    """The submitter's environment, plus what tells the job where it runs.

    A confined job sees only its own cards, numbered from 0, so
    CUDA_VISIBLE_DEVICES is 0..n-1; the host's numbers are in GPUQ_GPUS."""
    env = dict(job["env"])
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(job["cards"])))
    env["GPUQ_GPUS"] = ",".join(str(c) for c in job["cards"])
    env["GPUQ_JOB_ID"] = str(job["id"])
    env["GPUQ_DEADLINE"] = str(int(job["deadline"]))
    return env


def exit_of(report):
    """(exit code as a shell reports it, end reason) from the exit hook's report.

    systemd counts a job killed by SIGTERM as a clean stop; the shell, and
    the previous gpuq, report 143 (128 + 15). So does this."""
    code, status, result = report.get("code"), report.get("status"), report.get("result")
    rc = None
    if code == "exited":
        try:
            rc = int(status)
        except (TypeError, ValueError):
            rc = None
    elif code in ("killed", "dumped"):
        sig = getattr(signal, f"SIG{status}", None)
        rc = 128 + int(sig) if sig is not None else None
    if result == "timeout":
        reason = "timed_out"
    elif rc == 0:
        reason = "completed"
    elif code in ("killed", "dumped"):
        reason = "killed"
    else:
        reason = "failed"
    return rc, reason


def _common(job, host):
    return {
        "v": 2,
        "id": job["id"],
        "user": job["user"],
        "host": host,
        "command": command_text(job)[:COMMAND_LOG_MAX_CHARS],
        "name": job.get("name"),
        "gpus_requested": job["gpus"],
        "devices": job.get("devices"),
        "memory_gb": job.get("mem_gb"),
        "max_time_hours": MAX_RUNTIME_H,
        "priority": "normal",
        "submitted_at": iso(job["submitted"]),
    }


def end_record(job, ended, exit_code, reason, host, synthetic=False):
    """A ledger v2 "end" record, the format the previous gpuq wrote, so history,
    the replay and fair-share read every job alike."""
    started = job["started"]
    elapsed_h = max(0.0, ended - started) / HOUR
    rec = _common(job, host)
    rec.update({
        "event": "end",
        "gpus": list(job["cards"]),
        "over_quota_at_submit": False,
        "started_at": iso(started),
        "ended_at": iso(ended),
        "queue_wait_sec": max(0, int(started - job["submitted"])),
        "max_time_hours": round((job["deadline"] - started) / HOUR, 4),
        "elapsed_hours": round(elapsed_h, 4),
        "gpu_hours": round(elapsed_h * len(job["cards"]), 4),
        "exit_code": exit_code,
        "end_reason": reason,
    })
    if synthetic:
        rec["synthetic"] = True
    return rec


def cancelled_record(job, reason, now, host):
    """A job that left the queue without running. It uses "at", not
    "ended_at", so nothing ever charges it."""
    rec = _common(job, host)
    rec.update({
        "event": "cancelled",
        "at": iso(now),
        "wait_sec": max(0, int(now - job["submitted"])),
        "reason": reason,
    })
    return rec
