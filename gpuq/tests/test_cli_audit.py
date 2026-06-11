"""Tests for the `gpuq audit` subcommand."""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
USERSPACE_PY = HERE.parent / "userspace.py"

GPUS_2 = [
    {"index": 0, "name": "Fake-GPU-A", "uuid": "GPU-aaaa",
     "memory_used_mb": 0, "memory_total_mb": 81920, "utilization": 0},
    {"index": 1, "name": "Fake-GPU-B", "uuid": "GPU-bbbb",
     "memory_used_mb": 0, "memory_total_mb": 81920, "utilization": 0},
]


def run(args, env):
    return subprocess.run(
        [sys.executable, str(USERSPACE_PY)] + args,
        env=env, capture_output=True, text=True,
    )


def _current_owner():
    """The login name `gpuq` will see for this process (matches get_process_user)."""
    return subprocess.run(["ps", "-o", "user:32=", "-p", str(os.getpid())],
                          capture_output=True, text=True).stdout.strip()


def _write_untracked_cfg(path, **audit_extra):
    audit = {"max_gpus_per_user": 99, "max_total_memory_gb": 9999,
             "notify_untracked": True, "untracked_min_memory_mb": 512}
    audit.update(audit_extra)
    Path(path).write_text(json.dumps({
        "max_job_time_hours": 24, "max_memory_per_gpu_gb": 70,
        "notification_email": {"enabled": False}, "slack": {"enabled": False},
        "audit": audit,
    }))


def _proc_state(pid, used_memory_mb=40000, name="python"):
    return {"gpus": GPUS_2, "compute_apps": [
        {"pid": pid, "process_name": name, "gpu_uuid": "GPU-aaaa",
         "used_memory_mb": used_memory_mb}]}


def test_audit_clean(gpuq_env):
    r = run(["audit"], gpuq_env)
    assert r.returncode == 0
    assert "clean" in r.stdout


def test_audit_flags_gpu_count_hog(gpuq_env, tmp_queue_dir, userspace_module):
    # Simulate alice holding 3 GPUs (limit default 2).
    started = datetime.now().isoformat(timespec="seconds")
    running = [{
        "id": 1, "user": "alice", "host": userspace_module.HOST,
        "pid": os.getpid(), "gpus": [0, 1, 2], "memory_gb": 10,
        "started_at": started, "status": "running",
    }]
    (tmp_queue_dir / "running.json").write_text(json.dumps(running))
    r = run(["audit"], gpuq_env)
    assert r.returncode == 1
    assert "alice" in r.stdout
    assert "holds 3 GPUs" in r.stdout


def test_audit_flags_memory_hog(gpuq_env, tmp_queue_dir, userspace_module):
    # alice holds 2 GPUs * 30 GB = 60 GB > 50 GB limit.
    started = datetime.now().isoformat(timespec="seconds")
    running = [{
        "id": 1, "user": "alice", "host": userspace_module.HOST,
        "pid": os.getpid(), "gpus": [0, 1], "memory_gb": 30,
        "started_at": started, "status": "running",
    }]
    (tmp_queue_dir / "running.json").write_text(json.dumps(running))
    r = run(["audit"], gpuq_env)
    assert r.returncode == 1
    assert "60 GB total" in r.stdout


def test_audit_flags_quota_breach(gpuq_env, tmp_queue_dir):
    cfg = Path(gpuq_env["GPUQ_CONFIG_FILE"])
    cfg.write_text(json.dumps({
        "max_job_time_hours": 24,
        "max_memory_per_gpu_gb": 70,
        "notification_email": {"enabled": False},
        "slack": {"enabled": False},
        "audit": {"max_gpus_per_user": 99, "max_total_memory_gb": 9999},
        "quotas": {"users": {"alice": 50}},
    }))
    recent = (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")
    (tmp_queue_dir / "usage.jsonl").write_text(
        json.dumps({"user": "alice", "gpu_hours": 75, "ended_at": recent}) + "\n"
    )
    r = run(["audit"], gpuq_env)
    assert r.returncode == 1
    assert "alice" in r.stdout
    assert "75.0 GPU-hours" in r.stdout


# --- untracked-GPU detector (end-to-end through the CLI) -------------------

def test_untracked_disabled_by_default(gpuq_env, fake_gpus):
    """A GPU process present but notify_untracked unset -> still 'clean'."""
    _, write = fake_gpus
    write(_proc_state(os.getpid()))
    r = run(["audit"], gpuq_env)
    assert r.returncode == 0
    assert "clean" in r.stdout
    assert "UNTRACKED" not in r.stdout


def test_untracked_flagged_when_enabled(gpuq_env, fake_gpus, userspace_module):
    owner = _current_owner()
    if owner in userspace_module.SYSTEM_GPU_ACCOUNTS:
        pytest.skip(f"test user {owner!r} is in the system allowlist")
    _write_untracked_cfg(gpuq_env["GPUQ_CONFIG_FILE"])
    _, write = fake_gpus
    write(_proc_state(os.getpid()))       # this (alive, $USER-owned) process
    r = run(["audit"], gpuq_env)
    assert r.returncode == 1
    assert "UNTRACKED" in r.stdout
    assert owner in r.stdout              # full login, not 8-char-truncated
    assert "deadline" in r.stdout


def test_untracked_not_flagged_when_tracked(gpuq_env, tmp_queue_dir, fake_gpus,
                                            userspace_module):
    owner = _current_owner()
    if owner in userspace_module.SYSTEM_GPU_ACCOUNTS:
        pytest.skip(f"test user {owner!r} is in the system allowlist")
    _write_untracked_cfg(gpuq_env["GPUQ_CONFIG_FILE"])
    _, write = fake_gpus
    write(_proc_state(os.getpid()))
    # A tracked job whose child process group is this process's group.
    running = [{"id": 1, "user": owner, "host": userspace_module.HOST,
                "gpus": [0], "pid": os.getpid(),
                "child_pgid": os.getpgid(os.getpid()),
                "started_at": "2020-01-01T00:00:00", "status": "running"}]
    (tmp_queue_dir / "running.json").write_text(json.dumps(running))
    r = run(["audit"], gpuq_env)
    assert "UNTRACKED" not in r.stdout
    assert r.returncode == 0


def test_untracked_enforce_kills_past_deadline(gpuq_env, tmp_queue_dir, fake_gpus,
                                               userspace_module):
    owner = _current_owner()
    if owner in userspace_module.SYSTEM_GPU_ACCOUNTS:
        pytest.skip(f"test user {owner!r} is in the system allowlist")
    sleeper = subprocess.Popen(["sleep", "120"], start_new_session=True)
    try:
        pgid = os.getpgid(sleeper.pid)    # == sleeper.pid (setsid); its own session
        _write_untracked_cfg(gpuq_env["GPUQ_CONFIG_FILE"], untracked_grace_hours=24)
        _, write = fake_gpus
        write(_proc_state(sleeper.pid, name="sleep"))
        # Pre-seed state so this offender is already 48h old -> past 24h deadline.
        old = (datetime.now() - timedelta(hours=48)).isoformat(timespec="seconds")
        key = f"{userspace_module.HOST}|{owner}|{pgid}"
        (tmp_queue_dir / "untracked_state.json").write_text(json.dumps({
            key: {"host": userspace_module.HOST, "owner": owner, "pgid": pgid,
                  "first_seen": old, "last_email": old, "reminders": 0}}))
        r = run(["audit", "--enforce"], gpuq_env)
        assert r.returncode == 1
        assert "PAST DEADLINE" in r.stdout
        sleeper.wait(timeout=8)
        assert sleeper.poll() is not None  # the rogue process group was killed
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait()


# --- rebind detector (end-to-end through the CLI) -------------------------

def _write_rebind_cfg(path, **audit_extra):
    audit = {"max_gpus_per_user": 99, "max_total_memory_gb": 9999,
             "notify_rebind": True, "rebind_min_memory_mb": 512}
    audit.update(audit_extra)
    Path(path).write_text(json.dumps({
        "max_job_time_hours": 24, "max_memory_per_gpu_gb": 70,
        "notification_email": {"enabled": False}, "slack": {"enabled": False},
        "audit": audit,
    }))


def _rebind_job(owner, host, gpus):
    """A tracked job for `owner` whose child pgid == this process's group, so the
    fake GPU process (pid os.getpid()) attributes to it."""
    return {"id": 1, "user": owner, "host": host, "gpus": gpus,
            "pid": os.getpid(), "child_pgid": os.getpgid(os.getpid()),
            "started_at": "2020-01-01T00:00:00", "status": "running"}


def test_rebind_disabled_by_default(gpuq_env, tmp_queue_dir, fake_gpus,
                                    userspace_module):
    """A job on the wrong GPU but notify_rebind unset -> still 'clean'."""
    owner = _current_owner()
    _, write = fake_gpus
    write(_proc_state(os.getpid()))            # process is on GPU 0 (GPU-aaaa)
    running = [_rebind_job(owner, userspace_module.HOST, [1])]   # allocated GPU 1
    (tmp_queue_dir / "running.json").write_text(json.dumps(running))
    r = run(["audit"], gpuq_env)
    assert r.returncode == 0
    assert "clean" in r.stdout and "REBIND" not in r.stdout


def test_rebind_flagged_when_enabled(gpuq_env, tmp_queue_dir, fake_gpus,
                                     userspace_module):
    owner = _current_owner()
    _write_rebind_cfg(gpuq_env["GPUQ_CONFIG_FILE"])
    _, write = fake_gpus
    write(_proc_state(os.getpid()))            # process on GPU 0 ...
    running = [_rebind_job(owner, userspace_module.HOST, [1])]   # ... allocated GPU 1
    (tmp_queue_dir / "running.json").write_text(json.dumps(running))
    r = run(["audit"], gpuq_env)
    assert r.returncode == 1
    assert "REBIND" in r.stdout
    assert owner in r.stdout and "deadline" in r.stdout


def test_rebind_not_flagged_when_correctly_placed(gpuq_env, tmp_queue_dir,
                                                  fake_gpus, userspace_module):
    owner = _current_owner()
    _write_rebind_cfg(gpuq_env["GPUQ_CONFIG_FILE"])
    _, write = fake_gpus
    write(_proc_state(os.getpid()))            # process on GPU 0 ...
    running = [_rebind_job(owner, userspace_module.HOST, [0])]   # ... allocated GPU 0
    (tmp_queue_dir / "running.json").write_text(json.dumps(running))
    r = run(["audit"], gpuq_env)
    assert r.returncode == 0
    assert "REBIND" not in r.stdout


def test_rebind_enforce_kills_past_deadline(gpuq_env, tmp_queue_dir, fake_gpus,
                                            userspace_module):
    owner = _current_owner()
    sleeper = subprocess.Popen(["sleep", "120"], start_new_session=True)
    try:
        pgid = os.getpgid(sleeper.pid)         # == sleeper.pid (setsid)
        _write_rebind_cfg(gpuq_env["GPUQ_CONFIG_FILE"], rebind_grace_hours=24)
        _, write = fake_gpus
        write(_proc_state(sleeper.pid, name="sleep"))   # sleeper on GPU 0 ...
        running = [{"id": 1, "user": owner, "host": userspace_module.HOST,
                    "gpus": [1], "pid": sleeper.pid, "child_pgid": pgid,
                    "started_at": "2020-01-01T00:00:00", "status": "running"}]
        (tmp_queue_dir / "running.json").write_text(json.dumps(running))
        old = (datetime.now() - timedelta(hours=48)).isoformat(timespec="seconds")
        key = f"{userspace_module.HOST}|rebind|1|{pgid}"
        (tmp_queue_dir / "rebind_state.json").write_text(json.dumps({
            key: {"host": userspace_module.HOST, "owner": owner, "job_id": 1,
                  "pgid": pgid, "first_seen": old, "last_email": old,
                  "reminders": 0}}))
        r = run(["audit", "--enforce"], gpuq_env)
        assert r.returncode == 1
        assert "PAST DEADLINE" in r.stdout
        sleeper.wait(timeout=8)
        assert sleeper.poll() is not None      # the rebound job's group was killed
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait()
