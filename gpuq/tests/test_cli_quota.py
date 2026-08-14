"""CLI-level quota tests: ledger append on completion, deprioritization path."""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
USERSPACE_PY = HERE.parent / "userspace.py"


def _spawn(args, env):
    return subprocess.Popen(
        [sys.executable, str(USERSPACE_PY)] + args,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _run(args, env):
    return subprocess.run(
        [sys.executable, str(USERSPACE_PY)] + args,
        env=env, capture_output=True, text=True,
    )


def test_ledger_appended_after_submit(gpuq_env, tmp_queue_dir):
    r = _run(["submit", "-g", "1", "-t", "1", "--", "/bin/true"], gpuq_env)
    assert r.returncode == 0, r.stderr
    usage_path = tmp_queue_dir / "usage.jsonl"
    assert usage_path.exists()
    lines = [json.loads(l) for l in usage_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["user"] == os.environ.get("USER") or rec["user"]
    assert len(rec["gpus"]) == 1 and rec["gpus"][0] in (0, 1)  # random among free
    assert rec["gpu_hours"] >= 0
    assert "started_at" in rec and "ended_at" in rec


def test_over_quota_submit_is_deprioritized_and_queued(gpuq_env, tmp_queue_dir, tmp_path):
    # Pre-load the ledger so the user is already over budget.
    user = os.environ.get("USER") or "unknown"
    now = datetime.now()
    recent = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    (tmp_queue_dir / "usage.jsonl").write_text(
        json.dumps({"user": user, "gpu_hours": 999, "ended_at": recent}) + "\n"
    )
    # Lower the user's budget so they're over.
    cfg = Path(gpuq_env["GPUQ_CONFIG_FILE"])
    cfg.write_text(json.dumps({
        "max_job_time_hours": 24,
        "max_memory_per_gpu_gb": 70,
        "notification_email": {"enabled": False},
        "slack": {"enabled": False},
        "quotas": {"users": {user: 100}},
    }))

    # Without --queue, an over-quota submit is force-queued (not rejected),
    # so it will sit in the queue. Spawn, wait for queued.json, then SIGINT.
    proc = _spawn(["submit", "-g", "1", "-t", "1", "--", "/bin/true"], gpuq_env)
    deadline = time.time() + 10
    queued = []
    while time.time() < deadline:
        time.sleep(0.3)
        path = tmp_queue_dir / "jobs.json"
        if path.exists():
            try:
                queued = json.loads(path.read_text())
            except json.JSONDecodeError:
                queued = []
            if queued:
                break
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=5)
    err = proc.stderr.read()
    assert queued, f"expected over-quota submitter to be queued; stderr={err!r}"
    assert queued[0].get("priority") == "low"
    assert "DEPRIORITIZED" in err or "deprioritized" in err
    assert "over quota" in err


def _make_over_quota(gpuq_env, tmp_queue_dir, delay_hours):
    """Preload ledger + config so the current user is over budget, with the
    given quotas.delay_hours. Returns the user name."""
    user = os.environ.get("USER") or "unknown"
    now = datetime.now()
    recent = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    (tmp_queue_dir / "usage.jsonl").write_text(
        json.dumps({"user": user, "gpu_hours": 999, "ended_at": recent}) + "\n"
    )
    cfg = Path(gpuq_env["GPUQ_CONFIG_FILE"])
    cfg.write_text(json.dumps({
        "max_job_time_hours": 24,
        "max_memory_per_gpu_gb": 70,
        "notification_email": {"enabled": False},
        "slack": {"enabled": False},
        "quotas": {"users": {user: 100}, "delay_hours": delay_hours},
    }))
    return user


def test_over_quota_hold_defers_start(gpuq_env, tmp_queue_dir):
    """With quotas.delay_hours set, an over-quota submit is HELD: it must not
    claim a slot before submit-time + delay, even with every GPU free."""
    _make_over_quota(gpuq_env, tmp_queue_dir, delay_hours=1)
    env = dict(gpuq_env)
    env["GPUQ_DEPRIORITIZED_POLL_SEC"] = "1"
    t0 = datetime.now()
    proc = _spawn(["submit", "-g", "1", "-t", "1", "--", "/bin/sleep", "30"], env)
    deadline = time.time() + 10
    queued = []
    while time.time() < deadline:
        time.sleep(0.3)
        path = tmp_queue_dir / "jobs.json"
        if path.exists():
            try:
                queued = json.loads(path.read_text())
            except json.JSONDecodeError:
                queued = []
            if queued:
                break
    assert queued, "over-quota submitter never queued"
    assert queued[0].get("priority") == "low"
    hold = datetime.fromisoformat(queued[0]["hold_until"])
    assert timedelta(minutes=50) < (hold - t0) < timedelta(minutes=70)

    # Several polls elapse; without the hold the free GPU would be claimed.
    time.sleep(3.5)
    running_path = tmp_queue_dir / "running.json"
    running = (json.loads(running_path.read_text() or "[]")
               if running_path.exists() else [])
    assert running == [], "held job must not start before its hold expires"
    assert json.loads((tmp_queue_dir / "jobs.json").read_text()), \
        "held job must stay queued"

    r = _run(["status"], env)
    assert "held until" in r.stdout

    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=5)
    err = proc.stderr.read()
    assert "held until" in err


def test_over_quota_hold_expires_then_runs(gpuq_env, tmp_queue_dir):
    """Once the hold expires the job proceeds through the normal deprioritized
    path and runs."""
    _make_over_quota(gpuq_env, tmp_queue_dir, delay_hours=1.0 / 3600)  # 1 second
    env = dict(gpuq_env)
    env["GPUQ_DEPRIORITIZED_POLL_SEC"] = "1"
    proc = _spawn(["submit", "-g", "1", "-t", "1", "--", "/bin/true"], env)
    proc.wait(timeout=30)
    assert proc.returncode == 0, proc.stderr.read()
    usage = [json.loads(l)
             for l in (tmp_queue_dir / "usage.jsonl").read_text().splitlines()
             if l.strip()]
    end = [r for r in usage if r.get("event") == "end"]
    assert len(end) == 1
    assert end[0]["priority"] == "low"


def test_under_quota_runs_immediately(gpuq_env, tmp_queue_dir):
    user = os.environ.get("USER") or "unknown"
    cfg = Path(gpuq_env["GPUQ_CONFIG_FILE"])
    cfg.write_text(json.dumps({
        "max_job_time_hours": 24,
        "max_memory_per_gpu_gb": 70,
        "notification_email": {"enabled": False},
        "slack": {"enabled": False},
        "quotas": {"users": {user: 1000}},
    }))
    r = _run(["submit", "-g", "1", "-t", "1", "--", "/bin/true"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "DEPRIORITIZED" not in r.stderr
