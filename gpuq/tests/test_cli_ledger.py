"""Job-ledger tests: v2 end records, synthetic lost records, cancelled and
rejected events, and the history/quota commands that read them back."""
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

USER = os.environ.get("USER") or "unknown"


def run(args, env, **kw):
    return subprocess.run(
        [sys.executable, str(USERSPACE_PY)] + args,
        env=env, capture_output=True, text=True, **kw
    )


def _spawn(args, env):
    return subprocess.Popen(
        [sys.executable, str(USERSPACE_PY)] + args,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _ledger(tmp_queue_dir):
    path = tmp_queue_dir / "usage.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --- v2 end records ----------------------------------------------------------

def test_end_record_carries_job_context(gpuq_env, tmp_queue_dir):
    r = run(["submit", "-g", "1", "-t", "2", "--name", "ledger-test", "--",
             "/bin/true"], gpuq_env)
    assert r.returncode == 0, r.stderr
    recs = _ledger(tmp_queue_dir)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["v"] == 2 and rec["event"] == "end"
    assert rec["command"] == "/bin/true"
    assert rec["name"] == "ledger-test"
    assert rec["gpus_requested"] == 1 and len(rec["gpus"]) == 1
    assert rec["devices"] is None
    assert rec["max_time_hours"] == 2
    assert rec["exit_code"] == 0
    assert rec["end_reason"] == "completed"
    assert rec["queue_wait_sec"] is not None and rec["queue_wait_sec"] >= 0
    assert rec["submitted_at"] and rec["started_at"] and rec["ended_at"]
    assert rec["over_quota_at_submit"] is False


def test_end_record_failed_and_killed_reasons(gpuq_env, tmp_queue_dir):
    r = run(["submit", "-g", "1", "-t", "1", "--", "/bin/false"], gpuq_env)
    assert r.returncode == 1
    rec = _ledger(tmp_queue_dir)[-1]
    assert rec["end_reason"] == "failed" and rec["exit_code"] == 1

    # A SIGTERM to the supervisor forwards to the child; the child dies by
    # signal -> reason "killed", and gpuq exits 128+15 by shell convention.
    proc = _spawn(["submit", "-g", "1", "-t", "1", "--", "/bin/sleep", "30"],
                  gpuq_env)
    time.sleep(2.0)  # let it claim and start
    proc.send_signal(signal.SIGTERM)
    rc = proc.wait(timeout=10)
    assert rc == 143, f"expected shell-convention 143, got {rc}"
    rec = _ledger(tmp_queue_dir)[-1]
    assert rec["end_reason"] == "killed"
    assert rec["exit_code"] == -15


def test_devices_pin_recorded_in_ledger(gpuq_env, tmp_queue_dir):
    r = run(["submit", "--devices", "1", "-t", "1", "--", "/bin/true"],
            gpuq_env)
    assert r.returncode == 0, r.stderr
    rec = _ledger(tmp_queue_dir)[-1]
    assert rec["devices"] == [1]
    assert rec["gpus"] == [1]


# --- synthetic lost records --------------------------------------------------

def test_reap_writes_synthetic_lost_record(gpuq_env, tmp_queue_dir,
                                           userspace_module):
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    started = (datetime.now() - timedelta(hours=3)).isoformat(timespec="seconds")
    (tmp_queue_dir / "running.json").write_text(json.dumps([{
        "id": 424242, "user": USER, "host": userspace_module.HOST,
        "pid": pid, "gpus": [0], "gpu_count": 1, "started_at": started,
        "command": "python crashed.py", "max_time_hours": 2,
    }]))
    r = run(["status"], gpuq_env)   # any command that reaps
    assert r.returncode == 0, r.stderr
    recs = [x for x in _ledger(tmp_queue_dir) if x.get("id") == 424242]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["end_reason"] == "lost" and rec["synthetic"] is True
    assert rec["exit_code"] is None
    # Ran 3h but capped at its 2h limit.
    assert 1.9 <= rec["gpu_hours"] <= 2.1
    assert json.loads((tmp_queue_dir / "running.json").read_text()) == []


def test_dead_queued_waiter_reaped_and_logged(gpuq_env, tmp_queue_dir,
                                              userspace_module):
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    (tmp_queue_dir / "jobs.json").write_text(json.dumps([{
        "id": 555, "user": USER, "host": userspace_module.HOST, "pid": pid,
        "priority": "normal", "gpu_count": 1,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
    }]))
    r = run(["status"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "Queued Jobs (0)" in r.stdout
    assert json.loads((tmp_queue_dir / "jobs.json").read_text()) == []
    recs = [x for x in _ledger(tmp_queue_dir) if x.get("id") == 555]
    assert len(recs) == 1
    assert recs[0]["event"] == "cancelled" and recs[0]["reason"] == "lost"


# --- cancelled / rejected events ----------------------------------------------

def test_rejected_submit_logged(gpuq_env, tmp_queue_dir, fake_gpus):
    state_path, write = fake_gpus
    write({"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 80000, "memory_total_mb": 81920, "utilization": 0},
    ], "compute_apps": []})
    r = run(["submit", "-g", "1", "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    recs = _ledger(tmp_queue_dir)
    assert len(recs) == 1
    assert recs[0]["event"] == "rejected"
    assert recs[0]["reason"] == "no_free_gpu"
    assert recs[0]["user"] == USER


def test_cancelled_while_queued_logged(gpuq_env, tmp_queue_dir, fake_gpus):
    state_path, write = fake_gpus
    write({"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 80000, "memory_total_mb": 81920, "utilization": 0},
    ], "compute_apps": []})
    proc = _spawn(["submit", "--queue", "-g", "1", "--", "/bin/true"], gpuq_env)
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.2)
        path = tmp_queue_dir / "jobs.json"
        if path.exists() and json.loads(path.read_text() or "[]"):
            break
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=5)
    recs = [x for x in _ledger(tmp_queue_dir) if x.get("event") == "cancelled"]
    assert len(recs) == 1
    assert recs[0]["reason"] == "user"
    assert json.loads((tmp_queue_dir / "jobs.json").read_text()) == []


def test_sigterm_while_queued_cleans_up_and_logs(gpuq_env, tmp_queue_dir,
                                                 fake_gpus):
    state_path, write = fake_gpus
    write({"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 80000, "memory_total_mb": 81920, "utilization": 0},
    ], "compute_apps": []})
    proc = _spawn(["submit", "--queue", "-g", "1", "--", "/bin/true"], gpuq_env)
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.2)
        path = tmp_queue_dir / "jobs.json"
        if path.exists() and json.loads(path.read_text() or "[]"):
            break
    proc.send_signal(signal.SIGTERM)
    rc = proc.wait(timeout=5)
    assert rc == 128 + signal.SIGTERM
    assert json.loads((tmp_queue_dir / "jobs.json").read_text()) == []
    recs = [x for x in _ledger(tmp_queue_dir) if x.get("event") == "cancelled"]
    assert len(recs) == 1 and recs[0]["reason"] == "signal"


# --- history ------------------------------------------------------------------

def test_history_lists_own_jobs(gpuq_env, tmp_queue_dir):
    r = run(["submit", "-g", "1", "-t", "1", "--name", "histjob", "--",
             "/bin/true"], gpuq_env)
    assert r.returncode == 0, r.stderr
    r = run(["history"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "histjob" in r.stdout
    assert "completed" in r.stdout


def test_history_json_and_events(gpuq_env, tmp_queue_dir, fake_gpus):
    run(["submit", "-g", "1", "-t", "1", "--", "/bin/true"], gpuq_env)
    state_path, write = fake_gpus
    write({"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 80000, "memory_total_mb": 81920, "utilization": 0},
    ], "compute_apps": []})
    run(["submit", "-g", "1", "--", "/bin/true"], gpuq_env)  # rejected
    r = run(["history", "--events", "--json"], gpuq_env)
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    events = {x["event"] for x in recs}
    assert events == {"end", "rejected"}
    # Without --events, only end records appear.
    r = run(["history", "--json"], gpuq_env)
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    assert {x["event"] for x in recs} == {"end"}


# --- quota --------------------------------------------------------------------

def test_quota_data_gathering_mode(gpuq_env, tmp_queue_dir):
    run(["submit", "-g", "1", "-t", "1", "--", "/bin/true"], gpuq_env)
    r = run(["quota"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "unlimited" in r.stdout
    assert "not enforced" in r.stdout


def test_quota_with_budget_and_over(gpuq_env, tmp_queue_dir, tmp_path):
    now = datetime.now()
    recent = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    (tmp_queue_dir / "usage.jsonl").write_text(
        json.dumps({"user": USER, "gpu_hours": 150, "ended_at": recent}) + "\n"
    )
    cfg = Path(gpuq_env["GPUQ_CONFIG_FILE"])
    cfg.write_text(json.dumps({"quotas": {"users": {USER: 100}}}))
    r = run(["quota"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "OVER BUDGET" in r.stdout
    cfg.write_text(json.dumps({"quotas": {"users": {USER: 1000}}}))
    r = run(["quota"], gpuq_env)
    assert "OK" in r.stdout and "headroom" in r.stdout


def test_quota_all_table(gpuq_env, tmp_queue_dir):
    now = datetime.now()
    recent = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    lines = [
        json.dumps({"user": "alice", "gpu_hours": 12.5, "ended_at": recent}),
        json.dumps({"user": "bob", "gpu_hours": 3.25, "ended_at": recent}),
    ]
    (tmp_queue_dir / "usage.jsonl").write_text("\n".join(lines) + "\n")
    r = run(["quota", "--all"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "alice" in r.stdout and "bob" in r.stdout
    assert "12.5" in r.stdout
    assert "NOT enforced" in r.stdout


def test_quota_report_runs(gpuq_env, tmp_queue_dir):
    run(["submit", "-g", "1", "-t", "1", "--", "/bin/true"], gpuq_env)
    r = run(["quota", "--report"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "P50" in r.stdout and "P95" in r.stdout
    assert USER in r.stdout


# --- legacy compatibility -----------------------------------------------------

def test_legacy_v1_records_still_count(gpuq_env, tmp_queue_dir,
                                       userspace_module, monkeypatch):
    now = datetime.now()
    recent = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    legacy = {"id": 1, "user": "carol", "host": userspace_module.HOST,
              "gpus": [0], "started_at": (now - timedelta(hours=2)).isoformat(timespec="seconds"),
              "ended_at": recent, "elapsed_hours": 1.0, "gpu_hours": 1.0,
              "priority": "normal"}
    (tmp_queue_dir / "usage.jsonl").write_text(json.dumps(legacy) + "\n")
    assert userspace_module.usage_in_window("carol") == 1.0
    r = run(["history", "--user", "carol"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "1:00:00" in r.stdout      # legacy elapsed_hours renders
    assert "USER" not in r.stdout     # no USER column without --all
    r = run(["history", "--all"], gpuq_env)
    assert "USER" in r.stdout and "carol" in r.stdout


def test_history_and_report_survive_wrong_typed_fields(gpuq_env,
                                                       tmp_queue_dir):
    now = datetime.now()
    recent = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    bad = {"user": USER, "gpu_hours": "junk", "queue_wait_sec": "junk",
           "ended_at": recent, "elapsed_hours": "junk", "max_time_hours": "x"}
    good = {"user": USER, "gpu_hours": 1.5, "ended_at": recent}
    (tmp_queue_dir / "usage.jsonl").write_text(
        json.dumps(bad) + "\n" + json.dumps(good) + "\n")
    r = run(["history"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "1.50" in r.stdout
    r = run(["quota", "--report"], gpuq_env)
    assert r.returncode == 0, r.stderr
