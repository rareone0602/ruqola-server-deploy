"""Submit-argument validation and queued-job kill tests."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
USERSPACE_PY = HERE.parent / "userspace.py"


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


def test_time_zero_rejected(gpuq_env):
    r = run(["submit", "-t", "0", "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    assert "must be > 0" in r.stderr


def test_negative_time_rejected(gpuq_env):
    r = run(["submit", "-t", "-5", "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    assert "must be > 0" in r.stderr


def test_gpus_beyond_host_rejected(gpuq_env):
    # The fake host has 2 GPUs.
    r = run(["submit", "-g", "5", "--queue", "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    assert "this host has 2 GPU(s)" in r.stderr


def test_gpus_zero_rejected(gpuq_env):
    r = run(["submit", "-g", "0", "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    assert "at least 1" in r.stderr


def test_devices_out_of_range_rejected_even_with_queue(gpuq_env):
    r = run(["submit", "--devices", "7", "--queue", "--", "/bin/true"],
            gpuq_env)
    assert r.returncode != 0
    assert "no such GPU" in r.stderr


def test_command_and_dashdash_conflict(gpuq_env):
    r = run(["submit", "--command", "/bin/true", "--", "/bin/false"], gpuq_env)
    assert r.returncode != 0
    assert "not both" in r.stderr


def test_gpus_devices_mismatch_rejected(gpuq_env):
    r = run(["submit", "-g", "1", "--devices", "0,1", "--", "/bin/true"],
            gpuq_env)
    assert r.returncode != 0
    assert "implies -g 2" in r.stderr


def test_gpus_devices_match_accepted(gpuq_env):
    r = run(["submit", "-g", "2", "--devices", "0,1", "-t", "1", "--",
             "/bin/true"], gpuq_env)
    assert r.returncode == 0, r.stderr


def test_kill_cancels_queued_job_from_another_process(gpuq_env, tmp_queue_dir,
                                                      fake_gpus):
    state_path, write = fake_gpus
    write({"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 80000, "memory_total_mb": 81920, "utilization": 0},
    ], "compute_apps": []})
    proc = _spawn(["submit", "--queue", "-g", "1", "--", "/bin/true"], gpuq_env)
    queued = []
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.2)
        path = tmp_queue_dir / "jobs.json"
        if path.exists():
            queued = json.loads(path.read_text() or "[]")
            if queued:
                break
    assert queued, "waiter never queued"
    job_id = queued[0]["id"]

    r = run(["kill", str(job_id)], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "cancelled queued job" in r.stdout
    proc.wait(timeout=5)
    assert json.loads((tmp_queue_dir / "jobs.json").read_text()) == []


def test_kill_mine_with_nothing_running(gpuq_env):
    r = run(["kill", "--mine"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "no running or queued jobs" in r.stdout


def test_queued_id_preserved_on_claim(gpuq_env, tmp_queue_dir, fake_gpus):
    state_path, write = fake_gpus
    busy = {"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 80000, "memory_total_mb": 81920, "utilization": 0},
    ], "compute_apps": []}
    free = {"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 0, "memory_total_mb": 81920, "utilization": 0},
    ], "compute_apps": []}
    write(busy)
    env = dict(gpuq_env)
    env["GPUQ_DEPRIORITIZED_POLL_SEC"] = "1"
    proc = _spawn(["submit", "--queue", "-g", "1", "-t", "1", "--",
                   "/bin/true"], env)
    queued = []
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.2)
        path = tmp_queue_dir / "jobs.json"
        if path.exists():
            queued = json.loads(path.read_text() or "[]")
            if queued:
                break
    assert queued, "waiter never queued"
    qid = queued[0]["id"]
    write(free)   # free the GPU; the waiter polls every 30s — wait for claim
    proc.wait(timeout=45)
    usage = [json.loads(l)
             for l in (tmp_queue_dir / "usage.jsonl").read_text().splitlines()
             if l.strip()]
    end = [r for r in usage if r.get("event") == "end"]
    assert len(end) == 1
    assert end[0]["id"] == qid, "job must keep the id announced while queued"
    assert end[0]["queue_wait_sec"] >= 0


def test_config_force_without_init_fails_loudly(gpuq_env):
    r = run(["config", "--force"], gpuq_env)
    assert r.returncode != 0
    assert "config init --force" in r.stderr


def test_kill_mine_clears_running_and_queued(gpuq_env, tmp_queue_dir,
                                             fake_gpus):
    state_path, write = fake_gpus
    write({"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 0, "memory_total_mb": 81920, "utilization": 0},
    ], "compute_apps": []})
    runner = _spawn(["submit", "-g", "1", "-t", "1", "--", "/bin/sleep", "30"],
                    gpuq_env)
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.2)
        path = tmp_queue_dir / "running.json"
        if path.exists() and json.loads(path.read_text() or "[]"):
            break
    # Occupy the only GPU in fake state so the second submit queues.
    write({"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 80000, "memory_total_mb": 81920, "utilization": 99},
    ], "compute_apps": []})
    waiter = _spawn(["submit", "--queue", "-g", "1", "-t", "1", "--",
                     "/bin/true"], gpuq_env)
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.2)
        path = tmp_queue_dir / "jobs.json"
        if path.exists() and json.loads(path.read_text() or "[]"):
            break
    r = run(["kill", "--mine"], gpuq_env)
    assert r.returncode == 0, r.stderr + r.stdout
    runner.wait(timeout=15)
    waiter.wait(timeout=15)
    assert json.loads((tmp_queue_dir / "jobs.json").read_text()) == []
    assert json.loads((tmp_queue_dir / "running.json").read_text()) == []
