"""Submit-argument validation and queued-job kill tests."""
import json
import signal
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


def test_time_over_cap_rejected(gpuq_env):
    r = run(["submit", "-t", "200", "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    assert "exceeds" in r.stderr and "cap" in r.stderr


def test_time_at_cap_accepted(gpuq_env):
    r = run(["submit", "-t", "48", "--", "/bin/true"], gpuq_env)
    assert r.returncode == 0, r.stderr


def test_time_over_default_cap_rejected_at_48(gpuq_env):
    """The built-in wall-time ceiling is 48h (2 days); a config without the cap
    key must reject -t above it."""
    r = run(["submit", "-t", "60", "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    assert "48" in r.stderr and "cap" in r.stderr


def test_single_job_over_user_card_cap_rejected(gpuq_env, tmp_path):
    """A single submit asking for more GPUs than the per-user hard cap can
    never run; it is refused up front, --queue or not."""
    cfg = Path(gpuq_env["GPUQ_CONFIG_FILE"])
    cfg.write_text(json.dumps({
        "max_job_time_hours": 24,
        "max_memory_per_gpu_gb": 70,
        "max_gpus_per_user_hard": 1,
        "notification_email": {"enabled": False},
        "slack": {"enabled": False},
    }))
    for extra in ([], ["--queue"]):
        r = run(["submit", "-g", "2"] + extra + ["--", "/bin/true"], gpuq_env)
        assert r.returncode != 0
        assert "per-user" in r.stderr and "1" in r.stderr


def test_card_cap_stacks_second_job_on_same_card(gpuq_env, tmp_queue_dir):
    """With the per-user cap at 1 and two free GPUs, a user's second concurrent
    submit must stack onto the card they already hold, not claim a new one."""
    cfg = Path(gpuq_env["GPUQ_CONFIG_FILE"])
    cfg.write_text(json.dumps({
        "max_job_time_hours": 24,
        "max_memory_per_gpu_gb": 70,
        "max_gpus_per_user_hard": 1,
        "notification_email": {"enabled": False},
        "slack": {"enabled": False},
    }))
    runner = _spawn(["submit", "-g", "1", "-t", "1", "--", "/bin/sleep", "30"],
                    gpuq_env)
    first_gpus = None
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.2)
        path = tmp_queue_dir / "running.json"
        if path.exists():
            entries = json.loads(path.read_text() or "[]")
            if entries:
                first_gpus = entries[0]["gpus"]
                break
    assert first_gpus is not None, "first job never started"

    r = run(["submit", "-g", "1", "-t", "1", "--", "/bin/true"], gpuq_env)
    assert r.returncode == 0, r.stderr
    usage = [json.loads(l)
             for l in (tmp_queue_dir / "usage.jsonl").read_text().splitlines()
             if l.strip()]
    end = [rec for rec in usage if rec.get("event") == "end"]
    assert len(end) == 1
    assert end[0]["gpus"] == first_gpus, "second job must stack, not spread"

    run(["kill", "--mine"], gpuq_env)
    runner.wait(timeout=15)


def _cap1_config(gpuq_env):
    Path(gpuq_env["GPUQ_CONFIG_FILE"]).write_text(json.dumps({
        "max_job_time_hours": 24,
        "max_memory_per_gpu_gb": 70,
        "max_gpus_per_user_hard": 1,
        "notification_email": {"enabled": False},
        "slack": {"enabled": False},
    }))


def _start_runner_on_one_card(gpuq_env, tmp_queue_dir):
    """Spawn a sleep job, wait until it holds a card, return (proc, card)."""
    runner = _spawn(["submit", "-g", "1", "-t", "1", "--", "/bin/sleep", "30"],
                    gpuq_env)
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.2)
        path = tmp_queue_dir / "running.json"
        if path.exists():
            entries = json.loads(path.read_text() or "[]")
            if entries:
                return runner, entries[0]["gpus"][0]
    raise AssertionError("runner never started")


def test_cap_blocked_pin_rejection_names_the_cap(gpuq_env, tmp_queue_dir):
    """A pin refused purely by the per-user card cap must say so (not print a
    blank reason list) and be ledgered as user_card_cap, not
    devices_unavailable."""
    _cap1_config(gpuq_env)
    runner, card = _start_runner_on_one_card(gpuq_env, tmp_queue_dir)
    other = 1 - card
    r = run(["submit", "--devices", str(other), "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    assert "card cap" in r.stderr
    usage = [json.loads(l)
             for l in (tmp_queue_dir / "usage.jsonl").read_text().splitlines()
             if l.strip()]
    rejected = [rec for rec in usage if rec.get("event") == "rejected"]
    assert rejected and rejected[-1]["reason"] == "user_card_cap"
    run(["kill", "--mine"], gpuq_env)
    runner.wait(timeout=15)


def test_cap_blocked_unpinned_rejection_names_the_cap(gpuq_env, tmp_queue_dir,
                                                      fake_gpus):
    """An unpinned no-queue submit blocked by the cap (own card too full to
    stack, free card off-limits) must name the cap, not claim no VRAM exists."""
    _cap1_config(gpuq_env)
    state_path, write = fake_gpus
    runner, card = _start_runner_on_one_card(gpuq_env, tmp_queue_dir)
    # The held card fills up (no stacking headroom); the other stays free.
    gpus_state = []
    for i in range(2):
        full = (i == card)
        gpus_state.append({
            "index": i, "name": "X", "uuid": f"GPU-{i}",
            "memory_used_mb": 81000 if full else 0,
            "memory_total_mb": 81920,
            "utilization": 99 if full else 0,
        })
    write({"gpus": gpus_state, "compute_apps": []})
    r = run(["submit", "-g", "1", "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    assert "card cap" in r.stderr
    usage = [json.loads(l)
             for l in (tmp_queue_dir / "usage.jsonl").read_text().splitlines()
             if l.strip()]
    rejected = [rec for rec in usage if rec.get("event") == "rejected"]
    assert rejected and rejected[-1]["reason"] == "user_card_cap"
    run(["kill", "--mine"], gpuq_env)
    runner.wait(timeout=15)


def test_card_cap_enforced_on_queue_waiter_path(gpuq_env, tmp_queue_dir,
                                                fake_gpus):
    """The cap must bind in _wait_for_slot too (the ONLY enforcement point for
    --queue and over-quota submits): an at-cap user whose held card can't take
    a stack must QUEUE, not claim the free card."""
    _cap1_config(gpuq_env)
    state_path, write = fake_gpus
    runner, card = _start_runner_on_one_card(gpuq_env, tmp_queue_dir)
    gpus_state = []
    for i in range(2):
        full = (i == card)
        gpus_state.append({
            "index": i, "name": "X", "uuid": f"GPU-{i}",
            "memory_used_mb": 81000 if full else 0,
            "memory_total_mb": 81920,
            "utilization": 99 if full else 0,
        })
    write({"gpus": gpus_state, "compute_apps": []})
    waiter = _spawn(["submit", "--queue", "-g", "1", "-t", "1", "--",
                     "/bin/true"], gpuq_env)
    queued = []
    deadline = time.time() + 10
    while time.time() < deadline:
        time.sleep(0.2)
        path = tmp_queue_dir / "jobs.json"
        if path.exists():
            queued = json.loads(path.read_text() or "[]")
            if queued:
                break
    assert queued, "at-cap waiter should queue, not claim the free card"
    running = json.loads((tmp_queue_dir / "running.json").read_text() or "[]")
    assert len(running) == 1, "waiter must not have claimed a second card"
    waiter.send_signal(signal.SIGINT)
    waiter.wait(timeout=5)
    run(["kill", "--mine"], gpuq_env)
    runner.wait(timeout=15)


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
