"""End-to-end CLI tests for submit/status/kill against the fake GPU."""
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


def test_status_with_two_idle_gpus(gpuq_env):
    r = run(["status"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "Fake-GPU-A" in r.stdout
    assert "Fake-GPU-B" in r.stdout
    assert "Running Jobs (0)" in r.stdout
    assert "Queued Jobs (0)" in r.stdout


def test_submit_runs_and_clears(gpuq_env, tmp_queue_dir):
    r = run(["submit", "-g", "1", "-t", "1", "--", "/bin/true"], gpuq_env)
    assert r.returncode == 0, r.stderr
    running = tmp_queue_dir / "running.json"
    assert running.exists()
    assert json.loads(running.read_text()) == []


def test_submit_fails_when_no_slot(gpuq_env, fake_gpus):
    state_path, write = fake_gpus
    write({"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 80000, "memory_total_mb": 81920, "utilization": 0},
    ], "compute_apps": []})
    r = run(["submit", "-g", "1", "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    assert "no free GPU" in r.stderr


def test_kill_rejects_other_users_job(gpuq_env, tmp_queue_dir, userspace_module):
    foreign = {
        "id": 999, "user": "someone-else", "host": userspace_module.HOST,
        "pid": 1, "gpus": [0], "started_at": "2026-01-01T00:00:00",
        "command": "sleep 9999", "status": "running",
    }
    (tmp_queue_dir / "running.json").write_text(json.dumps([foreign]))
    r = run(["kill", "999"], gpuq_env)
    assert r.returncode != 0
    assert "you can only kill your own jobs" in r.stderr


def test_kill_unknown_job(gpuq_env):
    r = run(["kill", "12345"], gpuq_env)
    assert r.returncode != 0
    assert "no running or queued job" in r.stderr


def _write_four_gpus(write):
    write({
        "gpus": [
            {"index": i, "name": f"Fake-{i}", "uuid": f"GPU-{i}",
             "memory_used_mb": 0, "memory_total_mb": 81920, "utilization": 0}
            for i in range(4)
        ],
        "compute_apps": [],
    })


def test_cuda_visible_devices_is_set_for_child(gpuq_env, tmp_path, fake_gpus):
    # Pin GPUs 0,1 so the child's CUDA_VISIBLE_DEVICES is deterministic (the
    # default picker is now random among free GPUs).
    _, write = fake_gpus
    _write_four_gpus(write)
    out = tmp_path / "cvd.txt"
    # Single-quoted bash -c so $CUDA_VISIBLE_DEVICES is expanded by the child,
    # not by gpuq's argv handling.
    r = run([
        "submit", "--devices", "0,1", "-t", "1", "--",
        "/bin/bash", "-c", f"echo \"$CUDA_VISIBLE_DEVICES\" > {out}",
    ], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert out.read_text().strip() == "0,1"


def test_default_pick_is_random_among_free(gpuq_env, tmp_path, fake_gpus):
    # With 4 idle GPUs and no --devices, the chosen GPU should just be a valid
    # free one (not necessarily 0); over many runs it varies.
    _, write = fake_gpus
    _write_four_gpus(write)
    out = tmp_path / "cvd.txt"
    r = run([
        "submit", "-t", "1", "--",
        "/bin/bash", "-c", f"echo \"$CUDA_VISIBLE_DEVICES\" > {out}",
    ], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert out.read_text().strip() in {"0", "1", "2", "3"}


def test_devices_rejects_occupied_gpu(gpuq_env, tmp_queue_dir, fake_gpus, userspace_module):
    # GPU 0 is held by another gpuq job; pinning it must be rejected (not queued).
    _, write = fake_gpus
    _write_four_gpus(write)
    running = [{"id": 99, "user": "bob", "host": userspace_module.HOST,
                "gpus": [0], "pid": 1, "started_at": "2020-01-01T00:00:00"}]
    # pid 1 (init) is always alive, so reap keeps this entry.
    (tmp_queue_dir / "running.json").write_text(json.dumps(running))
    r = run(["submit", "--devices", "0", "-t", "1", "--", "/bin/true"], gpuq_env)
    assert r.returncode != 0
    blob = r.stdout + r.stderr
    assert "not available" in blob and "GPU 0" in blob
    assert "--queue" in blob                # tells the user how to wait instead


def test_devices_self_owned_succeeds(gpuq_env, tmp_queue_dir, userspace_module):
    # "You own your allocated GPU": stacking a second job onto a card YOU already
    # hold is allowed (where another user's card would be rejected).
    mine = {"id": 42, "user": userspace_module.USER, "host": userspace_module.HOST,
            "gpus": [0], "pid": 1, "started_at": "2020-01-01T00:00:00",
            "command": "sleep 9999", "status": "running"}
    (tmp_queue_dir / "running.json").write_text(json.dumps([mine]))
    r = run(["submit", "--devices", "0", "-t", "1", "--", "/bin/true"], gpuq_env)
    assert r.returncode == 0, r.stdout + r.stderr
    remaining = json.loads((tmp_queue_dir / "running.json").read_text())
    assert [j["id"] for j in remaining] == [42]   # my job stays; /bin/true cleared


def test_default_pick_stacks_on_owned_when_no_free(gpuq_env, tmp_queue_dir,
                                                   fake_gpus, userspace_module):
    # Single GPU, already owned by me: with no free card, the default picker
    # stacks onto my own instead of failing.
    _, write = fake_gpus
    write({"gpus": [
        {"index": 0, "name": "X", "uuid": "GPU-x",
         "memory_used_mb": 0, "memory_total_mb": 81920, "utilization": 0}],
        "compute_apps": []})
    mine = {"id": 42, "user": userspace_module.USER, "host": userspace_module.HOST,
            "gpus": [0], "pid": 1, "started_at": "2020-01-01T00:00:00",
            "command": "sleep 9999", "status": "running"}
    (tmp_queue_dir / "running.json").write_text(json.dumps([mine]))
    r = run(["submit", "-g", "1", "-t", "1", "--", "/bin/true"], gpuq_env)
    assert r.returncode == 0, r.stdout + r.stderr


def _own_busy_card(userspace_module):
    """A running job I own, pinning GPU 0, with the card reported nearly full."""
    return {"id": 42, "user": userspace_module.USER, "host": userspace_module.HOST,
            "gpus": [0], "pid": 1, "started_at": "2020-01-01T00:00:00",
            "command": "sleep 9999", "status": "running"}


def test_owned_busy_card_rejects_unmet_memory(gpuq_env, tmp_queue_dir,
                                              fake_gpus, userspace_module):
    # Issue #1 (Bug 2): a card I OWN but that is nearly full must NOT be auto-stacked
    # when -m asks for more free VRAM than it has. Pre-fix, -m was ignored on owned
    # cards and the job stacked onto the busy card; now it is rejected with a reason.
    _, write = fake_gpus
    write({"gpus": [{"index": 0, "name": "X", "uuid": "GPU-x",
                     "memory_used_mb": 75 * 1024, "memory_total_mb": 80 * 1024,
                     "utilization": 100}], "compute_apps": []})   # ~5 GB free
    (tmp_queue_dir / "running.json").write_text(
        json.dumps([_own_busy_card(userspace_module)]))
    r = run(["submit", "--devices", "0", "-m", "40", "-t", "1", "--", "/bin/true"],
            gpuq_env)
    assert r.returncode != 0
    blob = r.stdout + r.stderr
    assert "GPU 0" in blob and "40 GB" in blob          # honest per-card reason
    # nothing was claimed/stacked: my original job is the only running entry
    assert [j["id"] for j in
            json.loads((tmp_queue_dir / "running.json").read_text())] == [42]


def test_owned_busy_card_queue_waits_not_stacks(gpuq_env, tmp_queue_dir,
                                                fake_gpus, userspace_module):
    # Issue #1 (Bug 1): with --queue and an -m the owned busy card can't meet, gpuq
    # must WAIT (register a queued entry) for the card to free, not instantly stack.
    _, write = fake_gpus
    write({"gpus": [{"index": 0, "name": "X", "uuid": "GPU-x",
                     "memory_used_mb": 75 * 1024, "memory_total_mb": 80 * 1024,
                     "utilization": 100}], "compute_apps": []})
    (tmp_queue_dir / "running.json").write_text(
        json.dumps([_own_busy_card(userspace_module)]))
    proc = _spawn(["submit", "--devices", "0", "-m", "40", "--queue", "-t", "1",
                   "--", "/bin/true"], gpuq_env)
    try:
        deadline = time.time() + 5
        queued = []
        while time.time() < deadline:
            qf = tmp_queue_dir / "jobs.json"
            if qf.exists():
                queued = json.loads(qf.read_text())
                if queued:
                    break
            time.sleep(0.1)
        assert len(queued) == 1, queued          # waiting, not stacked
        assert proc.poll() is None
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)


def test_devices_queue_waits_for_other_user(gpuq_env, tmp_queue_dir, userspace_module):
    # GPU 0 is held by another user; --devices 0 --queue must WAIT (register a
    # queued entry) rather than reject.
    running = [{"id": 99, "user": "bob", "host": userspace_module.HOST,
                "gpus": [0], "pid": 1, "started_at": "2020-01-01T00:00:00"}]
    (tmp_queue_dir / "running.json").write_text(json.dumps(running))
    proc = _spawn(["submit", "--devices", "0", "--queue", "-t", "1",
                   "--", "/bin/true"], gpuq_env)
    try:
        deadline = time.time() + 5
        queued = []
        while time.time() < deadline:
            qf = tmp_queue_dir / "jobs.json"
            if qf.exists():
                queued = json.loads(qf.read_text())
                if queued:
                    break
            time.sleep(0.1)
        assert len(queued) == 1, queued          # waiting, not rejected
        assert proc.poll() is None
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
