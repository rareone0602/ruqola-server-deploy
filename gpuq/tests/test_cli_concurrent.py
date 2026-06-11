"""Concurrent submit tests — validate the flock contract across users.

Under the "you own your allocated GPU" policy, contention is now between
*different* users: a user may stack jobs on a card they already hold, but never
on one another user holds. Each subprocess is given a distinct USER so these
exercise real cross-user races; one test covers same-user stacking.
"""
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
USERSPACE_PY = HERE.parent / "userspace.py"


def _spawn(args, env, user=None):
    if user is not None:
        env = {**env, "USER": user, "LOGNAME": user}
    return subprocess.Popen(
        [sys.executable, str(USERSPACE_PY)] + args,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def test_two_gpus_two_winners_two_rejected(gpuq_env):
    # Four DIFFERENT users race for two GPUs. You can't stack on someone else's
    # card, so exactly two win and two are rejected.
    procs = [
        _spawn(["submit", "-g", "1", "-t", "1", "--", "/bin/sleep", "1"],
               gpuq_env, user=f"user{i}")
        for i in range(4)
    ]
    time.sleep(2.5)
    rcs = [p.wait(timeout=5) for p in procs]
    winners = sum(1 for rc in rcs if rc == 0)
    losers = sum(1 for rc in rcs if rc != 0)
    assert winners == 2, (rcs, [p.stderr.read() for p in procs])
    assert losers == 2


def test_queue_flag_waits_for_slot(gpuq_env, tmp_queue_dir):
    # A different user holds both GPUs; the waiter can't stack on them, so its
    # --queue submit waits (registers a queued entry), then runs.
    first = _spawn(["submit", "-g", "2", "-t", "1", "--", "/bin/sleep", "1"],
                   gpuq_env, user="holder")
    time.sleep(0.4)
    second = _spawn(["submit", "-g", "1", "-t", "1", "--queue", "--", "/bin/true"],
                    gpuq_env, user="waiter")
    time.sleep(0.6)
    queued = json.loads((tmp_queue_dir / "jobs.json").read_text())
    assert len(queued) == 1, queued
    assert first.wait(timeout=5) == 0
    # The second one polls every QUEUE_POLL_INTERVAL_SEC=30s; SIGINT triggers
    # the in-script cleanup (KeyboardInterrupt handler) so we don't leave
    # a stale queued entry in real flows.
    second.send_signal(signal.SIGINT)
    second.wait(timeout=5)


def test_same_user_stacks_on_single_gpu(gpuq_env, fake_gpus):
    # One GPU, one user: the first job holds it, the second STACKS onto the card
    # it owns (the old policy would have rejected the second). Both win.
    _, write = fake_gpus
    write({"gpus": [{"index": 0, "name": "X", "uuid": "GPU-x",
                     "memory_used_mb": 0, "memory_total_mb": 81920,
                     "utilization": 0}], "compute_apps": []})
    one = _spawn(["submit", "-g", "1", "-t", "1", "--", "/bin/sleep", "1"],
                 gpuq_env, user="solo")
    time.sleep(0.5)
    two = _spawn(["submit", "-g", "1", "-t", "1", "--", "/bin/true"],
                 gpuq_env, user="solo")
    assert two.wait(timeout=5) == 0, two.stderr.read()
    assert one.wait(timeout=5) == 0
