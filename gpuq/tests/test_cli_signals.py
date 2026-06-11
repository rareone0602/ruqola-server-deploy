"""Signal-forwarding test: SIGINT/SIGTERM to gpuq tears down the job tree."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
USERSPACE_PY = HERE.parent / "userspace.py"


def _spawn(args, env):
    return subprocess.Popen(
        [sys.executable, str(USERSPACE_PY)] + args,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def test_sigint_kills_child_tree(gpuq_env, tmp_queue_dir, tmp_path):
    # The child writes its own PID (after exec sleep) to a file we can poll,
    # so we know the grandchild has actually started before we signal gpuq.
    pidfile = tmp_path / "child.pid"
    proc = _spawn(
        ["submit", "-g", "1", "-t", "1", "--",
         "/bin/bash", "-c", f'echo $$ > {pidfile}; exec /bin/sleep 30'],
        gpuq_env,
    )

    # Wait for the bash to write its PID then exec into sleep.
    deadline = time.time() + 5
    while time.time() < deadline and not pidfile.exists():
        time.sleep(0.05)
    assert pidfile.exists(), "child never wrote its pidfile"
    child_pid = int(pidfile.read_text().strip())
    assert _alive(child_pid), "child should be alive before we signal gpuq"

    proc.send_signal(signal.SIGINT)
    rc = proc.wait(timeout=5)  # gpuq must exit quickly, not wait 30s
    assert rc != 0, "gpuq should report non-zero on signal-induced exit"

    # Give the kernel a beat to reap; then assert the child is gone.
    deadline = time.time() + 2
    while time.time() < deadline and _alive(child_pid):
        time.sleep(0.05)
    assert not _alive(child_pid), \
        f"child PID {child_pid} survived gpuq's SIGINT — signal forwarding broken"

    # And running.json was cleaned up.
    assert json.loads((tmp_queue_dir / "running.json").read_text()) == []
