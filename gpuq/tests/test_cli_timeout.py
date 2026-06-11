"""Timeout enforcement test."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
USERSPACE_PY = HERE.parent / "userspace.py"


def test_submit_times_out_and_clears(gpuq_env, tmp_queue_dir):
    # 0.001 hours = 3.6s; sleep 30 must be killed.
    start = time.time()
    r = subprocess.run(
        [sys.executable, str(USERSPACE_PY),
         "submit", "-g", "1", "-t", "0.001", "--", "/bin/sleep", "30"],
        env=gpuq_env, capture_output=True, text=True, timeout=20,
    )
    elapsed = time.time() - start
    assert elapsed < 15, f"timeout did not fire (took {elapsed:.1f}s)"
    # The child is killed by SIGTERM after timeout; gpuq forwards the negative
    # wait status via sys.exit (-15 -> 241, -9 -> 247), or 128+sig if the
    # exit code is normalized. Any non-zero is acceptable; what matters is
    # cleanup.
    assert r.returncode != 0, r.stderr
    assert json.loads((tmp_queue_dir / "running.json").read_text()) == []
