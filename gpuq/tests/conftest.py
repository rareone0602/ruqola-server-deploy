"""Test fixtures for the userspace gpuq."""
import json
import os
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
GPUQ_DIR = HERE.parent
USERSPACE_PY = GPUQ_DIR / "userspace.py"
FAKE_NVSMI = HERE / "fake_nvidia_smi.py"

DEFAULT_GPUS = [
    {"index": 0, "name": "Fake-GPU-A", "uuid": "GPU-aaaa",
     "memory_used_mb": 0, "memory_total_mb": 81920, "utilization": 0},
    {"index": 1, "name": "Fake-GPU-B", "uuid": "GPU-bbbb",
     "memory_used_mb": 0, "memory_total_mb": 81920, "utilization": 0},
]


@pytest.fixture
def tmp_queue_dir(tmp_path):
    d = tmp_path / "queue"
    d.mkdir()
    return d


@pytest.fixture
def fake_gpus(tmp_path):
    """Return (state_path, set_state(dict)). Defaults to 2 idle GPUs."""
    state_path = tmp_path / "nvsmi_state.json"

    def write(state):
        state_path.write_text(json.dumps(state))

    write({"gpus": list(DEFAULT_GPUS), "compute_apps": []})
    return state_path, write


@pytest.fixture
def gpuq_env(tmp_queue_dir, fake_gpus, tmp_path):
    """Compose the env dict for invoking userspace.py as a subprocess."""
    state_path, _ = fake_gpus
    cfg = tmp_path / "gpuq_config.json"
    cfg.write_text(json.dumps({
        "max_job_time_hours": 24,
        "max_memory_per_gpu_gb": 70,
        "notification_email": {"enabled": False},
        "slack": {"enabled": False},
    }))
    env = os.environ.copy()
    env["GPUQ_QUEUE_DIR"] = str(tmp_queue_dir)
    env["GPUQ_CONFIG_FILE"] = str(cfg)
    env["GPUQ_NVSMI"] = str(FAKE_NVSMI)
    env["FAKE_NVSMI_STATE"] = str(state_path)
    env["GPUQ_SCOPE"] = "off"          # deterministic fallback (no systemd scope)
    env["PYTHONUNBUFFERED"] = "1"
    return env


@pytest.fixture
def userspace_module(tmp_queue_dir, fake_gpus, tmp_path, monkeypatch):
    """Import userspace.py with env-driven paths set, for direct unit tests."""
    state_path, _ = fake_gpus
    monkeypatch.setenv("GPUQ_QUEUE_DIR", str(tmp_queue_dir))
    cfg = tmp_path / "gpuq_config.json"
    cfg.write_text("{}")
    monkeypatch.setenv("GPUQ_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("GPUQ_NVSMI", str(FAKE_NVSMI))
    monkeypatch.setenv("FAKE_NVSMI_STATE", str(state_path))
    monkeypatch.setenv("GPUQ_SCOPE", "off")   # deterministic fallback in unit tests
    sys.path.insert(0, str(GPUQ_DIR))
    sys.modules.pop("userspace", None)
    import userspace  # noqa: E402
    yield userspace
    sys.path.remove(str(GPUQ_DIR))
    sys.modules.pop("userspace", None)
