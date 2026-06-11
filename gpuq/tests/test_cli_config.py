"""Tests for `gpuq config` (writer + --show)."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
USERSPACE_PY = HERE.parent / "userspace.py"


def run(args, env):
    return subprocess.run(
        [sys.executable, str(USERSPACE_PY)] + args,
        env=env, capture_output=True, text=True,
    )


def test_config_writes_default_when_missing(gpuq_env, tmp_path):
    cfg = tmp_path / "fresh_config.json"
    env = dict(gpuq_env)
    env["GPUQ_CONFIG_FILE"] = str(cfg)
    r = run(["config"], env)
    assert r.returncode == 0, r.stderr
    assert cfg.exists()
    data = json.loads(cfg.read_text())
    assert "max_job_time_hours" in data
    assert "quotas" in data
    assert "audit" in data


def test_config_refuses_to_clobber(gpuq_env):
    # gpuq_env's config file already exists from the fixture.
    r = run(["config"], gpuq_env)
    assert r.returncode != 0
    assert "already exists" in r.stderr


def test_config_force_overwrites(gpuq_env, tmp_path):
    cfg = Path(gpuq_env["GPUQ_CONFIG_FILE"])
    cfg.write_text(json.dumps({"old": "data"}))
    r = run(["config", "--force"], gpuq_env)
    assert r.returncode == 0, r.stderr
    data = json.loads(cfg.read_text())
    assert "max_job_time_hours" in data
    assert "old" not in data


def test_config_show(gpuq_env):
    r = run(["config", "--show"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "Config file:" in r.stdout
    assert "Quotas:" in r.stdout
