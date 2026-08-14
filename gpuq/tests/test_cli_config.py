"""Tests for `gpuq config` (read-only default, `init` writer)."""
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


def test_config_init_writes_default_when_missing(gpuq_env, tmp_path):
    cfg = tmp_path / "fresh_config.json"
    env = dict(gpuq_env)
    env["GPUQ_CONFIG_FILE"] = str(cfg)
    r = run(["config", "init"], env)
    assert r.returncode == 0, r.stderr
    assert cfg.exists()
    data = json.loads(cfg.read_text())
    assert "max_job_time_hours" in data
    assert "default_min_free_gb" in data
    assert "quotas" in data
    assert "audit" in data


def test_config_init_template_carries_pressure_policy(gpuq_env, tmp_path):
    """The starter template ships the tightened limits: 48h wall cap, per-user
    3-card hard cap, quota hold, and the 4h/2h audit grace/reminder cadence."""
    cfg = tmp_path / "fresh_config.json"
    env = dict(gpuq_env)
    env["GPUQ_CONFIG_FILE"] = str(cfg)
    r = run(["config", "init"], env)
    assert r.returncode == 0, r.stderr
    data = json.loads(cfg.read_text())
    assert data["max_job_time_hours_cap"] == 48
    assert data["max_gpus_per_user_hard"] == 3
    assert data["quotas"]["delay_hours"] == 8
    audit = data["audit"]
    assert audit["untracked_grace_hours"] == 4
    assert audit["untracked_reminder_hours"] == 2
    assert audit["rebind_grace_hours"] == 4
    assert audit["rebind_reminder_hours"] == 2


def test_bare_config_is_read_only(gpuq_env, tmp_path):
    cfg = tmp_path / "fresh_config.json"
    env = dict(gpuq_env)
    env["GPUQ_CONFIG_FILE"] = str(cfg)
    r = run(["config"], env)
    assert r.returncode == 0, r.stderr
    assert not cfg.exists()          # bare `config` never writes
    assert "Config file:" in r.stdout
    assert "config init" in r.stdout + r.stderr  # hints at the writer


def test_config_init_refuses_to_clobber(gpuq_env):
    # gpuq_env's config file already exists from the fixture.
    r = run(["config", "init"], gpuq_env)
    assert r.returncode != 0
    assert "already exists" in r.stderr


def test_config_init_force_overwrites(gpuq_env, tmp_path):
    cfg = Path(gpuq_env["GPUQ_CONFIG_FILE"])
    cfg.write_text(json.dumps({"old": "data"}))
    r = run(["config", "init", "--force"], gpuq_env)
    assert r.returncode == 0, r.stderr
    data = json.loads(cfg.read_text())
    assert "max_job_time_hours" in data
    assert "old" not in data


def test_config_show(gpuq_env):
    r = run(["config", "--show"], gpuq_env)
    assert r.returncode == 0, r.stderr
    assert "Config file:" in r.stdout
    assert "Quotas:" in r.stdout
