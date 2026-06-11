"""Unit tests for quota helpers: ledger reads, window cutoff, budget logic."""
import json
from datetime import datetime, timedelta


def _write_ledger(usage_file, entries):
    with open(usage_file, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_no_quota_means_unlimited(userspace_module):
    config = {}
    over, used, budget = userspace_module.would_exceed_quota("alice", 999, config)
    assert over is False
    assert budget is None


def test_user_specific_quota_overrides_default(userspace_module):
    config = {"quotas": {
        "default_gpu_hours_per_week": 10,
        "users": {"alice": 100},
    }}
    assert userspace_module.quota_for_user("alice", config) == 100
    assert userspace_module.quota_for_user("bob", config) == 10


def test_zero_or_negative_quota_treated_as_unlimited(userspace_module):
    config = {"quotas": {"users": {"alice": 0, "bob": -1}}}
    assert userspace_module.quota_for_user("alice", config) is None
    assert userspace_module.quota_for_user("bob", config) is None


def test_usage_in_window_excludes_old_entries(userspace_module):
    now = datetime.now()
    old = (now - timedelta(hours=200)).isoformat(timespec="seconds")
    recent = (now - timedelta(hours=2)).isoformat(timespec="seconds")
    _write_ledger(userspace_module.USAGE_FILE, [
        {"user": "alice", "gpu_hours": 50, "ended_at": old},
        {"user": "alice", "gpu_hours": 5, "ended_at": recent},
        {"user": "bob", "gpu_hours": 100, "ended_at": recent},
    ])
    used = userspace_module.usage_in_window("alice")
    assert used == 5  # 50 was outside the 168h window


def test_running_jobs_charge_partial_usage(userspace_module):
    now = datetime.now()
    started = (now - timedelta(hours=3)).isoformat(timespec="seconds")
    userspace_module.save_running([{
        "id": 1, "user": "alice", "host": userspace_module.HOST,
        "pid": 99999, "gpus": [0, 1],
        "started_at": started, "status": "running",
    }])
    used = userspace_module.usage_in_window("alice")
    # 3h elapsed * 2 GPUs ≈ 6 gpu-hours
    assert 5.5 < used < 6.5


def test_would_exceed_quota_blocks_at_threshold(userspace_module):
    now = datetime.now()
    recent = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    _write_ledger(userspace_module.USAGE_FILE, [
        {"user": "alice", "gpu_hours": 95, "ended_at": recent},
    ])
    config = {"quotas": {"users": {"alice": 100}}}
    over, used, budget = userspace_module.would_exceed_quota("alice", 10, config)
    assert over is True
    assert used == 95
    assert budget == 100


def test_under_quota_allowed(userspace_module):
    now = datetime.now()
    recent = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    _write_ledger(userspace_module.USAGE_FILE, [
        {"user": "alice", "gpu_hours": 10, "ended_at": recent},
    ])
    config = {"quotas": {"users": {"alice": 100}}}
    over, _, _ = userspace_module.would_exceed_quota("alice", 10, config)
    assert over is False
