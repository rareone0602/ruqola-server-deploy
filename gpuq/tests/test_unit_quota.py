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


def test_window_clamps_jobs_straddling_the_cutoff(userspace_module):
    # A 20h job that ended 10h ago: with a 168h window all of it counts; but
    # shrink the window to 12h and only the in-window fraction (2h of runtime
    # before end... actually the last 12h minus nothing) should count.
    now = datetime.now()
    started = now - timedelta(hours=20)
    ended = now - timedelta(hours=10)
    _write_ledger(userspace_module.USAGE_FILE, [{
        "user": "alice", "gpu_hours": 10.0,
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
    }])
    # window 168h: full 10 gpu-hours
    assert abs(userspace_module.usage_in_window("alice") - 10.0) < 0.01
    # window 12h: the job ran from t-20h to t-10h; overlap with [t-12h, t] is
    # 2h of its 10h span -> 2 gpu-hours
    used = userspace_module.usage_in_window("alice", window_hours=12)
    assert abs(used - 2.0) < 0.05


def test_running_job_clamped_to_window(userspace_module):
    now = datetime.now()
    started = (now - timedelta(hours=200)).isoformat(timespec="seconds")
    userspace_module.save_running([{
        "id": 1, "user": "alice", "host": userspace_module.HOST,
        "pid": 99999, "gpus": [1], "started_at": started, "status": "running",
    }])
    used = userspace_module.usage_in_window("alice")  # 168h window
    assert 167 < used < 169  # not 200


def test_malformed_gpu_hours_never_crashes_the_scan(userspace_module):
    now = datetime.now()
    recent = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    _write_ledger(userspace_module.USAGE_FILE, [
        {"user": "alice", "gpu_hours": None, "ended_at": recent},
        {"user": "alice", "gpu_hours": "junk", "ended_at": recent},
        {"user": "alice", "gpu_hours": 3, "ended_at": recent},
    ])
    assert userspace_module.usage_in_window("alice") == 3


def test_event_records_never_charged(userspace_module):
    now = datetime.now()
    _write_ledger(userspace_module.USAGE_FILE, [
        {"event": "cancelled", "user": "alice", "gpu_hours": 99,
         "at": now.isoformat(timespec="seconds")},
        {"event": "rejected", "user": "alice",
         "at": now.isoformat(timespec="seconds")},
    ])
    assert userspace_module.usage_in_window("alice") == 0.0


def test_rotated_ledger_files_are_read(userspace_module, tmp_queue_dir):
    now = datetime.now()
    recent = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    rotated = tmp_queue_dir / "usage-2026-05.jsonl"
    rotated.write_text(json.dumps(
        {"user": "alice", "gpu_hours": 4, "ended_at": recent}) + "\n")
    _write_ledger(userspace_module.USAGE_FILE, [
        {"user": "alice", "gpu_hours": 3, "ended_at": recent},
    ])
    assert userspace_module.usage_in_window("alice") == 7
