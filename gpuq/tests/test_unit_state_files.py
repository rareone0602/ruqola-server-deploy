"""Files in the shared queue dir are opened safely (todo E1).

The queue dir is group-writable with no sticky bit, so any gpuqueue member can
put a symlink, FIFO or hard link where a state file should be. Root's audit
opens these files, so gpuq must never follow, block on, or chmod through one.
Each test plants the trap in a tmp queue dir and checks the target is untouched.
"""
import json
import os
import stat
import threading

import pytest


def mode(path):
    return stat.S_IMODE(os.lstat(path).st_mode)


def finishes(fn, timeout=5):
    """Run fn in a daemon thread; True if it returned (or raised) in time."""
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    t.join(timeout)
    return not t.is_alive()


@pytest.fixture
def victim(tmp_path):
    """A private file outside the queue dir, standing in for /etc/shadow."""
    v = tmp_path / "shadow"
    v.write_text("secret\n")
    v.chmod(0o600)
    return v


# --- the lock -----------------------------------------------------------------
def test_the_lock_refuses_a_symlink_and_leaves_its_target_alone(userspace_module, victim):
    us = userspace_module
    us.LOCK_FILE.symlink_to(victim)
    with pytest.raises(OSError, match="symlink"):
        with us.file_lock(us.LOCK_FILE):
            pass
    assert mode(victim) == 0o600


def test_the_lock_does_not_create_a_file_through_a_dangling_symlink(userspace_module, tmp_path):
    us = userspace_module
    planted = tmp_path / "cron.d-evil"
    us.LOCK_FILE.symlink_to(planted)
    with pytest.raises(OSError):
        with us.file_lock(us.LOCK_FILE):
            pass
    assert not planted.exists()


def test_the_lock_refuses_a_hard_link_to_a_file_elsewhere(userspace_module, victim):
    us = userspace_module
    os.link(victim, us.LOCK_FILE)
    with pytest.raises(OSError, match="plain file"):
        with us.file_lock(us.LOCK_FILE):
            pass
    assert mode(victim) == 0o600


def test_the_lock_refuses_a_fifo(userspace_module):
    us = userspace_module
    os.mkfifo(us.LOCK_FILE)
    with pytest.raises(OSError, match="plain file"):
        with us.file_lock(us.LOCK_FILE):
            pass


def test_a_new_lock_is_group_rw_even_under_umask_022(userspace_module):
    # Root creates it 0644 otherwise, and no one else could open it.
    us = userspace_module
    old = os.umask(0o022)
    try:
        with us.file_lock(us.LOCK_FILE):
            pass
    finally:
        os.umask(old)
    assert mode(us.LOCK_FILE) == 0o664


# --- the usage ledger -----------------------------------------------------------
def test_appending_usage_refuses_a_symlink_and_leaves_its_target_alone(userspace_module, victim):
    us = userspace_module
    us.USAGE_FILE.symlink_to(victim)
    with pytest.raises(OSError, match="symlink"):
        us.append_usage({"user": "mallory"})
    assert victim.read_text() == "secret\n" and mode(victim) == 0o600


def test_appending_usage_creates_a_group_rw_ledger(userspace_module):
    us = userspace_module
    old = os.umask(0o022)
    try:
        us.append_usage({"user": "a"})
        us.append_usage({"user": "b"})
    finally:
        os.umask(old)
    assert [json.loads(l)["user"] for l in us.USAGE_FILE.read_text().splitlines()] == ["a", "b"]
    assert mode(us.USAGE_FILE) == 0o664


def test_reading_usage_skips_a_symlinked_or_fifo_rotation(userspace_module, tmp_path):
    us = userspace_module
    elsewhere = tmp_path / "elsewhere.jsonl"
    elsewhere.write_text(json.dumps({"user": "forged"}) + "\n")
    (us.QUEUE_DIR / "usage-2026-01.jsonl").symlink_to(elsewhere)
    os.mkfifo(us.QUEUE_DIR / "usage-2026-02.jsonl")
    us.append_usage({"user": "real"})
    seen = []
    assert finishes(lambda: seen.extend(us.iter_usage_records()))
    assert [r["user"] for r in seen] == ["real"]


def test_reading_usage_survives_a_file_that_is_not_text(userspace_module):
    us = userspace_module
    (us.QUEUE_DIR / "usage-2026-01.jsonl").write_bytes(b"\xff\xfe\x00junk\n")
    us.append_usage({"user": "real"})
    assert [r["user"] for r in us.iter_usage_records()] == ["real"]


# --- JSON state -------------------------------------------------------------------
def test_state_behind_a_symlink_reads_as_unreadable(userspace_module, tmp_path, capsys):
    us = userspace_module
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps([{"id": "forged"}]))
    us.RUNNING_FILE.symlink_to(elsewhere)
    assert us.load_running() == []
    assert "symlink" in capsys.readouterr().err


def test_state_that_is_a_fifo_does_not_hang_the_reader(userspace_module):
    us = userspace_module
    os.mkfifo(us.RUNNING_FILE)
    got = []
    assert finishes(lambda: got.append(us.load_running()))
    assert got == [[]]


def test_saving_state_replaces_a_planted_symlink_not_its_target(userspace_module, victim):
    us = userspace_module
    us.RUNNING_FILE.symlink_to(victim)
    us.save_running([{"id": "1"}])
    assert victim.read_text() == "secret\n"
    assert not us.RUNNING_FILE.is_symlink() and us.load_running() == [{"id": "1"}]


def test_missing_state_reads_as_empty_without_a_warning(userspace_module, capsys):
    assert userspace_module.load_running() == []
    assert capsys.readouterr().err == ""
