"""Unit tests for reap_running / reap_queued."""
import os


def _dead_pid():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def test_drops_dead_pid_on_this_host(userspace_module):
    pid = _dead_pid()
    jobs = [
        {"id": 1, "host": userspace_module.HOST, "pid": pid},
        {"id": 2, "host": userspace_module.HOST, "pid": os.getpid()},
    ]
    kept, lost = userspace_module.reap_running(jobs)
    assert [j["id"] for j in kept] == [2]
    assert [j["id"] for j in lost] == [1]


def test_keeps_foreign_host_entries(userspace_module):
    jobs = [{"id": 1, "host": "some-other-host", "pid": 99999999}]
    kept, lost = userspace_module.reap_running(jobs)
    assert kept == jobs and lost == []


def test_keeps_live_pid(userspace_module):
    jobs = [{"id": 1, "host": userspace_module.HOST, "pid": os.getpid()}]
    kept, lost = userspace_module.reap_running(jobs)
    assert kept == jobs and lost == []


def test_keeps_orphan_while_child_group_alive(userspace_module):
    # Supervisor dead, but child pgid (our own group) still alive -> kept.
    pid = _dead_pid()
    jobs = [{"id": 1, "host": userspace_module.HOST, "pid": pid,
             "child_pgid": os.getpgid(0)}]
    kept, lost = userspace_module.reap_running(jobs)
    assert kept == jobs and lost == []


def test_lost_when_supervisor_and_child_both_dead(userspace_module):
    pid = _dead_pid()
    child = _dead_pid()
    jobs = [{"id": 1, "host": userspace_module.HOST, "pid": pid,
             "child_pgid": child}]
    kept, lost = userspace_module.reap_running(jobs)
    assert kept == [] and [j["id"] for j in lost] == [1]


def test_pid_reuse_detected_via_start_time(userspace_module):
    # Entry claims our PID but with a start time that cannot match.
    jobs = [{"id": 1, "host": userspace_module.HOST, "pid": os.getpid(),
             "pid_start": -1}]
    kept, lost = userspace_module.reap_running(jobs)
    assert kept == [] and [j["id"] for j in lost] == [1]


def test_reap_queued_drops_dead_waiter(userspace_module):
    pid = _dead_pid()
    jobs = [
        {"id": 1, "host": userspace_module.HOST, "pid": pid},
        {"id": 2, "host": userspace_module.HOST, "pid": os.getpid()},
        {"id": 3, "host": "elsewhere", "pid": 99999999},
    ]
    kept, lost = userspace_module.reap_queued(jobs)
    assert [j["id"] for j in kept] == [2, 3]
    assert [j["id"] for j in lost] == [1]


def test_recycled_child_pgid_not_kept(userspace_module):
    # Supervisor dead; child_pgid points at OUR live group but with a start
    # time that cannot match -> recycled pgid, entry must be reaped.
    pid = _dead_pid()
    jobs = [{"id": 1, "host": userspace_module.HOST, "pid": pid,
             "child_pgid": os.getpgid(0), "child_pgid_start": -1}]
    kept, lost = userspace_module.reap_running(jobs)
    assert kept == [] and [j["id"] for j in lost] == [1]


def test_orphan_kept_when_child_identity_matches(userspace_module):
    pid = _dead_pid()
    pgid = os.getpgid(0)
    start = userspace_module.pid_start_time(pgid)
    jobs = [{"id": 1, "host": userspace_module.HOST, "pid": pid,
             "child_pgid": pgid, "child_pgid_start": start}]
    kept, lost = userspace_module.reap_running(jobs)
    assert [j["id"] for j in kept] == [1] and lost == []
