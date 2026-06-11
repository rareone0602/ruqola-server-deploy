"""Unit tests for reap_running."""
import os


def test_drops_dead_pid_on_this_host(userspace_module):
    # PID 1 (init) is alive; pick a definitely-dead one by spawning + waiting.
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    jobs = [
        {"id": 1, "host": userspace_module.HOST, "pid": pid},
        {"id": 2, "host": userspace_module.HOST, "pid": os.getpid()},
    ]
    out = userspace_module.reap_running(jobs)
    assert [j["id"] for j in out] == [2]


def test_keeps_foreign_host_entries(userspace_module):
    jobs = [{"id": 1, "host": "some-other-host", "pid": 99999999}]
    assert userspace_module.reap_running(jobs) == jobs


def test_keeps_live_pid(userspace_module):
    jobs = [{"id": 1, "host": userspace_module.HOST, "pid": os.getpid()}]
    assert userspace_module.reap_running(jobs) == jobs
