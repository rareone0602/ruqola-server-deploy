"""Unit tests for the untracked-GPU detector and warn->kill state machine.

These exercise check_untracked() in-process with the OS seams monkeypatched
(get_process_user / get_process_pgid / list_gpu_processes_with_owner), since the
fake nvidia-smi can fabricate a GPU process row but cannot tie it to a real
process group/owner. Time is injected via `now=` so the deadline can be crossed
without waiting. One test does a real same-user kill to cover enforce_kill_pgid.
"""
import json
import os
import subprocess
import types
from datetime import datetime, timedelta


def _args(enforce=False):
    return types.SimpleNamespace(enforce=enforce, quiet=True)


def _proc(owner="mallory", gpu_idx=0, pgid=9999, memory_mb=40000,
          name="python", pid=4242, gpu_uuid="GPU-aaaa"):
    return {"owner": owner, "gpu_idx": gpu_idx, "pgid": pgid,
            "memory_mb": memory_mb, "name": name, "pid": pid,
            "gpu_uuid": gpu_uuid}


def _run_check(us, running, procs, monkeypatch, *, audit_cfg=None,
               enforce=False, now=None, kill_ok=True):
    """Drive check_untracked with controlled procs; capture emails + kills."""
    monkeypatch.setattr(us, "list_gpu_processes_with_owner", lambda: procs)
    sent = []
    monkeypatch.setattr(
        us, "notify_untracked",
        lambda owner, sample, deadline, config, kind: sent.append(
            {"owner": owner, "kind": kind, "sample": sample}) or True)
    killed = []
    monkeypatch.setattr(
        us, "enforce_kill_pgid", lambda pgid: (killed.append(pgid) or kill_ok))
    cfg = dict(audit_cfg or {})
    breaches = us.check_untracked(running, cfg, _args(enforce), {"audit": cfg}, now=now)
    state = (json.loads(us.UNTRACKED_STATE_FILE.read_text())
             if us.UNTRACKED_STATE_FILE.exists() else {})
    return breaches, sent, killed, state


# --- detection / filtering -------------------------------------------------

def test_flags_new_offender(userspace_module, monkeypatch):
    breaches, sent, killed, state = _run_check(
        userspace_module, [], [_proc()], monkeypatch)
    assert any("UNTRACKED" in b and "mallory" in b for b in breaches)
    assert len(sent) == 1 and sent[0]["kind"] == "warn"
    assert killed == []
    assert len(state) == 1


def test_tracked_pgid_not_flagged(userspace_module, monkeypatch):
    us = userspace_module
    running = [{"id": 1, "user": "mallory", "host": us.HOST, "gpus": [0],
                "child_pgid": 9999, "started_at": "2020-01-01T00:00:00"}]
    breaches, sent, _, state = _run_check(us, running, [_proc(pgid=9999)], monkeypatch)
    assert breaches == [] and sent == [] and state == {}


def test_allowlisted_system_account_skipped(userspace_module, monkeypatch):
    breaches, sent, _, _ = _run_check(
        userspace_module, [], [_proc(owner="root")], monkeypatch)
    assert breaches == [] and sent == []


def test_admin_allowlist_skipped(userspace_module, monkeypatch):
    breaches, sent, _, _ = _run_check(
        userspace_module, [], [_proc(owner="carol")], monkeypatch,
        audit_cfg={"untracked_allowlist": ["carol"]})
    assert breaches == [] and sent == []


def test_below_min_memory_skipped(userspace_module, monkeypatch):
    breaches, _, _, _ = _run_check(
        userspace_module, [], [_proc(memory_mb=100)], monkeypatch)
    assert breaches == []


def test_unattributable_procs_skipped(userspace_module, monkeypatch):
    procs = [_proc(gpu_idx=None), _proc(pgid=None), _proc(owner=None)]
    breaches, sent, _, _ = _run_check(userspace_module, [], procs, monkeypatch)
    assert breaches == [] and sent == []


def test_legacy_job_without_pgid_skipped(userspace_module, monkeypatch):
    """A pre-feature running job (no child_pgid) protects its (user, gpu)."""
    us = userspace_module
    running = [{"id": 1, "user": "mallory", "host": us.HOST, "gpus": [0],
                "started_at": "2020-01-01T00:00:00"}]
    breaches, sent, _, _ = _run_check(
        us, running, [_proc(pgid=12345, gpu_idx=0)], monkeypatch)
    assert breaches == [] and sent == []


def test_scope_supported_off_in_tests(userspace_module):
    # conftest sets GPUQ_SCOPE=off, so jobs use the plain-process-group fallback.
    assert userspace_module.scope_supported() is False


def test_proc_in_scope_matches_real_cgroup(userspace_module):
    us = userspace_module
    cg = open(f"/proc/{os.getpid()}/cgroup").read().strip()
    leaf = cg.rsplit("/", 1)[-1]                 # e.g. session-77.scope
    assert us._proc_in_scope(os.getpid(), {leaf}) is True
    assert us._proc_in_scope(os.getpid(), {"gpuq-nope.scope"}) is False
    assert us._proc_in_scope(os.getpid(), set()) is False
    assert us._proc_in_scope(2 ** 31, {leaf}) is False   # no such pid


def test_scoped_worker_not_flagged(userspace_module, monkeypatch):
    """A detached worker (own pgid, no ancestry to the job) is cleared because
    its cgroup is inside the job's scope."""
    us = userspace_module
    running = [{"id": 1, "user": "yyq", "host": us.HOST, "gpus": [1],
                "cgroup_scope": "gpuq-1.scope", "started_at": "2020-01-01T00:00:00"}]
    monkeypatch.setattr(us, "_proc_in_scope",
                        lambda pid, scopes: "gpuq-1.scope" in scopes)
    breaches, sent, _, _ = _run_check(
        us, running, [_proc(owner="yyq", gpu_idx=0, pgid=999, pid=999)], monkeypatch)
    assert breaches == [] and sent == []


def test_descendant_that_resetsid_is_protected(userspace_module, monkeypatch):
    """A worker that re-ran setsid (pgid != child_pgid) is protected by ancestry."""
    us = userspace_module
    running = [{"id": 1, "user": "mallory", "host": us.HOST, "gpus": [0],
                "child_pid": 1000, "child_pgid": 1000,
                "started_at": "2020-01-01T00:00:00"}]
    # GPU proc 1050 has its own pgid 1050 (re-setsid'd); parent chain -> 1000.
    monkeypatch.setattr(us, "get_parent_pid",
                        lambda pid: {1050: 1000, 1000: 1}.get(int(pid)))
    breaches, sent, _, _ = _run_check(
        us, running, [_proc(pgid=1050, pid=1050)], monkeypatch)
    assert breaches == [] and sent == []


def test_same_gpu_rogue_still_flagged_despite_ancestry(userspace_module, monkeypatch):
    """A genuine rogue by the same user on the same GPU has no ancestry to the
    tracked job, so the same-GPU blind-spot closure is preserved."""
    us = userspace_module
    running = [{"id": 1, "user": "mallory", "host": us.HOST, "gpus": [0],
                "child_pid": 1000, "child_pgid": 1000,
                "started_at": "2020-01-01T00:00:00"}]
    # rogue 2050 -> shell 2000 -> init; never reaches the tracked child 1000.
    monkeypatch.setattr(us, "get_parent_pid",
                        lambda pid: {2050: 2000, 2000: 1}.get(int(pid)))
    breaches, sent, _, _ = _run_check(
        us, running, [_proc(pgid=2050, pid=2050)], monkeypatch)
    assert any("UNTRACKED" in b for b in breaches)
    assert len(sent) == 1


def test_foreign_host_job_does_not_clear_local_rogue(userspace_module, monkeypatch):
    us = userspace_module
    running = [{"id": 1, "user": "mallory", "host": "other-node", "gpus": [0],
                "child_pgid": 9999, "started_at": "2020-01-01T00:00:00"}]
    breaches, sent, _, _ = _run_check(us, running, [_proc(pgid=9999)], monkeypatch)
    assert any("UNTRACKED" in b for b in breaches)
    assert len(sent) == 1


def test_submit_grace_suppresses_flag(userspace_module, monkeypatch):
    us = userspace_module
    now = datetime(2026, 6, 10, 12, 0, 0)
    running = [{"id": 1, "user": "mallory", "host": us.HOST, "gpus": [1],
                "child_pgid": 777, "started_at": now.isoformat()}]
    breaches, sent, _, _ = _run_check(
        us, running, [_proc(pgid=888, gpu_idx=0)], monkeypatch, now=now)
    assert breaches == [] and sent == []


# --- state machine ---------------------------------------------------------

def test_warn_then_remind_then_overdue_kill(userspace_module, monkeypatch):
    us = userspace_module
    proc = _proc()
    cfg = {"untracked_grace_hours": 24, "untracked_reminder_hours": 6}
    t0 = datetime(2026, 6, 10, 12, 0, 0)

    _, s1, _, _ = _run_check(us, [], [proc], monkeypatch, now=t0, audit_cfg=cfg)
    assert s1 and s1[0]["kind"] == "warn"

    # +1h: within window, before reminder cadence -> breach but no email
    b2, s2, _, _ = _run_check(us, [], [proc], monkeypatch,
                              now=t0 + timedelta(hours=1), audit_cfg=cfg)
    assert s2 == [] and any("deadline" in b for b in b2)

    # +7h: reminder due
    _, s3, _, _ = _run_check(us, [], [proc], monkeypatch,
                             now=t0 + timedelta(hours=7), audit_cfg=cfg)
    assert s3 and s3[0]["kind"] == "remind"

    # +25h: past deadline, enforce -> kill + "killed" email
    b4, s4, k4, _ = _run_check(us, [], [proc], monkeypatch,
                               now=t0 + timedelta(hours=25), enforce=True,
                               audit_cfg=cfg)
    assert any("PAST DEADLINE" in b and "enforcing" in b for b in b4)
    assert k4 == [proc["pgid"]]
    assert s4 and s4[0]["kind"] == "killed"


def test_overdue_email_when_kill_fails(userspace_module, monkeypatch):
    us = userspace_module
    proc = _proc()
    cfg = {"untracked_grace_hours": 24}
    t0 = datetime(2026, 6, 10, 12, 0, 0)
    _run_check(us, [], [proc], monkeypatch, now=t0, audit_cfg=cfg)  # warn
    b, s, k, _ = _run_check(us, [], [proc], monkeypatch,
                            now=t0 + timedelta(hours=25), enforce=True,
                            audit_cfg=cfg, kill_ok=False)
    assert any("PAST DEADLINE" in x for x in b)
    assert k == [proc["pgid"]]            # kill was attempted
    assert s and s[0]["kind"] == "overdue"  # but reported as not-killed


def test_past_deadline_without_enforce_escalates_no_kill(userspace_module, monkeypatch):
    us = userspace_module
    proc = _proc()
    cfg = {"untracked_grace_hours": 24}
    t0 = datetime(2026, 6, 10, 12, 0, 0)
    _run_check(us, [], [proc], monkeypatch, now=t0, audit_cfg=cfg)
    b, s, k, _ = _run_check(us, [], [proc], monkeypatch,
                            now=t0 + timedelta(hours=25), enforce=False,
                            audit_cfg=cfg)
    assert any("PAST DEADLINE" in x and "escalated to admin" in x for x in b)
    assert k == []                        # nothing killed without --enforce
    assert s and s[0]["kind"] == "overdue"


def test_kill_always_emails_even_when_throttle_unexpired(userspace_module, monkeypatch):
    """Past-deadline kill must notify the user even if the reminder cadence has
    not re-fired (grace_hours < reminder_hours), else the kill is silent."""
    us = userspace_module
    proc = _proc()
    cfg = {"untracked_grace_hours": 1, "untracked_reminder_hours": 10}
    t0 = datetime(2026, 6, 10, 12, 0, 0)
    _run_check(us, [], [proc], monkeypatch, now=t0, audit_cfg=cfg)  # warn; last_email=t0
    # Just past the 1h deadline, but the 10h reminder cadence is not met (due=False).
    _, s, k, _ = _run_check(us, [], [proc], monkeypatch,
                            now=t0 + timedelta(hours=1, minutes=1),
                            enforce=True, audit_cfg=cfg)
    assert k == [proc["pgid"]]               # the group was killed
    assert s and s[0]["kind"] == "killed"    # and the user was told, despite throttle


def test_state_self_heals_when_process_gone(userspace_module, monkeypatch):
    us = userspace_module
    proc = _proc()
    t0 = datetime(2026, 6, 10, 12, 0, 0)
    _, _, _, st1 = _run_check(us, [], [proc], monkeypatch, now=t0)
    assert len(st1) == 1
    _, _, _, st2 = _run_check(us, [], [], monkeypatch, now=t0 + timedelta(hours=1))
    assert st2 == {}


# --- OS-seam helpers -------------------------------------------------------

def test_get_process_user_not_truncated(userspace_module, monkeypatch):
    us = userspace_module
    seen = {}
    real_run = us.subprocess.run

    def fake_run(argv, *a, **k):
        seen["argv"] = list(argv)
        return real_run(argv, *a, **k)

    monkeypatch.setattr(us.subprocess, "run", fake_run)
    owner = us.get_process_user(os.getpid())
    assert "user:32=" in seen["argv"]      # wide format, not the 8-char `user=`
    assert owner and not owner.endswith("+")


def test_list_gpu_processes_with_owner_join(userspace_module, fake_gpus, monkeypatch):
    us = userspace_module
    _, write = fake_gpus
    write({
        "gpus": [{"index": 0, "name": "A", "uuid": "GPU-aaaa",
                  "memory_used_mb": 0, "memory_total_mb": 81920, "utilization": 0}],
        "compute_apps": [{"pid": 4242, "process_name": "python",
                          "gpu_uuid": "GPU-aaaa", "used_memory_mb": 40000}],
    })
    monkeypatch.setattr(us, "get_process_user", lambda pid: "mallory")
    monkeypatch.setattr(us, "get_process_pgid", lambda pid: 7777)
    out = us.list_gpu_processes_with_owner()
    assert len(out) == 1
    p = out[0]
    assert p["owner"] == "mallory"
    assert p["gpu_idx"] == 0
    assert p["pgid"] == 7777
    assert p["memory_mb"] == 40000        # python key, though state uses used_memory_mb


def test_enforce_kill_pgid_same_user(userspace_module):
    """Real kill of a same-user process group (no privilege needed)."""
    sleeper = subprocess.Popen(["sleep", "120"], start_new_session=True)
    try:
        pgid = os.getpgid(sleeper.pid)   # == sleeper.pid because of setsid
        assert userspace_module.enforce_kill_pgid(pgid) is True
        sleeper.wait(timeout=8)
        assert sleeper.poll() is not None
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait()


def test_notify_untracked_emails_offender_and_skips_unknown(userspace_module, monkeypatch):
    from test_email import _patch_smtp
    us = userspace_module
    cfg = {"notification_email": {"enabled": True, "smtp_server": "x", "smtp_port": 1,
                                  "username": "bot@lab.com", "password": "p"}}
    monkeypatch.setattr(us, "email_for_user",
                        lambda u: "mallory@lab.com" if u == "mallory" else None)
    sample = {"pid": 4242, "gpus": [0], "memory_mb": 40000, "name": "python"}
    deadline = datetime(2026, 6, 11, 12, 0)
    with _patch_smtp(monkeypatch) as fake:
        assert us.notify_untracked("mallory", sample, deadline, cfg, "warn") is True
    assert fake.instances[0].sent[0]["to"] == ["mallory@lab.com"]
    raw = fake.instances[0].sent[0]["msg"]
    assert "untracked" in raw.lower() and "mallory" in raw

    with _patch_smtp(monkeypatch) as fake2:
        assert us.notify_untracked("carol", sample, deadline, cfg, "warn") is False
    assert fake2.instances == []
