"""Unit tests for the GPU-rebind detector and its warn->kill state machine.

A "rebind" is a process attributable to a tracked gpuq job (by the job's cgroup
scope / child pgid / ancestry) but running on a physical GPU outside that job's
allocation. These exercise check_rebind() in-process with the OS seams
monkeypatched, time injected via `now=`. One CLI test (test_cli_audit.py) does a
real same-user kill to cover enforcement end to end.
"""
import json
import types
from datetime import datetime, timedelta


def _args(enforce=False):
    return types.SimpleNamespace(enforce=enforce, quiet=True)


def _proc(owner="mallory", gpu_idx=0, pgid=9999, memory_mb=40000,
          name="python", pid=4242, gpu_uuid="GPU-aaaa"):
    return {"owner": owner, "gpu_idx": gpu_idx, "pgid": pgid,
            "memory_mb": memory_mb, "name": name, "pid": pid,
            "gpu_uuid": gpu_uuid}


def _job(us, gpus, *, id=7, user="mallory", host=None, child_pgid=9999,
         started="2020-01-01T00:00:00", **extra):
    j = {"id": id, "user": user, "host": host or us.HOST, "gpus": gpus,
         "child_pgid": child_pgid, "started_at": started}
    j.update(extra)
    return j


def _run_rebind(us, running, procs, monkeypatch, *, audit_cfg=None,
                enforce=False, now=None, kill_ok=True):
    """Drive check_rebind with controlled procs; capture emails + kills."""
    monkeypatch.setattr(us, "list_gpu_processes_with_owner", lambda: procs)
    sent = []
    monkeypatch.setattr(
        us, "notify_rebind",
        lambda owner, sample, deadline, config, kind: sent.append(
            {"owner": owner, "kind": kind, "sample": sample}) or True)
    killed = []
    monkeypatch.setattr(
        us, "enforce_kill_pgid", lambda pgid, **kw: (killed.append(pgid) or kill_ok))
    cfg = dict(audit_cfg or {})
    breaches = us.check_rebind(running, cfg, _args(enforce), {"audit": cfg}, now=now)
    state = (json.loads(us.REBIND_STATE_FILE.read_text())
             if us.REBIND_STATE_FILE.exists() else {})
    return breaches, sent, killed, state


# --- detection / filtering -------------------------------------------------

def test_rebind_flagged_when_process_on_wrong_gpu(userspace_module, monkeypatch):
    us = userspace_module
    running = [_job(us, [1], child_pgid=9999)]
    breaches, sent, killed, state = _run_rebind(
        us, running, [_proc(pgid=9999, gpu_idx=0)], monkeypatch)
    assert any("REBIND" in b and "mallory" in b for b in breaches)
    assert len(sent) == 1 and sent[0]["kind"] == "warn"
    assert killed == [] and len(state) == 1
    # the sample records both the allocation and where it actually ran
    assert sent[0]["sample"]["allocated"] == [1]
    assert sent[0]["sample"]["gpus"] == [0]


def test_correctly_placed_job_no_breach(userspace_module, monkeypatch):
    us = userspace_module
    running = [_job(us, [0], child_pgid=9999)]
    breaches, sent, _, state = _run_rebind(
        us, running, [_proc(pgid=9999, gpu_idx=0)], monkeypatch)
    assert breaches == [] and sent == [] and state == {}


def test_multi_gpu_allocation_member_ok(userspace_module, monkeypatch):
    us = userspace_module
    running = [_job(us, [1, 3], child_pgid=9999)]
    # a worker on an allocated member is fine ...
    b1, s1, _, _ = _run_rebind(us, running, [_proc(pgid=9999, gpu_idx=3)], monkeypatch)
    assert b1 == [] and s1 == []
    # ... but one outside the whole set is a rebind.
    b2, s2, _, _ = _run_rebind(us, running, [_proc(pgid=9999, gpu_idx=2)], monkeypatch)
    assert any("REBIND" in b for b in b2) and len(s2) == 1


def test_stacked_same_user_jobs_no_breach(userspace_module, monkeypatch):
    """Two of a user's jobs stacked on one card: each process is on a GPU in its
    OWN job's allocation, so neither is a rebind (the policy-critical case)."""
    us = userspace_module
    running = [_job(us, [0], id=1, child_pgid=100),
               _job(us, [0], id=2, child_pgid=200)]
    procs = [_proc(pgid=100, gpu_idx=0, pid=100),
             _proc(pgid=200, gpu_idx=0, pid=200)]
    breaches, sent, _, state = _run_rebind(us, running, procs, monkeypatch)
    assert breaches == [] and sent == [] and state == {}


def test_unattributable_proc_not_rebind(userspace_module, monkeypatch):
    """A process matching no tracked job is the untracked detector's concern."""
    us = userspace_module
    running = [_job(us, [1], child_pgid=9999)]
    breaches, sent, _, _ = _run_rebind(
        us, running, [_proc(pgid=12345, gpu_idx=0)], monkeypatch)
    assert breaches == [] and sent == []


def test_legacy_job_without_pgid_skipped(userspace_module, monkeypatch):
    """A pre-feature job (no scope/pgid/child_pid) is unattributable -> skipped."""
    us = userspace_module
    running = [{"id": 1, "user": "mallory", "host": us.HOST, "gpus": [1],
                "started_at": "2020-01-01T00:00:00"}]
    breaches, sent, _, _ = _run_rebind(
        us, running, [_proc(pgid=12345, gpu_idx=0)], monkeypatch)
    assert breaches == [] and sent == []


def test_unattributable_fields_skipped(userspace_module, monkeypatch):
    us = userspace_module
    running = [_job(us, [1], child_pgid=9999)]
    procs = [_proc(gpu_idx=None), _proc(pgid=None), _proc(owner=None)]
    breaches, sent, _, _ = _run_rebind(us, running, procs, monkeypatch)
    assert breaches == [] and sent == []


def test_below_min_memory_skipped(userspace_module, monkeypatch):
    us = userspace_module
    running = [_job(us, [1], child_pgid=9999)]
    breaches, _, _, _ = _run_rebind(
        us, running, [_proc(pgid=9999, gpu_idx=0, memory_mb=100)], monkeypatch)
    assert breaches == []


def test_foreign_host_job_not_attributed(userspace_module, monkeypatch):
    us = userspace_module
    running = [_job(us, [1], host="other-node", child_pgid=9999)]
    breaches, sent, _, _ = _run_rebind(
        us, running, [_proc(pgid=9999, gpu_idx=0)], monkeypatch)
    assert breaches == [] and sent == []


def test_submit_grace_suppresses_rebind(userspace_module, monkeypatch):
    us = userspace_module
    now = datetime(2026, 6, 10, 12, 0, 0)
    running = [_job(us, [1], child_pgid=777, started=now.isoformat())]
    cfg = {"rebind_grace_seconds": 120}
    b1, s1, _, _ = _run_rebind(
        us, running, [_proc(pgid=777, gpu_idx=0)], monkeypatch, now=now, audit_cfg=cfg)
    assert b1 == [] and s1 == []           # within grace
    b2, s2, _, _ = _run_rebind(
        us, running, [_proc(pgid=777, gpu_idx=0)], monkeypatch,
        now=now + timedelta(seconds=200), audit_cfg=cfg)
    assert any("REBIND" in b for b in b2) and len(s2) == 1   # past grace


def test_scope_attribution_judges_allocation(userspace_module, monkeypatch):
    """A worker matched by cgroup scope is still judged against its allocation."""
    us = userspace_module
    running = [{"id": 1, "user": "yyq", "host": us.HOST, "gpus": [1],
                "cgroup_scope": "gpuq-1.scope", "started_at": "2020-01-01T00:00:00"}]
    monkeypatch.setattr(us, "_proc_in_scope",
                        lambda pid, scopes: "gpuq-1.scope" in scopes)
    breaches, sent, _, _ = _run_rebind(
        us, running, [_proc(owner="yyq", gpu_idx=0, pgid=999, pid=999)], monkeypatch)
    assert any("REBIND" in b for b in breaches) and len(sent) == 1


# --- state machine ---------------------------------------------------------

def test_warn_then_remind_then_overdue_kill(userspace_module, monkeypatch):
    us = userspace_module
    running = [_job(us, [1], child_pgid=9999)]
    proc = _proc(pgid=9999, gpu_idx=0)
    cfg = {"rebind_grace_hours": 24, "rebind_reminder_hours": 6}
    t0 = datetime(2026, 6, 10, 12, 0, 0)

    _, s1, _, _ = _run_rebind(us, running, [proc], monkeypatch, now=t0, audit_cfg=cfg)
    assert s1 and s1[0]["kind"] == "warn"

    b2, s2, _, _ = _run_rebind(us, running, [proc], monkeypatch,
                               now=t0 + timedelta(hours=1), audit_cfg=cfg)
    assert s2 == [] and any("deadline" in b for b in b2)

    _, s3, _, _ = _run_rebind(us, running, [proc], monkeypatch,
                              now=t0 + timedelta(hours=7), audit_cfg=cfg)
    assert s3 and s3[0]["kind"] == "remind"

    b4, s4, k4, _ = _run_rebind(us, running, [proc], monkeypatch,
                                now=t0 + timedelta(hours=25), enforce=True,
                                audit_cfg=cfg)
    assert any("PAST DEADLINE" in b and "enforcing" in b for b in b4)
    assert k4 == [proc["pgid"]]
    assert s4 and s4[0]["kind"] == "killed"


def test_overdue_email_when_kill_fails(userspace_module, monkeypatch):
    us = userspace_module
    running = [_job(us, [1], child_pgid=9999)]
    proc = _proc(pgid=9999, gpu_idx=0)
    cfg = {"rebind_grace_hours": 24}
    t0 = datetime(2026, 6, 10, 12, 0, 0)
    _run_rebind(us, running, [proc], monkeypatch, now=t0, audit_cfg=cfg)  # warn
    b, s, k, _ = _run_rebind(us, running, [proc], monkeypatch,
                             now=t0 + timedelta(hours=25), enforce=True,
                             audit_cfg=cfg, kill_ok=False)
    assert any("PAST DEADLINE" in x for x in b)
    assert k == [proc["pgid"]]              # kill attempted
    assert s and s[0]["kind"] == "overdue"  # reported as not-killed


def test_past_deadline_without_enforce_escalates_no_kill(userspace_module, monkeypatch):
    us = userspace_module
    running = [_job(us, [1], child_pgid=9999)]
    proc = _proc(pgid=9999, gpu_idx=0)
    cfg = {"rebind_grace_hours": 24}
    t0 = datetime(2026, 6, 10, 12, 0, 0)
    _run_rebind(us, running, [proc], monkeypatch, now=t0, audit_cfg=cfg)
    b, s, k, _ = _run_rebind(us, running, [proc], monkeypatch,
                             now=t0 + timedelta(hours=25), enforce=False,
                             audit_cfg=cfg)
    assert any("PAST DEADLINE" in x and "escalated to admin" in x for x in b)
    assert k == []
    assert s and s[0]["kind"] == "overdue"


def test_state_self_heals_when_process_gone(userspace_module, monkeypatch):
    us = userspace_module
    running = [_job(us, [1], child_pgid=9999)]
    proc = _proc(pgid=9999, gpu_idx=0)
    t0 = datetime(2026, 6, 10, 12, 0, 0)
    _, _, _, st1 = _run_rebind(us, running, [proc], monkeypatch, now=t0)
    assert len(st1) == 1
    _, _, _, st2 = _run_rebind(us, running, [], monkeypatch,
                               now=t0 + timedelta(hours=1))
    assert st2 == {}


# --- email -----------------------------------------------------------------

def test_notify_rebind_emails_offender_and_skips_unknown(userspace_module, monkeypatch):
    from test_email import _patch_smtp
    us = userspace_module
    cfg = {"notification_email": {"enabled": True, "smtp_server": "x", "smtp_port": 1,
                                  "username": "bot@lab.com", "password": "p"}}
    monkeypatch.setattr(us, "email_for_user",
                        lambda u: "mallory@lab.com" if u == "mallory" else None)
    sample = {"pid": 4242, "gpus": [0], "allocated": [1], "job_id": 7,
              "memory_mb": 40000, "name": "python"}
    deadline = datetime(2026, 6, 11, 12, 0)
    with _patch_smtp(monkeypatch) as fake:
        assert us.notify_rebind("mallory", sample, deadline, cfg, "warn") is True
    assert fake.instances[0].sent[0]["to"] == ["mallory@lab.com"]
    raw = fake.instances[0].sent[0]["msg"]
    assert "rebind" in raw.lower() and "mallory" in raw     # subject (7-bit header)
    # The body carries an em-dash, so MIMEText base64-encodes it; decode to check.
    import email as emaillib
    body = emaillib.message_from_string(raw).get_payload(decode=True).decode()
    assert "GPU [1]" in body and "GPU [0]" in body          # allocated vs actual

    with _patch_smtp(monkeypatch) as fake2:
        assert us.notify_rebind("carol", sample, deadline, cfg, "warn") is False
    assert fake2.instances == []


def test_builtin_grace_default_is_fifteen_minutes(userspace_module, monkeypatch):
    """An audit block with no rebind_grace_hours must fall back to the
    15-minute policy, not the old 24h one. Pinned from both sides so a stale
    fallback in either direction fails loudly."""
    us = userspace_module
    running = [_job(us, [1], child_pgid=9999)]
    proc = _proc(pgid=9999, gpu_idx=0)
    t0 = datetime(2026, 6, 10, 12, 0, 0)
    _run_rebind(us, running, [proc], monkeypatch, now=t0, audit_cfg={})  # warn

    # +14m: still inside the window -> no kill.
    _, _, k_early, _ = _run_rebind(us, running, [proc], monkeypatch,
                                   now=t0 + timedelta(minutes=14),
                                   enforce=True, audit_cfg={})
    assert k_early == []

    # +16m: past the deadline -> killed, and the user is told.
    b, s, k, _ = _run_rebind(us, running, [proc], monkeypatch,
                             now=t0 + timedelta(minutes=16),
                             enforce=True, audit_cfg={})
    assert any("PAST DEADLINE" in x for x in b)
    assert k == [proc["pgid"]]
    assert s and s[0]["kind"] == "killed"
