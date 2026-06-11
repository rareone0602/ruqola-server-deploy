"""Email tests: capture outbound SMTP via a fake smtplib.SMTP.

Doesn't run a real SMTP server (avoids STARTTLS + cert wrangling). Exercises
in-process via the `userspace_module` fixture.
"""
import smtplib
from contextlib import contextmanager


class _FakeSMTP:
    """Records every SMTP method call. Used as a drop-in for smtplib.SMTP."""

    instances = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.calls = []
        self.sent = []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        self.calls.append(("ctx_enter",))
        return self

    def __exit__(self, *exc):
        self.calls.append(("ctx_exit",))
        return False

    def starttls(self):
        self.calls.append(("starttls",))

    def login(self, user, pw):
        self.calls.append(("login", user, pw))

    def sendmail(self, sender, recipients, msg):
        self.calls.append(("sendmail", sender, tuple(recipients)))
        self.sent.append({"from": sender, "to": list(recipients), "msg": msg})


@contextmanager
def _patch_smtp(monkeypatch):
    _FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    yield _FakeSMTP


def test_send_email_skipped_when_disabled(userspace_module, monkeypatch):
    with _patch_smtp(monkeypatch) as fake:
        ok = userspace_module.send_email(
            "alice@lab.com", "subj", "body",
            {"notification_email": {"enabled": False}},
        )
        assert ok is False
        assert fake.instances == []


def test_send_email_skipped_when_no_recipient(userspace_module, monkeypatch):
    with _patch_smtp(monkeypatch) as fake:
        ok = userspace_module.send_email(
            "", "subj", "body",
            {"notification_email": {"enabled": True, "smtp_server": "x"}},
        )
        assert ok is False
        assert fake.instances == []


def test_send_email_full_envelope(userspace_module, monkeypatch):
    config = {"notification_email": {
        "enabled": True,
        "smtp_server": "smtp.example.com",
        "smtp_port": 1234,
        "username": "bot@lab.com",
        "password": "hunter2",
    }}
    with _patch_smtp(monkeypatch) as fake:
        ok = userspace_module.send_email(
            "alice@lab.com", "[gpuq] hello", "body line 1\nbody line 2",
            config,
        )
    assert ok is True
    assert len(fake.instances) == 1
    smtp = fake.instances[0]
    assert smtp.host == "smtp.example.com"
    assert smtp.port == 1234
    # Order matters: starttls, login, sendmail.
    method_seq = [c[0] for c in smtp.calls]
    assert "starttls" in method_seq
    assert method_seq.index("starttls") < method_seq.index("login")
    assert method_seq.index("login") < method_seq.index("sendmail")
    assert smtp.sent[0]["from"] == "bot@lab.com"
    assert smtp.sent[0]["to"] == ["alice@lab.com"]
    raw = smtp.sent[0]["msg"]
    assert "Subject: [gpuq] hello" in raw
    assert "To: alice@lab.com" in raw
    assert "body line 1" in raw
    assert "body line 2" in raw


def test_quota_exceeded_email_uses_gecos(userspace_module, monkeypatch):
    """notify_quota_exceeded falls back to the GECOS email when no override given."""
    config = {"notification_email": {
        "enabled": True, "smtp_server": "smtp.example.com",
        "smtp_port": 587, "username": "bot@lab.com", "password": "x",
    }}
    monkeypatch.setattr(userspace_module, "email_for_user",
                        lambda u: "alice@lab.com" if u == "alice" else None)
    with _patch_smtp(monkeypatch) as fake:
        userspace_module.notify_quota_exceeded(
            user="alice", notify_email=None,
            used_hours=200.0, requested_hours=10.0, budget=100.0,
            config=config,
        )
    assert len(fake.instances) == 1
    sent = fake.instances[0].sent[0]
    assert sent["to"] == ["alice@lab.com"]
    raw = sent["msg"]
    assert "deprioritized" in raw.lower()
    assert "200.0" in raw and "100.0" in raw and "alice" in raw
    assert "[gpuq] alice: GPU-hour quota exceeded" in raw


def test_quota_exceeded_no_email_when_no_gecos_email(userspace_module, monkeypatch):
    config = {"notification_email": {"enabled": True, "smtp_server": "x"}}
    monkeypatch.setattr(userspace_module, "email_for_user", lambda u: None)
    with _patch_smtp(monkeypatch) as fake:
        userspace_module.notify_quota_exceeded(
            user="carol", notify_email=None,
            used_hours=200.0, requested_hours=10.0, budget=100.0,
            config=config,
        )
    assert fake.instances == []     # no SMTP session at all


def test_quota_exceeded_explicit_notify_email_wins(userspace_module, monkeypatch):
    """An explicit --notify EMAIL on submit overrides GECOS resolution."""
    config = {"notification_email": {
        "enabled": True, "smtp_server": "smtp.example.com",
        "smtp_port": 587, "username": "bot@lab.com", "password": "x",
    }}
    # GECOS would resolve alice@lab.com, but the explicit address must win.
    monkeypatch.setattr(userspace_module, "email_for_user", lambda u: "alice@lab.com")
    with _patch_smtp(monkeypatch) as fake:
        userspace_module.notify_quota_exceeded(
            user="alice", notify_email="alice-personal@gmail.com",
            used_hours=200.0, requested_hours=10.0, budget=100.0,
            config=config,
        )
    sent = fake.instances[0].sent[0]
    assert sent["to"] == ["alice-personal@gmail.com"]


def test_email_for_user_parses_gecos(userspace_module, monkeypatch):
    """email_for_user pulls the address from GECOS (5th comma-subfield or whole)."""
    import types
    table = {
        "alice": "alice:x:1001:1001:Alice A,,,,alice@lab.com:/home/alice:/bin/bash\n",
        "bob":   "bob:x:1002:1002:bob@e.ntu.edu.sg:/home/bob:/bin/bash\n",
        "carol": "carol:x:1003:1003:Carol,,,:/home/carol:/bin/bash\n",
    }

    def fake_run(argv, *a, **k):
        user = argv[-1]
        if user not in table:
            raise userspace_module.subprocess.CalledProcessError(2, argv)
        return types.SimpleNamespace(stdout=table[user], returncode=0)

    monkeypatch.setattr(userspace_module.subprocess, "run", fake_run)
    assert userspace_module.email_for_user("alice") == "alice@lab.com"    # ,,,,addr
    assert userspace_module.email_for_user("bob") == "bob@e.ntu.edu.sg"   # whole field
    assert userspace_module.email_for_user("carol") is None              # no email in GECOS
    assert userspace_module.email_for_user("nosuch") is None             # getent fails
    assert userspace_module.email_for_user("") is None
