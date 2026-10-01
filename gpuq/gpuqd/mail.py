"""Email from gpuqd. Mail goes only to the account's own address (todo E8), and
the SMTP password lives in a root-only file, /etc/gpuq/mail.json, never in a
file members can read (todo B1). Sending happens on a background thread, so a
slow mail server never stalls scheduling.

/etc/gpuq/mail.json is the previous gpuq's config's "notification_email" section:
{"enabled": true, "smtp_server": ..., "smtp_port": 587, "username": ..., "password": ...}
"""
import json
import pwd
import re
import smtplib
import sys
import threading
from email.mime.text import MIMEText

MAIL_CONFIG = "/etc/gpuq/mail.json"

_ADDRESS = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def address_of(user):
    """The address in the account's GECOS field, as the previous gpuq read it."""
    try:
        gecos = pwd.getpwnam(user).pw_gecos
    except KeyError:
        return None
    m = _ADDRESS.search(gecos or "")
    return m.group(0) if m else None


def load(path=MAIL_CONFIG):
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


class Mailer:
    def __init__(self, path=MAIL_CONFIG, host=""):
        self.cfg = load(path)
        self.host = host

    @property
    def enabled(self):
        return bool(self.cfg.get("enabled"))

    def send(self, user, subject, body):
        to = address_of(user)
        if not self.enabled or not to:
            return
        threading.Thread(target=self._send, args=(to, subject, body), daemon=True).start()

    def _send(self, to, subject, body):
        cfg = self.cfg
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = cfg.get("username", "")
            msg["To"] = to
            with smtplib.SMTP(cfg["smtp_server"], cfg.get("smtp_port", 587), timeout=30) as s:
                s.starttls()
                s.login(cfg["username"], cfg["password"])
                s.sendmail(cfg["username"], [to], msg.as_string())
        except Exception as e:      # mail must never take the daemon down
            print(f"mail to {subject!r} failed: {type(e).__name__}: {e}", file=sys.stderr)
