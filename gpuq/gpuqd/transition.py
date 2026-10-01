"""Jobs the previous gpuq was running at cutover. They finish as they would have.

install_v3.sh copies the previous gpuq's running.json to /var/lib/gpuq/legacy.json (root
only) at the moment of cutover. Each of those jobs holds its cards while its
supervisor process (pid + kernel start time) lives. gpuqd never trusts that
running.json after that: the folder stays writable by every member until it is
retired, so anything added there later is not honoured, only reported.

When a supervisor ends, its job's end record is copied from the previous gpuq's ledger,
where the old client wrote it, into gpuqd's ledger. Times are clamped to what
gpuqd saw (from the job's start to the moment its supervisor was found gone),
since that ledger is writable by every member too. If no record is found, an
end record marked synthetic is written instead, as the previous gpuq's reaper did.
"""
import json
from datetime import datetime
from pathlib import Path

from . import legacy
from .jobs import iso
from .procinfo import read_proc
from .safeio import read_text

LEGACY_DIR = "/var/lib/gpu_queue"


def _epoch(s):
    try:
        return datetime.fromisoformat(s).timestamp()
    except (TypeError, ValueError):
        return None


class Transition:
    def __init__(self, snapshot_path, legacy_dir=LEGACY_DIR, host="", proc="/proc", done=()):
        self.legacy_dir = Path(legacy_dir)
        self.host = host
        self.proc = proc
        self.done = set(done)           # ids whose end is already recorded
        self.reported = set()           # stray entry ids already logged
        try:
            with open(snapshot_path) as f:
                snap = json.load(f)
        except (OSError, ValueError):
            snap = []
        self.snapshot = [e for e in snap if isinstance(e, dict) and e.get("id") is not None
                         and e.get("host", host) == host] if isinstance(snap, list) else []

    def running(self):
        """Snapshot entries whose end is not yet recorded."""
        return [e for e in self.snapshot if e["id"] not in self.done]

    def _alive(self, e):
        try:
            info = read_proc(int(e["pid"]), self.proc)
        except (KeyError, TypeError, ValueError):
            return False
        if info is None:
            return False
        return e.get("pid_start") is None or info.start == int(e["pid_start"])

    def ended(self, now):
        """End records for snapshot jobs whose supervisor is gone (marks them done)."""
        records = []
        for e in self.running():
            if self._alive(e):
                continue
            self.done.add(e["id"])
            records.append(self._end_record(e, now))
        return records

    def _old_record(self, e):
        """The old client's end record for this job, or None."""
        try:
            text = read_text(self.legacy_dir / "usage.jsonl")
        except OSError:
            return None
        found = None
        for line in text.splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if (isinstance(rec, dict) and rec.get("event") == "end"
                    and rec.get("id") == e["id"] and rec.get("user") == e.get("user")):
                found = rec
        return found

    def _end_record(self, e, now):
        started = _epoch(e.get("started_at")) or now
        rec = self._old_record(e)
        if rec is None:
            rec = {"v": 2, "event": "end", "id": e["id"], "user": e.get("user"),
                   "host": e.get("host", self.host), "command": e.get("command"),
                   "name": e.get("name"), "gpus_requested": e.get("gpu_count"),
                   "devices": e.get("devices"), "memory_gb": e.get("memory_gb"),
                   "max_time_hours": e.get("max_time_hours"), "priority": e.get("priority"),
                   "over_quota_at_submit": e.get("over_quota_at_submit", False),
                   "submitted_at": e.get("submitted_at"), "queue_wait_sec": e.get("queue_wait_sec"),
                   "exit_code": None, "end_reason": "lost", "synthetic": True}
            ended = now
        else:
            ended = min(max(_epoch(rec.get("ended_at")) or now, started), now)
        rec = dict(rec)
        gpus = list(e.get("gpus") or [])
        hours = round((ended - started) / 3600.0, 4)
        rec.update({"id": e["id"], "user": e.get("user"), "gpus": gpus,
                    "started_at": iso(started), "ended_at": iso(ended),
                    "elapsed_hours": hours, "gpu_hours": round(hours * len(gpus), 4)})
        return rec

    def strays(self):
        """Entries in the previous gpuq's state files that gpuqd does not honour: someone is
        still running an old copy of gpuq. Each is returned once, for the log."""
        try:
            state = legacy.read_state(self.legacy_dir, self.host)
        except OSError:
            return []
        if state is None:
            return []
        known = {e["id"] for e in self.snapshot}
        out = []
        for kind, entries in zip(("running", "queued"), state):
            for e in entries:
                if e.get("id") in known or e.get("id") in self.reported:
                    continue
                self.reported.add(e.get("id"))
                out.append((kind, e))
        return out
