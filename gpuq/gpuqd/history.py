"""Fair-share history: the ledger (usage.jsonl) as card-hours held.

Only finished jobs are in the ledger; the caller adds the jobs still running.
The ledger is read, never written, and re-read only when a file changes.
"""
import json
from datetime import datetime
from pathlib import Path

from scheduler.fairshare import decayed_usage, held_intervals
from scheduler.model import MAX_RUNTIME_H, Job, Running


def _epoch(iso):
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return None


def _run(rec):
    """(Running, end_t) for one finished-job record, or None if it held nothing."""
    if rec.get("event", "end") != "end":
        return None
    start, stop = _epoch(rec.get("started_at")), _epoch(rec.get("ended_at"))
    try:
        cards = tuple(int(c) for c in rec.get("gpus") or ())
    except (TypeError, ValueError):
        return None
    if start is None or stop is None or stop < start or not cards or not rec.get("user"):
        return None
    job = Job(str(rec.get("id")), str(rec["user"]), len(cards), 0.0, MAX_RUNTIME_H, start)
    return Running(job, cards, start), stop


class Ledger:
    """The ledger files: rotated usage-*.jsonl oldest first, then usage.jsonl."""

    def __init__(self, queue_dir):
        self.dir = Path(queue_dir)
        self.runs = []          # [(Running, end_t)]
        self.ended = {}         # job id -> (Running, end_t), the latest record per id
        self.bad_lines = 0      # lines that are not JSON (skipped, never fatal)
        self.parses = 0
        self._seen = None

    def _files(self):
        return sorted(self.dir.glob("usage-*.jsonl")) + [self.dir / "usage.jsonl"]

    def _signature(self):
        sig = []
        for p in self._files():
            try:
                st = p.stat()
            except OSError:
                continue
            sig.append((p.name, st.st_size, st.st_mtime_ns))
        return tuple(sig)

    def refresh(self):
        sig = self._signature()
        if sig == self._seen:
            return
        runs, bad = [], 0
        for p in self._files():
            try:
                with open(p) as f:
                    lines = f.readlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                run = _run(rec) if isinstance(rec, dict) else None
                if run is not None:
                    runs.append(run)
        self.runs, self.bad_lines, self._seen = runs, bad, sig
        self.ended = {run.job.id: (run, end) for run, end in runs}
        self.parses += 1


def usage(runs, running_now, now):
    """user -> card-hours held, fading by half each week (rule 1)."""
    history = list(runs) + [(r, None) for r in running_now]
    return decayed_usage(held_intervals(history, now), now)
