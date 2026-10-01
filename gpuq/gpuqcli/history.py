"""gpuq history: recent jobs from the ledger, as the previous gpuq showed them.

The ledger (/var/lib/gpuq/usage.jsonl, the v2 format) is readable by
everyone, so this needs no daemon. Rotated usage-*.jsonl files are read first.
"""
import getpass
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from .common import STATE_DIR, printable


def records(state_dir=STATE_DIR):
    d = Path(state_dir)
    for path in sorted(d.glob("usage-*.jsonl")) + [d / "usage.jsonl"]:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        rec.setdefault("event", "end")
                        yield rec
        except OSError:
            continue


def _float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _secs(sec):
    try:
        sec = int(sec)
    except (TypeError, ValueError):
        return "-"
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def _hours(h):
    try:
        return str(timedelta(seconds=int(float(h) * 3600)))
    except (TypeError, ValueError):
        return "-"


def _when(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%m-%d %H:%M")
    except (TypeError, ValueError):
        return "-"


def _trunc(text, width):
    text = text or ""
    return text if len(text) <= width else text[:width - 1] + "…"


def cmd_history(args):
    me = os.environ.get("USER") or getpass.getuser()
    who = None if args.all else (args.user or me)
    recs = [r for r in records() if who is None or r.get("user") == who]
    recs.sort(key=lambda r: str(r.get("ended_at") or r.get("at") or ""))
    if not args.events:
        recs = [r for r in recs if r.get("event") == "end"]
    if args.limit > 0:
        recs = recs[-args.limit:]
    if args.json:
        for r in recs:
            print(json.dumps(r, separators=(",", ":")))
        return 0
    if not recs:
        print(f"no ledger records{f' for {who}' if who else ''} yet.")
        return 0
    host = os.uname().nodename
    print(f"Job history on {host} — {'all users' if who is None else who}, oldest first "
          "(see `gpuq history -h` for filters):")
    print(f"{'JOB':<11} " + (f"{'USER':<10} " if args.all else "")
          + f"{'ENDED':<12} {'WAIT':>7} {'RUNTIME':>9} {'GPUS':<5} "
            f"{'GPU-H':>7} {'EXIT':>4} {'RESULT':<10} NAME/COMMAND")
    # Each row is made printable whole (todo E9): the ledger lines copied at cutover
    # came from a file every member could write, so any field may hold anything.
    for r in recs:
        name = r.get("name") or ""
        command = r.get("command") or ""
        label = f"{name}: {command}" if name else (command or "-")
        user = f"{(r.get('user') or '?'):<10} " if args.all else ""
        if r.get("event") != "end":
            want = r.get("gpus_requested")
            print(printable(
                f"{str(r.get('id') or '-'):<11} {user}{_when(r.get('at')):<12} "
                f"{_secs(r.get('wait_sec')):>7} {'-':>9} {('?' if want is None else want):<5} "
                f"{'-':>7} {'-':>4} {r['event'] + ' (' + str(r.get('reason', '?')) + ')':<10} "
                f"{_trunc(label, 48)}"))
            continue
        gpus = ",".join(map(str, r.get("gpus") or [])) or "-"
        code = r.get("exit_code")
        result = (r.get("end_reason") or "-") + ("*" if r.get("synthetic") else "")
        print(printable(
            f"{str(r.get('id') or '-'):<11} {user}{_when(r.get('ended_at')):<12} "
            f"{_secs(r.get('queue_wait_sec')):>7} {_hours(r.get('elapsed_hours')):>9} "
            f"{gpus:<5} {_float(r.get('gpu_hours')):>7.2f} {('-' if code is None else code):>4} "
            f"{result:<10} {_trunc(label, 48)}"))
    if any(r.get("synthetic") for r in recs):
        print("(*synthetic record: the job ended without normal accounting; "
              "charged up to when it was found gone)")
    return 0
