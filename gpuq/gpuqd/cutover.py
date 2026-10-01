"""The data steps of install_v3.sh, in Python rather than shell.

    python3 -m gpuqd.cutover STEP ARGS...

  waiting  LEGACY            list the old queue's live waiting submits; exit 1 if any
  snapshot LEGACY STATE      the old running jobs -> STATE/legacy.json (once; transition.py)
  ledger   LEGACY STATE      the old ledger -> STATE/usage.jsonl (once), so fair-share and
                             `gpuq history` start with the full history
  mail     OLD_CONFIG OUT    the "notification_email" section -> OUT, root only (once)
  jobs     STATE             how many jobs gpuqd has queued or running
  old      STATE             how many of the cutover's old jobs are still running
  unledger LEGACY STATE      rollback: append gpuqd's records to the previous gpuq's ledger

LEGACY is the previous gpuq's /var/lib/gpu_queue: every member can write there, so it is read
with safeio and written only through a checked file descriptor.
"""
import json
import os
import stat
import sys
from pathlib import Path

from .procinfo import read_proc
from .safeio import read_text

PROC = os.environ.get("GPUQD_PROC", "/proc")


def _list(path):
    try:
        data = json.loads(read_text(path))
    except FileNotFoundError:
        return []
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def _alive(e):
    try:
        info = read_proc(int(e["pid"]), PROC)
    except (KeyError, TypeError, ValueError):
        return False
    return info is not None and (e.get("pid_start") is None or info.start == int(e["pid_start"]))


def _lines(legacy):
    d = Path(legacy)
    out = []
    for p in sorted(d.glob("usage-*.jsonl")) + [d / "usage.jsonl"]:
        try:
            out += [l for l in read_text(p).splitlines() if l.strip()]
        except FileNotFoundError:
            pass
    return out


def _write_new(path, text, mode):
    """Create `path` with `text`; never overwrite."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(path, mode)


def waiting(legacy):
    alive = [e for e in _list(Path(legacy) / "jobs.json") if _alive(e)]
    for e in alive:
        print(f"  job {e.get('id')} of {e.get('user')}, waiting since {e.get('submitted_at')}")
    return 1 if alive else 0


def snapshot(legacy, state):
    out = Path(state) / "legacy.json"
    if out.exists():
        print(f"{out} already taken; kept")
        return 0
    running = [e for e in _list(Path(legacy) / "running.json") if _alive(e)]
    _write_new(out, json.dumps(running), 0o600)
    print(f"{len(running)} job(s) started by the previous gpuq keep their cards until they end")
    return 0


def ledger(legacy, state):
    out = Path(state) / "usage.jsonl"
    if out.exists():
        print(f"{out} exists; kept")
        return 0
    lines = _lines(legacy)
    _write_new(out, "".join(l + "\n" for l in lines), 0o644)
    print(f"copied {len(lines)} ledger line(s) to {out}")
    return 0


def mail(old_config, out):
    if Path(out).exists():
        print(f"{out} exists; kept")
        return 0
    try:
        cfg = json.loads(read_text(old_config)).get("notification_email") or {}
    except (OSError, ValueError, AttributeError):
        cfg = {}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_new(out, json.dumps(cfg, indent=2) + "\n", 0o600)
    print(f"mail settings -> {out} (root only); email {'on' if cfg.get('enabled') else 'off'}")
    return 0


def _state(state):
    try:
        with open(Path(state) / "state.json") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def jobs(state):
    print(len(_state(state).get("jobs") or []))
    return 0


def old(state):
    try:
        with open(Path(state) / "legacy.json") as f:
            snap = json.load(f)
    except (OSError, ValueError):
        snap = []
    done = set(_state(state).get("legacy_done") or ())
    print(len([e for e in snap if e.get("id") not in done and _alive(e)]))
    return 0


def unledger(legacy, state):
    """Rollback: the previous gpuq charges quota from its own ledger, so give it the
    jobs gpuqd ran. Appends through a descriptor checked to be a regular file:
    never through a planted symlink or FIFO."""
    have = set(_lines(legacy))
    try:
        mine = [l for l in (Path(state) / "usage.jsonl").read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        mine = []
    new = [l for l in mine if l not in have]
    if not new:
        print("the previous gpuq's ledger already has every record")
        return 0
    path = Path(legacy) / "usage.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                 0o664)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            print(f"{path} is not a regular file; not touched", file=sys.stderr)
            return 1
        os.write(fd, "".join(l + "\n" for l in new).encode())
    finally:
        os.close(fd)
    print(f"appended {len(new)} record(s) to {path}")
    return 0


STEPS = {"waiting": waiting, "snapshot": snapshot, "ledger": ledger, "mail": mail,
         "jobs": jobs, "old": old, "unledger": unledger}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in STEPS:
        print(__doc__, file=sys.stderr)
        return 2
    return STEPS[argv[0]](*argv[1:])


if __name__ == "__main__":
    sys.exit(main())
