"""gpuqd's files under /var/lib/gpuq. Only gpuqd (root) writes here, so there is
no lock, no group-writable file, and no path a member can plant (todo E1, E4).

  state.json      the queue and the running jobs, with their environments (0600)
  legacy.json     the previous gpuq's running.json at cutover, from install_v3.sh (transition.py)
  usage.jsonl     the ledger: the v2 format, readable by all (0644)
  exits/ID.json   how a job ended, written by its unit's exit hook (dir 0700)
  launch/ID.json  what a starting job runs, readable only by its owner (dir 0711)
"""
import json
import os
from pathlib import Path

STATE_DIR = os.environ.get("GPUQD_STATE_DIR", "/var/lib/gpuq")


def _write_atomic(path, data, mode):
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, mode)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.chmod(tmp, mode)
    os.replace(tmp, path)


class Store:
    def __init__(self, root=STATE_DIR):
        self.root = Path(root)
        self.state = self.root / "state.json"
        self.ledger = self.root / "usage.jsonl"
        self.exits = self.root / "exits"
        self.launch = self.root / "launch"

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o755)
        for d, mode in ((self.exits, 0o700), (self.launch, 0o711)):
            d.mkdir(exist_ok=True)
            os.chmod(d, mode)

    # -- state --------------------------------------------------------------------

    def load(self):
        try:
            with open(self.state) as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        return {k: list(data.get(k) or []) for k in ("jobs", "promised", "legacy_done")}

    def save(self, jobs, promised, legacy_done=()):
        data = {"jobs": jobs, "promised": list(promised), "legacy_done": list(legacy_done)}
        _write_atomic(self.state, json.dumps(data), 0o600)

    # -- ledger -------------------------------------------------------------------

    def append_ledger(self, rec):
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        fd = os.open(self.ledger, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "a") as f:
            f.write(line)

    # -- exit reports -------------------------------------------------------------

    def exit_path(self, job_id):
        return self.exits / f"{int(job_id)}.json"

    def read_exit(self, job_id):
        try:
            with open(self.exit_path(job_id)) as f:
                rep = json.load(f)
        except (OSError, ValueError):
            return None
        return rep if isinstance(rep, dict) else None

    def drop_exit(self, job_id):
        self.exit_path(job_id).unlink(missing_ok=True)

    # -- launch specs ---------------------------------------------------------------

    def launch_path(self, job_id):
        return self.launch / f"{int(job_id)}.json"

    def write_launch(self, job_id, spec, uid, gid):
        """The spec the job's first process reads, as its owner: nobody else can."""
        path = self.launch_path(job_id)
        path.unlink(missing_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
        with os.fdopen(fd, "w") as f:
            if os.geteuid() == 0:
                os.fchown(f.fileno(), uid, gid)
            json.dump(spec, f)
        return path

    def drop_launch(self, job_id):
        self.launch_path(job_id).unlink(missing_ok=True)
