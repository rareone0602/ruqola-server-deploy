"""Start, signal and stop jobs as systemd units, one per job.

Each job is a transient system service, gpuq-job-ID.service in gpuq.slice, run
as its submitter (docs/v3-design.md §6, §7):

  - it may open only its own cards' /dev/nvidiaN (verified on this host, §6);
  - systemd stops it at its deadline even if gpuqd is down (RuntimeMaxSec);
  - its processes are exactly the cgroup /gpuq.slice/gpuq-job-ID.service, so
    "is this process part of a gpuq job?" has an exact answer;
  - a root exit hook records how it ended (exithook.py).

An attached job writes to the submitter's own terminal (systemd-run --pipe with
the files the client passed); a shell gets a terminal of its own, forwarded to
the client's (--pty); a detached job writes to a log file. The systemd-run
helper of an attached job or shell is gpuqd's child. If it dies (gpuqd
restarting), the job runs on: it holds its own copy of the terminal.
"""
import os
import subprocess

from .devices import device_properties
from .jobs import job_env

LIB = os.environ.get("GPUQD_LIB", "/usr/local/lib/gpuq-v3")
PYTHON = "/usr/bin/python3"

# SIGTERM, then SIGKILL this long after: the previous gpuq's KILL_GRACE_SEC.
STOP_GRACE_S = 10

ACTIVE = ("active", "activating", "deactivating", "reloading")


class StartError(Exception):
    pass


def unit_name(job_id):
    return f"gpuq-job-{int(job_id)}.service"


class SystemdRunner:
    def __init__(self, store, lib=LIB, python=PYTHON, systemd_run="systemd-run",
                 systemctl="systemctl"):
        self.store = store
        self.lib = lib
        self.python = python
        self.systemd_run = systemd_run
        self.systemctl = systemctl

    def argv(self, job, minors, spec_path, now):
        runtime = max(1, int(job["deadline"] - now))
        argv = [self.systemd_run, f"--unit={unit_name(job['id'])}", "--slice=gpuq",
                f"--uid={job['uid']}", "--collect", "--quiet", "--expand-environment=no",
                f"--description=gpuq job {job['id']} of {job['user']}"]
        argv += {"attached": ["--pipe", "--wait"], "shell": ["--pty", "--wait"],
                 "detached": []}[job["mode"]]
        props = [f"RuntimeMaxSec={runtime}", f"TimeoutStopSec={STOP_GRACE_S}",
                 "OOMPolicy=continue", f"UMask={job['umask']:04o}"]
        props += [f"{k}={v}" for k, v in sorted(job.get("limits", {}).items())]
        if minors is not None:          # None only in smoke.py's diagnosis
            props += device_properties(minors)
        props.append(f"ExecStopPost=+{self.python} -I {self.lib}/gpuqd/exithook.py "
                     f"{self.store.exits} {job['id']}")
        argv += [f"--property={p}" for p in props]
        return argv + ["--", self.python, "-I", f"{self.lib}/gpuqd/launch.py", str(spec_path)]

    def start(self, job, minors, fds, now):
        """Start the job's unit. Returns the systemd-run helper (attached, shell)
        or None (detached). Raises StartError if the unit could not start."""
        spec = {"argv": job["argv"], "env": job_env(job), "cwd": job["cwd"],
                "log": job.get("log")}
        path = self.store.write_launch(job["id"], spec, job["uid"], job["gid"])
        argv = self.argv(job, minors, path, now)
        try:
            if job["mode"] == "detached":
                r = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                                   text=True, timeout=60)
                if r.returncode != 0:
                    raise StartError(r.stderr.strip() or f"systemd-run exited {r.returncode}")
                return None
            return subprocess.Popen(argv, stdin=fds[0], stdout=fds[1], stderr=fds[2],
                                    start_new_session=True, close_fds=True)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise StartError(str(e)) from None

    def _systemctl(self, *args):
        try:
            return subprocess.run([self.systemctl, *args], stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None

    def signal(self, job_id, sig):
        """Send `sig` to every process of the job, as the previous gpuq did."""
        self._systemctl("kill", "--kill-whom=all", f"--signal={int(sig)}", unit_name(job_id))

    def stop(self, job_id):
        """SIGTERM to every process, SIGKILL STOP_GRACE_S later. Returns at once."""
        self._systemctl("stop", "--no-block", unit_name(job_id))

    def active(self, job_id):
        """True while the unit runs, False once it is gone, None if unknown."""
        r = self._systemctl("show", "-P", "ActiveState", unit_name(job_id))
        if r is None or r.returncode != 0:
            return None
        return r.stdout.strip() in ACTIVE
