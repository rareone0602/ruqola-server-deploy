"""Rule 4, enforced: a GPU process outside every gpuq job is stopped (§6, D3).

If people see others use the GPUs without the queue and get away with it, soon
nobody queues. So gpuqd stops such a process on its second sighting, at least a
minute after the first (sightings.py), with no warning period:

  1. The process is pinned with a pidfd, then checked again: the same kernel
     start time (not a recycled pid) and the same owner, a person's (uid >=
     1000). Every signal goes through the pidfd, so it can only reach that
     process.
  2. SIGTERM, then SIGKILL STOP_GRACE_S later if it is still there.
  3. Its owner gets an email: what was stopped, and the gpuq command that runs
     it properly. At most one an hour per person; every stop is in the journal.

Root can switch stopping off at once, without a restart: while REPORT_ONLY
exists, offenders are only logged.
"""
import os
import shlex
import signal

from scheduler.model import HOUR

from .procinfo import read_proc
from .runner import STOP_GRACE_S
from .untracked import SYSTEM_UID_MAX

REPORT_ONLY = "/etc/gpuq/report-only"
# One email an hour at most per person, however often they restart the process.
MAIL_EVERY_S = HOUR
GUIDE = "https://rareone0602.github.io/ruqola-server-deploy/#gpuq/gpu-queue-guide"
CMD_MAX = 300


def _cmdline(pid, proc):
    try:
        with open(f"{proc}/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        return None
    argv = [a.decode(errors="replace") for a in raw.split(b"\0") if a]
    return shlex.join(argv)[:CMD_MAX] if argv else None


class Stopper:
    def __init__(self, proc="/proc", log=print, mailer=None, host="", name=str,
                 report_only=REPORT_ONLY):
        self.proc = proc
        self.log = log
        self.mailer = mailer
        self.host = host
        self.name = name                # uid -> login name
        self.report_only = report_only
        self.pending = {}               # offender key -> [pidfd, SIGKILL due at, label]
        self.mailed = {}                # uid -> when last emailed

    @property
    def enforcing(self):
        return not os.path.exists(self.report_only)

    def stop(self, offenders, now):
        """SIGTERM each offender not already being stopped; email the owners.
        Returns the offenders signalled."""
        done = []
        for o in offenders:
            if o.key in self.pending:
                continue
            pinned = self._pin(o)
            if pinned is None:
                continue
            fd, command = pinned
            where = "outside every gpuq job" if o.job is None else f"off job {o.job}'s cards"
            label = (f"pid {o.pid} of {self.name(o.uid)} on GPU {o.card} "
                     f"({where}, {o.used_mb} MiB, {o.name})")
            try:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
            except ProcessLookupError:
                os.close(fd)
                continue
            except OSError as e:
                os.close(fd)
                self.log(f"rule 4: could not stop {label}: {e}")
                continue
            self.pending[o.key] = [fd, now + STOP_GRACE_S, label]
            self.log(f"rule 4: stopped {label}: SIGTERM, SIGKILL in {STOP_GRACE_S} s "
                     "if it is still there")
            done.append((o, command))
        self._mail(done, now)
        return [o for o, _ in done]

    def _pin(self, o):
        """(pidfd, command line) for exactly the process the offender describes,
        or None if it is gone or is no longer that process."""
        if self.proc != "/proc":
            # A pidfd names a real process; a test's fake /proc must never pick one.
            self.log(f"rule 4: would stop pid {o.pid}, but {self.proc} is not /proc")
            return None
        try:
            fd = os.pidfd_open(o.pid)
        except OSError:
            return None                     # it has exited already
        info = read_proc(o.pid, self.proc)
        if (info is None or info.start != o.start or info.uid != o.uid
                or info.uid <= SYSTEM_UID_MAX):
            os.close(fd)
            self.log(f"rule 4: pid {o.pid} ended or changed before it could be stopped; "
                     "nothing done")
            return None
        return fd, _cmdline(o.pid, self.proc)

    def tick(self, now):
        """SIGKILL what outlived its SIGTERM grace; forget what has exited."""
        for key, entry in list(self.pending.items()):
            fd, kill_at, label = entry
            try:
                if kill_at is not None and now >= kill_at:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                    self.log(f"rule 4: {label} was still there: SIGKILL")
                    entry[1] = None
                else:
                    signal.pidfd_send_signal(fd, 0)     # still there?
            except ProcessLookupError:
                os.close(fd)
                del self.pending[key]
            except OSError as e:
                self.log(f"rule 4: could not signal {label}: {e}")

    def _mail(self, done, now):
        if self.mailer is None:
            return
        by_uid = {}
        for o, command in done:
            by_uid.setdefault(o.uid, []).append((o, command))
        for uid, items in by_uid.items():
            if now - self.mailed.get(uid, -MAIL_EVERY_S) < MAIL_EVERY_S:
                continue
            self.mailed[uid] = now
            self.mailer.send(self.name(uid), f"[gpuq] stopped your GPU process on {self.host}",
                             mail_body(items, self.host))


def mail_body(items, host):
    lines = [f"Host: {host}", "",
             "gpuq stopped these processes of yours. They were using a GPU outside gpuq:", ""]
    for o, command in items:
        where = "" if o.job is None else f", which its job {o.job} was not given"
        lines.append(f"  pid {o.pid} on GPU {o.card}{where} ({o.used_mb} MiB): "
                     f"{command or o.name}")
    command = next((c for _, c in items if c), "python train.py")
    lines += ["",
              "Every GPU job goes through gpuq, so that everyone gets a fair turn. To run it",
              "again:", "",
              f"  gpuq submit -- {command}",
              f"  gpuq submit --detach -- {command}    (keeps running after you log out)",
              "  gpuq shell    (an interactive shell on a GPU, for notebooks and debugging)", "",
              f"A process gets SIGTERM, then SIGKILL {STOP_GRACE_S} seconds later. More: {GUIDE}",
              ""]
    return "\n".join(lines)
