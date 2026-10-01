"""Who each GPU process belongs to, and which cards something other than a
gpuqd job is holding.

A process belongs to gpuqd job ID exactly when its cgroup is
/gpuq.slice/gpuq-job-ID.service: gpuqd creates that cgroup before the job runs
its first instruction (§6). A process of a job the previous gpuq was running at
cutover is recognised by that gpuq's rules (legacy.owner_of).

Any other GPU process of a person (uid >= 1000) is outside every job. Rule 4
stops it about a minute after it is first seen (stopper.py). Until it is gone,
it holds its card like a job would, so the planner starts nothing on top of it.
"""
import re
from dataclasses import dataclass, field

from scheduler.model import MAX_RUNTIME_H, Job, Running

from . import legacy
from .procinfo import read_proc
from .untracked import SYSTEM_UID_MAX, offenders

JOB_CGROUP = re.compile(r"/gpuq\.slice/gpuq-job-(\d+)\.service(?:/|$)")


def job_of(cgroup):
    """The gpuqd job id a cgroup path belongs to, or None."""
    m = JOB_CGROUP.match(cgroup or "")
    return int(m.group(1)) if m else None


@dataclass
class Picture:
    untracked: list = field(default_factory=list)   # [(user, card, used_mb, since, pids)]
    holding: list = field(default_factory=list)     # untracked holdings, as Running
    offenders: list = field(default_factory=list)   # untracked.Offender, for rule 4
    job_mb: dict = field(default_factory=dict)      # job id -> card -> MiB measured


class Holders:
    def __init__(self, proc="/proc", users=None):
        self.proc = proc
        self.since = {}             # (user, card) -> first seen
        self.users = users          # uid -> name; pwd by default (tests pass a dict)

    def _name(self, uid):
        if self.users is not None:
            return self.users.get(uid, str(uid))
        import pwd
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:
            return str(uid)

    def read(self, now, procs, jobs, legacy_entries):
        """procs: nvsmi.read_procs(). jobs: gpuqd's running jobs by id.
        legacy_entries: the previous gpuq's jobs still running from cutover."""
        pic = Picture()
        seen, held = [], {}
        for p in procs:
            info = read_proc(p.pid, self.proc)
            if info is None:
                continue                # exited between nvidia-smi and /proc
            jid = job_of(info.cgroup)
            owner = None
            if jid is not None and jid in jobs:
                owner = (str(jid), tuple(jobs[jid].get("cards") or ()))
                if p.card is not None:
                    by_card = pic.job_mb.setdefault(jid, {})
                    by_card[p.card] = by_card.get(p.card, 0) + p.used_mb
            elif legacy_entries:
                e = legacy.owner_of(info, legacy_entries, self.proc)
                if e is not None:
                    owner = (str(e["id"]), tuple(int(c) for c in e.get("gpus") or ()))
            seen.append((p, info, owner))
            if owner is None and info.uid > SYSTEM_UID_MAX and p.card is not None:
                key = (self._name(info.uid), p.card)
                mb, pids = held.get(key, (0, []))
                held[key] = (mb + p.used_mb, pids + [p.pid])
        self.since = {k: self.since.get(k, now) for k in held}
        for (user, card), (mb, pids) in sorted(held.items()):
            since = self.since[(user, card)]
            pic.untracked.append((user, card, mb, since, pids))
            job = Job(f"untracked:{user}:{card}", user, 1, mb / 1024, MAX_RUNTIME_H, since)
            pic.holding.append(Running(job, (card,), since))
        pic.offenders = offenders(seen)
        return pic
