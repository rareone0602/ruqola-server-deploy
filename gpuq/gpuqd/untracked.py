"""Rule 4, GPUs only through gpuq: which GPU processes are outside every job.

A process is an offender when no gpuq job owns it, or when its job owns it but
it is on a card the job was not given. Accounts below uid 1000 (the display
server, driver daemons) are exempt. Who owns a process is the caller's
question: the previous gpuq's rules in shadow mode, the job's own cgroup after cutover.
"""
import pwd
from dataclasses import dataclass

# Accounts below 1000 are the system's, not people's.
SYSTEM_UID_MAX = 999


@dataclass(frozen=True)
class Offender:
    pid: int
    start: int
    uid: int
    card: object        # card index, or None if nvidia-smi could not map it
    used_mb: int
    name: str
    job: object         # None: outside every job; a job id: on a card that job does not hold

    @property
    def key(self):
        """One process, even if its pid number is later reused."""
        return (self.pid, self.start)

    @property
    def user(self):
        try:
            return pwd.getpwuid(self.uid).pw_name
        except KeyError:
            return str(self.uid)


def offenders(seen):
    """seen: [(GpuProc, ProcInfo, owner)] where owner is None for a process no
    job owns, else (job_id, cards the job holds)."""
    out = []
    for proc, info, owner in seen:
        if info.uid <= SYSTEM_UID_MAX:
            continue
        if owner is None:
            job = None
        else:
            job_id, cards = owner
            if proc.card is None or not cards or proc.card in cards:
                continue
            job = job_id
        out.append(Offender(proc.pid, info.start, info.uid, proc.card, proc.used_mb,
                            proc.name, job))
    return out
