"""One process as /proc describes it. Everything read here is world-readable."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    uid: int
    ppid: int
    pgid: int
    start: int          # kernel start time: with pid, names one process across pid reuse
    cgroup: str         # the cgroup v2 path


def read_proc(pid, proc="/proc"):
    """The process, or None if it is gone or its entry is half-written."""
    base = f"{proc}/{int(pid)}"
    try:
        with open(f"{base}/stat") as f:
            stat = f.read()
        with open(f"{base}/status") as f:
            status = f.read()
        with open(f"{base}/cgroup") as f:
            cgroup = f.read()
    except OSError:
        return None
    try:
        # The command name is in parentheses and may contain anything, so the
        # fields are counted from the LAST ')': state ppid pgrp ... starttime.
        after = stat[stat.rindex(")") + 1:].split()
        ppid, pgid, start = int(after[1]), int(after[2]), int(after[19])
        uid = next(int(line.split()[1]) for line in status.splitlines()
                   if line.startswith("Uid:"))
    except (ValueError, IndexError, StopIteration):
        return None
    path = next((line[3:] for line in cgroup.splitlines() if line.startswith("0::")), "")
    return ProcInfo(int(pid), uid, ppid, pgid, start, path.strip())


# /proc/PID/limits row -> the systemd property that sets the same limit. A job
# runs with the limits of the shell it was submitted from, as the previous gpuq's jobs did:
# systemd's defaults are far lower (1,024 open files, 8 MiB locked memory).
LIMITS = {
    "Max open files": "LimitNOFILE",
    "Max locked memory": "LimitMEMLOCK",
    "Max stack size": "LimitSTACK",
    "Max core file size": "LimitCORE",
    "Max processes": "LimitNPROC",
    "Max address space": "LimitAS",
    "Max data size": "LimitDATA",
    "Max file size": "LimitFSIZE",
    "Max cpu time": "LimitCPU",
}


def read_limits(pid, proc="/proc"):
    """systemd property -> "soft:hard" for each limit in LIMITS; {} if unreadable."""
    try:
        with open(f"{proc}/{int(pid)}/limits") as f:
            lines = f.read().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        for row, prop in LIMITS.items():
            if line.startswith(row):
                try:
                    soft, hard = line[len(row):].split()[:2]
                    out[prop] = ":".join("infinity" if v == "unlimited" else str(int(v))
                                         for v in (soft, hard))
                except ValueError:
                    pass
    return out


def read_umask(pid, proc="/proc"):
    """The process's umask, or 0o022 if /proc does not say."""
    try:
        with open(f"{proc}/{int(pid)}/status") as f:
            for line in f:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    return 0o022
