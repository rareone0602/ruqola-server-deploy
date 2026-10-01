"""What nvidia-smi says: the cards, and the processes using them.

Every reader returns None when the driver cannot answer (missing binary, error
exit, a wedged driver past the timeout, or no cards at all). None means
"unknown", and callers must then plan and stop nothing; it is never "idle".
"""
import os
import subprocess
from dataclasses import dataclass

# A wedged driver must not hang a pass. Same bound as the previous gpuq.
TIMEOUT_S = 15


@dataclass(frozen=True)
class Card:
    index: int
    uuid: str
    total_mb: int
    used_mb: int

    @property
    def total_gb(self):
        return self.total_mb / 1024

    @property
    def free_gb(self):
        return (self.total_mb - self.used_mb) / 1024


@dataclass(frozen=True)
class GpuProc:
    pid: int
    card: object        # card index, or None for a uuid not in the card list (MIG)
    used_mb: int
    name: str


def _query(binary, kind, fields, timeout):
    binary = binary or os.environ.get("GPUQ_NVSMI", "nvidia-smi")
    try:
        r = subprocess.run([binary, f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, timeout=timeout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return [line for line in r.stdout.splitlines() if line.strip()]


def read_cards(binary=None, timeout=TIMEOUT_S):
    """card index -> Card, or None if the driver did not answer."""
    lines = _query(binary, "gpu", "index,uuid,memory.used,memory.total", timeout)
    if lines is None:
        return None
    cards = {}
    for line in lines:
        parts = [s.strip() for s in line.split(",")]
        try:
            idx, uuid, used, total = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        except (IndexError, ValueError):
            return None         # a half-read table is not a table
        cards[idx] = Card(idx, uuid, total, used)
    return cards or None


def read_procs(cards, binary=None, timeout=TIMEOUT_S):
    """Every compute process on a GPU, or None if the driver did not answer."""
    lines = _query(binary, "compute-apps", "pid,gpu_uuid,used_memory,process_name", timeout)
    if lines is None:
        return None
    by_uuid = {c.uuid: c.index for c in cards.values()}
    procs = []
    for line in lines:
        parts = [s.strip() for s in line.split(",", 3)]
        try:
            pid = int(parts[0])
        except (IndexError, ValueError):
            continue            # a process that exited mid-read shows as [N/A]
        try:
            used = int(parts[2])
        except (IndexError, ValueError):
            used = 0
        name = parts[3] if len(parts) > 3 else ""
        procs.append(GpuProc(pid, by_uuid.get(parts[1]), used, name))
    return procs
