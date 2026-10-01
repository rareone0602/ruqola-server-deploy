"""The plain data the scheduler reasons about.

Nothing here reads a clock, a file or nvidia-smi: callers pass `now` and the
current picture, so the same code runs in the daemon, the simulator and tests.
Times are epoch seconds; durations are hours.
"""
from dataclasses import dataclass, field

HOUR = 3600.0

# The one policy number: how long one job may hold a card. A real trade-off
# (long runs vs. how long a newcomer can wait); the lab chose 48 h.
MAX_RUNTIME_H = 48.0


@dataclass(frozen=True)
class Job:
    """A request for `gpus` whole cards with `mem_gb` VRAM each, for at most `limit_h` hours.

    `join` names cards the user already holds (`--devices`): the job runs on
    exactly those, and `limit_h` ends when the user's hold on them does."""
    id: str
    user: str
    gpus: int
    mem_gb: float
    limit_h: float
    submit_t: float
    join: tuple = ()


@dataclass(frozen=True)
class Running:
    """A job that holds `cards` since `start_t`. It is stopped at its limit."""
    job: Job
    cards: tuple
    start_t: float

    @property
    def end_by(self) -> float:
        return self.start_t + self.job.limit_h * HOUR


@dataclass(frozen=True)
class Start:
    job_id: str
    cards: tuple


@dataclass(frozen=True)
class Reservation:
    """A promise: these cards are held back so the job can start by `start_t`."""
    job_id: str
    start_t: float
    cards: tuple


@dataclass
class Plan:
    starts: list = field(default_factory=list)
    reservations: list = field(default_factory=list)
    # job id -> "queued" (behind a promised job), "impossible" (asks for
    # more than the host has) or "room" (joining its user's own cards, which
    # do not have enough VRAM free yet).
    waiting: dict = field(default_factory=dict)
