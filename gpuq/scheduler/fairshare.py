"""Fair-share: how much each user has recently kept from everyone else.

The charge is card-hours HELD. A card you own is unusable by others however many
of your own jobs share it, so stacking is charged once, per card. Usage fades
exponentially: after one half-life an hour counts half. This is the same idea
Slurm (PriorityDecayHalfLife) and HTCondor (PRIORITY_HALFLIFE) use.
"""
import math
from collections import defaultdict

from .model import HOUR

# Usage a week old counts half. 3 days and 7 days replayed identically, so the
# week wins: it is the horizon people already think in.
HALF_LIFE_H = 168.0


def held_intervals(history, now):
    """Merge job runs into the spans each user held each card.

    history: iterable of (Running, end_t or None-if-still-running).
    Returns [(user, card, start_t, end_t)], overlapping spans merged.
    """
    spans = defaultdict(list)
    for run, end_t in history:
        end = now if end_t is None else end_t
        for card in run.cards:
            spans[(run.job.user, card)].append((run.start_t, end))
    merged = []
    for (user, card), items in spans.items():
        items.sort()
        cur_s, cur_e = items[0]
        for s, e in items[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((user, card, cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((user, card, cur_s, cur_e))
    return merged


def _decayed_hours(start_t, end_t, now, half_life_h):
    """Integral of 2^(-age/half_life) over the span, in hours."""
    if math.isinf(half_life_h):
        return (end_t - start_t) / HOUR
    near = (now - end_t) / HOUR
    far = (now - start_t) / HOUR
    k = half_life_h / math.log(2)
    return k * (2 ** (-near / half_life_h) - 2 ** (-far / half_life_h))


def decayed_usage(spans, now, half_life_h=HALF_LIFE_H):
    """user -> decayed card-hours, from held_intervals() output."""
    usage = defaultdict(float)
    for user, _card, s, e in spans:
        usage[user] += _decayed_hours(s, e, now, half_life_h)
    return dict(usage)


class UsageMeter:
    """The same accounting, updated as time passes instead of recomputed.

    Call advance(t, held) where `held` is user -> distinct cards held since the
    previous call; the holding is assumed constant over that stretch.
    """

    def __init__(self, now, half_life_h=HALF_LIFE_H, usage=None):
        self.half_life_h = half_life_h
        self.now = now
        self.usage = dict(usage or {})

    def advance(self, t, held):
        dt = (t - self.now) / HOUR
        if dt < 0:
            raise ValueError("time went backwards")
        if math.isinf(self.half_life_h):
            fade, gain = 1.0, dt
        else:
            fade = 2 ** (-dt / self.half_life_h)
            gain = self.half_life_h / math.log(2) * (1 - fade)
        for user in list(self.usage):
            self.usage[user] *= fade
        for user, cards in held.items():
            if cards:
                self.usage[user] = self.usage.get(user, 0.0) + cards * gain
        self.now = t
