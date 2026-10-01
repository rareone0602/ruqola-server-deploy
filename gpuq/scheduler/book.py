"""Who holds or is promised each card, and when.

Whole cards: a user owns a card while any of their jobs run on it and may stack
more of their OWN jobs there while VRAM allows. Nobody else may use it, and a
card promised to a waiting job is off-limits to everyone else until then.
"""
from .model import HOUR

# A job still running past its limit is about to be stopped; treat its card as
# freeing shortly, never as free already.
OVERRUN_S = 60.0


class CardBook:
    """Every card's spans for one planning pass: running jobs, then promises."""

    def __init__(self, cards, running, now, free_gb=None):
        self.cap = dict(cards)
        self.now = now
        # What nvidia-smi says is free on each card right now. When known, it
        # decides whether a job fits NOW: declared sizes are loose guesses
        # (226 of 412 real stacks exceeded the card on paper and ran fine).
        self.free_now = dict(free_gb) if free_gb is not None else None
        # card -> [(start, end, user, mem_gb, promised?)]
        self.spans = {c: [] for c in cards}
        for r in running:
            end = max(r.end_by, now + OVERRUN_S)
            for c in r.cards:
                self.spans[c].append((r.start_t, end, r.job.user, r.job.mem_gb, False))

    def add(self, cards, start, end, user, mem, promised):
        for c in cards:
            self.spans[c].append((start, end, user, mem, promised))
            if start == self.now and self.free_now is not None:
                self.free_now[c] = self.free_now.get(c, self.cap[c]) - mem

    def times(self):
        """Moments worth trying: now, and whenever some card frees up."""
        ends = {e for spans in self.spans.values() for _, e, *_ in spans if e > self.now}
        return [self.now] + sorted(ends)

    def held_at(self, user, t):
        return {c for c, spans in self.spans.items()
                if any(u == user and s <= t < e for s, e, u, _, _ in spans)}

    def fits(self, c, user, start, end, mem):
        """Can `user` use card c over [start, end) with `mem` GB?"""
        if self.cap[c] < mem:
            return False
        used = 0.0
        for s, e, u, gb, promised in self.spans[c]:
            if s < end and start < e:
                if u != user or promised:
                    return False
                used += gb          # stacking on your own running job
        if start == self.now and self.free_now is not None:
            return mem <= self.free_now.get(c, self.cap[c])
        return used + mem <= self.cap[c]

    def next_claim(self, c, after):
        """When card c is next needed by anyone, at or after `after`."""
        return min((s for s, *_ in self.spans[c] if s >= after), default=float("inf"))

    def choose(self, job, t, max_cards=None):
        """Cards for `job` starting at t, or None. Own cards first, then best fit.
        A job joining its user's cards (job.join) gets exactly those or nothing.

        max_cards caps the distinct cards the user would hold. v3 has no cap;
        it exists so sim/ can model the previous gpuq's rules (hard cap 3).
        """
        end = t + job.limit_h * HOUR
        if job.join:
            ok = all(c in self.cap and self.fits(c, job.user, t, end, job.mem_gb)
                     for c in job.join)
            return tuple(sorted(job.join)) if ok else None
        held = self.held_at(job.user, t)
        ok = [c for c in self.cap if self.fits(c, job.user, t, end, job.mem_gb)]
        own = [c for c in ok if c in held]
        fresh = sorted((c for c in ok if c not in held),
                       key=lambda c: (self.next_claim(c, end), c))
        if max_cards is not None:
            fresh = fresh[:max(0, max_cards - len(held))]
        picked = (own + fresh)[:job.gpus]
        return tuple(sorted(picked)) if len(picked) == job.gpus else None
