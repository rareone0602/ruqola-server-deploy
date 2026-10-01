"""One planning pass: who starts now, who is promised a start, who waits and why.

The rules, in the words `gpuq why` will use:

1. When more people want cards than there are cards, whoever has held the
   fewest card-hours lately goes first (fair-share, see fairshare.py). Within
   one pass everyone waiting gets one turn before anyone gets a second.
2. A job that fits now starts now, unless it would still be running when a
   card it wants is promised to someone ahead of it.
3. The first job that does not fit is promised the earliest moment enough
   cards free up, and those cards are held for it (EASY backfill, Lifka 1995).
   This is what stops a 2-card job being overtaken forever by 1-card jobs.
4. A multi-card job keeps its promise: later arrivals, however light, plan
   around it (Maui's RESERVATIONPOLICY HIGHEST). Without this the replay had a
   2-card job lose its promise to each new 1-card arrival and wait 74.96 h.
   A 1-card job starts on the first card that frees, so it cannot starve that
   way; its place is re-decided each pass, which keeps fair-share working.
5. A job joining cards its user already holds (`--devices`) takes nothing from
   anyone, so it does not wait in line: it starts as soon as those cards have
   room. It must end when the user's hold on them does, so it never runs into
   a promise, and stacking can never keep a card past its 48 h.

There are no settings: each rule won its replay (docs/v3-design.md).
"""
from collections import Counter

from .book import CardBook
from .model import HOUR, Plan, Reservation, Start


def _keeps_promise(job):
    return job.gpus >= 2


def _order(job, usage):
    return (usage.get(job.user, 0.0), job.submit_t, job.id)


def plan(now, cards, running, queue, usage, free_gb=None, promised=()):
    """Decide this pass.

    cards: card index -> VRAM GB. usage: user -> fair-share card-hours.
    free_gb: optional card index -> VRAM measured free right now.
    promised: ids of jobs promised in earlier passes, oldest promise first.
      Pass back the ids of this plan's reservations, in order, next time.
    """
    book = CardBook(cards, running, now, free_gb)
    out = Plan()
    pending = [j for j in queue if not j.join]
    for job in (j for j in queue if j.join):
        picked = book.choose(job, now)
        if picked is None:
            out.waiting[job.id] = "room"
            continue
        book.add(picked, now, now + job.limit_h * HOUR, job.user, job.mem_gb, False)
        out.starts.append(Start(job.id, picked))
    by_id = {j.id: j for j in pending}
    kept = [by_id[i] for i in promised if i in by_id and _keeps_promise(by_id[i])]
    kept_ids = {j.id for j in kept}
    turns = Counter()
    biggest = max(cards.values(), default=0.0)
    while pending:
        job = kept.pop(0) if kept else min(
            pending, key=lambda j: (turns[j.user], _order(j, usage)))
        pending.remove(job)

        if job.gpus > len(cards) or job.mem_gb > biggest:
            out.waiting[job.id] = "impossible"
            continue

        picked = book.choose(job, now)
        if picked is not None:
            book.add(picked, now, now + job.limit_h * HOUR, job.user, job.mem_gb, False)
            out.starts.append(Start(job.id, picked))
            turns[job.user] += 1
            continue

        # One new promise per pass; promises kept from earlier passes count.
        if job.id not in kept_ids and out.reservations:
            out.waiting[job.id] = "queued"
            continue

        for t in book.times()[1:]:
            picked = book.choose(job, t)
            if picked is not None:
                book.add(picked, t, t + job.limit_h * HOUR, job.user, job.mem_gb, True)
                out.reservations.append(Reservation(job.id, t, picked))
                turns[job.user] += 1
                break
        else:
            out.waiting[job.id] = "queued"
    return out
