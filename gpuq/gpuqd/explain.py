"""What gpuqd tells people: the queue in order with promises and estimates
(`gpuq status`), why a job waits (`gpuq why`), recent use (`gpuq share`), and
the text of its emails. Everything here reads the last pass; nothing changes it.

A start time for a job of 2 or more cards is a PROMISE: the cards are held for
it and the time can only get earlier (rule 2). A time for a 1-card job is an
ESTIMATE: its place is re-decided each pass, so it can move either way (§5.1).
A job joining its user's own cards (--devices) is not in line at all: it waits
only for room on those cards (§4.5).
"""
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from scheduler.book import CardBook
from scheduler.model import HOUR, MAX_RUNTIME_H

from .jobs import command_text, iso, join_gb
from .runner import STOP_GRACE_S


@dataclass
class View:
    t: float
    cards: dict                 # index -> nvsmi.Card; None when the driver did not answer
    cap: dict = None
    free: dict = None
    usage: dict = field(default_factory=dict)
    plan: object = None
    queue: list = field(default_factory=list)       # scheduler Jobs planned this pass
    untracked: list = field(default_factory=list)   # (user, card, used_mb, since, pids)
    held: list = field(default_factory=list)        # every holding, as scheduler Running


def when(t, now=None):
    if t is None:
        return "-"
    d = datetime.fromtimestamp(t)
    if now is not None and abs(t - now) > 6 * 24 * HOUR:
        return d.strftime("%m-%d %H:%M")
    return d.strftime("%a %H:%M")


def span(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def order(queue, usage, promised):
    """Waiting jobs in the planner's order: kept promises first, then least
    recent use, one turn per person before anyone gets a second."""
    by_id = {j.id: j for j in queue}
    kept = [by_id[i] for i in promised if i in by_id and by_id[i].gpus >= 2]
    rest = [j for j in queue if j not in kept]
    turns = Counter(j.user for j in kept)
    out = list(kept)
    while rest:
        j = min(rest, key=lambda j: (turns[j.user], usage.get(j.user, 0.0), j.submit_t, j.id))
        rest.remove(j)
        out.append(j)
        turns[j.user] += 1
    return out


def line(view):
    """[(Job, kind, t, cards)] in order. kind: promise | estimate | impossible | unknown."""
    if view.plan is None:
        return [(j, "unknown", None, None) for j in view.queue]
    p = view.plan
    resv = {r.job_id: r for r in p.reservations}
    started = {s.job_id for s in p.starts}
    queue = [j for j in view.queue if j.id not in started and not j.join]
    book = CardBook(view.cap, view.held, view.t)
    for j in queue:
        r = resv.get(j.id)
        if r is not None:
            book.add(r.cards, r.start_t, r.start_t + j.limit_h * HOUR, j.user, j.mem_gb, True)
    out = []
    for j in order(queue, view.usage, [r.job_id for r in p.reservations]):
        r = resv.get(j.id)
        if r is not None:
            out.append((j, "promise" if j.gpus >= 2 else "estimate", r.start_t, r.cards))
        elif p.waiting.get(j.id) == "impossible":
            out.append((j, "impossible", None, None))
        else:
            for t in book.times()[1:]:
                cards = book.choose(j, t)
                if cards is not None:
                    book.add(cards, t, t + j.limit_h * HOUR, j.user, j.mem_gb, True)
                    out.append((j, "estimate", t, cards))
                    break
            else:
                out.append((j, "unknown", None, None))
    return out


def joining(view, jobs):
    """Jobs waiting to join their user's own cards: [(job, free GB, must end by)].
    Free GB is the least free on any of its cards; the end is None when the
    hold is about to end and the job is not planned this pass."""
    planned = {j.id: j for j in view.queue if j.join}
    started = set() if view.plan is None else {s.job_id for s in view.plan.starts}
    out = []
    for jid, job in sorted(jobs.items()):
        if job["state"] != "queued" or not job.get("devices") or str(jid) in started:
            continue
        free = min((view.free or {}).get(c, 0.0) for c in job["devices"])
        j = planned.get(str(jid))
        out.append((job, free, None if j is None else view.t + j.limit_h * HOUR))
    return out


def _holders(view, jobs):
    """card -> [(user, label, end_by)] for everything holding it now."""
    out = {}
    for j in jobs.values():
        if j["state"] == "running":
            for c in j["cards"]:
                out.setdefault(c, []).append((j["user"], f"job {j['id']}", j["deadline"]))
    for user, card, _mb, _since, _pids in view.untracked:
        out.setdefault(card, []).append((user, "outside gpuq", None))     # no deadline
    return out


def snapshot(view, jobs, host, orphans=()):
    """Everything `gpuq status` prints, as plain data."""
    if view is None or view.cards is None:
        return {"host": host, "t": view.t if view else None, "driver": False,
                "running": _running(jobs), "queue": [], "cards": []}
    holders = _holders(view, jobs)
    resv = {}
    for r in view.plan.reservations:
        for c in r.cards:
            resv[c] = r
    users = {str(j["id"]): j["user"] for j in jobs.values()}
    cards = []
    for i, c in sorted(view.cards.items()):
        hs = holders.get(i, [])
        r = resv.get(i)
        ends = [e for _, _, e in hs]
        cards.append({
            "index": i, "used_mb": c.used_mb, "total_mb": c.total_mb,
            "holders": [{"user": u, "what": w} for u, w, _ in hs],
            # None: free now. "?": something outside gpuq holds it, with no end time.
            "free_by": "?" if None in ends else max(ends, default=None),
            "held_for": None if r is None else {"job": int(r.job_id), "user": users.get(r.job_id),
                                                "t": r.start_t},
        })
    queue = []
    for j, kind, t, cs in line(view):
        queue.append({"job": int(j.id), "user": j.user, "gpus": j.gpus,
                      "mem_gb": jobs[int(j.id)].get("mem_gb"), "name": jobs[int(j.id)].get("name"),
                      "submitted": j.submit_t, "kind": kind, "t": t,
                      "cards": None if cs is None else list(cs),
                      "used": round(view.usage.get(j.user, 0.0), 1)})
    held_back = [jid for jid, j in jobs.items() if j["state"] == "queued" and jid in orphans]
    joins = [{"job": j["id"], "user": j["user"], "cards": j["devices"], "need_gb": join_gb(j),
              "free_gb": round(free, 1), "until": until, "name": j.get("name")}
             for j, free, until in joining(view, jobs)]
    return {"host": host, "t": view.t, "driver": True, "cards": cards,
            "running": _running(jobs), "queue": queue, "joining": joins,
            "waiting_for_client": held_back,
            "untracked": [{"user": u, "card": c, "used_mb": mb, "since": s, "pids": p}
                          for u, c, mb, s, p in view.untracked]}


def _running(jobs):
    return [{"job": j["id"], "user": j["user"], "name": j.get("name"), "cards": j["cards"],
             "started": j["started"], "deadline": j["deadline"], "mode": j["mode"],
             "joined": bool(j.get("devices")), "command": command_text(j)}
            for j in sorted(jobs.values(), key=lambda j: j.get("started", 0))
            if j["state"] == "running"]


def place(view, job):
    """Where a just-queued job stands: for the submitter's first message."""
    out = {"need": job["gpus"], "free": 0, "kind": "unknown", "t": None, "cards": None,
           "ahead": None}
    if view is None or view.cards is None:
        return out
    if job.get("devices"):
        for j, free, until in joining(view, {job["id"]: job}):
            out.update(kind="room", cards=j["devices"], need_gb=join_gb(j), free_gb=free,
                       until=until)
        return out
    holders = _holders(view, {})
    taken = {c for c in view.cards if c in holders} | {
        c for r in view.plan.reservations for c in r.cards}
    taken |= {c for s in view.plan.starts for c in s.cards}
    out["free"] = len([c for c in view.cards if c not in taken])
    for n, (j, kind, t, cards) in enumerate(line(view)):
        if j.id == str(job["id"]):
            out.update(kind=kind, t=t, cards=None if cards is None else list(cards), ahead=n)
    return out


def why(view, jobs, jid, host):
    """Plain sentences on where job `jid` stands, and why."""
    job = jobs.get(jid)
    if job is None:
        return f"no running or queued job {jid} on {host} (or it already finished)."
    if job["state"] == "running":
        since = (f"job {jid} is running on GPU(s) {_list(job['cards'])} since "
                 f"{when(job['started'], view.t if view else None)}.")
        if job.get("devices"):
            return (f"{since} It joined {job['user']}'s own card(s) there (--devices), so it "
                    f"must end when that hold does: {when(job['deadline'])}.")
        return f"{since} It may run until {when(job['deadline'])} ({MAX_RUNTIME_H:g} h)."
    if view is None or view.cards is None:
        return f"job {jid} is waiting. nvidia-smi is not answering, so nothing starts until it does."
    if job.get("devices"):
        return _why_joining(view, job)
    entries = line(view)
    mine = next(((n, e) for n, e in enumerate(entries) if e[0].id == str(jid)), None)
    if mine is None:
        return (f"job {jid} is waiting for its terminal: gpuqd restarted and the submitting "
                "`gpuq submit` has not come back yet.")
    n, (j, kind, t, cards) = mine
    used = view.usage.get(j.user, 0.0)
    lines = []
    if kind == "impossible":
        return f"job {jid} asks for {j.gpus} cards of {j.mem_gb:.0f} GB; this host cannot run it."
    if kind == "promise":
        lines.append(f"job {jid} is promised GPU(s) {_list(cards)} by {when(t, view.t)}. "
                     "Those cards are held for it, so that time can only get earlier.")
    else:
        lines.append(f"job {jid} is number {n + 1} in line.")
    ahead = entries[:n]
    if ahead:
        lines.append("Ahead of it:")
        for a, akind, at, acards in ahead:
            what = (f"promised GPU(s) {_list(acards)} by {when(at, view.t)}" if akind == "promise"
                    else f"estimate {when(at, view.t)}" if at else akind)
            lines.append(f"  {a.user:<10} job {a.id}  {a.gpus} GPU  "
                         f"used lately: {view.usage.get(a.user, 0.0):.1f} card-h  ({what})")
    lines.append(f"{j.user} has used {used:.1f} card-hours lately. When more people want cards "
                 "than there are cards, whoever has used the fewest GPU-hours lately goes first; "
                 "use fades by half every 7 days.")
    holders = _holders(view, jobs)
    resv = {c: r for r in view.plan.reservations for c in r.cards}
    users = {str(x["id"]): x["user"] for x in jobs.values()}
    for c in sorted(view.cards):
        if c in holders:
            continue
        r = resv.get(c)
        if r is not None and r.job_id != str(jid):
            lines.append(f"GPU {c} is free but held for {users.get(r.job_id, '?')}'s job "
                         f"{r.job_id}, promised by {when(r.start_t, view.t)}.")
        elif r is None and j.mem_gb > view.free.get(c, 0.0):
            lines.append(f"GPU {c} has {view.free.get(c, 0.0):.0f} GB free; job {jid} "
                         + ("wants a card to itself (no -m)." if job.get("mem_gb") is None
                            else f"asked for {j.mem_gb:g} GB (-m)."))
    if kind == "estimate" and t is not None:
        lines.append(f"Estimated start: {when(t, view.t)} on GPU(s) {_list(cards)}, assuming every "
                     f"running job uses its full {MAX_RUNTIME_H:g} h; earlier if they end sooner.")
    return "\n".join(lines)


def _why_joining(view, job):
    cards = _list(job["devices"])
    need = join_gb(job)
    asked = (f"-m {job['mem_gb']:g}" if job.get("mem_gb") and job["mem_gb"] >= need
             else f"at least {need:g} GB, the least any job joining a card needs")
    [(_, free, until)] = joining(view, {job["id"]: job})
    lines = [f"job {job['id']} is joining {job['user']}'s own GPU {cards} (--devices). It does "
             "not wait in line: it starts as soon as there is room on that card, whoever else "
             "is waiting.",
             f"It needs {need:g} GB free there ({asked}); nvidia-smi measures {free:.0f} GB free now."]
    if until is not None:
        lines.append(f"Once started, it must end by {when(until, view.t)}, when {job['user']}'s "
                     f"job(s) on GPU {cards} reach their {MAX_RUNTIME_H:g} h.")
    lines.append(f"If {job['user']}'s jobs on GPU {cards} all end before there is room, it is "
                 "cancelled.")
    return "\n".join(lines)


def share(view, user, everyone):
    use = dict(view.usage) if view and view.usage else {}
    rows = sorted(use.items(), key=lambda kv: kv[1])
    if not everyone:
        rows = [(u, v) for u, v in rows if u == user] or [(user, 0.0)]
    places = []
    if view and view.plan is not None:
        for n, (j, kind, t, _) in enumerate(line(view)):
            if j.user == user:
                places.append({"job": int(j.id), "place": n + 1, "kind": kind, "t": t})
    return {"usage": [{"user": u, "card_hours": round(v, 1)} for u, v in rows],
            "you": user, "places": places, "half_life_days": 7}


def _list(cards):
    return ",".join(str(c) for c in cards or ())


def _label(job):
    return f"{job['id']}" + (f" ({job['name']})" if job.get("name") else "")


def _until(job):
    if job.get("devices"):
        return (f"It joined your own card(s) (--devices), so it may run until your hold on "
                f"them ends: {iso(job['deadline'])}.")
    return f"It may run until {iso(job['deadline'])} ({MAX_RUNTIME_H:g} hours)."


def started_mail(job, host):
    return (f"Host: {host}\nJob: {_label(job)}\nCommand: {command_text(job)}\n\n"
            f"Started {iso(job['started'])} on GPU(s) {_list(job['cards'])}, after waiting "
            f"{span(job['started'] - job['submitted'])}.\n"
            f"{_until(job)} You will get a warning an hour before.\n")


def deadline_mail(job, host):
    if job.get("devices"):
        when_ = (f"This job joined your own card(s) (--devices), so it ends when your hold on "
                 f"them does: at {iso(job['deadline'])}.")
    else:
        when_ = f"This job reaches its {MAX_RUNTIME_H:g}-hour limit at {iso(job['deadline'])}."
    return (f"Host: {host}\nJob: {_label(job)}\nCommand: {command_text(job)}\n\n{when_}\n"
            f"It will be stopped then: SIGTERM, then SIGKILL {STOP_GRACE_S} seconds later.\n"
            "Save a checkpoint now. Inside the job, GPUQ_DEADLINE holds the same moment "
            "(epoch seconds).\n")


def end_mail(job, t, rc, reason, host):
    return (f"Host: {host}\nUser: {job['user']}\nJob: {_label(job)}\n"
            f"Command: {command_text(job)}\nGPUs: {job['cards']}\n"
            f"Started: {iso(job['started'])}\nEnded: {iso(t)}\n"
            f"Reason: {reason} (exit code {rc})\n")
