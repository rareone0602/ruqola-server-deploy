"""gpuq status, gpuq why, gpuq share (and its old name, gpuq quota)."""
import time
from datetime import datetime

from .common import ask


def when(t, now=None):
    if t is None:
        return "now"
    if t == "?":
        return "?"
    if now is not None and t <= now:
        return "now"
    d = datetime.fromtimestamp(t)
    if now is not None and t - now > 6 * 86400:
        return d.strftime("%m-%d %H:%M")
    return d.strftime("%a %H:%M")


def _run_for(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}"


def _trunc(text, width):
    text = (text or "").replace("\n", " ")
    return text if len(text) <= width else text[:width - 1] + "…"


def cmd_status(args):
    s = ask({"op": "status"})
    now = s.get("t") or time.time()
    print(f"=== GPU queue on {s['host']} ===")
    if not s.get("driver"):
        print("\nnvidia-smi is not answering: nothing starts until it does.")
    else:
        print(f"\n{'GPU':<4} {'holder':<16} {'free by':<12} {'memory':<11} next")
        for c in s["cards"]:
            holders = sorted({h["user"] for h in c["holders"]})
            nxt = ""
            if c["held_for"]:
                h = c["held_for"]
                verb = "promised to" if c["holders"] else "held for"
                nxt = f"{verb} {h['user']} {h['job']}"
            mem = f"{c['used_mb'] / 1024:.0f}/{c['total_mb'] / 1024:.0f} GB"
            print(f"{c['index']:<4} {', '.join(holders) or '-':<16} "
                  f"{when(c['free_by'], now):<12} {mem:<11} {nxt}")
    print(f"\nrunning ({len(s['running']) + len(s.get('old') or ())})")
    for r in s["running"]:
        label = f"{r['name']}: " if r.get("name") else ""
        print(f"  {r['job']:<11} {r['user']:<10} GPU {','.join(map(str, r['cards'])):<6} "
              f"ran {_run_for(now - r['started']):>6}  until {when(r['deadline'], now):<10} "
              f"{_trunc(label + r['command'], 50)}")
    for o in s.get("old") or ():
        print(f"  {o['job']:<11} {o['user']:<10} GPU {','.join(map(str, o['cards'] or ())):<6} "
              f"started by the old gpuq at {o['started']}")
    print(f"\nqueue, in the order cards go out ({len(s['queue'])})")
    for n, q in enumerate(s["queue"], 1):
        if q["kind"] == "promise":
            what = f"promised: by {when(q['t'], now)} on GPU {','.join(map(str, q['cards']))}"
        elif q["kind"] == "estimate" and q["t"]:
            what = f"estimate: ~{when(q['t'], now)}"
        elif q["kind"] == "impossible":
            what = "cannot run on this host as asked"
        else:
            what = "waiting"
        label = f"  {q['name']}" if q.get("name") else ""
        print(f"  {n}. {q['user']:<10} {q['job']:<11} {q['gpus']} GPU  {what:<34} "
              f"used lately: {q['used']:.1f} card-h{label}")
    for jid in s.get("waiting_for_client") or ():
        print(f"     {jid}: waiting for its `gpuq submit` to reconnect after a gpuqd restart")
    if s["queue"]:
        print("  A promise (2+ cards) can only get earlier; an estimate can move either way.")
        print("  `gpuq why JOB` explains a place in line.")
    if s.get("joining"):
        print(f"\njoining their own card, not in line: each starts once its card has room "
              f"({len(s['joining'])})")
        for j in s["joining"]:
            label = f"  {j['name']}" if j.get("name") else ""
            until = f"  must end by {when(j['until'], now)}" if j.get("until") else ""
            print(f"  {j['user']:<10} {j['job']:<11} GPU {','.join(map(str, j['cards'])):<6} "
                  f"needs {j['need_gb']:g} GB, {j['free_gb']:.0f} GB free{until}{label}")
    if s.get("untracked"):
        print("\nGPU use outside gpuq: not allowed. "
              + ("Stopped about a minute after it is first seen."
                 if s.get("enforcing", True) else "Logged; root has switched stopping off."))
        for u in s["untracked"]:
            print(f"  {u['user']:<10} GPU {u['card']}  {u['used_mb'] / 1024:.1f} GB  "
                  f"pids {','.join(map(str, u['pids']))}")
    return 0


def cmd_why(args):
    print(ask({"op": "why", "job": args.job_id})["text"])
    return 0


def cmd_share(args):
    r = ask({"op": "share", "user": args.user, "all": bool(args.all)})
    print("Recent use, in card-hours held (fading by half every "
          f"{r['half_life_days']} days). Least first goes first:")
    for row in r["usage"]:
        mark = "  <- you" if row["user"] == r["you"] and args.all else ""
        print(f"  {row['user']:<12} {row['card_hours']:>8.1f}{mark}")
    for p in r["places"]:
        print(f"{r['you']}'s job {p['job']} is number {p['place']} in line.")
    return 0
