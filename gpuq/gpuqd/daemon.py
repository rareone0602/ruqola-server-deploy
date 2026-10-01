"""gpuqd: the one owner of the GPU queue (docs/v3-design.md §7).

Every PASS_S, and at once when a job is submitted or ends, it reads the cards,
asks the v3 planner what to do (scheduler/planner.py), and starts what the plan
says, each job as its own systemd unit run as its submitter (runner.py). It is
the only writer of the queue and the ledger (store.py).

This module is the queue's logic, driven by explicit times so tests can run
it pass by pass; server.py is the socket loop around it.
"""
import os
import pwd
import secrets
import signal
import socket
import sys

from scheduler.model import HOUR, MAX_RUNTIME_H
from scheduler.planner import plan

from . import explain
from .devices import device_minors
from .history import Ledger, usage
from .holders import Holders
from .jobs import MODES, as_job, as_running, cancelled_record, end_record, exit_of
from .nvsmi import read_cards, read_procs
from .procinfo import read_limits, read_umask
from .runner import STOP_GRACE_S, StartError
from .sightings import Sightings
from .stopper import Stopper

PASS_S = 30.0
# Passes asked for by submits and job ends run at most this often.
MIN_PASS_GAP_S = 1.0
# After gpuqd restarts, an attached client has this long to come back for its job.
REATTACH_S = 120.0
# The deadline warning: an hour, so jobs can checkpoint (§4.3).
WARN_BEFORE_S = HOUR
# "Your job started" mail goes out only after a wait this long; shorter waits
# are seen at the terminal.
STARTED_MAIL_AFTER_S = 600.0
# A card using at most this much VRAM with no person's process on it counts as
# empty (driver housekeeping), so a job that wants a card to itself can start.
IDLE_MB = 1024
# status, why and share run a fresh pass when the last is older than this.
STATUS_FRESH_S = 5.0
# A job's --name is kept to this many characters.
NAME_MAX = 200
# How many ended jobs are remembered for clients coming back after a restart.
ENDED_KEEP = 1000
# A running job whose unit is gone with no exit report is lost after this many passes.
LOST_AFTER_PASSES = 2
# A job joining your card (--devices) starts only while your hold on it has at
# least this long left; a hold closer to its end is being stopped.
JOIN_MIN_LEFT_S = 60.0


def _err(message):
    return {"ok": False, "error": message}


def _str_list(v):
    return isinstance(v, list) and v and all(isinstance(x, str) for x in v)


def _cards(cards):
    return ",".join(str(c) for c in cards)


def _str_dict(v):
    return isinstance(v, dict) and all(isinstance(k, str) and isinstance(x, str)
                                       for k, x in v.items())


class Daemon:
    def __init__(self, store, runner, mailer, *, host=None, nvsmi=None, proc="/proc",
                 minors=device_minors, users=None, log=None, stopper=None):
        self.store = store
        self.runner = runner
        self.mailer = mailer
        self.host = host or socket.gethostname()
        self.nvsmi = nvsmi
        self.proc = proc
        self.minors = minors
        self.log = log or (lambda line: print(line, flush=True))
        self.jobs = {}              # id -> job (see jobs.py)
        self.promised = ()          # the last plan's reservations, oldest first
        self.clients = {}           # job id -> the connection waiting on it
        self.fds = {}               # job id -> the terminal of an attached job, until it starts
        self.helpers = {}           # job id -> systemd-run helper (attached jobs, shells)
        self.orphans = {}           # job id -> when its client must be back by (after a restart)
        self.ended = {}             # job id -> {"exit", "reason"}: for clients coming back
        self.missing = {}           # job id -> passes its unit was seen gone with no report
        self.first = set()          # submitted, first outcome not yet told
        self.view = None
        self.want_pass = False
        self.last_pass = None
        self.ledger = Ledger(store.root)
        self.holders = Holders(proc, users)
        self.sightings = Sightings()
        self.reported = set()
        self.stopper = stopper or Stopper(proc, self.log, mailer, self.host,
                                          self.holders._name)
        self.driver_down = False

    # -- state --------------------------------------------------------------------

    def recover(self, now):
        """Load the saved queue after a (re)start. Jobs are their own units and
        kept running; attached clients get REATTACH_S to come back."""
        data = self.store.load()
        for job in data["jobs"]:
            self.jobs[int(job["id"])] = job
            if job["mode"] != "detached":
                self.orphans[int(job["id"])] = now + REATTACH_S
        self.promised = tuple(data["promised"])
        self.want_pass = True

    def save(self):
        self.store.save(list(self.jobs.values()), self.promised)

    def tell(self, job_id, msg):
        conn = self.clients.get(job_id)
        if conn is not None:
            conn.send(dict(msg, job=job_id))

    def _close_fds(self, job_id):
        for fd in self.fds.pop(job_id, ()):
            try:
                os.close(fd)
            except OSError:
                pass

    def _remember(self, jid, rc, reason):
        self.ended[jid] = {"exit": rc, "reason": reason}
        while len(self.ended) > ENDED_KEEP:
            del self.ended[next(iter(self.ended))]

    def _new_id(self):
        while True:
            jid = 10 ** 8 + secrets.randbelow(2 ** 31 - 10 ** 8)
            if jid not in self.jobs and jid not in self.ended:
                return jid

    # -- requests -------------------------------------------------------------------

    def submit(self, conn, msg, fds, now):
        def refuse(message):
            for fd in fds:
                os.close(fd)
            return _err(message)

        pid, uid, _ = conn.peer
        try:
            pw = pwd.getpwuid(uid)
        except KeyError:
            return refuse("your account has no passwd entry")
        mode = msg.get("mode", "attached")
        argv, env, cwd = msg.get("argv"), msg.get("env"), msg.get("cwd")
        if mode not in MODES:
            return refuse(f"unknown mode {mode!r}")
        if not _str_list(argv):
            return refuse("no command given. Use:  gpuq submit [opts] -- COMMAND ARGS...")
        if not _str_dict(env) or not isinstance(cwd, str):
            return refuse("the request has no environment or folder")
        try:
            gpus = 1 if msg.get("gpus") is None else int(msg["gpus"])
            mem = None if msg.get("mem_gb") is None else float(msg["mem_gb"])
            time_h = None if msg.get("time_h") is None else float(msg["time_h"])
            if not isinstance(msg.get("devices") or [], list):
                raise TypeError("devices")
            devices = sorted({int(d) for d in msg.get("devices") or ()}) or None
        except (TypeError, ValueError):
            return refuse("-g, -m, -t and --devices must be numbers")
        if devices is not None:
            if msg.get("gpus") is not None and gpus != len(devices):
                return refuse(f"--devices {_cards(devices)} is {len(devices)} card(s), "
                              f"but -g says {gpus}.")
            gpus = len(devices)
        if gpus < 1:
            return refuse("-g/--gpus must be at least 1.")
        if mem is not None and not mem > 0:
            return refuse("-m/--memory must be more than 0 GB.")
        if mode != "detached" and len(fds) != 3:
            return refuse("an attached job needs the terminal (stdin, stdout, stderr)")
        if self.view and self.view.cards:
            n = len(self.view.cards)
            biggest = max(c.total_gb for c in self.view.cards.values())
            if gpus > n:
                return refuse(f"this host has {n} GPU(s); you asked for {gpus}.")
            if mem is not None and mem > biggest:
                return refuse(f"-m {mem:g}: the largest card here has {biggest:.0f} GB.")
        if devices is not None:
            problem = self._join_problem(pw.pw_name, devices)
            if problem:
                return refuse(problem)
        name = msg.get("name")
        name = None if name is None else str(name)[:NAME_MAX]
        jid = self._new_id()
        job = {"id": jid, "user": pw.pw_name, "uid": uid, "gid": pw.pw_gid, "home": pw.pw_dir,
               "argv": argv, "cwd": cwd, "env": env,
               "umask": read_umask(pid, self.proc), "limits": read_limits(pid, self.proc),
               "gpus": gpus, "mem_gb": mem, "mode": mode, "name": name,
               "notify": bool(msg.get("notify")), "devices": devices,
               "time_h": time_h, "submitted": now, "state": "queued"}
        if mode == "detached":
            job["log"] = os.path.join(pw.pw_dir, "gpuq-logs", f"{jid}.log")
            for fd in fds:
                os.close(fd)
        else:
            self.fds[jid] = list(fds)
        self.jobs[jid] = job
        self.clients[jid] = conn
        conn.job = jid
        self.first.add(jid)
        self.log(f"submit {jid} {job['user']} {gpus} GPU {mode}"
                 + ("" if mem is None else f" -m {mem:g}")
                 + ("" if devices is None else f" joining GPU {_cards(devices)}"))
        self.save()
        self.want_pass = True
        return {"ok": True, "job": jid, "log": job.get("log")}

    def _holds(self, user):
        """card -> when `user`'s hold on it ends: the latest deadline of their
        gpuq jobs running there."""
        held = [(j["cards"], j["deadline"]) for j in self.jobs.values()
                if j["state"] == "running" and j["user"] == user]
        out = {}
        for cards, end in held:
            for c in cards:
                out[c] = max(out.get(c, end), end)
        return out

    def _join_problem(self, user, devices):
        """Why `user` may not join `devices`, or None. --devices adds a job to
        cards the user's gpuq jobs are running on now (§4.5); it no longer picks
        a free card, so an old script that did gets told what to use instead."""
        holds = self._holds(user)
        missing = [c for c in devices if c not in holds]
        if not missing:
            return None
        yours = (f"Your jobs are running on GPU {_cards(sorted(holds))}." if holds
                 else "You have no job running now.")
        return (f"--devices {_cards(devices)}: you have no job running on GPU "
                f"{_cards(missing)}. --devices adds a job to a card one of your jobs is "
                f"already running on. {yours} For a card of its own, leave out --devices "
                "(-g N asks for N cards; gpuq picks which).")

    def attach(self, conn, msg, fds, now):
        """A client coming back for its job, after gpuqd restarted."""
        try:
            jid = int(msg.get("job"))
        except (TypeError, ValueError):
            jid = None
        job = self.jobs.get(jid)
        if job is None:
            for fd in fds:
                os.close(fd)
            if jid in self.ended:
                return {"ok": True, "state": "ended", **self.ended[jid]}
            return _err(f"no job {jid} here")
        if conn.peer[1] != job["uid"]:
            for fd in fds:
                os.close(fd)
            return _err(f"job {jid} belongs to {job['user']}")
        if job["state"] == "queued" and job["mode"] != "detached" and len(fds) == 3:
            self._close_fds(jid)
            self.fds[jid] = list(fds)
        else:
            for fd in fds:
                os.close(fd)
        self.clients[jid] = conn
        conn.job = jid
        self.orphans.pop(jid, None)
        self.want_pass = True
        return {"ok": True, "state": job["state"], "cards": job.get("cards")}

    def signal(self, conn, sig, now):
        """The attached client got a signal: cancel a waiting job, or pass the
        signal on to a running one, as the previous gpuq did."""
        job = self.jobs.get(conn.job)
        if job is None or self.clients.get(job["id"]) is not conn:
            return
        if job["state"] == "queued":
            interrupted = sig == signal.SIGINT
            self.cancel(job, "user" if interrupted else "signal", now,
                        130 if interrupted else 128 + sig)
        else:
            self.runner.signal(job["id"], sig)

    def winch(self, conn):
        """A shell's terminal was resized: tell its forwarder to look again."""
        helper = self.helpers.get(conn.job)
        if helper is not None and helper.poll() is None:
            try:
                helper.send_signal(signal.SIGWINCH)
            except OSError:
                pass

    def disconnected(self, conn, now):
        """An attached client is gone: its terminal closed, or it was killed.
        The job goes with it, as today."""
        jid = conn.job
        if jid is None or self.clients.get(jid) is not conn:
            return
        del self.clients[jid]
        job = self.jobs.get(jid)
        if job is None or job["mode"] == "detached":
            return
        if job["state"] == "queued":
            self.cancel(job, "lost", now)
        elif not job.get("stop_reason"):
            job["stop_reason"] = "killed"
            self.log(f"stop {jid}: its terminal is gone")
            self.runner.stop(jid)
            self.save()

    def kill(self, conn, msg, now):
        uid = conn.peer[1]
        if msg.get("mine"):
            mine = [j for j in self.jobs.values() if j["uid"] == uid]
            ids = ([j["id"] for j in mine if j["state"] == "queued"]
                   + [j["id"] for j in mine if j["state"] == "running"])
        else:
            ids = msg.get("jobs") or []
        results = []
        for jid in ids:
            try:
                jid = int(jid)
            except (TypeError, ValueError):
                results.append({"job": jid, "ok": False, "message": f"{jid!r} is not a job id"})
                continue
            job = self.jobs.get(jid)
            if job is None:
                results.append({"job": jid, "ok": False, "message":
                                f"no running or queued job {jid} (or it already finished)."})
            elif uid not in (0, job["uid"]):
                results.append({"job": jid, "ok": False, "message":
                                f"job {jid} belongs to {job['user']}; you can only kill your own jobs."})
            elif job["state"] == "queued":
                self.cancel(job, "user", now, 130)
                results.append({"job": jid, "ok": True, "message": f"cancelled queued job {jid}."})
            else:
                job["stop_reason"] = "killed"
                self.runner.stop(jid)
                self.log(f"stop {jid}: gpuq kill by uid {uid}")
                results.append({"job": jid, "ok": True, "message":
                                f"stopping job {jid}: SIGTERM now, SIGKILL in {STOP_GRACE_S} s if it is still running."})
        self.save()
        return {"ok": True, "results": results}

    def _fresh(self, now):
        """Answer from a pass that has seen every request so far."""
        if (self.want_pass or self.last_pass is None
                or now - self.last_pass >= STATUS_FRESH_S):
            self.step(now)

    def status(self, conn, msg, now):
        self._fresh(now)
        return {"ok": True, **explain.snapshot(self.view, self.jobs, self.host, self.orphans),
                "enforcing": self.stopper.enforcing}

    def why(self, conn, msg, now):
        try:
            jid = int(msg.get("job"))
        except (TypeError, ValueError):
            return _err("why: give a job id, e.g. `gpuq why 12345`")
        self._fresh(now)
        return {"ok": True, "text": explain.why(self.view, self.jobs, jid, self.host)}

    def share(self, conn, msg, now):
        user = msg.get("user")
        if not user:
            try:
                user = pwd.getpwuid(conn.peer[1]).pw_name
            except KeyError:
                user = str(conn.peer[1])
        self._fresh(now)
        return {"ok": True, **explain.share(self.view, user, bool(msg.get("all")))}

    def config(self, conn, msg, now):
        return {"ok": True, "host": self.host, "max_runtime_h": MAX_RUNTIME_H,
                "state_dir": str(self.store.root), "ledger": str(self.store.ledger),
                "mail": self.mailer.enabled, "enforcing": self.stopper.enforcing}

    # -- the pass -------------------------------------------------------------------

    def tick(self, now):
        """Called about once a second: notice ended jobs, finish stopping GPU use
        outside gpuq, and run a pass when due."""
        self.reap(now)
        self.stopper.tick(now)
        since = None if self.last_pass is None else now - self.last_pass
        if since is None or since >= PASS_S or (self.want_pass and since >= MIN_PASS_GAP_S):
            self.step(now)

    def step(self, now):
        self.last_pass = now
        self.want_pass = False
        self.reap(now)
        self._check_units(now)
        cards = read_cards(self.nvsmi)
        procs = read_procs(cards, self.nvsmi) if cards else None
        if procs is None:
            if not self.driver_down:
                self.log("nvidia-smi did not answer: starting nothing until it does")
            self.driver_down = True
            self.view = explain.View(now, None)
            self._first_outcomes()
            self.save()
            return
        self.driver_down = False
        whole = max(c.total_gb for c in cards.values())
        cap = {i: c.total_gb for i, c in cards.items()}
        running = {j["id"]: j for j in self.jobs.values() if j["state"] == "running"}
        pic = self.holders.read(now, procs, running)
        idle = {i for i, c in cards.items()
                if c.used_mb <= IDLE_MB and not any(u[1] == i for u in pic.untracked)}
        free = {i: (c.total_gb if i in idle else c.free_gb) for i, c in cards.items()}
        mine = [as_running(j, whole) for j in running.values()]
        # A card running a job is never empty, however little the job has
        # allocated yet, so a job without -m (a card to itself) needs a card with
        # no job on it. That holds for use outside gpuq too. You own your allocated card: more of your own jobs
        # with -m join it whenever that much VRAM is measured free (§4.5),
        # whatever its first job asked for, and fair-share charges the card once.
        for r in mine + pic.holding:
            for c in r.cards:
                free[c] = min(free[c], cap[c] - 1.0)
        # A waiting attached job whose client has not come back after a restart
        # cannot start: it has no terminal to run in.
        waiting = [j for j in self.jobs.values() if j["state"] == "queued"
                   and (j["mode"] == "detached" or j["id"] in self.fds)]
        queue, join_ends = [], {}
        for j in waiting:
            if not j.get("devices"):
                queue.append(as_job(j, whole))
                continue
            holds = self._holds(j["user"])
            if not all(c in holds for c in j["devices"]):
                self._released(j, now)
                continue
            end = min(holds[c] for c in j["devices"])
            if end - now >= JOIN_MIN_LEFT_S:
                join_ends[j["id"]] = end
                queue.append(as_job(j, whole, (end - now) / HOUR))
        self.ledger.refresh()
        use = usage(self.ledger.runs, mine, now)
        p = plan(now, cap, mine + pic.holding, queue, use, free, self.promised)
        self.promised = tuple(r.job_id for r in p.reservations)
        for s in p.starts:
            self._launch(self.jobs[int(s.job_id)], s.cards, cards, now,
                         join_ends.get(int(s.job_id)))
        self.view = explain.View(now, cards, cap=cap, free=free, usage=use, plan=p,
                                 queue=queue, untracked=pic.untracked,
                                 held=mine + pic.holding)
        self._first_outcomes()
        self._watch(now, pic)
        self.save()

    def _first_outcomes(self):
        """Tell each new submitter where its job stands after its first pass."""
        for jid in sorted(self.first):
            job = self.jobs.get(jid)
            if job is not None and job["state"] == "queued":
                self.tell(jid, {"event": "queued", **explain.place(self.view, job)})
        self.first.clear()

    def _launch(self, job, cards, card_info, now, deadline=None):
        """Start `job` on `cards`. A job joining your cards passes `deadline`,
        when your hold on them ends; any other job may run MAX_RUNTIME_H."""
        jid = job["id"]
        table = self.minors()
        try:
            minors = [table[card_info[c].uuid] for c in cards]
        except (TypeError, KeyError):
            # Without the device table a job cannot be confined to its cards.
            self.log(f"cannot start {jid}: the driver's device table is unreadable")
            return
        job.update(state="running", cards=list(cards), started=now,
                   deadline=now + MAX_RUNTIME_H * HOUR if deadline is None else deadline)
        try:
            helper = self.runner.start(job, minors, self.fds.get(jid), now)
        except StartError as e:
            self.log(f"start {jid} failed: {e}")
            self.tell(jid, {"event": "error", "message": f"the job could not start: {e}"})
            self.finish(job, now, None, "failed")
            return
        finally:
            self._close_fds(jid)
        if helper is not None:
            self.helpers[jid] = helper
        self.first.discard(jid)
        waited = now - job["submitted"]
        self.log(f"start {jid} {job['user']} on GPU {_cards(cards)} after {waited / HOUR:.2f} h"
                 + (" (joining their own card)" if job.get("devices") else ""))
        self.tell(jid, {"event": "started", "cards": list(cards), "deadline": job["deadline"],
                        "joined": bool(job.get("devices"))})
        if waited >= STARTED_MAIL_AFTER_S:
            self.mailer.send(job["user"], f"[gpuq] job {jid} started on GPU(s) "
                             f"{','.join(map(str, cards))}", explain.started_mail(job, self.host))

    def reap(self, now):
        """Finish jobs whose units have ended; warn and stop at the deadline;
        let go of attached jobs whose clients never came back."""
        for jid, job in list(self.jobs.items()):
            if job["state"] != "running":
                continue
            rep = self.store.read_exit(jid)
            if rep is not None:
                rc, reason = exit_of(rep)
                if job.get("stop_reason") and rc != 0:
                    reason = job["stop_reason"]
                self.finish(job, min(float(rep.get("t") or now), now), rc, reason)
                continue
            helper = self.helpers.get(jid)
            if helper is not None and helper.poll() is not None:
                del self.helpers[jid]
                if job["mode"] == "shell":
                    self.runner.stop(jid)       # the terminal forwarder ended
            if not job.get("warned") and now >= job["deadline"] - WARN_BEFORE_S:
                job["warned"] = True
                self.mailer.send(job["user"], f"[gpuq] job {jid}: 1 hour left",
                                 explain.deadline_mail(job, self.host))
                self.save()
            if now >= job["deadline"] + STOP_GRACE_S + 60 and not job.get("stop_reason"):
                # systemd's RuntimeMaxSec should have done this already.
                job["stop_reason"] = "timed_out"
                self.runner.stop(jid)
                self.save()
        for jid, until in list(self.orphans.items()):
            job = self.jobs.get(jid)
            if job is None:
                del self.orphans[jid]
            elif now >= until:
                del self.orphans[jid]
                if job["state"] == "queued":
                    self.cancel(job, "lost", now)
                elif not job.get("stop_reason"):
                    job["stop_reason"] = "killed"
                    self.log(f"stop {jid}: its client did not come back after the restart")
                    self.runner.stop(jid)
                    self.save()

    def _check_units(self, now):
        """A running job with no exit report whose unit is gone: lost."""
        for jid, job in list(self.jobs.items()):
            if job["state"] != "running" or jid in self.helpers:
                self.missing.pop(jid, None)
                continue
            if self.runner.active(jid) is False:
                self.missing[jid] = self.missing.get(jid, 0) + 1
                if self.missing[jid] >= LOST_AFTER_PASSES and self.store.read_exit(jid) is None:
                    self.finish(job, now, None, "lost", synthetic=True)
            else:
                self.missing.pop(jid, None)

    def finish(self, job, t, rc, reason, synthetic=False):
        jid = job["id"]
        self.jobs.pop(jid, None)
        self.store.append_ledger(end_record(job, t, rc, reason, self.host, synthetic))
        self.store.drop_exit(jid)
        self.store.drop_launch(jid)
        for d in (self.helpers, self.missing, self.orphans, self.fds):
            d.pop(jid, None)
        self.first.discard(jid)
        self._remember(jid, rc, reason)
        self.log(f"end {jid} {job['user']}: {reason} (exit {rc})")
        self.tell(jid, {"event": "ended", "exit": rc, "reason": reason, "cards": job["cards"],
                        "started": job["started"], "ended": t})
        self.clients.pop(jid, None)
        if job.get("notify"):
            self.mailer.send(job["user"], f"[gpuq] job {jid} {reason}",
                             explain.end_mail(job, t, rc, reason, self.host))
        self.want_pass = True
        self.save()

    def _released(self, job, now):
        """A job waiting to join cards its user no longer holds: cancelled, never
        moved elsewhere (it asked for those cards)."""
        cards = _cards(job["devices"])
        self.cancel(job, "released", now, message=(
            f"job {job['id']} was cancelled: your job(s) on GPU {cards} ended before there "
            f"was room for it to join them (--devices {cards}). To run it on a card of its "
            "own, submit it again without --devices."))

    def cancel(self, job, reason, now, exit_code=None, message=None):
        jid = job["id"]
        self.jobs.pop(jid, None)
        self.store.append_ledger(cancelled_record(job, reason, now, self.host))
        self._close_fds(jid)
        self.orphans.pop(jid, None)
        self.first.discard(jid)
        self._remember(jid, exit_code, f"cancelled ({reason})")
        self.log(f"cancel {jid} {job['user']}: {reason}")
        self.tell(jid, {"event": "cancelled", "reason": reason, "exit": exit_code,
                        "message": message})
        self.clients.pop(jid, None)
        if message and job["mode"] == "detached":
            self.mailer.send(job["user"], f"[gpuq] job {jid} cancelled",
                             f"Host: {self.host}\nJob: {jid}\n\n{message}\n")
        self.want_pass = True
        self.save()

    def _watch(self, now, pic):
        """Rule 4 (§6): a GPU process outside every gpuq job is stopped on its
        second sighting, at least a minute after the first (stopper.py). While
        root has switched stopping off, it is only logged."""
        due = self.sightings.observe(now, {o.key for o in pic.offenders})
        if self.stopper.enforcing:
            self.stopper.stop([o for o in pic.offenders if o.key in due], now)
            return
        for o in pic.offenders:
            if o.key in due and o.key not in self.reported:
                self.reported.add(o.key)
                where = "outside every gpuq job" if o.job is None else f"off job {o.job}'s cards"
                self.log(f"rule 4 would stop pid {o.pid} of {o.user} on GPU {o.card} "
                         f"({where}, {o.used_mb} MiB, {o.name})")
        self.reported &= {o.key for o in pic.offenders}


def main(argv=None):
    from .mail import Mailer
    from .runner import SystemdRunner
    from .server import serve
    from .store import Store
    store = Store()
    store.prepare()
    daemon = Daemon(store, SystemdRunner(store), Mailer(host=socket.gethostname()))
    return serve(daemon)


if __name__ == "__main__":
    sys.exit(main())
