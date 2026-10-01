"""gpuq submit and gpuq shell.

An attached submit passes its terminal (stdin, stdout, stderr) to gpuqd with the
request. The job writes to it directly, exactly as the previous gpuq's jobs did. The client
then waits: it prints when the job is queued, starts and ends, and exits with
the job's exit code (128+N if a signal N ended it), because scripts chain on
it. Ctrl-C, or closing the terminal, cancels a waiting job or passes the signal
on to a running one, as today.

If gpuqd restarts meanwhile, the client reconnects and asks for its job back.
The job itself keeps running: it is its own systemd unit.
"""
import os
import pwd
import shlex
import signal
import sys
import time
from datetime import datetime, timedelta

from gpuqd import protocol

from .common import connect, die

# gpuqd gives a returning client 120 s after it restarts; try for most of that.
RECONNECT_FOR_S = 100


def _when(t):
    return datetime.fromtimestamp(t).strftime("%a %H:%M") if t else "?"


def _cards(cards):
    return ",".join(str(c) for c in cards or ())


def say(msg):
    print(f"[gpuq] {msg}", file=sys.stderr, flush=True)


def command_of(args):
    cmd = list(args.cmd_args or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if args.command:
        if cmd:
            die("give the command either after `--` or via --command, not both.")
        cmd = shlex.split(args.command)
    if not cmd:
        die("no command given. Use:  gpuq submit [opts] -- COMMAND ARGS...")
    return cmd


def gpus_of(args):
    """(-g, the cards of --devices). --devices adds the job to cards your own
    jobs are running on (§4.5); gpuqd checks that you hold them."""
    devices = None
    if getattr(args, "devices", None):
        try:
            devices = sorted({int(x) for x in args.devices.split(",") if x.strip()})
        except ValueError:
            die(f"--devices must be comma-separated GPU indices, got {args.devices!r}")
        if not devices:
            die("--devices was given but empty")
        if args.gpus is not None and args.gpus != len(devices):
            die(f"--devices {args.devices} implies -g {len(devices)}; drop -g or make them match.")
        return len(devices), devices
    return (1 if args.gpus is None else args.gpus), None


def notes(args):
    """One line for each old flag whose meaning changed."""
    if args.time is not None:
        say(f"-t {args.time:g} is ignored: every job may run 48 h "
            "(for a shorter limit: timeout 4h python ...).")
    if args.notify:
        say("--notify: the email goes to your account's address.")


def cmd_submit(args):
    cmd = command_of(args)
    gpus, devices = gpus_of(args)
    notes(args)
    req = {"op": "submit", "argv": cmd, "gpus": gpus, "mem_gb": args.memory, "name": args.name,
           "notify": args.notify is not None, "devices": devices, "time_h": args.time,
           "mode": "detached" if args.detach else "attached"}
    return run(req)


def cmd_shell(args):
    shell = os.environ.get("SHELL")
    if not shell:
        try:
            shell = pwd.getpwuid(os.getuid()).pw_shell
        except KeyError:
            shell = "/bin/bash"
    if not os.isatty(0) or not os.isatty(1):
        die("gpuq shell needs a terminal; use `gpuq submit` for scripts.")
    gpus, devices = gpus_of(args)
    req = {"op": "submit", "argv": [shell], "gpus": gpus, "mem_gb": args.memory,
           "devices": devices, "name": args.name or "shell", "mode": "shell"}
    return run(req)


def run(req):
    req.update(env=dict(os.environ), cwd=os.getcwd())
    detached = req["mode"] == "detached"
    sock = connect()
    fds = [] if detached else [0, 1, 2]
    try:
        protocol.send(sock, req, fds)
        reply, _ = protocol.recv(sock)
    except (OSError, protocol.Closed) as e:
        die(f"gpuqd did not take the job: {e}")
    if not reply.get("ok"):
        die(reply.get("error") or "gpuqd refused the job")
    job = reply["job"]
    if detached:
        return _detached(sock, job, reply.get("log"))
    return Waiter(sock, job, req["mode"]).wait()


def _detached(sock, job, log):
    try:
        ev, _ = protocol.recv(sock)
    except (OSError, protocol.Closed):
        ev = {}
    if ev.get("event") == "started":
        say(f"job {job} started on GPU(s) {_cards(ev['cards'])}{_joined(ev)}; output: {log}")
    elif ev.get("event") == "queued":
        say(f"job {job} queued{_placement(ev)}; output will go to {log}")
    else:
        say(f"job {job} submitted; output: {log}")
    print(job)
    return 0


def _joined(ev):
    if not ev.get("joined"):
        return ""
    return (f", beside your job(s) there. It must end by {_when(ev['deadline'])}, when they "
            "reach their 48 h")


def _room(ev):
    return (f"joins your GPU {_cards(ev['cards'])} once {ev['need_gb']:g} GB is free there "
            f"({ev['free_gb']:.0f} GB is now). It does not wait in line")


def _placement(ev):
    if ev.get("kind") == "room":
        return f": {_room(ev)}"
    if ev.get("kind") == "promise":
        return (f". Promised: starts by {_when(ev['t'])} on GPUs {_cards(ev['cards'])}. "
                "It can only get earlier")
    if ev.get("kind") == "estimate" and ev.get("t"):
        return f". Estimate: ~{_when(ev['t'])}"
    if ev.get("kind") == "impossible":
        return ". This host cannot run it as asked"
    return ""


class Waiter:
    def __init__(self, sock, job, mode):
        self.sock = sock
        self.job = job
        self.mode = mode
        self.started = None

    def _forward(self, sig, _frame):
        try:
            protocol.send(self.sock, {"op": "signal", "sig": int(sig)})
        except OSError:
            pass

    def _winch(self, _sig, _frame):
        try:
            protocol.send(self.sock, {"op": "winch"})
        except OSError:
            pass

    def wait(self):
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, self._forward)
        if self.mode == "shell":
            signal.signal(signal.SIGWINCH, self._winch)
        while True:
            try:
                ev, fds = protocol.recv(self.sock)
            except (protocol.Closed, OSError):
                ev = self._come_back()
                if ev is None:
                    continue
            else:
                for fd in fds:
                    os.close(fd)
            done = self.show(ev)
            if done is not None:
                return done

    def show(self, ev):
        """Print one event; return the exit code once the job is over."""
        kind = ev.get("event")
        if kind == "queued" and ev.get("kind") == "room":
            say(f"job {self.job} {_room(ev)}. Ctrl-C cancels (or `gpuq kill {self.job}`).")
        elif kind == "queued":
            free = ev.get("free")
            need = ev.get("need")
            say(f"job {self.job} queued: needs {need} card(s), {free} free"
                f"{_placement(ev)}. Ctrl-C cancels (or `gpuq kill {self.job}`).")
        elif kind == "started":
            self.started = time.time()
            say(f"job {self.job} starting on GPU(s) {_cards(ev['cards'])}{_joined(ev)}.")
        elif kind == "error":
            say(ev.get("message", "error"))
        elif kind == "cancelled":
            say(ev.get("message") or "cancelled while queued.")
            return 1 if ev.get("exit") is None else ev["exit"]
        elif kind == "ended":
            rc, reason = ev.get("exit"), ev.get("reason")
            if ev.get("started") and ev.get("ended"):
                ran = timedelta(seconds=int(ev["ended"] - ev["started"]))
                hours = (ev["ended"] - ev["started"]) / 3600 * len(ev.get("cards") or ())
                say(f"job {self.job} {reason}: ran {ran} on GPU(s) {_cards(ev.get('cards'))}, "
                    f"{hours:.2f} GPU-hours recorded (exit {rc}).")
            else:
                say(f"job {self.job} {reason} (exit {rc}).")
            return 1 if rc is None else rc
        return None

    def _come_back(self):
        """gpuqd went away (a restart): reconnect and ask for the job back."""
        say("lost contact with gpuqd; reconnecting ...")
        try:
            self.sock.close()
        except OSError:
            pass
        give_up = time.time() + RECONNECT_FOR_S
        while time.time() < give_up:
            time.sleep(1)
            try:
                sock = protocol.connect()
                protocol.send(sock, {"op": "attach", "job": self.job}, [0, 1, 2])
                reply, _ = protocol.recv(sock)
            except (OSError, protocol.Closed):
                continue
            if not reply.get("ok"):
                die(f"gpuqd no longer knows job {self.job}: {reply.get('error')}")
            self.sock = sock
            say(f"reconnected; job {self.job} is {reply['state']}.")
            if reply["state"] == "ended":
                return {"event": "ended", "exit": reply.get("exit"),
                        "reason": reply.get("reason")}
            return None
        die(f"could not reach gpuqd for {RECONNECT_FOR_S} s; job {self.job} is stopped "
            "once gpuqd is back, since its terminal is gone.")
