"""Before cutting over: run real jobs the way gpuqd will, as root, on this host,
without touching the live queue or /var/lib/gpuq.

    sudo env PYTHONPATH=$PWD python3 -m gpuqd.smoke --cards 0,1 [--user NAME] [--python PY]

Each check starts a real unit through gpuqd's own runner (runner.py), as
--user, given the two cards in --cards, with state in a temporary root-only
folder that is removed afterwards:

  1. the job runs as the user, in their folder, with the environment, umask
     and open-file limit it was given;
  2. its exit code comes back through the root exit hook (exit 3 -> 3);
  3. CUDA sees exactly its two cards, numbered from 0;
  4. nvidia-smi inside lists only those two;
  5. NCCL all-reduces across the two cards (needs --python: a Python with torch).
     If not, it says how far each worker got, shows NCCL's log, and tries again
     with GPU-to-GPU transfers off and without the device list, to find the cause;
  6. gpuq kill's stop: SIGTERM, reported as 143;
  7. the deadline stops it (RuntimeMaxSec), reported as timed_out;
  8. a detached job writes its log as the user;
  9. a shell gets a terminal of its own, sized, and typed input reaches it.

The CUDA probe creates no context. The NCCL check allocates 4 MiB per card for
a few seconds. It refuses cards running anyone's process unless --force.
"""
import argparse
import json
import os
import pty
import pwd
import re
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .confine import PROBE, _probe, _smi
from .devices import device_minors
from .jobs import exit_of
from .nvsmi import read_cards, read_procs
from .procinfo import read_limits
from .runner import SystemdRunner
from .store import Store

LIB = str(Path(__file__).resolve().parent.parent)
# An NCCL run that has not finished by then is stopped (a hang shows as timed_out).
NCCL_LIMIT_S = 90
FIRST_ID = 990_000_000          # unit names gpuq-job-99xxxxxxx, apart from real ids

NCCL = r"""
import datetime, json, os, socket, sys
import torch, torch.distributed as dist, torch.multiprocessing as mp

def step(rank, what):
    print(f"rank {rank}: {what}", file=sys.stderr, flush=True)

def run(rank, world, port):
    step(rank, f"torch {torch.__version__}, nccl {torch.cuda.nccl.version()}, "
               f"{torch.cuda.device_count()} device(s)")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world, timeout=datetime.timedelta(seconds=60),
                            device_id=torch.device("cuda", rank))
    step(rank, "process group up")
    t = torch.full((1 << 20,), float(rank + 1), device="cuda")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    step(rank, "all-reduce done")
    if rank == 0:
        # On stderr, as one short write: NCCL_DEBUG fills stdout from both
        # workers at once, and their lines interleave mid-line there.
        want = float(sum(range(1, world + 1)))
        step(rank, "result " + json.dumps({"ok": bool((t == want).all()), "value": t[0].item(),
                                           "devices": torch.cuda.device_count()}))
    dist.destroy_process_group()

if __name__ == "__main__":
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(run, args=(2, port), nprocs=2)
"""


class Smoke:
    def __init__(self, user, cards, card_info, python):
        self.pw = pwd.getpwnam(user)
        self.cards = cards
        self.card_info = card_info
        self.python = python
        self.tmp = Path(tempfile.mkdtemp(prefix="gpuq-smoke-"))
        os.chmod(self.tmp, 0o755)
        self.store = Store(self.tmp / "state")
        self.store.prepare()
        self.runner = SystemdRunner(self.store, lib=LIB)
        table = device_minors()
        self.minors = [table[card_info[c].uuid] for c in cards]
        # A fresh range each run, so a unit left by an interrupted run never clashes.
        self.first_id = self.next_id = FIRST_ID + (os.getpid() % 9000) * 100

    def job(self, argv, mode="attached", deadline_s=600, env=None, log=None):
        jid = self.next_id
        self.next_id += 1
        base = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": self.pw.pw_dir,
                "USER": self.pw.pw_name, "LOGNAME": self.pw.pw_name, "SMOKE": "a b $c"}
        return {"id": jid, "user": self.pw.pw_name, "uid": self.pw.pw_uid, "gid": self.pw.pw_gid,
                "argv": argv, "cwd": self.pw.pw_dir, "env": {**base, **(env or {})},
                "umask": 0o027, "limits": {"LimitNOFILE": "4096:8192"}, "mode": mode,
                "cards": list(self.cards), "deadline": time.time() + deadline_s, "log": log}

    def run(self, job, timeout=120, after_start=None, minors="own"):
        """Run an attached job; (stdout, stderr, (exit code, reason) or None)."""
        null = os.open(os.devnull, os.O_RDONLY)
        r_out, w_out = os.pipe()
        r_err, w_err = os.pipe()
        minors = self.minors if minors == "own" else minors
        p = self.runner.start(job, minors, [null, w_out, w_err], time.time())
        for fd in (null, w_out, w_err):
            os.close(fd)
        if after_start:
            threading.Timer(3.0, after_start).start()
        out, err = _drain([r_out, r_err], timeout)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
        return out, err, self._report(job["id"])

    def _report(self, jid):
        for _ in range(50):
            rep = self.store.read_exit(jid)
            if rep is not None:
                return exit_of(rep)
            time.sleep(0.1)
        return None

    def cleanup(self):
        for jid in range(self.first_id, self.next_id):
            if self.runner.active(jid):
                self.runner.stop(jid)
        shutil.rmtree(self.tmp, ignore_errors=True)


def _drain(fds, timeout):
    bufs = {fd: b"" for fd in fds}
    end = time.time() + timeout
    open_fds = list(fds)
    while open_fds and time.time() < end:
        ready, _, _ = select.select(open_fds, [], [], 0.2)
        for fd in ready:
            chunk = os.read(fd, 65536)
            if chunk:
                bufs[fd] += chunk
            else:
                open_fds.remove(fd)
                os.close(fd)
    for fd in open_fds:
        os.close(fd)
    return tuple(bufs[fd].decode(errors="replace") for fd in fds)


def check_identity(s):
    out, err, rep = s.run(s.job(["sh", "-c", 'id -un; pwd; echo "$SMOKE"; umask; ulimit -n; '
                                 'echo "$CUDA_VISIBLE_DEVICES $GPUQ_GPUS"; exit 3']))
    want = [s.pw.pw_name, s.pw.pw_dir, "a b $c", "0027", "4096", f"0,1 {s.cards[0]},{s.cards[1]}"]
    return [("runs as the user, in their folder, with its environment, umask and limits",
             out.splitlines() == want, out.splitlines() or err.strip()[-300:]),
            ("the exit code comes back through the root exit hook (exit 3 -> 3)",
             rep == (3, "failed"), rep)]


def check_cuda(s):
    out, err, rep = s.run(s.job([sys.executable, "-I", "-c", PROBE]))
    ans = _probe({"rc": 0 if rep and rep[0] == 0 else 1, "out": out})
    want = [s.card_info[c].uuid for c in s.cards]
    return [("CUDA sees exactly its two cards, numbered from 0",
             ans is not None and sorted(ans["uuids"]) == sorted(want),
             ans["uuids"] if ans else err.strip()[-300:])]


def check_smi(s):
    smi = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    out, err, rep = s.run(s.job([smi, "--query-gpu=index,uuid", "--format=csv,noheader"]))
    rows = _smi({"rc": 0 if rep and rep[0] == 0 else 1, "out": out})
    want = sorted(s.card_info[c].uuid for c in s.cards)
    return [("nvidia-smi inside lists only those two",
             rows is not None and sorted(u for _, u in rows) == want, rows or err.strip()[-300:])]


def _nccl(s, script, minors, extra_env):
    """One NCCL run: (passed, what was seen)."""
    job = s.job([s.python, str(script)], env={"NCCL_DEBUG": "INFO", **extra_env},
                deadline_s=NCCL_LIMIT_S)
    # Real jobs get the submitting shell's limits; systemd's own default of
    # 8 MiB locked memory is not what any job will run with.
    job["limits"] = read_limits(os.getpid())
    start = time.time()
    out, err, rep = s.run(job, timeout=NCCL_LIMIT_S + 30, minors=minors)
    ans = None
    for m in re.finditer(r"rank 0: result (\{.*?\})", out + "\n" + err):
        try:
            ans = json.loads(m.group(1))
        except ValueError:
            pass
    ok = bool(ans and ans.get("ok") and ans.get("devices") == 2 and rep == (0, "completed"))
    via = sorted({l.split(" via ", 1)[1].strip() for l in out.splitlines() if " via " in l})
    seen = {"exit": rep, "seconds": round(time.time() - start, 1), "result": ans,
            "transport": via[:4]}
    if not ok:
        tail = lambda text: [l for l in text.splitlines() if l.strip()][-12:]
        seen["steps"] = [l for l in err.splitlines() if l.startswith("rank ")]
        seen["stdout_tail"] = tail(out)
        seen["stderr_tail"] = [l for l in tail(err) if not l.startswith("rank ")]
    return ok, seen


def check_nccl(s):
    name = "NCCL all-reduces across the two cards"
    if not s.python:
        return [(name, None, "skipped: give --python, a Python with torch")]
    # A file, not -c: torch.multiprocessing's workers re-import the main script.
    script = s.tmp / "nccl.py"
    script.write_text(NCCL)
    script.chmod(0o644)
    ok, seen = _nccl(s, script, s.minors, {})
    rows = [(name, ok, _show(seen))]
    if not ok:
        # Which part breaks it: GPU-to-GPU transfers, or the device list itself?
        for label, minors, env in (
                ("confined, GPU-to-GPU (P2P) transfers off", s.minors, {"NCCL_P2P_DISABLE": "1"}),
                ("not confined (no device list), as a comparison", None, {})):
            ok2, seen2 = _nccl(s, script, minors, env)
            rows.append((f"diagnosis, {label}: {'works' if ok2 else 'fails too'}", "info",
                         _show(seen2)))
    return rows


def _show(seen):
    return "\n      ".join(f"{k}: {v}" if not isinstance(v, list) or len(v) < 2
                            else f"{k}:\n        " + "\n        ".join(map(str, v))
                            for k, v in seen.items())


def check_stop(s):
    job = s.job(["sleep", "60"])
    _, _, rep = s.run(job, after_start=lambda: s.runner.stop(job["id"]))
    return [("gpuq kill's stop: SIGTERM, reported as 143", rep == (143, "killed"), rep)]


def check_deadline(s):
    _, _, rep = s.run(s.job(["sleep", "60"], deadline_s=3))
    return [("the deadline stops it, reported as timed_out", rep == (143, "timed_out"), rep)]


def check_detached(s):
    folder = s.tmp / "logs"
    folder.mkdir()
    os.chown(folder, s.pw.pw_uid, s.pw.pw_gid)
    log = folder / "job.log"
    job = s.job(["sh", "-c", "echo to-the-log; id -un"], mode="detached", log=str(log))
    s.runner.start(job, s.minors, None, time.time())
    rep = s._report(job["id"])
    for _ in range(50):
        if rep is not None:
            break
        time.sleep(0.1)
        rep = s._report(job["id"])
    text = log.read_text() if log.exists() else ""
    owner = log.stat().st_uid if log.exists() else None
    return [("a detached job writes its log as the user",
             text == f"to-the-log\n{s.pw.pw_name}\n" and owner == s.pw.pw_uid and rep == (0, "completed"),
             {"log": text, "owner_uid": owner, "exit": rep})]


def check_shell(s):
    import fcntl
    import termios
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 132, 0, 0))
    job = s.job(["sh", "-c", 'tty >/dev/null && echo is-a-tty; stty size; read x; echo "got:$x"'],
                mode="shell")
    p = s.runner.start(job, s.minors, [slave, slave, slave], time.time())
    os.close(slave)
    out = b""
    end = time.time() + 20
    sent = False
    while time.time() < end:
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
        if not sent and b"132" in out:
            os.write(master, b"hello\r")
            sent = True
        if b"got:hello" in out:
            break
    p.wait(timeout=30)
    os.close(master)
    text = out.decode(errors="replace").replace("\r", "")
    ok = "is-a-tty" in text and "40 132" in text and "got:hello" in text
    return [("a shell gets a terminal of its own, sized, and input reaches it", ok, text.strip())]


CHECKS = [check_identity, check_cuda, check_smi, check_nccl, check_stop, check_deadline,
          check_detached, check_shell]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cards", default="0,1", help="two cards, as nvidia-smi numbers them")
    ap.add_argument("--user", default=os.environ.get("SUDO_USER"),
                    help="the account the jobs run as (default: whoever ran sudo)")
    ap.add_argument("--python", help="a Python with torch, for the NCCL check (run as --user)")
    ap.add_argument("--force", action="store_true", help="use cards that have processes on them")
    ap.add_argument("--only", help="run only these checks, e.g. --only nccl,shell")
    args = ap.parse_args(argv)
    if os.geteuid() != 0:
        print("needs root: run it with sudo", file=sys.stderr)
        return 2
    if not args.user:
        print("say which account to test as: --user NAME", file=sys.stderr)
        return 2
    try:
        cards = [int(c) for c in args.cards.split(",")]
    except ValueError:
        cards = []
    info = read_cards()
    if info is None or len(cards) != 2 or len(set(cards)) != 2 or not set(cards) <= set(info):
        print("give two different cards that nvidia-smi lists: --cards 0,1", file=sys.stderr)
        return 2
    busy = sorted({p.card for p in read_procs(info) or () if p.card in cards})
    if busy and not args.force:
        print(f"GPU {busy} has processes on it; pick idle cards or pass --force", file=sys.stderr)
        return 2
    s = Smoke(args.user, cards, info, args.python)
    print(f"testing as {args.user} on GPUs {cards} (/dev/nvidia{s.minors[0]}, "
          f"/dev/nvidia{s.minors[1]}); state in {s.tmp}")
    results = []
    try:
        chosen = [c for c in CHECKS
                  if not args.only or c.__name__[len("check_"):] in args.only.split(",")]
        for check in chosen:
            try:
                got = check(s)
            except Exception as e:      # one broken check must not hide the rest
                got = [(check.__name__, False, f"{type(e).__name__}: {e}")]
            for name, ok, seen in got:
                results.append(ok)
                mark = ("INFO" if ok == "info" else "SKIP" if ok is None
                        else "PASS" if ok else "FAIL")
                print(f"{mark}  {name}\n      saw: {seen}", flush=True)
    finally:
        s.cleanup()
    return 0 if all(r is not False for r in results if r != "info") else 1


if __name__ == "__main__":
    sys.exit(main())
