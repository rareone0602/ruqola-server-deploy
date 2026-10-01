"""Check, once, on this host, that a job given one card cannot reach the others.

    sudo env PYTHONPATH=<gpuq folder> python3 -m gpuqd.confine [--user NAME] [--card 0] [--other 1]

v3 runs each job as a systemd unit that may open only its own /dev/nvidiaN plus
the shared control devices (DevicePolicy=closed + DeviceAllow=), the way Slurm
confines GPUs. Cards are named as nvidia-smi numbers them; devices.py finds
each one's device file. This starts four tiny units as --user and checks:

  1. with no device list, CUDA sees every card     (the probe itself works)
  2. given --card, CUDA sees exactly that card
  3. inside that job, asking for --other by UUID finds nothing   (no rebinding)
  4. inside that job, nvidia-smi works and lists only --card     (NVML users such
     as vLLM keep working)

The probe only asks the driver which cards it can see (cuInit, cuDeviceGetUuid).
It creates no CUDA context and allocates no GPU memory, so it is safe to run
while the cards are busy. Needs root, because only root can start a unit as
another user with a device list.
"""
import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys

from .devices import device_minors, device_properties
from .nvsmi import read_cards

def format_uuid(raw):
    """CUDA's 16 uuid bytes in nvidia-smi's spelling."""
    h = bytes(raw).hex()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


PROBE = inspect.getsource(format_uuid) + """
import ctypes, json
out = {"init": None, "uuids": []}
cu = ctypes.CDLL("libcuda.so.1")
out["init"] = cu.cuInit(0)
if out["init"] == 0:
    n = ctypes.c_int()
    cu.cuDeviceGetCount(ctypes.byref(n))
    for i in range(n.value):
        d = ctypes.c_int()
        cu.cuDeviceGet(ctypes.byref(d), i)
        u = (ctypes.c_ubyte * 16)()
        cu.cuDeviceGetUuid(u, d)
        out["uuids"].append(format_uuid(u))
print(json.dumps(out))
"""


def unit_argv(user, minors=None, env=None, program=None):
    """A transient unit running `program` (default: the CUDA probe) as `user`,
    allowed only the /dev/nvidiaN in `minors` unless minors is None."""
    argv = ["systemd-run", "--quiet", "--wait", "--pipe", "--collect", f"--uid={user}"]
    if minors is not None:
        argv += [f"--property={p}" for p in device_properties(minors)]
    argv += [f"--setenv={k}={v}" for k, v in (env or {}).items()]
    return argv + ["--"] + list(program or [sys.executable, "-c", PROBE])


def _run(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=120,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"rc": None, "out": "", "err": str(e)}
    return {"rc": r.returncode, "out": r.stdout, "err": r.stderr[-2000:]}


# cuInit's answers that are clean: success, or "no CUDA device here".
CUDA_SUCCESS, CUDA_ERROR_NO_DEVICE = 0, 100


def _probe(res):
    """The probe's answer from one run, or None if the run did not complete."""
    if res["rc"] != 0:
        return None
    for line in reversed(res["out"].splitlines()):
        if line.startswith("{"):
            try:
                ans = json.loads(line)
            except json.JSONDecodeError:
                return None
            ok = ans.get("init") in (CUDA_SUCCESS, CUDA_ERROR_NO_DEVICE)
            return ans if ok and isinstance(ans.get("uuids"), list) else None
    return None


def _smi(res):
    """nvidia-smi's (index, uuid) rows, or None if it failed."""
    if res["rc"] != 0:
        return None
    rows = [l.split(",") for l in res["out"].splitlines() if "GPU-" in l]
    return [(r[0].strip(), r[1].strip()) for r in rows if len(r) == 2]


def verdicts(results, cards, card, other):
    """[(check, passed, what was seen)] from the four runs. A run that crashed,
    timed out or printed no answer fails its check: it proves nothing."""
    everyone = [c.uuid for _, c in sorted(cards.items())]
    mine, theirs = cards[card].uuid, cards[other].uuid
    seen = {k: _probe(results[k]) for k in ("open", "confined", "rebind")}
    uuids = {k: (v["uuids"] if v else None) for k, v in seen.items()}
    smi = _smi(results["smi"])
    return [
        ("CUDA sees every card with no device list (the probe works)",
         uuids["open"] == everyone, uuids["open"]),
        (f"CUDA sees only GPU {card} when the job is given GPU {card}",
         uuids["confined"] == [mine], uuids["confined"]),
        (f"Inside that job, GPU {other} asked for by UUID is out of reach",
         uuids["rebind"] is not None and theirs not in uuids["rebind"], seen["rebind"]),
        (f"Inside that job, nvidia-smi works and lists only GPU {card}",
         smi is not None and [u for _, u in smi] == [mine], smi),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--user", default=os.environ.get("SUDO_USER"),
                    help="the account the test jobs run as (default: whoever ran sudo)")
    ap.add_argument("--card", type=int, default=0, help="the card the test job is given")
    ap.add_argument("--other", type=int, default=1, help="a card it must not reach")
    args = ap.parse_args(argv)
    if os.geteuid() != 0:
        print("needs root: run it with sudo", file=sys.stderr)
        return 2
    if not args.user:
        print("say which account to test as: --user NAME", file=sys.stderr)
        return 2
    cards = read_cards()
    if cards is None or args.card not in cards or args.other not in cards:
        print("nvidia-smi did not list both cards", file=sys.stderr)
        return 2
    minors = device_minors()
    if minors is None or cards[args.card].uuid not in minors:
        print("the driver's table in /proc/driver/nvidia/gpus is unreadable", file=sys.stderr)
        return 2
    mine = [minors[cards[args.card].uuid]]
    print(f"GPU {args.card} is /dev/nvidia{mine[0]}")
    smi = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    runs = {
        "open": unit_argv(args.user),
        "confined": unit_argv(args.user, mine),
        "rebind": unit_argv(args.user, mine,
                            env={"CUDA_VISIBLE_DEVICES": cards[args.other].uuid}),
        "smi": unit_argv(args.user, mine,
                         program=[smi, "--query-gpu=index,uuid", "--format=csv,noheader"]),
    }
    results = {k: _run(v) for k, v in runs.items()}
    checks = verdicts(results, cards, args.card, args.other)
    for name, ok, seen in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      saw: {seen}")
    for k, r in results.items():
        if r["rc"] not in (0,) and r["err"].strip():
            print(f"      [{k}] exit {r['rc']}: {r['err'].strip().splitlines()[-1]}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
