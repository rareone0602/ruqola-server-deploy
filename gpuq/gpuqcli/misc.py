"""gpuq kill, gpuq config, gpuq audit."""
import sys

from gpuqd import protocol

from .common import STATE_DIR, ask, die


def cmd_kill(args):
    ids = list(args.job_id or [])
    if args.job_id_flag is not None:
        ids.append(args.job_id_flag)
    if not ids and not args.mine:
        die("kill: provide a job ID (see `gpuq status`), e.g. `gpuq kill 12345`, or use --mine.")
    r = ask({"op": "kill", "jobs": ids, "mine": bool(args.mine)})
    if args.mine and not r["results"]:
        print("you have no running or queued jobs.")
    failed = 0
    for res in r["results"]:
        print(res["message"], file=sys.stdout if res["ok"] else sys.stderr)
        failed += not res["ok"]
    return 1 if failed else 0


def cmd_config(args):
    if args.config_action == "init":
        print("gpuq needs no config file any more: the installer sets everything up.",
              file=sys.stderr)
        return 0
    if args.force:
        die("--force only applies to `gpuq config init`; "
            "use `gpuq config init --force` to overwrite the config.")
    c = ask({"op": "config"})
    print(f"Host:            {c['host']}")
    print(f"Job time limit:  {c['max_runtime_h']:g} h (the one policy number)")
    print(f"gpuqd socket:    {protocol.SOCKET_PATH}")
    print(f"State and ledger: {STATE_DIR}  (ledger: {c['ledger']})")
    print(f"Email:           {'on' if c['mail'] else 'off'}")
    print("GPU use outside gpuq: " + ("stopped about a minute after it is first seen"
                                       if c.get("enforcing") else
                                       "only logged (root switched stopping off)"))
    print("Rules: least recent use goes first (7-day half-life); 2+ card jobs are promised "
          "a start; every job may run 48 h.")
    return 0


def cmd_audit(args):
    if not args.quiet:
        print("gpuq audit does nothing now: gpuqd checks every GPU process itself.")
    return 0
