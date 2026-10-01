"""gpuq: the GPU queue's command line (docs/v3-design.md §8).

Every command and flag the previous gpuq accepted still works (§8.1). A flag v3 no
longer needs is accepted and ignored, so no script that ran before breaks. The
one exception is --devices on a card you do not hold: it is refused (§4.5).
"""
import argparse
import sys

from . import history, misc, status, submit

DESCRIPTION = """\
gpuq: the GPU queue on this host.

`gpuq submit` waits for card(s), then runs your command on them as you, in your
folder, with your environment, attached to your terminal. When more people want
cards than there are cards, whoever has used the fewest GPU-hours lately goes
first. A job needing 2+ cards is promised a start time, and the first cards to
free up wait for it. Every job may run up to 48 h.

A card your job is running on is yours: nobody else's job joins it. To run
more of your own jobs on it (a quick test beside your training, say), use
`--devices N`. That job skips the line, the card is charged once however many
of your jobs share it, and the job must end when your job(s) on card N reach
their 48 h."""

EPILOG = """\
examples:
  gpuq submit -- python train.py             one card to yourself
  gpuq submit -g 2 -- python x.py            two cards
  gpuq submit -m 30 -- python small.py       on one of your own cards if 30 GB is free there,
                                             else on a free card
  gpuq submit --devices 2 -m 10 -- python test.py
                                             add this job to card 2, which a job of yours
                                             is running on, once 10 GB is free there
  gpuq shell --devices 2                     a shell on your card 2, beside your job there
  gpuq submit --detach -- python train.py    return at once; output to ~/gpuq-logs/
  gpuq shell                                 an interactive shell with a card
  gpuq status                                cards, running jobs, and the queue in order
  gpuq why 12345                             why that job is waiting
  gpuq share                                 your recent use and your place in line
  gpuq history                               your recent jobs from the ledger
  gpuq kill 12345                            stop or cancel your job

Run `gpuq <command> -h` for a command's options, e.g. `gpuq submit -h`.
"""


def _job_flags(p):
    p.add_argument("-g", "--gpus", type=int, default=None,
                   help="Number of whole cards (default: 1).")
    p.add_argument("-m", "--memory", type=float, default=None,
                   help="The GB of VRAM you need. With it, gpuq first tries the cards your "
                        "own jobs are running on, and starts the job beside them where that "
                        "much is measured free; if none has room, it gets a free card like "
                        "any job, waiting in line if needed. Without it, the job gets a card "
                        "to itself.")
    p.add_argument("--devices", default=None, metavar="N[,N...]",
                   help="Add this job to your card N: a GPU one of your jobs is running on "
                        "now (see `gpuq status`). It skips the line and starts once N has "
                        "the -m GB free (at least 2 GB). The card is charged once, not per job. It "
                        "must end when your job(s) on N reach their 48 h, and is cancelled "
                        "if they end before there is room. A card you have no job on is "
                        "refused: use -g for new cards. Inside the job, N is cuda:0.")
    p.add_argument("--name", default=None, help="A label shown in `gpuq status`.")


def parser():
    ap = argparse.ArgumentParser(prog="gpuq", description=DESCRIPTION, epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(
        dest="action",
        metavar="{submit,shell,status,why,share,history,kill,quota,config,audit}",
        help="run `gpuq <command> -h` for that command's options")

    p = sub.add_parser("submit", help="Wait for card(s), then run a command on them.")
    _job_flags(p)
    p.add_argument("--detach", action="store_true",
                   help="Return at once. The job survives closing the terminal; its output "
                        "goes to ~/gpuq-logs/JOB.log.")
    p.add_argument("--notify", nargs="?", const="", default=None,
                   help="Also email when the job ends, to your account's address. (An "
                        "address given here is ignored.)")
    p.add_argument("--command", default=None,
                   help="The command as one string, split like a shell would (not run by one).")
    p.add_argument("-t", "--time", type=float, default=None,
                   help="Accepted for old scripts and ignored: every job may run 48 h. For a "
                        "shorter limit, run `timeout 4h python ...`.")
    p.add_argument("--queue", action="store_true",
                   help="Accepted for old scripts and ignored: every submit waits in line.")
    p.add_argument("cmd_args", nargs=argparse.REMAINDER,
                   help="The command and its arguments, after `--`.")

    p = sub.add_parser("shell", help="An interactive shell with card(s), through the queue.")
    _job_flags(p)

    sub.add_parser("status", help="Cards, running jobs, and the queue in order.")

    p = sub.add_parser("why", help="Why a job is waiting, and when it should start.")
    p.add_argument("job_id", type=int)

    for name, helptext in (("share", "Your recent GPU use and your place in line."),
                           ("quota", "Same as `gpuq share` (there is no quota any more).")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--user", default=None, help="Someone else's use instead of yours.")
        p.add_argument("--all", action="store_true", help="Everyone's recent use.")
        p.add_argument("--report", action="store_true", help=argparse.SUPPRESS)
        p.add_argument("--weeks", type=int, default=8, help=argparse.SUPPRESS)

    p = sub.add_parser("history", help="Your recent jobs from the ledger.")
    p.add_argument("-n", "--limit", type=int, default=20,
                   help="Show at most N records (default: 20; 0 = all).")
    p.add_argument("--all", action="store_true", help="All users' jobs, not just yours.")
    p.add_argument("--user", default=None, help="Another user's jobs.")
    p.add_argument("--events", action="store_true",
                   help="Also show jobs that left the queue without running.")
    p.add_argument("--json", action="store_true", help="Print raw ledger records as JSON lines.")

    p = sub.add_parser("kill", help="Stop your running jobs or cancel your queued ones.")
    p.add_argument("job_id", type=int, nargs="*", help="ID(s) of your job(s).")
    p.add_argument("--job-id", dest="job_id_flag", type=int, help="A job ID, as a flag.")
    p.add_argument("--mine", action="store_true",
                   help="Stop all your running jobs and cancel all your queued ones.")

    p = sub.add_parser("config", help="The one policy number, and where things are.")
    p.add_argument("config_action", nargs="?", choices=["init"], help=argparse.SUPPRESS)
    p.add_argument("--show", action="store_true", help="(default) Show it.")
    p.add_argument("--force", action="store_true", help=argparse.SUPPRESS)

    p = sub.add_parser("audit", help="Does nothing now: gpuqd checks GPU use itself.")
    p.add_argument("--quiet", "-q", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--enforce", action="store_true", help=argparse.SUPPRESS)
    return ap


HANDLERS = {
    "submit": submit.cmd_submit,
    "shell": submit.cmd_shell,
    "status": status.cmd_status,
    "why": status.cmd_why,
    "share": status.cmd_share,
    "quota": status.cmd_share,
    "history": history.cmd_history,
    "kill": misc.cmd_kill,
    "config": misc.cmd_config,
    "audit": misc.cmd_audit,
}


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if not args.action:
        ap.print_help(sys.stderr)
        return 2
    try:
        return HANDLERS[args.action](args) or 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
