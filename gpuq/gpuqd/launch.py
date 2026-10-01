"""The first thing a job runs, inside its own unit, as its own user. It restores
what the submitter's shell had (environment and folder; for a detached job,
the log file), then becomes the submitter's command.

    /usr/bin/python3 -I launch.py SPEC

SPEC is written by gpuqd and readable only by this user:
{"argv": [...], "env": {...}, "cwd": "...", "log": null or "..."}.
-I keeps the submitter's environment from changing this script. Standard
library only: it runs from the installed copy, outside any package.
"""
import json
import os
import signal
import sys


def _to_log(path):
    """Send stdout and stderr to `path` (appended), stdin from /dev/null. The
    file is opened as the job's user, so it is theirs and follows their umask."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    for x in (fd, null):
        if x > 2:
            os.close(x)


def main(argv):
    with open(argv[1]) as f:
        spec = json.load(f)
    if spec.get("log"):
        try:
            _to_log(spec["log"])
        except OSError as e:
            print(f"gpuq: cannot write the log {spec['log']}: {e.strerror}", file=sys.stderr)
            return 1
    try:
        os.chdir(spec["cwd"])
    except OSError as e:
        print(f"gpuq: cannot enter {spec['cwd']}: {e.strerror}", file=sys.stderr)
        return 1
    # Python ignores these two; a program started from a shell does not.
    for s in (signal.SIGPIPE, signal.SIGXFSZ):
        signal.signal(s, signal.SIG_DFL)
    args = spec["argv"]
    try:
        os.execvpe(args[0], args, spec["env"])
    except FileNotFoundError:
        print(f"gpuq: {args[0]}: command not found", file=sys.stderr)
        return 127
    except OSError as e:
        print(f"gpuq: {args[0]}: {e.strerror}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    sys.exit(main(sys.argv))
