"""Run by systemd as root when a job's unit has stopped (ExecStopPost=+): record
how the job ended, so gpuqd learns it even if it was restarting at the time.

    /usr/bin/python3 -I exithook.py EXITS_DIR JOB_ID

systemd sets SERVICE_RESULT, EXIT_CODE and EXIT_STATUS (systemd.exec(5)). By
the time this runs, every process of the job is gone, so its cards are free.
Standard library only, like launch.py.
"""
import json
import os
import sys
import time


def main(argv):
    folder, job = argv[1], int(argv[2])
    report = {"result": os.environ.get("SERVICE_RESULT"),
              "code": os.environ.get("EXIT_CODE"),
              "status": os.environ.get("EXIT_STATUS"),
              "t": time.time()}
    tmp = os.path.join(folder, f".{job}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(report, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(folder, f"{job}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
