"""What every gpuq command shares: reaching gpuqd, and dying politely."""
import errno
import os
import sys
import time

from gpuqd import protocol

# gpuqd restarts in about 2 s (an update, a crash). Wait that out rather than
# failing someone's script.
CONNECT_FOR_S = float(os.environ.get("GPUQ_CONNECT_FOR_S", 15))   # tests shorten it

STATE_DIR = os.environ.get("GPUQD_STATE_DIR", "/var/lib/gpuq")


def printable(text):
    """`text` safe to show on someone else's terminal (todo E9). A job's name and
    command are typed by its owner, and an escape sequence, a carriage return or a
    right-to-left override in them could hide or forge a line of `gpuq status`. So
    line breaks and tabs become a space, any other control character a `?`."""
    return "".join(c if c.isprintable() else " " if c.isspace() else "?" for c in str(text))


class Die(SystemExit):
    pass


def die(msg, code=1):
    print(f"gpuq: {msg}", file=sys.stderr)
    raise Die(code)


def unreachable(e):
    die(f"cannot reach gpuqd at {protocol.SOCKET_PATH} ({e.strerror or e}). "
        "Is it running? `systemctl status gpuqd`")


def connect(timeout=None):
    end = time.time() + CONNECT_FOR_S
    while True:
        try:
            return protocol.connect(timeout=timeout)
        except OSError as e:
            down = e.errno in (errno.ENOENT, errno.ECONNREFUSED)
            if not down or time.time() >= end:
                unreachable(e)
            time.sleep(0.5)


def ask(request, timeout=60):
    """One request, one reply; dies on an error reply."""
    sock = connect(timeout)
    try:
        with sock:
            protocol.send(sock, request)
            reply, fds = protocol.recv(sock)
            for fd in fds:
                os.close(fd)
    except protocol.Closed:
        die("gpuqd closed the connection before answering")
    except OSError as e:
        unreachable(e)
    if not reply.get("ok"):
        die(reply.get("error") or "gpuqd refused the request")
    return reply
