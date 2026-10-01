"""How the gpuq client and gpuqd talk: one JSON message per packet on a Unix
SOCK_SEQPACKET socket, with open files passed alongside (SCM_RIGHTS).

Seqpacket keeps message boundaries, so a message is never half-read. The kernel
tells the daemon who is calling (SO_PEERCRED), so nobody can claim to be
someone else, and an attached client passes its terminal itself: the job
writes to it directly, as the previous gpuq's jobs did.
"""
import json
import os
import socket
import struct

SOCKET_PATH = os.environ.get("GPUQ_SOCKET", "/run/gpuq/gpuqd.sock")

# One message holds at most one job's command line and environment. 1 MiB is
# many times the largest environment a shell can pass to a program.
MAX_MSG = 1 << 20
MAX_FDS = 3


class Closed(Exception):
    """The other end hung up."""


def send(sock, obj, fds=()):
    data = json.dumps(obj, separators=(",", ":")).encode()
    if len(data) > MAX_MSG:
        raise ValueError(f"message of {len(data)} bytes is over the {MAX_MSG}-byte limit")
    if fds:
        socket.send_fds(sock, [data], list(fds))
    else:
        sock.send(data)


def recv(sock):
    """(message, fds). Raises Closed at end of stream. Files that arrive with a
    message that is not JSON are closed, never leaked."""
    try:
        data, fds, flags, _ = socket.recv_fds(sock, MAX_MSG, MAX_FDS)
    except ConnectionResetError:
        raise Closed() from None
    if not data and not fds:
        raise Closed()
    try:
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise ValueError("message cut short")
        msg = json.loads(data)
        if not isinstance(msg, dict):
            raise ValueError("message is not an object")
    except ValueError:
        for fd in fds:
            os.close(fd)
        raise
    return msg, fds


def peer(sock):
    """(pid, uid, gid) of the process at the other end, from the kernel."""
    raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return struct.unpack("3i", raw)


def connect(path=None, timeout=None):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    s.settimeout(timeout)
    try:
        s.connect(path or SOCKET_PATH)
    except OSError:
        s.close()
        raise
    return s


def call(request, path=None, timeout=30):
    """One request, one reply."""
    with connect(path, timeout) as s:
        send(s, request)
        reply, fds = recv(s)
        for fd in fds:
            os.close(fd)
        return reply
