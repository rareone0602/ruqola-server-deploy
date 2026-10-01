"""gpuqd's socket loop: accept clients, hand their requests to the Daemon, and
tick it about once a second.

The socket is /run/gpuq/gpuqd.sock, mode 0666: open to every local account,
with no group and no sign-up (D4). Who is calling comes from the kernel
(SO_PEERCRED), never from the request.
"""
import os
import selectors
import signal
import socket
import sys
import time
import traceback

from . import protocol

TICK_S = 1.0
# A client that stops reading (its terminal suspended, say) never stalls gpuqd
# for longer than this; it is dropped instead.
SEND_TIMEOUT_S = 2.0

ONE_SHOT = ("status", "why", "share", "kill", "config")


class Conn:
    def __init__(self, sock):
        self.sock = sock
        self.peer = protocol.peer(sock)     # (pid, uid, gid)
        self.job = None
        self.alive = True

    def send(self, msg):
        if not self.alive:
            return
        try:
            protocol.send(self.sock, msg)
        except OSError:
            self.alive = False


def handle(daemon, conn, msg, fds, now):
    """The reply to one request, or None for a notice that needs none."""
    op = msg.get("op")
    if op == "submit":
        return daemon.submit(conn, msg, fds, now)
    if op == "attach":
        return daemon.attach(conn, msg, fds, now)
    for fd in fds:
        os.close(fd)
    if op == "signal":
        try:
            sig = int(msg.get("sig"))
        except (TypeError, ValueError):
            return None
        if sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT,
                   signal.SIGUSR1, signal.SIGUSR2):
            daemon.signal(conn, sig, now)
        return None
    if op == "winch":
        daemon.winch(conn)
        return None
    if op in ONE_SHOT:
        return getattr(daemon, op)(conn, msg, now)
    return {"ok": False, "error": f"unknown request {op!r}; is this gpuq older than gpuqd?"}


def listen(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    s.bind(path)
    os.chmod(path, 0o666)
    s.listen(64)
    s.setblocking(False)
    return s


def serve(daemon, path=None, clock=time.time, stop=None):
    """Run until SIGTERM (or `stop`, a threading.Event, is set: tests)."""
    listener = listen(path or protocol.SOCKET_PATH)
    sel = selectors.DefaultSelector()
    sel.register(listener, selectors.EVENT_READ, None)
    conns = set()
    done = []
    if stop is None:
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, lambda *_: done.append(True))
    daemon.recover(clock())

    def drop(conn):
        conns.discard(conn)
        try:
            sel.unregister(conn.sock)
        except (KeyError, ValueError):
            pass
        conn.sock.close()
        try:
            daemon.disconnected(conn, clock())
        except Exception:
            traceback.print_exc()

    while not done and not (stop is not None and stop.is_set()):
        for key, _ in sel.select(TICK_S):
            if key.data is None:
                try:
                    sock, _ = listener.accept()
                    sock.settimeout(SEND_TIMEOUT_S)
                    conn = Conn(sock)
                except OSError:
                    continue
                conns.add(conn)
                sel.register(sock, selectors.EVENT_READ, conn)
                continue
            conn = key.data
            try:
                msg, fds = protocol.recv(conn.sock)
            except (protocol.Closed, OSError):
                drop(conn)
                continue
            except ValueError as e:
                conn.send({"ok": False, "error": f"bad request: {e}"})
                continue
            try:
                reply = handle(daemon, conn, msg, fds, clock())
            except Exception as e:
                traceback.print_exc()
                reply = {"ok": False, "error": f"gpuqd failed on this request: {type(e).__name__}"}
            if reply is not None:
                conn.send(reply)
        try:
            daemon.tick(clock())
        except Exception:
            traceback.print_exc()
        sys.stdout.flush()
        for conn in [c for c in conns if not c.alive]:
            drop(conn)
    # Shutting down is not every client going away: their jobs carry on, and
    # the clients come back to the next gpuqd (submit.py, Waiter._come_back).
    for conn in conns:
        conn.sock.close()
    daemon.save()
    listener.close()
    return 0
