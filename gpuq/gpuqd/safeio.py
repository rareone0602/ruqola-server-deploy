"""Reading a file in a folder other people can write to, as root, without being
tricked (todo E1).

The previous gpuq's /var/lib/gpu_queue is writable by every gpuqueue member. A member can
put a symlink there (pointing root at a file they cannot read) or a FIFO (which
blocks whoever opens it, forever). So: never follow a symlink, never block on
open, and read only regular files, up to a size limit.
"""
import errno
import os
import stat

MAX_BYTES = 256 << 20


def read_text(path, limit=MAX_BYTES):
    """The file's text. FileNotFoundError if it is missing; OSError if it is a
    symlink, not a regular file, or larger than `limit`."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        if st.st_size > limit:
            raise OSError(errno.EFBIG, "file too large", str(path))
        chunks, size = [], 0
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            size += len(b)
            if size > limit:
                raise OSError(errno.EFBIG, "file too large", str(path))
            chunks.append(b)
    finally:
        os.close(fd)
    return b"".join(chunks).decode("utf-8", errors="replace")
