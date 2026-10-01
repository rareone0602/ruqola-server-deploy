"""Which /dev/nvidiaN file is which card, and the systemd properties that let a
job open only its own cards.

The driver numbers its device files in its own order, not nvidia-smi's (PCI
bus) order. On wsserver1 (checked 2026-09-30): nvidia-smi GPU 0 is
/dev/nvidia1, GPU 1 is /dev/nvidia3, GPU 2 is /dev/nvidia2, GPU 3 is
/dev/nvidia0. So a card's device file is looked up by its UUID in the driver's
own table, never assumed from its index.
"""
from pathlib import Path

DRIVER_GPUS = "/proc/driver/nvidia/gpus"

# What every CUDA program needs besides its own cards, plus the terminal, so a
# job attached to one can still open it.
SHARED_DEVICES = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools",
                  "/dev/nvidia-modeset", "/dev/tty", "char-pts")


def device_minors(root=DRIVER_GPUS):
    """GPU UUID -> the N of its /dev/nvidiaN, from the driver; None if unreadable."""
    minors = {}
    try:
        infos = sorted(Path(root).glob("*/information"))
        for info in infos:
            fields = {}
            for line in info.read_text().splitlines():
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()
            minors[fields["GPU UUID"]] = int(fields["Device Minor"])
    except (OSError, KeyError, ValueError):
        return None
    return minors or None


def device_properties(minors):
    """systemd properties that let a job open only the /dev/nvidiaN in `minors`."""
    return (["DevicePolicy=closed"]
            + [f"DeviceAllow={d} rw" for d in SHARED_DEVICES]
            + [f"DeviceAllow=/dev/nvidia{int(m)} rw" for m in sorted(minors)])
