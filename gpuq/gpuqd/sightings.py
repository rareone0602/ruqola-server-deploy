"""When an offending process is due to be stopped: on its second sighting, at
least MIN_GAP_S after the first.

One sighting is never enough: a single bad nvidia-smi sample, or a process
that exited mid-read, must not cost anyone their work. A process that drops out
of sight starts over. A pass where the driver did not answer is not a pass: the
caller skips observe() so no clock moves on a bad reading.
"""

# About a minute, not the previous gpuq's 15: with a cgroup per job the answer to "is this
# a gpuq job?" is exact, so a longer grace would only be free time for a bypass.
MIN_GAP_S = 60.0


class Sightings:
    def __init__(self):
        self.first = {}         # key -> when first seen in the current run of sightings

    def observe(self, now, keys):
        """Record this pass's offenders; return the keys now due."""
        self.first = {k: self.first.get(k, now) for k in keys}
        return {k for k, t in self.first.items() if now - t >= MIN_GAP_S}
