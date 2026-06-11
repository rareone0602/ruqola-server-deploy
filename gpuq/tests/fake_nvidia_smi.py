#!/usr/bin/env python3
"""
Fake `nvidia-smi` that emits CSV from a JSON state file.

State path: env var FAKE_NVSMI_STATE (required).
State schema:
  {
    "gpus": [
      {"index": 0, "name": "Fake-GPU", "uuid": "GPU-aaaa",
       "memory_used_mb": 1024, "memory_total_mb": 81920, "utilization": 0},
      ...
    ],
    "compute_apps": [
      {"pid": 1234, "process_name": "python", "gpu_uuid": "GPU-aaaa",
       "used_memory_mb": 1024},
      ...
    ]
  }

Supported invocations (the only forms userspace.py uses):
  nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits
  nvidia-smi --query-compute-apps=<fields> --format=csv,noheader,nounits
"""
import json
import os
import sys


GPU_FIELD = {
    "index": lambda g: g["index"],
    "name": lambda g: g["name"],
    "uuid": lambda g: g["uuid"],
    "memory.used": lambda g: g["memory_used_mb"],
    "memory.total": lambda g: g["memory_total_mb"],
    "memory.free": lambda g: g["memory_total_mb"] - g["memory_used_mb"],
    "utilization.gpu": lambda g: g["utilization"],
}

PROC_FIELD = {
    "pid": lambda p: p["pid"],
    "process_name": lambda p: p["process_name"],
    "gpu_uuid": lambda p: p["gpu_uuid"],
    "used_memory": lambda p: p["used_memory_mb"],
}


def load_state():
    path = os.environ.get("FAKE_NVSMI_STATE")
    if not path:
        sys.stderr.write("fake-nvidia-smi: FAKE_NVSMI_STATE not set\n")
        sys.exit(2)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.stderr.write(f"fake-nvidia-smi: cannot read {path}: {e}\n")
        sys.exit(2)


def emit(rows, fields, table):
    out_lines = []
    for r in rows:
        vals = [str(table[f](r)) for f in fields if f in table]
        out_lines.append(", ".join(vals))
    if out_lines:
        sys.stdout.write("\n".join(out_lines) + "\n")


def main(argv):
    query = None
    fields = None
    for a in argv[1:]:
        if a.startswith("--query-gpu="):
            query = "gpu"
            fields = a.split("=", 1)[1].split(",")
        elif a.startswith("--query-compute-apps="):
            query = "compute-apps"
            fields = a.split("=", 1)[1].split(",")
        elif a in ("--format=csv,noheader,nounits", "--format=csv"):
            pass
        elif a.startswith("--"):
            pass
        else:
            sys.stderr.write(f"fake-nvidia-smi: unsupported arg {a!r}\n")
            return 2
    if query is None:
        sys.stderr.write("fake-nvidia-smi: missing --query-gpu or --query-compute-apps\n")
        return 2

    state = load_state()
    if query == "gpu":
        emit(state.get("gpus", []), fields, GPU_FIELD)
    else:
        emit(state.get("compute_apps", []), fields, PROC_FIELD)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
