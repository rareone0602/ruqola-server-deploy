# Troubleshooting

Start with two commands:

```bash
gpuq history     # how your recent jobs ended: RESULT and exit code
gpuq status      # who holds which GPU, what is queued, every GPU process
```

For messages printed by `gpuq submit`, see the table in the
[GPU Queue guide](gpu-queue-guide.md#troubleshooting).

## My job ended unexpectedly

Find the job in `gpuq history` and read its RESULT:

| RESULT | What happened | Fix |
|---|---|---|
| `timed_out` | It hit its `-t` limit. | Resubmit with a larger `-t` (at most 48). Save checkpoints so long runs can resume. |
| `killed` | A signal ended it: `gpuq kill`, Ctrl-C, a closed terminal or dropped SSH session, or the kernel running out of RAM. | Run jobs inside `tmux` or `screen`. If none of these fits, ask the admin to check the kernel log. |
| `failed` | Your program exited with an error. | Read the output you saved. Look for a Python traceback or `CUDA out of memory`. |
| `lost*` | The `gpuq submit` process itself died. | Run inside `tmux`; resubmit. |

If you got a `... was KILLED` email, the audit killed a GPU process that gpuq
did not start, or a job running on a card it was not given. See
[Notifications](notifications-faq.md).

gpuq keeps no copy of your job's output. Save it next time:
`gpuq submit -m 40 -- python train.py 2>&1 | tee run.log`.

## Python and packages

| Error | Fix |
|---|---|
| `python: command not found` | The system has `python3` only. Use a venv, where `python` exists. |
| `error: externally-managed-environment` from `pip install` | The system Python is locked. Install into a venv. |
| `conda: command not found` | conda is not installed. Use a venv. |
| `python: command not found` inside `gpuq submit -- bash -lc "..."` | A login shell resets `PATH` and drops your venv. Use `bash -c`. |

Create and use a venv:

```bash
python3 -m venv ~/venvs/myenv
source ~/venvs/myenv/bin/activate
pip install -r requirements.txt        # now pip and python work
```

The GPU driver is 575.57.08 (CUDA 12.9), and `nvcc` 12.9 is in `/usr/local/cuda`.
Wheels built for CUDA 12.x work. Wheels built for CUDA 13 need a newer driver
than this host has; the [PyTorch guide](pytorch-guide.md) shows how to pick a
CUDA 12 build.

## CUDA out of memory

- Lower the batch size, train in BF16, or turn on gradient checkpointing.
  The [framework guides](pytorch-guide.md) show how.
- Check `gpuq status`: another of your jobs may share the card.
- JAX takes 75% of the card at start by default. Set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` before you submit.
- TensorFlow takes the whole card by default. Set
  `TF_FORCE_GPU_ALLOW_GROWTH=true`.

## Disk full

- **`Disk quota exceeded` in your home:** the limit is 90 GiB soft, 100 GiB hard.
  Run `quota -s`, then find what is big with `du -h --max-depth=1 ~ | sort -hr | head`.
  Common culprits are `~/.cache/pip` (clear with `pip cache purge`) and
  `~/.cache/huggingface` (set `export HF_HOME=/scratch/users/$USER/hf` in `~/.bashrc`,
  then move the old folder's contents there).
- **A file vanished from `/scratch`:** it may have hit the scratch deletion rule.
  See [Scratch Storage](scratch-folder.md); the admin can check the deletion log.

## Slow training

- Watch `nvidia-smi -l 1`. Low utilization usually means the GPU is waiting for
  data: raise the DataLoader's `num_workers`, and use `pin_memory=True`.
- The 256 CPUs are shared. Use as many workers as you need, not all of them;
  `htop` shows the load.
- `utilization.gpu` is the share of time a kernel was running, not how
  efficiently it ran. Use a profiler to measure real throughput.

## Tools you may miss

- `dmesg` fails with `Operation not permitted`: only root can read the kernel
  log. Ask the admin to check for out-of-memory kills or GPU (Xid) errors.
- `journalctl` shows only your own messages; the `adm` and `systemd-journal`
  groups see the rest.
- Installed: `nvidia-smi`, `htop`, `tmux`, `screen`. Not installed: `iotop`,
  `nethogs`, `nvtop`, `dcgmi`, `conda`.
- Ordinary accounts have no `sudo`. Ask the admin for system packages.

## Contact the admin

Contact the admin for GPU errors, a GPU missing from `nvidia-smi`, `gpuq status`
disagreeing with `nvidia-smi`, a kill you think was wrong, or a deleted file you
need. Send:

- your username and the job id
- the exact command and the error text
- the job's line from `gpuq history`
- the output of `nvidia-smi` and `gpuq status`
