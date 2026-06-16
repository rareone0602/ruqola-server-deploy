# GPU Queue System User Guide

Comprehensive guide to using the Ruqola server's custom GPU queue management system (`gpuq`) for efficient resource sharing.

## 📖 Table of Contents

1. [Overview](#overview)
2. [Basic Usage](#basic-usage)
3. [Job Submission](#job-submission)
4. [Choosing Which GPU](#choosing-which-gpu)
5. [Monitoring and Management](#monitoring-and-management)
6. [Job History](#job-history)
7. [GPU-Hour Quotas](#gpu-hour-quotas)
8. [Advanced Usage](#advanced-usage)
9. [Best Practices](#best-practices)
10. [Common Workflows](#common-workflows)
11. [Troubleshooting](#troubleshooting)

## Overview

### What is the GPU Queue System?

Our custom GPU queue system (`gpuq`) coordinates fair access to the server's 4 H200 NVL GPUs among everyone in the `gpuqueue` group, ensuring:

- **Cooperative allocation** - Each `gpuq submit` claims free GPU(s) and runs your job; jobs queue and wait when nothing is free
- **Ownership policy** - You may stack more jobs on GPUs you already hold; GPUs held by *other* users are off-limits until they free up
- **Self-supervising time limits** - Each job's own `gpuq submit` process kills it when the time limit is reached
- **Usage monitoring** - `gpuq status` shows who is using what, and `gpuq audit` (run on cron) reports resource hogs and policy breaches
- **Job ledger** - Every job is logged; `gpuq history` shows how your past jobs ended and what they cost, `gpuq quota` shows your rolling 7-day GPU-hours
- **Per-user quotas** - Rolling 7-day GPU-hour budgets keep one user from monopolizing the box over time (not yet enforced — usage is being recorded to set fair budgets)
- **Opt-in notifications** - Email alerts for job completion and policy issues

> **gpuq is daemonless.** There is no background service. Each `gpuq submit` runs
> your command **in the foreground** in your terminal; that same process supervises
> the job and enforces its time limit. Server-side policy checks (resource hogs,
> over-quota users, untracked/rebound GPU jobs) run periodically via `gpuq audit`
> on cron — typically every 15 minutes — not from a live monitor.

### Key Features

- **~140 GiB memory per H200 NVL GPU** - Each card reports 143771 MiB (~141 GB); about ~564 GB total across the 4 GPUs. Massive memory for large models.
- **Default 24-hour time limit** - The default `--time` is 24h; you can request more or less, up to a hard **96-hour (4-day) cap**
- **Queue management** - With `--queue`, jobs wait for resources instead of being rejected
- **Resource monitoring** - Real-time `gpuq status` plus periodic `gpuq audit` policy checks
- **Flexible submission** - Pass the command after `--`, or as a single `--command "..."` string

## Basic Usage

### Quick Start Commands

```bash
# Check current GPU status and queue
gpuq status

# Submit a simple training job on one random free GPU
gpuq submit -- python train.py

# Submit with specific requirements (2 GPUs, need >=40 GB free each, 12h limit)
gpuq submit -g 2 -m 40 -t 12 -- python large_model.py

# Check your jobs
gpuq status | grep "$USER"

# Your recent jobs: runtime, GPU-hours, exit code, how they ended
gpuq history

# Your rolling 7-day GPU-hour usage (vs budget, once quotas are set)
gpuq quota

# Kill a running job — or cancel a queued one — by id
gpuq kill 12345

# Stop/cancel everything of yours on this host
gpuq kill --mine
```

> **Pass your command after `--`.** `gpuq` runs the command **directly, with no
> shell**, so shell features (`&&`, `||`, `cd`, `>`, `2>&1`, trailing `&`,
> `VAR=value` prefixes) are **not** interpreted — they would be handed to your
> program as literal arguments. If you need any of those, wrap the whole pipeline
> in an explicit shell: `gpuq submit -- bash -lc "cd /path && python train.py"`.

### First Time Setup

1. **Check system access**:
```bash
gpuq status
nvidia-smi
```

2. **Test job submission**:
```bash
gpuq submit -t 1 -- python -c 'print("Hello GPU!")'
```

3. **Monitor job progress**:
```bash
watch -n 5 gpuq status
```

## Job Submission

`gpuq submit` accepts the command in two equivalent ways:

```bash
# Preferred: everything after `--` is the command and its arguments (no shell)
gpuq submit -- python train.py --epochs 100

# Or as one string (parsed with shell-style quoting, but still run without a shell)
gpuq submit --command "python train.py --epochs 100"
```

### Basic Job Submission

```bash
# Minimal submission (uses defaults: 1 GPU, default memory/time)
gpuq submit -- python train.py

# Specify the common parameters
gpuq submit \
  -g 1 \
  -m 40 \
  -t 8 \
  --notify "your-email@example.com" \
  -- python train.py --epochs 100 --batch-size 32
```

> `--notify` is **optional**. If you omit it, the completion email goes to the
> address on your account (read from your GECOS field). Use `--notify` only to
> send it somewhere else. There is **no** `--email` flag.

### Resource Specification

#### GPU Requirements

```bash
# Single GPU (default)
gpuq submit -g 1 -- python train.py

# Multi-GPU training (2 GPUs)
gpuq submit -g 2 -- torchrun --nproc_per_node=2 train.py

# All four GPUs (only succeeds if all are free or already yours)
gpuq submit -g 4 -- torchrun --nproc_per_node=4 multi_gpu_train.py
```

`-g/--gpus N` asks for `N` GPUs; `gpuq` picks them at random among the GPUs
selectable for you (see [Choosing Which GPU](#choosing-which-gpu)). To pin exact
indices, use `--devices` instead.

#### Memory Requirements (a selection filter, not a reservation)

`-m/--memory GB` is the **minimum free VRAM a candidate GPU must have to be
chosen** — it is an admission *filter*, not a cap or a reservation. Your job is
**not** held to that number; it can use as much VRAM as is physically on the card.
The default comes from the config (`default_min_free_gb`, 16 in the current
template; configs that predate that key fall back to `max_memory_per_gpu_gb`).
If a bare submit is rejected for "no free GPU", the error tells you the filter
that was applied and reminds you to pass a smaller `-m` if your job needs less.

```bash
# Only pick a GPU that currently has >= 60 GB free
gpuq submit -m 60 -- python big_model.py

# Memory-hungry job: require a near-empty card (>= 120 GB free)
gpuq submit -m 120 -- python huge_model.py

# Modest filter for a small model
gpuq submit -m 20 -- python small_model.py
```

#### Time Limits

`-t/--time HOURS` sets how long the job may run before `gpuq` kills it. The
default is 24h (from the config). It must be **greater than zero** (there is no
unlimited mode) and **may not exceed the 96-hour (4-day) wall-time cap** —
`gpuq` rejects a larger `-t` at submit. When the limit fires, gpuq prints a
`time limit reached` notice to your terminal before terminating the job, so a
timeout is never confused with a crash.

```bash
# Short experiment (1 hour)
gpuq submit -t 1 -- python quick_test.py

# Medium training (8 hours)
gpuq submit -t 8 -- python train.py

# Long training (48 hours)
gpuq submit -t 48 -- python long_train.py
```

> Because `gpuq` is daemonless, the time limit is enforced by the **foreground
> `gpuq submit` process itself** (a timer inside it). Closing your terminal
> sends SIGHUP, which gpuq **forwards to the job** — so an SSH drop normally
> kills the job along with the timer. Run long jobs under `tmux`/`screen`, or
> with [user-linger](#keeping-jobs-alive-after-logout) enabled, so the
> supervising process stays alive. (If a supervisor dies anyway — e.g.
> SIGKILL — the job's GPU-hours are still recorded to the ledger as a `lost`
> record when the dead entry is reaped.)

### Command Examples

#### Training Scripts

```bash
# PyTorch training
gpuq submit -g 1 -m 30 -t 12 -- python train.py --model resnet50 --epochs 100

# TensorFlow training
gpuq submit -g 1 -m 25 -t 8 -- python tf_train.py --model_dir ./models

# Distributed training (2 GPUs)
gpuq submit -g 2 -m 40 -t 16 -- torchrun --nproc_per_node=2 distributed_train.py
```

#### Jupyter Notebooks

```bash
# Start Jupyter on port 8888 (holds your terminal — see note below)
gpuq submit -g 1 -t 8 -- jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# JupyterLab with custom port
gpuq submit -g 1 -m 30 -t 4 -- jupyter lab --ip=0.0.0.0 --port=9999 --no-browser

# Jupyter with a specific working directory (needs a shell for `cd`)
gpuq submit -g 1 -t 6 -- bash -lc "cd /path/to/project && jupyter notebook --ip=0.0.0.0 --port=8888"
```

#### Data Processing

```bash
# Large dataset preprocessing
gpuq submit -g 1 -m 50 -t 4 -- python preprocess_data.py --dataset imagenet

# Feature extraction
gpuq submit -g 1 -m 35 -t 3 -- python extract_features.py --model vit_large
```

## Choosing Which GPU

`gpuq` hands out two kinds of GPU:

- **Free** cards — held by no `gpuq` job, with enough free VRAM (your `-m` filter)
  and under 10% utilization.
- Cards **you already own** — held by one of your own running jobs. You may
  **stack** more jobs onto your own cards. An owned card skips the utilization
  check (your own job legitimately drives it up) but must still clear the **same
  `-m` free-VRAM filter** you asked for — it needs `max(2 GB, your -m)` free — so
  `-m` is honored when stacking too, and a 2 GB floor always guards against an
  instant out-of-memory when `-m` is tiny.

A GPU held by **another** user is **never** handed to you until it frees up.

> **Want a genuinely *free* card instead of stacking onto your own?** Add `--queue`
> with an `-m` larger than your busy card's current free VRAM (a free H200 NVL has
> ~141 GB). Your own card won't clear the filter, so the job **waits** for a free
> one rather than stacking. (Pinning a card you don't hold — `--devices <n>
> --queue` — also waits.)

- **Default picker:** `gpuq submit -g N` picks `N` GPUs, preferring **free** cards
  (chosen at random, to spread load across the box) and only stacking onto cards
  you already own when there aren't enough free ones.
- **Pin specific GPUs:** `gpuq submit --devices 1,3 -- …` runs on exactly those
  indices (the GPU count is taken from the list). Each must be free or already
  yours. If any is held by **another user**, the submit is **rejected
  immediately** with a per-GPU reason — *unless* you add `--queue`, in which case
  it **waits** until all the pinned GPUs become available.

```bash
gpuq submit -g 2 -- python train.py              # 2 free GPUs (spread), else stack on yours
gpuq submit --devices 0,2 -- python x.py         # exactly GPU 0 and 2 (else rejected now)
gpuq submit --devices 0 -- python b.py           # stack another job onto your own GPU 0
gpuq submit --devices 1,3 --queue -- python y.py # wait for exactly GPU 1 and 3
```

> **`gpuq` sets `CUDA_VISIBLE_DEVICES` for you** to the GPU(s) it allocated. Do
> **not** override it (and do not pass a hard-coded `--gpu N` / `device=N` to your
> script), or your job will run on a card it wasn't allocated — leaving the
> allocated card reserved-but-idle. The `gpuq audit` **rebind detector** flags
> this (warn → remind → overdue → kill, with kill only under `--enforce`). If you
> want a specific card, pin it with `--devices`.

## Monitoring and Management

### Checking Job Status

```bash
# Full status: each GPU's state, running jobs, queued jobs, and every live
# GPU compute process from nvidia-smi
gpuq status

# Monitor in real-time
watch -n 5 gpuq status

# Check only your jobs
gpuq status | grep "$USER"
```

> Plain `gpuq status` already shows everything — GPUs, running jobs, queued jobs,
> and all live `nvidia-smi` compute processes. There is **no** `--detailed` flag.

### GPU Monitoring

```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# GPU utilization and memory
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

# Continuous monitoring with compact formatting
watch -n 2 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

### Job Output (no per-job log files)

`gpuq` runs your job in the foreground and **inherits your terminal's
stdin/stdout/stderr** — it does **not** write per-job log files anywhere. When
the job ends, gpuq prints a one-line summary to stderr (`job <id> completed:
ran H:MM:SS on GPU(s) ..., N GPU-hours recorded (exit 0)`), and the same facts
land in the ledger (`gpuq history`). To keep the job's own output, redirect it
yourself, or run inside `tmux`/`screen`:

```bash
# Tee output to a file while still seeing it
gpuq submit -- bash -lc 'python train.py' 2>&1 | tee run.log

# Or just redirect with your shell
gpuq submit -- python train.py > train.out 2> train.err

# Detached, survives logout
tmux new -s train 'gpuq submit -- python train.py 2>&1 | tee run.log'
```

### Managing Jobs

```bash
# Kill one of YOUR running jobs (positional job id; preferred form)
gpuq kill 12345

# The same, as a flag
gpuq kill --job-id 12345

# Cancel one of YOUR queued jobs from any terminal — same command.
# (The waiting submit is signalled and removes itself from the queue.)
gpuq kill 67890

# Several ids at once
gpuq kill 12345 67890

# Stop ALL of your running jobs and cancel all your queued ones
gpuq kill --mine

# Check exit status of a foreground job after it finishes
echo $?   # 0 = success; 128+N = died by signal N (143 = SIGTERM, e.g. timeout)
```

> You can only kill your **own** jobs; `gpuq kill` refuses someone else's. A
> job killed by signal (including a `-t` timeout) exits with the shell
> convention `128 + signal` — a timed-out job typically surfaces as `143`.

## Job History

Every job leaves a record in a shared ledger; `gpuq history` reads it back:

```bash
gpuq history              # your last 20 jobs, oldest first
gpuq history -n 50        # more of them
gpuq history --all        # everyone's jobs (adds a USER column)
gpuq history --user bob   # one specific user
gpuq history --events     # also show cancelled/rejected submits
gpuq history --json       # raw records, one JSON object per line
```

Each row shows when the job ended, how long it queued (WAIT) and ran
(RUNTIME), the GPUs it held, the **GPU-hours charged**, the exit code, and the
RESULT — one of:

- `completed` — exit code 0
- `failed` — non-zero exit
- `timed_out` — killed at its `-t` limit
- `killed` — died by signal (e.g. `gpuq kill`, Ctrl-C)
- `lost*` — the supervising process died (e.g. SSH drop without tmux and a
  later SIGKILL); the job was charged up to the moment it was reaped, capped
  at its time limit

So "did my job finish overnight? when? why did it stop?" is one command — no
more digging through redirected logs to find out *whether* something ran.

## GPU-Hour Quotas

`gpuq` enforces a **rolling 7-day GPU-hour budget** per user (set by the admin in
the config). A budget of `0` or missing means **unlimited** — which is the
**current state**: quotas are not enforced yet, while usage data is collected
to set fair budgets. Charging is by **actual runtime × GPUs held**, recorded
when each job ends (jobs straddling the 7-day cutoff are only charged for the
in-window portion, and running jobs count at their current elapsed time).

Check where you stand at any time:

```bash
gpuq quota          # your usage: finished + running, budget, headroom
gpuq quota --all    # one row per user + host capacity utilisation
```

While budgets are unset, `gpuq quota` reports your usage with an explicit
"quotas are not enforced yet" note, and the `gpuq status` footer shows your
7-day total after every status check.

When budgets are active and a `gpuq submit` would push you over, the job is
**not rejected** — instead it is **deprioritized**:

1. Marked low-priority and forced into the queue (a warning is printed).
2. You are emailed once (if notifications are enabled and your account has an
   email).
3. Polled at a longer interval and made to yield to any normal-priority submitter
   waiting on the same host.

A solo over-quota submit still runs eventually (after its first poll delay); a
contended one waits until the normal-priority queue clears. Pinning GPUs with
`--devices` does **not** skip the quota gate — it chooses *which* card you
get, not *whether* you wait like everyone else.

## Advanced Usage

### Environment Variables

`gpuq` runs your command with **no shell**, so you cannot prefix it with
`VAR=value` (that token would be taken as the program name). Two correct
approaches:

```bash
# 1) Export in your own shell BEFORE submitting — gpuq inherits your environment
export PYTHONPATH=/path/to/modules
gpuq submit -- python train.py

# 2) Or wrap the command in an explicit shell
gpuq submit -- bash -lc "export PYTHONPATH=/path/to/modules && python train.py"
```

> Do **not** set `CUDA_VISIBLE_DEVICES` yourself — `gpuq` sets it to the GPU(s) it
> allocated, and overriding it triggers the rebind detector (see
> [Choosing Which GPU](#choosing-which-gpu)).

### Virtual Environments

Activate your conda/venv environment **in your shell first**, then submit; `gpuq`
inherits the active environment:

```bash
conda activate myenv
gpuq submit -- python train.py
```

Or wrap activation in a shell so it travels with the job:

```bash
gpuq submit -- bash -lc "conda activate myenv && python train.py"
```

### Complex Commands

Because there is no shell, `&&`, `||`, `cd`, redirects (`>`), and background `&`
are not interpreted. Wrap any pipeline in `bash -lc`:

```bash
# Chain multiple commands
gpuq submit -- bash -lc "cd /path/to/project && python preprocess.py && python train.py"

# Conditional execution
gpuq submit -- bash -lc "python train.py && python evaluate.py || echo 'Training failed'"

# Redirect output inside the job
gpuq submit -- bash -lc "python train.py > output.log 2>&1"
```

### Resource Optimization

```bash
# Memory-efficient training with gradient checkpointing
gpuq submit -m 35 -- python train.py --gradient-checkpointing --batch-size 16

# Mixed-precision training
gpuq submit -m 25 -- python train.py --amp --batch-size 64

# Model parallelism across 2 GPUs
gpuq submit -g 2 -m 60 -- python train.py --model-parallel
```

### Interactive Jobs

Interactive sessions work because `gpuq` passes through stdin, but the session is
**tied to your terminal** — there is no detach, and if the job has to queue first,
the submit blocks while it polls. Run interactive jobs inside `tmux`/`screen`.

```bash
# Interactive Python session
gpuq submit -g 1 -t 2 -- python -i

# Interactive shell with GPU access
gpuq submit -g 1 -t 1 -- bash

# Remote development session (code-server)
gpuq submit -g 1 -t 8 -- code-server --bind-addr 0.0.0.0:8080
```

### Keeping Jobs Alive After Logout

The supervising `gpuq submit` process must stay alive for the job's time limit to
be enforced and for output to be captured. Options:

- Run inside **`tmux`** or **`screen`** and detach.
- Ask the admin to **enable user-linger** (`sudo loginctl enable-linger <you>`),
  which keeps your processes — and `gpuq`'s per-job systemd `--user` scope — alive
  after you log out, and gives `gpuq audit` the most robust subprocess tracking.

## Best Practices

### Resource Management

1. **Request only what you need**:
   ```bash
   # Good: specific, modest requirements
   gpuq submit -g 1 -m 30 -t 8 -- python train.py

   # Avoid: grabbing all four GPUs and a giant time window for a small job
   gpuq submit -g 4 -m 120 -t 48 -- python train.py
   ```

2. **Use appropriate time limits**:
   - Quick tests: 1-2 hours
   - Medium experiments: 4-8 hours
   - Long training: 12-48 hours

3. **Mind your quota**: GPU-hours are charged as runtime × GPUs; check
   `gpuq quota` (and `gpuq history` for what past jobs cost) and avoid holding
   GPUs idle.

4. **Monitor resource usage**:
   ```bash
   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 10
   ```

### Job Submission Best Practices

1. **Test locally first**:
   ```bash
   # Quick check with a small dataset / short training before a full run
   python train.py --epochs 1 --batch-size 8
   ```

2. **Use absolute paths** (and a shell when you need `cd`):
   ```bash
   # Good
   gpuq submit -- bash -lc "cd /home/user/project && python train.py"

   # Fragile: relative paths resolve against gpuq's working directory
   gpuq submit -- python ../train.py
   ```

3. **Specify output directories**:
   ```bash
   gpuq submit -- python train.py --output-dir /home/user/results/exp1
   ```

### Code Organization

1. **Use configuration files**:
   ```yaml
   # config.yaml
   model:
     name: "resnet50"
     batch_size: 32
   training:
     epochs: 100
     lr: 0.001
   ```

2. **Enable checkpointing**:
   ```python
   # Save checkpoints regularly
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
   }, f'checkpoint_epoch_{epoch}.pth')
   ```

3. **Log important metrics**:
   ```python
   import wandb  # or tensorboard, mlflow
   wandb.log({"loss": loss, "accuracy": acc, "epoch": epoch})
   ```

## Common Workflows

### Training a Deep Learning Model

```bash
# 1. Prepare data and code
cd /home/user/myproject
ls -la  # check files are ready

# 2. Test locally with a small dataset
python train.py --epochs 1 --batch-size 4 --debug

# 3. Submit the full training job under tmux, teeing the output to a file
tmux new -s train \
  'gpuq submit -g 1 -m 40 -t 12 --notify "user@example.com" \
     -- python train.py --epochs 100 --batch-size 32 --save-dir ./models \
   2>&1 | tee train.log'

# 4. Monitor progress (from another shell)
watch -n 10 gpuq status
tail -f /home/user/myproject/train.log
```

### Hyperparameter Tuning

```bash
# Submit multiple jobs with different hyperparameters.
# Each gpuq submit is foreground, so run them with --queue in the background,
# or launch each in its own tmux window.
for lr in 0.001 0.01 0.1; do
  for bs in 16 32 64; do
    tmux new -d -s "lr${lr}_bs${bs}" \
      "gpuq submit -g 1 -m 30 -t 8 --queue --name lr${lr}_bs${bs} \
         -- python train.py --lr $lr --batch-size $bs"
  done
done

# Abort the whole sweep (running + queued) in one go:
gpuq kill --mine

# Afterwards, compare what each run cost:
gpuq history -n 20
```

### Model Inference

```bash
# Batch inference on a large dataset
gpuq submit -g 1 -m 25 -t 4 \
  -- python inference.py --model-path ./best_model.pth --data-dir ./test_data
```

### Interactive Development

```bash
# Start Jupyter for development (run it under tmux so it survives logout)
tmux new -s jlab \
  'gpuq submit -g 1 -m 30 -t 8 \
     -- jupyter lab --ip=0.0.0.0 --port=8888 --no-browser'

# Connect via SSH tunnel (from your local machine)
ssh -L 8888:localhost:8888 user@server.com

# Open http://localhost:8888 in your browser
```

## Troubleshooting

### Common Issues

#### Job Won't Start / Is Rejected

```bash
# Check queue status
gpuq status

# Common causes:
# 1. No selectable GPU right now — all are busy or held by other users.
#    Add --queue to wait instead of being rejected.
# 2. Your -m/--memory filter is higher than any card's current free VRAM.
#    Lower it, or wait for a card to free up. (With no -m, the config default
#    applies — the rejection message tells you the number it used.)
# 3. --devices pinned a GPU held by someone else (rejected immediately
#    unless you also pass --queue).
# 4. The request fails validation: -t must be > 0, -g must fit the host's
#    GPU count, --devices indices must exist, and -g must match --devices.
#    These are rejected with a specific message before anything is queued.
# 5. You are over your GPU-hour quota — the job is deprioritized, not rejected.

# Debugging smoke test
gpuq submit -g 1 -t 1 -- echo "Test job"
```

#### Job Killed Unexpectedly

```bash
# First: ask gpuq how the job ended (RESULT column + exit code)
gpuq history

# gpuq has no per-job log files — check the output you redirected yourself
tail -100 train.err

# What the RESULT tells you:
# - timed_out: gpuq killed it at --time (it also printed a "time limit
#   reached" notice to the submitting terminal); request more time with -t
# - failed: your code exited non-zero — inspect your redirected stderr (OOM,
#   exceptions, ...)
# - killed: it died by signal (gpuq kill, Ctrl-C, or a closed terminal whose
#   SIGHUP was forwarded to the job — use tmux/screen next time)
# - lost*: the supervising gpuq process itself died (e.g. SIGKILL); the job's
#   hours were still charged when the dead entry was reaped
```

#### GPU Out of Memory

```bash
# Check current GPU memory usage
nvidia-smi

# Solutions:
# 1. Reduce batch size
# 2. Use gradient accumulation
# 3. Enable gradient checkpointing
# 4. Use mixed precision (AMP / bf16)

# Example with memory optimization
gpuq submit -g 1 -m 30 \
  -- python train.py --batch-size 16 --gradient-checkpointing --amp
```

#### Can't See Job Output

```bash
# Check if the job is still running
gpuq status | grep "$USER"

# gpuq does NOT create log files — its output went to your terminal.
# If you redirected it, read that file:
tail -f train.log

# Next time, capture it: `... | tee run.log`, or run under tmux/screen.
```

### Performance Issues

#### Slow Training

```bash
# Check GPU utilization
nvidia-smi -l 1

# If utilization is low:
# 1. Increase batch size
# 2. Check data loading (use more workers)
# 3. Profile your code

# Example with optimized data loading
gpuq submit -g 1 -- python train.py --num-workers 8 --batch-size 64
```

#### Memory Leaks

```bash
# Monitor memory usage over time
nvidia-smi --query-gpu=memory.used --format=csv -l 60 > memory_usage.log

# In Python code, use memory profiling
pip install memory-profiler
python -m memory_profiler train.py
```

### Getting Help

1. **Check system status**: `gpuq status`
2. **Check how the job ended**: `gpuq history` (RESULT + exit code), and
   `gpuq quota` for your usage
3. **Inspect your own redirected output** (there are no per-job log files)
4. **Test with simple commands**: `gpuq submit -- nvidia-smi`
5. **See active settings**: `gpuq config`
6. **Contact the administrator** with:
   - Job ID (from `gpuq history` — it's the same id announced at submit)
   - Command used
   - Error output you captured
   - Expected vs actual behavior

---

**Next Steps**: Once you're comfortable with the queue system, check out the framework-specific guides:
- [PyTorch with H200](pytorch-guide.md)
- [TensorFlow with H200](tensorflow-guide.md)
- [JAX with H200](jax-guide.md)
