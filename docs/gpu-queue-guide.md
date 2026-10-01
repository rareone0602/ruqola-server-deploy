# GPU Queue (gpuq) Guide

`gpuq` shares Mjölnir's 4 GPUs. Start every GPU job with `gpuq submit` (or
`gpuq shell`). It waits for your turn, gives the job GPUs of its own, and runs
your command as you, in your folder, with your environment. Start all GPU work
through gpuq (see [Rules](#rules)).

## Quick start

```bash
tmux new -s train                     # a job lives as long as its terminal
source ~/venvs/myenv/bin/activate     # gpuq runs your command in this environment
gpuq submit -- python train.py        # 1 GPU to itself, for up to 48 h
```

Detach with `Ctrl-b d`. Come back with `tmux attach -t train`.

Or let it run without a terminal, and read its output from a file:

```bash
gpuq submit --detach -- python train.py   # prints the job id; output in ~/gpuq-logs/<id>.log
```

The system has `python3` but no `python`; `python` exists inside a venv. To set
one up, see [Best Practices](best-practices.md#set-up-once).

Everyday commands:

```bash
gpuq status          # each GPU, running jobs, and the queue in order
gpuq why 12345       # why a job is waiting, and when it should start
gpuq share           # your recent GPU use and your place in line
gpuq history         # how your recent jobs ended
gpuq kill 12345      # stop a running job or cancel a waiting one (yours only)
gpuq kill --mine     # stop and cancel all of your jobs
gpuq submit -h       # every submit option
```

## Rules

| Rule | What happens |
|---|---|
| Job time: 48 hours | Every job may run 48 h. Then it is stopped: SIGTERM, then SIGKILL 10 s later. You get an email 1 hour before. |
| Who goes first | When more people want GPUs than there are, whoever has used the fewest GPU-hours lately goes first. Use fades by half every 7 days. |
| Jobs needing 2+ GPUs | get a promised start time, and the first GPUs to free up are held for them. A promise only moves earlier. |
| Your GPU is yours | While your job runs on a GPU, nobody else's job is put on it. You may add more of your own jobs ([Sharing your own GPU](#sharing-your-own-gpu)). |
| GPU work outside gpuq | Not allowed. gpuq records it, and will stop it automatically in a later step. |

- There is no weekly quota and no limit on GPUs per person. Fair-share keeps it
  fair: the more you have used lately, the later your turn when others wait.
- What counts as use: GPU-hours **held**. A GPU counts once, however many of
  your jobs share it.
- "All GPU work" means everything: a notebook kernel, a quick `python3 -c`
  test, an IDE session. Use `gpuq shell` for interactive work
  ([Background jobs and shells](#background-jobs-and-shells)).
- A job can only reach its own GPUs. The other GPUs do not exist for it.

## Submitting a job

Put your command after `--`. gpuq runs it directly, with no shell.

```bash
gpuq submit -g 2 -- torchrun --nproc_per_node=2 train.py
```

| Option | Meaning | Default |
|---|---|---|
| `-g N`, `--gpus N` | number of whole GPUs | 1 |
| `-m GB`, `--memory GB` | the VRAM you need. With it, the job may run beside your own running job where that much is free ([Sharing your own GPU](#sharing-your-own-gpu)) | none: a GPU to itself |
| `--devices N` | add the job to your own GPU N, which one of your jobs is running on ([Sharing your own GPU](#sharing-your-own-gpu)) | |
| `--detach` | return at once; output goes to `~/gpuq-logs/<id>.log` | off |
| `--name TEXT` | label shown in `gpuq status` and `gpuq history` | |
| `--notify` | email you when the job ends | off |
| `--command "..."` | the command as one string, instead of after `--` | |
| `-t H`, `--queue` | accepted so old scripts still run, and ignored | |

**Leave out `-m` for a normal run.** Without it, the job gets a GPU to itself.
With `-m`, gpuq first puts the job beside a job of yours that is already
running, if that GPU has room, and the two share its compute.

**There is no time limit flag.** Every job may run 48 h. For a shorter limit,
wrap the command: `gpuq submit -- timeout 4h python train.py`. For longer work,
save checkpoints and resubmit. Inside the job, `GPUQ_DEADLINE` holds the stop
time (epoch seconds).

**There is no shell.** `&&`, `|`, `>`, `cd`, and `VAR=value` prefixes reach your
program as plain arguments. Wrap them in `bash -c`:

```bash
gpuq submit -- bash -c "cd ~/proj && python train.py > run.log 2>&1"
```

Use `bash -c`, not `bash -lc`. A login shell resets `PATH` on this host, which
drops your venv. To set a variable, `export` it before you submit; gpuq passes
your environment to the job.

**Waiting.** When no GPU is free for you, the job waits its turn and gpuq says
where it stands, for example
`[gpuq] job <id> queued: needs 1 card(s), 0 free. Estimate: ~Fri 16:34.`
An estimate can move either way; a promise (2+ GPUs) only moves earlier.
`gpuq why <id>` explains the wait. Ctrl-C cancels it.

When the job starts, gpuq prints `[gpuq] job <id> starting on GPU(s) <list>.`
The id is what `gpuq kill`, `gpuq why` and `gpuq history` use.

**Inside the job:**

- `CUDA_VISIBLE_DEVICES` is `0`, `0,1`, …: `cuda:0` is your first GPU.
- `GPUQ_GPUS` holds the host's numbers for your GPUs, as `gpuq status` and
  `nvidia-smi` show them. `GPUQ_JOB_ID` is the job id.
- `nvidia-smi` inside the job lists only your GPUs.
- Do not set `CUDA_VISIBLE_DEVICES` yourself. Another number finds no GPU,
  because the job cannot reach GPUs it was not given.

**Exit code.** `gpuq submit` exits with the job's exit code. A job ended by a
signal exits with 128 + the signal number: 143 for SIGTERM, which is what the
48-hour stop and `gpuq kill` send first.

## Sharing your own GPU

A GPU your job is running on is yours, and you can add more of your own jobs
to it: a quick test of an idea beside your training, an evaluation, a debugger.
Adding them is never charged extra.

| you type | where the job runs | waits its turn? | may run |
|---|---|---|---|
| neither flag | a GPU to itself | yes | 48 h |
| `-m X` | beside a job of yours, on a GPU of yours with X GB free; if none has room, a free GPU | yes | 48 h |
| `--devices N`, with or without `-m X` | your GPU N, and nowhere else | no | until your job(s) on GPU N reach their 48 h |

`--devices` is the way to pick the GPU:

```bash
gpuq status                                            # your training runs on GPU 2
gpuq submit --devices 2 -m 10 -- python test_idea.py   # beside it, once 10 GB is free there
gpuq shell --devices 2                                 # a shell on GPU 2, beside it
```

- It skips the queue: the job starts as soon as GPU 2 has the `-m` GB free,
  and never with less than 2 GB free. Until then `gpuq status` lists it under
  "joining their own card".
- It must end when your job(s) on GPU 2 reach their 48 h. The start message
  says when.
- If your jobs on GPU 2 all end before there is room, it is cancelled.
- `--devices` on a GPU you have no job on is refused. For new GPUs, use `-g N`:
  all 4 GPUs are identical, so gpuq picks which.
- Jobs sharing a GPU share its compute: each runs slower than it would alone.

## Background jobs and shells

**Attached (the default).** The job writes to your terminal and lives as long
as the `gpuq submit` that started it. Closing the terminal, or a dropped SSH
session, stops it. Run attached jobs inside `tmux` or `screen`; `nohup` does
not help.

**Detached.** `gpuq submit --detach -- ...` returns at once and prints the job
id. The job keeps running after you log out. Its output goes to
`~/gpuq-logs/<id>.log`:

```bash
gpuq submit --detach --name run1 -- python train.py
tail -f ~/gpuq-logs/<id>.log       # follow it; Ctrl-C stops tail, not the job
gpuq kill <id>                     # stop it
```

**Interactive.** `gpuq shell` waits its turn like any job, then opens a shell on
a GPU. Everything you start in it is part of the job. The job ends when you
exit the shell. It takes `-g`, `-m` and `--devices`, and needs a terminal.

Notebooks: inside tmux, run `gpuq shell`, then `jupyter lab --no-browser --port 8888`
in that shell. Install `jupyter` in your venv first. From your own computer, open
a tunnel with `ssh -N -L 8888:localhost:8888 <you>@<server>`, then browse to
`http://localhost:8888`.

## Monitoring and history

```bash
gpuq status                  # GPUs, running jobs, the queue, GPU use outside gpuq
watch -n 10 gpuq status      # refresh every 10 s
gpuq why <id>                # why a job is waiting, and when it should start
gpuq share --all             # everyone's recent use: least goes first
nvidia-smi -l 1              # raw GPU usage, every second
```

**Output.** An attached job writes to your terminal, a detached one to
`~/gpuq-logs/<id>.log`. To keep a copy of an attached job's output:

```bash
gpuq submit -- python train.py 2>&1 | tee run.log
```

When the job ends, gpuq prints one line such as
`[gpuq] job <id> completed: ran 1:02:03 on GPU(s) 0, 1.03 GPU-hours recorded (exit 0).`

**Email** goes to the address in your account details:

- when a job that waited 10 minutes or more starts;
- 1 hour before a job's 48-hour stop;
- when a job ends, only if you passed `--notify`.

**Stopping jobs.** `gpuq kill <id> [<id> ...]` stops your running jobs (SIGTERM,
then SIGKILL 10 s later) or cancels waiting ones, from any terminal.
`gpuq kill --mine` does all of yours. You cannot kill another user's job.

**History.**

```bash
gpuq history              # your last 20 jobs, oldest first
gpuq history -n 50        # more (0 = all)
gpuq history --all        # everyone's jobs
gpuq history --user bob   # one user's jobs
gpuq history --events     # also jobs that left the queue without running
gpuq history --json       # raw records
```

The RESULT column says how each job ended:

| RESULT | Meaning |
|---|---|
| `completed` | exit code 0 |
| `failed` | non-zero exit code |
| `timed_out` | stopped at 48 h |
| `killed` | ended by a signal: `gpuq kill`, Ctrl-C, or a closed terminal |
| `lost*` | gpuq lost track of the job (rare) |

## Common workflows

### Multi-GPU training

```bash
gpuq submit -g 2 -- torchrun --nproc_per_node=2 train.py
```

Keep `--nproc_per_node` equal to `-g`. The job's GPUs talk over NVLink.

### A sweep

Detached jobs need no tmux session each. Activate the venv first: each job
takes the environment of the shell that submitted it.

```bash
source ~/venvs/myenv/bin/activate
for lr in 0.001 0.01 0.1; do
  gpuq submit --detach --name lr$lr -- python train.py --lr $lr
done
gpuq status          # where each one stands
gpuq kill --mine     # stops the whole sweep, running and waiting
```

## Troubleshooting

| You see | Cause and fix |
|---|---|
| `[gpuq] job <id> queued: needs 1 card(s), 0 free. ...` | Normal: no GPU is free for you yet. `gpuq why <id>` says why and when. |
| `gpuq: --devices 2: you have no job running on GPU 2. ...` | `--devices` only adds a job to a GPU your own job runs on. For a new GPU, leave it out (`-g N` for N GPUs). |
| `[gpuq] -t 8 is ignored: every job may run 48 h ...` | Just a note. For a shorter limit: `gpuq submit -- timeout 8h python ...`. |
| `gpuq: this host has 4 GPU(s); you asked for 5.` | Ask for 4 or fewer. |
| `gpuq: -m 200: the largest card here has 140 GB.` | Ask for what fits on one GPU. |
| ``gpuq: gpuq shell needs a terminal; use `gpuq submit` for scripts.`` | Run `gpuq shell` from an interactive terminal (tmux is fine). |
| `gpuq: cannot reach gpuqd at /run/gpuq/gpuqd.sock ...` | The queue service is down. Tell the admin. |
| Job ended with RESULT `timed_out`, exit 143 | It hit 48 h. Save checkpoints and resubmit. |
| CUDA finds no GPU, or "invalid device ordinal" | The job set `CUDA_VISIBLE_DEVICES`, or used `cuda:N` beyond what `-g` gave it. Use `cuda:0` … `cuda:<g-1>`. |
| `python: command not found` | Activate your venv before `gpuq submit`, or use `bash -c`, not `bash -lc`. |
| Job died, reason unknown | Run `gpuq history`. Read the RESULT and exit code, then your saved output (`~/gpuq-logs/<id>.log` for detached jobs). |

More problems: [Troubleshooting](troubleshooting.md). For the admin, include
the job id, the command, and the output you saved.
