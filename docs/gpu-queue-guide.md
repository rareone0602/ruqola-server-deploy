# GPU Queue (gpuq) Guide

`gpuq` shares Mjölnir's 4 GPUs. Start every GPU job with `gpuq submit`. It picks
free cards, sets `CUDA_VISIBLE_DEVICES`, and runs your command in your terminal
until the command ends. GPU work started any other way is killed (see
[Rules](#rules)).

## Quick start

```bash
tmux new -s train                          # the job survives a dropped SSH session
source ~/venvs/myenv/bin/activate          # gpuq runs your command in this environment
gpuq submit -m 40 -t 8 -- python train.py  # 1 GPU with >= 40 GB free, stop after 8 h
```

Detach with `Ctrl-b d`. Come back with `tmux attach -t train`.

The system has `python3` but no `python`; `python` exists inside a venv. To set
one up, see [Best Practices](best-practices.md#set-up-once).

Everyday commands:

```bash
gpuq status          # each GPU, running and queued jobs, your 7-day usage
gpuq history         # how your recent jobs ended and what they cost
gpuq quota           # your GPU-hours in the last 7 days against the budget
gpuq kill 12345      # stop a running job or cancel a queued one (yours only)
gpuq kill --mine     # stop and cancel all of your jobs
gpuq config          # the defaults in force (time limit, -m filter)
gpuq submit -h       # every submit option
```

## Rules

| Rule | Limit | What happens |
|---|---|---|
| Job time | 48 hours | gpuq stops the job at its `-t` limit. A `-t` above 48 is refused. |
| Cards per user | 3 at once | gpuq never gives you a 4th card. Holding 3 alerts the admin, so use 2 unless you need 3. |
| GPU-hours | 168 per rolling 7 days | Over budget, a new job waits 15 minutes, then runs at low priority. It is never refused. |
| GPU work outside gpuq | not allowed | You get a warning email. The process is killed at the first audit 15 minutes or more after the warning. |

- A root audit, `gpuq audit --enforce`, runs every 15 minutes. It finds GPU
  processes that gpuq did not start, and gpuq jobs running on a card they were
  not given (a "rebind"). Each gets a warning email and is killed at the first
  audit 15 minutes or more later. Reminders go out every 2 hours, so in practice
  you see the warning and then the kill.
- This covers everything: a notebook kernel, a quick `python3 -c` test, an IDE
  session. Start them through gpuq ([examples](#interactive-work-and-notebooks)).
- A card held by another user is never given to you. You may stack more of your
  own jobs on cards you already hold.

## Submitting a job

Put your command after `--`. gpuq runs it directly, with no shell.

```bash
gpuq submit -g 2 -m 60 -t 12 -- torchrun --nproc_per_node=2 train.py
```

| Option | Meaning | Default |
|---|---|---|
| `-g N`, `--gpus N` | number of whole GPUs | 1 |
| `-m GB`, `--memory GB` | only use a card with at least this much free VRAM. A filter, not a reservation: the job may use more. | 120 |
| `-t H`, `--time H` | stop the job after H hours. Must be above 0 and at most 48. Decimals work: `-t 0.5`. | 48 |
| `--devices 1,3` | use exactly these GPU indices. Sets the GPU count. | |
| `--queue` | wait for a slot instead of failing | off |
| `--name TEXT` | label shown in `gpuq status` and `gpuq history` | |
| `--notify EMAIL` | send the end-of-job email here instead of to your account's address | |
| `--command "..."` | the command as one string, instead of after `--` | |

**Always pass `-m`.** The default filter is 120 GB, and a card someone else
has been using rarely has that much free. Ask for what your job needs.
`gpuq config` shows the filter in force as `min_free_vram_filter`.

**Pass a realistic `-t`.** It is the kill timer, and the quota check counts it
in advance: a 1-GPU job with the default 48 h counts as 48 GPU-hours at submit.

**There is no shell.** `&&`, `|`, `>`, `cd`, and `VAR=value` prefixes reach your
program as plain arguments. Wrap them in `bash -c`:

```bash
gpuq submit -m 40 -- bash -c "cd ~/proj && python train.py > run.log 2>&1"
```

Use `bash -c`, not `bash -lc`. A login shell resets `PATH` on this host, which
drops your venv. To set a variable, `export` it before you submit; gpuq passes
your environment to the job.

**Jobs longer than 48 hours** must save checkpoints and be resubmitted.

When the job starts, gpuq prints `[gpuq] job <id> starting on GPU(s) <list>`.
The id is what `gpuq kill` and `gpuq history` use.

## Choosing Which GPU

gpuq gives you two kinds of card:

- A **free** card: no gpuq job on it, utilization under 10%, and at least `-m`
  GB free.
- A card **you already hold**: one of your running jobs is on it. Another job
  can stack there if the card has at least `-m` GB free (never less than 2 GB).
  Utilization does not matter.

`-g N` takes free cards first, chosen at random. It stacks on your own cards
only when there are not enough free ones. Once you hold 3 cards, new jobs can
only stack.

`--devices` pins exact cards. Each must be free or already yours. If one is not,
the submit fails at once and names the reason for each card. Add `--queue` to
wait for them instead.

```bash
gpuq submit -g 2 -m 40 -- python train.py               # 2 cards, free ones first
gpuq submit --devices 0,2 -m 40 -- python train.py      # exactly GPUs 0 and 2
gpuq submit --devices 1 --queue -m 40 -- python eval.py # wait for GPU 1
```

To wait for a fresh card instead of stacking on your busy one, use `--queue` with
an `-m` larger than your busy card has free.

gpuq sets `CUDA_VISIBLE_DEVICES` for the job, so inside it your cards are
numbered from 0: `cuda:0` is your first card. Do not set the variable again
inside the job, for example `bash -c "CUDA_VISIBLE_DEVICES=0 python ..."`: that
names physical GPU 0, which may not be yours. A job on a card it was not given
is a rebind and is killed (see [Rules](#rules)). To choose a card, use
`--devices`.

## Keeping Jobs Alive After Logout

A job lives only as long as the `gpuq submit` that started it. If the terminal
closes or SSH drops, gpuq passes the hangup to the job and the job dies. Run
jobs inside `tmux` or `screen`. `nohup` does not protect a gpuq job.

## Monitoring and Management

```bash
gpuq status                  # everything: GPUs, jobs, queue, all GPU processes
watch -n 10 gpuq status      # refresh every 10 s
nvidia-smi -l 1              # raw GPU usage, every second
```

**Output.** The job's output goes to your terminal. gpuq writes no log files. To
keep a copy:

```bash
gpuq submit -m 40 -- python train.py 2>&1 | tee run.log
```

When the job ends, gpuq prints one line such as
`[gpuq] job <id> completed: ran 1:02:03 on GPU(s) 0, 1.03 GPU-hours recorded (exit 0).`
and emails you (see [Notifications](notifications-faq.md)).

**Stopping jobs.** `gpuq kill <id> [<id> ...]` stops your running jobs or
cancels queued ones, from any terminal. `gpuq kill --mine` does all of yours.
You cannot kill another user's job.

**Exit code.** `gpuq submit` exits with the job's exit code. A job killed by a
signal exits with 128 + the signal number: 143 for SIGTERM, which is what a
time-limit stop sends first.

## Job History

```bash
gpuq history              # your last 20 jobs, oldest first
gpuq history -n 50        # more (0 = all)
gpuq history --all        # everyone's jobs
gpuq history --user bob   # one user's jobs
gpuq history --events     # also cancelled and rejected submits
gpuq history --json       # raw records
```

The RESULT column says how each job ended:

| RESULT | Meaning |
|---|---|
| `completed` | exit code 0 |
| `failed` | non-zero exit code |
| `timed_out` | stopped at its `-t` limit |
| `killed` | ended by a signal: `gpuq kill`, Ctrl-C, or a closed terminal |
| `lost*` | the `gpuq submit` process died; the job was charged until gpuq noticed, at most its `-t` |

## GPU-Hour Quotas

Each user has **168 GPU-hours per rolling 7 days**. That is one GPU running
around the clock.

- **What is charged:** runtime × number of cards, for every job. Stacked jobs are
  charged separately: two 1-GPU jobs sharing one card cost 2 GPU-hours per hour.
  Running jobs count as they go.
- **The check at submit:** gpuq compares your usage plus `cards × -t` with 168.
- **Over budget:** the job is not refused. gpuq prints and emails the time until
  which it is held. It waits 15 minutes from submit, then queues at low priority:
  it starts only when no normal-priority job waiting could take the slot. It
  checks for a slot every 120 seconds.
- `--devices` does not skip the check.

```bash
gpuq quota          # your finished + running GPU-hours, budget, status
gpuq quota --all    # every user, and how full the host is
```

`gpuq quota` can say OK while a submit is still held, because the submit check
adds `cards × -t`. Pass a realistic `-t`.

## Common Workflows

### Multi-GPU training

```bash
gpuq submit -g 2 -m 60 -t 12 -- torchrun --nproc_per_node=2 train.py
```

Keep `--nproc_per_node` equal to `-g`.

### A sweep

Each `gpuq submit` holds its terminal, so give each run its own tmux session
and let them wait with `--queue`. Activate the venv inside each session:

```bash
for lr in 0.001 0.01 0.1; do
  tmux new -d -s "lr$lr" "source ~/venvs/myenv/bin/activate && gpuq submit -m 30 -t 4 --queue --name lr$lr -- python train.py --lr $lr"
done
gpuq kill --mine     # stops the whole sweep, running and queued
```

### Interactive work and notebooks

```bash
gpuq submit -m 20 -t 1 -- bash       # a shell on 1 GPU; everything you start in it is tracked
gpuq submit -m 40 -t 4 -- jupyter lab --no-browser --port 8888   # inside tmux
```

Install `jupyter` in your venv first. From your own computer, open a tunnel with
`ssh -N -L 8888:localhost:8888 <you>@<server>`, then browse to
`http://localhost:8888`.

## Troubleshooting

| You see | Cause and fix |
|---|---|
| `gpuq: no free GPU matches your request (need 1 GPU(s) with >= 120 GB free; ...)` | No card meets your `-m`. Pass a smaller `-m`, or add `--queue` to wait. |
| `gpuq: requested GPU(s) not available — not submitting:` | A `--devices` card is held or too full; the next lines say why. Add `--queue`, or pick other cards. |
| `Per-user card cap: you already hold GPU(s) [...] of the 3-card cap` | You hold 3 cards. The job can only stack on them. Wait for one of your jobs to end, or add `--queue`. |
| `gpuq: this host caps each user at 3 concurrent GPU(s)` | `-g` above 3. Use 3 or fewer. |
| `gpuq: -t/--time ...h exceeds the 48h (2-day) wall-time cap on wsserver1.` | Use `-t 48` or less; checkpoint and resubmit. |
| `[gpuq] over quota: used ...` | Over 168 GPU-hours. The job waits 15 minutes, then runs at low priority. See [GPU-Hour Quotas](#gpu-hour-quotas). |
| `[gpuq] no slot; queued as job <id> ..., polling every 30s.` | Normal with `--queue`: it starts when a card frees up. |
| `[gpuq] job <id> reached its 8.0h time limit — sending SIGTERM (SIGKILL in 10s).` | The job hit `-t`. Resubmit with a larger `-t` (at most 48). |
| `python: command not found` | Activate your venv before `gpuq submit`, or use `bash -c`, not `bash -lc`. |
| Job died, reason unknown | Run `gpuq history`. Read the RESULT and exit code, then your saved output. |

More problems: [Troubleshooting](troubleshooting.md). For the admin, include
the job id, the command, and the output you saved.
