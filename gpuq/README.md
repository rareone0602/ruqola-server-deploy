# Custom GPU Queue Management

This subfolder contains the custom script implementing the GPU queuing and monitoring system set up on the Mjölnir server.
The script is contained in `userspace.py` (deployed as `gpuq`). The older `gpu_queue.py` is the **retired legacy daemon**, kept only in the development repo for reference — it is no longer what `gpuq` runs. Below instructions show how the script was set up on the server (and how it can be set up again, in case), as well as some basic usage examples and other useful information for GPU monitoring on the server.

> **gpuq is daemonless.** Each `gpuq submit` runs your command in the **foreground**
> in your own terminal; that process claims the GPU(s), sets
> `CUDA_VISIBLE_DEVICES`, supervises the job, and enforces its time limit. There is
> no background service, no `gpuq daemon`, and no per-job log files written by gpuq —
> your job's stdout/stderr go straight to your terminal (redirect them yourself if you
> want a file). Periodic enforcement (resource-hog / quota / untracked / rebind
> checks) is done by `gpuq audit`, scheduled from cron — see "Admin audits" below.

## The server (Mjölnir)

The host (`wsserver1`) has **4 × NVIDIA H200 NVL** GPUs (indices `0,1,2,3`),
each with **~141 GB** of VRAM (`nvidia-smi` reports ~143771 MiB), for **~564 GB**
total across the box. These are Hopper-class cards (compute capability **9.0**,
GH100 die), PCIe/NVL form factor, 600 W power cap, max SM clock ~1785 MHz. Tensor-core
BF16/FP16 dense matmul tops out around **~836 TFLOP/s** empirically on this server
(datasheet ~989 TFLOP/s FP16 with sparsity); the ~134 TFLOPS figure you may see quoted
is the non-tensor CUDA-core number, not the headline throughput.

Host basics: 256 logical CPUs, 755 GiB RAM, Ubuntu 24.04.4 LTS, GPU driver
575.57.08, CUDA (driver) 12.9. "All GPUs" therefore means indices `0,1,2,3`
(e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3`, `torchrun --nproc_per_node=4`) — but note
that gpuq caps each user at **3 concurrent cards**, so no single gpuq job can
take all four (`gpuq submit -g 4` is refused; `-g 3` is the per-user maximum).

# GPU Queue Management System Setup

## Installation

The script is installed with the two provided installers — do **not** copy it by hand,
and do **not** use the legacy `gpu_queue.py`.

1. **System install** (deploys `userspace.py` to `/usr/local/bin/gpuq`, needs root).
   It validates the source compiles, installs atomically, ensures the shared
   coordination dir `/var/lib/gpu_queue/` exists (group `gpuqueue`, SGID), and seeds a
   starter config only if none exists:
   ```bash
   sudo ./install_system.sh
   ```

2. **Per-user install** (no root). This publishes the shared master to
   `/var/lib/gpu_queue/gpuq.py` and shadows `~/.local/bin/gpuq` so it precedes the
   system binary on `PATH`:
   ```bash
   ./install_user.sh --publish-shared
   ```
   On subsequent machines/users, once the shared master exists, a plain
   `./install_user.sh` symlinks `~/.local/bin/gpuq` to it. Other modes:
   `--copy-shared` (per-user copy of the shared master instead of a symlink)
   and `--copy-from-repo` (per-user copy straight from this repo).

3. (Optional) Install `requests` if you want Slack notifications from `gpuq audit`:
   ```bash
   pip install requests  # for Slack notifications
   ```

> **Upgrading: replace EVERY deployed copy in one pass.** gpuq lives in three
> places (system `/usr/local/bin/gpuq`, shared master
> `/var/lib/gpu_queue/gpuq.py`, and each user's `~/.local/bin/gpuq`). Symlink-
> mode users pick up a republished master automatically, but `--copy-shared` /
> `--copy-from-repo` users keep private copies — chase those down. Old
> binaries sharing `/var/lib/gpu_queue` with new ones are *mostly* compatible
> (the ledger format tolerates both directions), but an old binary's reaper
> will silently drop entries the new version deliberately keeps (orphaned
> jobs whose workload is still on the GPU), losing their ledger hours — so
> don't rely on the job-log accounting until every copy is current:
> `sudo ./install_system.sh && ./install_user.sh --publish-shared`.

4. Create the configuration file. `gpuq config init` writes the **canonical
   default** template (the source of truth lives in `userspace.py`), but only if
   no config exists yet — otherwise it refuses and tells you to use `--force`.
   A bare `gpuq config` is **read-only**: it shows the loaded settings and paths
   (safe for any user to run):
   ```bash
   gpuq config            # show the loaded config, paths, and host/user
   gpuq config init       # (admin) write the default config if none exists
   gpuq config init --force   # (admin) overwrite an existing config
   ```
   Always prefer the generated template over copying any on-disk sample by hand.
   `install_system.sh` already seeds this config the first time it runs.

5. Edit the configuration file to enable notifications, quotas, and audit
   thresholds. The full template `gpuq config init` writes looks like this
   (`username`/`password` are seeded empty — fill in the SMTP sender account):
   ```json
   {
     "max_job_time_hours": 24,
     "max_job_time_hours_cap": 48,
     "max_memory_per_gpu_gb": 70,
     "default_min_free_gb": 16,
     "max_gpus_per_user_hard": 3,
     "notification_email": {
       "enabled": false,
       "smtp_server": "smtp.gmail.com",
       "smtp_port": 587,
       "username": "",
       "password": "",
       "admin_email": "admin@yourlab.com"
     },
     "slack": {
       "enabled": false,
       "webhook_url": "",
       "channel": "#gpu-alerts"
     },
     "quotas": {
       "default_gpu_hours_per_week": 0,
       "delay_hours": 0.25,
       "users": {}
     },
     "audit": {
       "max_gpus_per_user": 2,
       "max_total_memory_gb": 50,
       "notify_untracked": false,
       "untracked_min_memory_mb": 512,
       "untracked_grace_seconds": 120,
       "untracked_grace_hours": 0.25,
       "untracked_reminder_hours": 2,
       "untracked_allowlist": [],
       "notify_rebind": false,
       "rebind_min_memory_mb": 512,
       "rebind_grace_seconds": 120,
       "rebind_grace_hours": 0.25,
       "rebind_reminder_hours": 2
     }
   }
   ```
   See the **GPU-hour quotas** and **Admin audits** sections lower in this file for
   what the `quotas` and `audit` blocks do. `default_min_free_gb` is the `-m`
   filter applied when a submit passes no `-m` (configs that predate this key
   fall back to `max_memory_per_gpu_gb`). The webhook in the `slack` block posts
   to whatever channel it was created for; the `channel` key is a human-readable
   note, not something the code reads.

> The userspace `gpuq` resolves a user's email from their account's **GECOS**
> field (`getent passwd <user>`), not from a config map — so there is **no**
> `user_emails` block. The `notification_email` block holds only the SMTP
> **sender** credentials plus `admin_email`. `--notify EMAIL` on `gpuq submit`
> overrides the per-job completion-notice address.

## No daemon

There is nothing to "set up as a service" — gpuq has no daemon. Each `gpuq submit`
is its own foreground supervisor, and shared coordination state lives in
`/var/lib/gpu_queue/` (SGID-writable by the `gpuqueue` group), created by
`install_system.sh`. The only thing you schedule is the **auditor** (`gpuq audit`)
via cron; see the "Admin audits" section.

## Usage Examples

### Virtual Environment
When using the gpuq tool, always remember to activate your virtual environment first, so that the required libraries for your experiments will be available to the script. gpuq runs your command in the foreground, inheriting your current environment.

#### Activate the environment with conda
```bash
conda activate $your_environment
```

#### Activate the environment with venv
```bash
source venvs/$your_environment/bin/activate
```

### Basic Job Submission
```bash
# Submit a simple training job (one random free GPU)
gpuq submit -- python train.py --epochs 100

# Request 2 GPUs, each needing >= 40GB free VRAM, for up to 12 hours
gpuq submit -g 2 -m 40 -t 12 -- python big_model.py

# Submit with an email completion notice
gpuq submit --notify "user@lab.com" -- python experiment.py

# Capture output to a file (gpuq writes no logs itself — redirect yourself)
gpuq submit -- python train.py > train.log 2>&1
```

When the job ends, gpuq prints a one-line summary to stderr with the runtime,
the GPUs used, the GPU-hours recorded to the ledger, and the exit code.

### Monitoring
```bash
# Check current status (GPUs + who holds them + running/queued jobs +
# live GPU processes + your own 7-day usage footer)
gpuq status

# Monitor in real-time
watch -n 5 gpuq status

# Your recent jobs from the ledger (runtime, GPU-hours, exit, end reason)
gpuq history

# Your rolling 7-day GPU-hours vs budget
gpuq quota
```

### Advanced Usage
```bash
# Kill one of your own RUNNING jobs (positional job id, or --job-id)
gpuq kill 12345

# Cancel one of your QUEUED jobs from any terminal (same command)
gpuq kill 67890

# Several at once, or everything of yours on this host
gpuq kill 12345 67890
gpuq kill --mine

# Submit interactive job (for Jupyter notebooks)
gpuq submit -t 4 -- jupyter notebook --ip=0.0.0.0 --port=8888
```

## Creating Wrapper Scripts

Create convenient aliases for common tasks:

### `/usr/local/bin/gpu-train` - Training wrapper
```bash
#!/bin/bash
# Wrapper for training jobs
if [ $# -eq 0 ]; then
    echo "Usage: gpu-train <script.py> [-g N] [-t H]"
    exit 1
fi

SCRIPT=$1
shift
gpuq submit "$@" -- python "$SCRIPT"
```

### `/usr/local/bin/gpu-jupyter` - Jupyter wrapper
```bash
#!/bin/bash
# Start Jupyter with GPU access
PORT=${1:-8888}
gpuq submit -t 8 -- jupyter notebook --ip=0.0.0.0 --port="$PORT" --no-browser
```

> Because gpuq runs in the foreground, these wrappers block in the terminal that
> launched them (run them under `tmux`/`screen`, or `nohup ... &`, if you want to
> detach). Job output is the command's own stdout/stderr.

## User Guidelines

### For Users
1. **Always specify resource requirements** - don't hog more than you need
2. **Use time limits** - helps others plan their work
3. **Monitor your jobs** - check `gpuq status` regularly; `gpuq history` shows
   how past jobs ended and what they cost in GPU-hours
4. **Kill finished jobs** - if something hangs, use `gpuq kill <job-id>`
   (works on queued jobs too; `gpuq kill --mine` clears everything of yours)

### For Admins
1. **Monitor the Slack channel** - on the #gpu-alerts channel you'll get `gpuq audit` alerts about resource hogs, over-quota users, and untracked/rebound jobs
2. **Schedule `gpuq audit`** from cron (see "Admin audits") and check its output/email summary
3. **Adjust limits** in the config file as needed
4. **Job output is the user's terminal** - gpuq writes no central log files; ask users to redirect their own output if you need it persisted

## Slack Integration

To set up Slack notifications (used by `gpuq audit`):

1. Go to your Slack workspace
2. Create a new app at https://api.slack.com/apps
3. Add "Incoming Webhooks" feature
4. Create a webhook for your channel
5. Copy the webhook URL to your config file (the `slack` block) and set `"enabled": true`

`gpuq audit` will send a Slack message when it finds:
- resource hogs (users over the GPU/memory thresholds)
- users over their GPU-hour quota
- untracked GPU jobs and GPU rebinds (when those detectors are enabled)

## Email Notifications

For Gmail, you'll need to:
1. Enable 2-factor authentication
2. Generate an "App Password"
3. Use the app password in the config (not your regular password)

The config's `notification_email` block holds only the **sender** account
(username + app password) and `admin_email`. **Recipient** addresses are read from
each user's own account (GECOS field, via `getent passwd`) — there is no per-user
email map in the config.

At the moment we use a new gmail account **mjolnirruqola@gmail.com** to send email notifications via smtp.
All information related to this account (passwords/app password, etc.) are securely stored on the OneDrive folder named mjolnir.

## Resource Limits

What `gpuq submit` itself enforces (no background system is involved):
- **Refuses to start a job** when no GPU meets the request (`-g N` GPUs each with
  `-m GB` free VRAM and utilization below the idle threshold) — pass `--queue` to
  wait instead of being rejected.
- **Kills a job after its time limit** (`-t HOURS`; the default is the config's
  `max_job_time_hours` — 48h on this host, 24h in the shipped template —
  hard-capped at 48h / 2 days): the supervising foreground process arms a timer
  and terminates the job when it expires.
- **Caps concurrent cards per user** (`max_gpus_per_user_hard`, 0 = off): a user
  at the cap can still submit, but new jobs stack onto cards they already hold
  instead of claiming free ones, and a single job asking for more cards than the
  cap is refused outright. Cards you share with another user count toward your
  total. This is the *hard* companion to the warn-only `audit.max_gpus_per_user`.

What `gpuq audit` enforces (scheduled from cron, see below):
- **Flags resource hogs** — users holding **more than** `audit.max_gpus_per_user`
  GPUs (default 2, i.e. it fires at 3+) or **more than** `audit.max_total_memory_gb`
  total (default 50 GB). Both thresholds are configurable.
- **Flags over-quota users** — see "GPU-hour quotas".

Scheduling is **opportunistic**: a free slot goes to whichever submit asks for
it first (waiters poll every 30s; the queue shown by `gpuq status` is
informational, not a strict order). The one hard rule is that **over-quota
submissions are deprioritized** and always yield to normal-priority waiters
(see "GPU-hour quotas" below).

## Monitoring Commands

```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Check who's using what (gpuq status lists every live GPU compute process)
gpuq status

# Just the running jobs
gpuq status | grep -A 10 "Running Jobs"

# Want a job's output in a file? Redirect at submit time (gpuq writes no logs):
gpuq submit -- python train.py > train.log 2>&1
tail -f train.log
```

## Choosing which GPU

`gpuq` hands out two kinds of GPU: ones that are **free** for anyone (held by no
gpuq job, with enough free VRAM and under 10% utilization) and ones **you already
own** (held by one of your own running jobs) — you may stack more jobs on your own
cards. A GPU held by *another* user is never handed to you until it frees up.

An owned card skips the 10%-utilization check (your own job legitimately drives it
up) but must still clear the **same `-m` free-VRAM filter** you asked for — it needs
`max(2 GB, your -m)` free — so `-m` is honored when stacking too, and a 2 GB floor
always guards against an instant out-of-memory when `-m` is tiny.

To wait for a genuinely **free** card instead of stacking onto your own busy one,
add `--queue` with an `-m` larger than that card's current free VRAM (a free H200
NVL has ~141 GB) — your own card won't clear the filter, so the job waits for a
free one. (Pinning a card you don't hold — `--devices <n> --queue` — also waits.)

- **Default:** `gpuq submit -g N` picks N GPUs, preferring **free** cards (chosen
  at random, to spread load across the box) and only stacking onto cards you
  already own when there aren't enough free ones.
- **Pin specific GPUs:** `gpuq submit --devices 1,3 -- …` runs on exactly those
  indices (the count is taken from the list). Each must be free or already yours.
  If any is held by **another user**, the submit is **rejected** with a per-GPU
  reason — *unless* you add `--queue`, in which case it **waits** until all the
  pinned GPUs are available.

```bash
gpuq submit -g 2 -- python train.py              # 2 free GPUs (spread), else stack on yours
gpuq submit --devices 0,2 -- python x.py         # exactly GPU 0 and 2 (else rejected)
gpuq submit --devices 0 -- python b.py           # stack another job onto your own GPU 0
gpuq submit --devices 1,3 --queue -- python y.py # wait for exactly GPU 1 and 3
```

`-m/--memory` is the **minimum free VRAM** a candidate GPU must have to be selected
(an admission check at submit time) — it is **not** a hard cap or reservation the
job is later held to; once the job runs it can use as much VRAM as the card has.
When `-m` is omitted, the filter defaults to the config's `default_min_free_gb`
(16 in the current template; older configs fall back to `max_memory_per_gpu_gb`).

## The job ledger (`usage.jsonl`)

Every job leaves a JSON line in `/var/lib/gpu_queue/usage.jsonl` — this is both
the quota ledger and the job log that future budgets will be calibrated from.
Record types (the `event` key; a missing `event` means `end`, which is how
pre-ledger-v2 lines are absorbed):

- **`end`** — a job finished. Carries the full job context: `command`, `name`,
  `gpus_requested` vs the granted `gpus`, `devices` (when pinned), `memory_gb`
  filter, `max_time_hours`, `priority`, `over_quota_at_submit`, `submitted_at` /
  `started_at` / `ended_at`, `queue_wait_sec`, `elapsed_hours`, `gpu_hours`,
  `exit_code`, and `end_reason` — one of `completed`, `failed`, `timed_out`,
  `killed` (died by signal, e.g. `gpuq kill`), or `lost`.
- **`end` with `end_reason: "lost"`, `synthetic: true`** — written by the
  reaper when a job's supervising process AND its child both died without
  reaching normal accounting (e.g. the supervisor was SIGKILLed). The job is
  charged from its start to the reap, capped at its own time limit, so killed
  supervisors cannot erase GPU-hours from the ledger.
- **`cancelled`** — a queued submit that never ran (Ctrl-C, SIGTERM, killed
  via `gpuq kill`, or its waiting process died). Uses `at`/`wait_sec`; never
  quota-charged.
- **`rejected`** — a submit refused outright (no free slot without `--queue`,
  pinned `--devices` unavailable, or the request was blocked by the per-user
  card cap — reason `user_card_cap`). Records unmet demand — a user with low
  usage but many rejections is starved, not idle.

Users read the ledger with `gpuq history` (`--all`, `--user U`, `--events` to
include cancelled/rejected, `--json` for scripting). The file is group-readable
and append-only under the shared lock; old and new gpuq versions can write it
concurrently (old readers skip what they don't know). If it ever grows
unwieldy, rotate it manually to `usage-YYYY-MM.jsonl` in the same directory —
all readers pick rotated files up automatically.

## GPU-hour quotas

The userspace `gpuq` enforces a rolling 7-day GPU-hour budget per user. Set it
in the config file:

```json
{
  "quotas": {
    "default_gpu_hours_per_week": 168,
    "delay_hours": 0.25,
    "users": { "alice": 250, "bob": 50 }
  }
}
```

A budget of `0` (or missing) means unlimited. `delay_hours` is the
**over-quota hold**: how long an over-budget submit must wait after submission
before it may claim any slot at all (0 or missing = no hold, deprioritization
only). It is a float, so sub-hour holds are legal — the shipped `0.25` is
**15 minutes**.

Charging is by **actual runtime × GPUs held**, recorded in the ledger at job
end. The rolling window is exact: a job straddling the 7-day cutoff is only
charged for the in-window fraction, and running jobs count at their current
elapsed time. Pinned (`--devices`) submits pass the same quota gate as the
random picker — pinning chooses *which* card you get, not *whether* you skip
the line.

When a `gpuq submit` would push the user over budget, the job is **not
rejected** — instead it is:

1. Marked `priority: low` and forced into the queue (warning printed), and —
   with `quotas.delay_hours` set — **held**: it may not claim a slot before
   submit-time + that many hours (shipped default `0.25` = **15 minutes**).
   `gpuq status` shows the hold deadline on the queued entry.
2. Sent an email (if `notification_email` is enabled and the user has an email
   on their account, read from GECOS), including the hold deadline.
3. Polled at a longer interval (default 120s, override with
   `GPUQ_DEPRIORITIZED_POLL_SEC`) and made to yield to any normal-priority
   submitter waiting on the same host.

Solo over-quota submitters still run eventually (once their hold expires and
after their first poll delay); contended ones wait until the normal-priority
queue clears.

### Checking usage (`gpuq quota`)

```bash
gpuq quota             # your own window: finished + running + budget/headroom
gpuq quota --all       # one row per user, plus host capacity utilisation
gpuq quota --report    # weekly per-user stats for SETTING budgets (see below)
```

While budgets are unset, `gpuq quota` says so explicitly ("usage is recorded so
budgets can be set from real data") instead of just "unlimited", and the
`gpuq status` footer shows your own 7-day total after every status check.

### Setting budgets from data (`gpuq quota --report`)

After a few weeks of ledger data, `gpuq quota --report --weeks 8` prints, per
user: active weeks, weekly GPU-hour p50/p95/max/mean, queue-wait p50/p95,
timeout/lost percentages, and the share of hours via pinned `--devices` — plus
a host row as % of capacity (4 GPUs × 168 h = 672 GPU-h/week). A practical
recipe: set each user's budget near their **P95** weekly hours (normal weeks
unaffected, outlier weeks deprioritized) and sanity-check that the sum of
budgets stays within ~1.5× capacity. Then enable enforcement by editing
`quotas.default_gpu_hours_per_week` / `quotas.users` in the config — no
redeploy needed, the next submit picks it up.

## Admin audits (`gpuq audit`)

The old systemd daemon's resource-hog alerts are replaced by the cron-driven
`gpuq audit` subcommand (no daemon; it keeps small per-offender state files,
`untracked_state.json` / `rebind_state.json`, so offenders are not re-emailed
every run). It reports:

- users holding more GPUs than `audit.max_gpus_per_user`
- users holding more "requested memory" than `audit.max_total_memory_gb`
  (the sum of each job's `-m` filter × its GPUs — the request, not measured
  VRAM; nvidia-smi truth is in the `gpuq status` process list)
- users whose 7-day GPU-hour usage exceeds their quota — including users idle
  at audit time (candidates come from running jobs, the config, AND the ledger)

Each run also performs housekeeping: dead running/queued entries are reaped,
and jobs whose supervisor and child both vanished get a synthetic `lost`
ledger record so their GPU-hours are not silently dropped. If `nvidia-smi`
itself fails (wedged driver), the untracked/rebind checks skip that run rather
than mistaking the failure for "all processes gone" — offender deadlines are
preserved. Before any `--enforce` kill, the target process group is re-checked
against the recorded owner so a recycled pgid is never signalled.

Exit code is `0` when clean, `1` when any breach is reported. Slack and email
alerts use the existing `slack` / `notification_email` config blocks.

Schedule it from any admin's user-cron:

```
0 * * * * /home/admin/.local/bin/gpuq audit --quiet
```

(Hourly matches the live schedule on this host; run it more often if you want
faster detection.)

### Catching untracked GPU jobs

`gpuq audit` can also flag and email users who run GPU work **without** going
through `gpuq submit` (i.e. they launched a script directly and gpuq isn't
tracking it). It is **off by default**; enable it in the `audit` block:

```json
{
  "audit": {
    "notify_untracked": true,
    "untracked_min_memory_mb": 512,
    "untracked_grace_seconds": 120,
    "untracked_grace_hours": 0.25,
    "untracked_reminder_hours": 2,
    "untracked_allowlist": ["serviceacct"]
  }
}
```

How a process is judged "untracked": audit lists every GPU compute process,
resolves each to its owner (`ps`), and flags any that belongs to **no tracked
gpuq job** on this host. A GPU process is recognised as part of a job by, in
order:

1. **cgroup scope** — `gpuq submit` launches each job in its own
   `systemd --user --scope` (a dedicated cgroup). *Every* descendant stays in
   that cgroup no matter how it forks, so framework workers that `setsid` /
   double-fork / detach (vLLM, Ray, torchrun) are still recognised. This is the
   robust path and needs the user to **linger** (see below).
2. **process group / ancestry** — the fallback when scoping is off: the job's
   child process group (created with `setsid`) plus a walk of each GPU process's
   parent chain back to the job's child PID. Catches in-tree workers, including
   ones that re-`setsid` while their parent chain is intact.

A rogue process is still caught **even on a GPU where the same user holds a
legitimate gpuq job** — it is in neither the job's cgroup nor its process tree —
while a legitimate worker is not falsely flagged.

**Activating cgroup tracking:** scoping turns on for a user only when they
*linger* (`sudo loginctl enable-linger <user>`), which is also what keeps a job
alive after the user logs out — so it never creates a scope that would die on
logout. Until then, the job uses the process-group fallback. Note that closing
the submitting terminal sends SIGHUP, which gpuq **forwards to the job** —
i.e. a job does *not* survive a closed terminal by orphaning; run under
`tmux`/`screen` (or with linger) to keep jobs alive across disconnects. Enable
linger for your GPU users to get robust subprocess tracking fleet-wide.
`GPUQ_SCOPE=off` forces the fallback; `on` forces scoping regardless of linger.

The lifecycle for each offending process group:

1. **First seen** → email the offender (subject `[gpuq] <user>: untracked GPU
   process on <host>`) stating a deadline `first_seen + untracked_grace_hours`
   (shipped default `0.25` = **15 minutes**).
2. **During the window** → a reminder email at most every
   `untracked_reminder_hours`. With the shipped 15-minute grace and 2h reminder
   cadence no reminder can fit inside the window, so in practice the offender
   gets the first-seen mail and then the kill mail; widen the grace if you want
   reminders.
3. **Past the deadline** → with `--enforce`, the process group is killed
   (`SIGTERM`, grace, `SIGKILL`) and the user is emailed that it was killed;
   without `--enforce`, it is escalated to the admin summary instead.

Offender emails require `notification_email.enabled` and an email on the user's
account (resolved from GECOS via `getent passwd`); a user with no email is still
listed in the admin breach summary. Per-offender state lives in `untracked_state.json` (so a
long-running rogue is not re-emailed every run) and self-heals once the process
goes away.

**Enforcement needs privilege.** Signalling *another user's* process requires
root, so run the killing form from **root's** crontab:

```
*/5 * * * * /usr/local/bin/gpuq audit --enforce --quiet
```

**The grace window is only as tight as the cron.** Both `first_seen` and the
kill happen *on an audit run*, never between them: an offender is first seen at
the first run after it starts, and killed at the first run after
`first_seen + <grace>_grace_hours`. So the shipped 15-minute grace only means
15 minutes if the enforcing audit runs at least every ~5 minutes — on an hourly
cron the same config kills after roughly two hours. Avoid setting the cron
period equal to the grace (`*/15` with `0.25`): the deadline then lands within
a second of a run, and whether it fires at 15 or 30 minutes is a coin flip.

Run unprivileged, `--enforce` kills only your own processes and prints
`cannot signal process group N (needs sudo/root)` for the rest — which is
exactly what lets you test enforcement locally without sudo.

Caveats: it is **per-host** (install the cron on every GPU node), the
`untracked_allowlist` must use **full** login names (audit reads
`ps -o user:32=`, so they are not truncated to 8 chars), and **MIG**
instance processes are never flagged (their `MIG-*` UUIDs are not in the
full-GPU index map). Built-in system accounts (`root`, `gdm`, `lightdm`,
`nvidia-persistenced`, …) are always exempt.

### Catching GPU rebinds

A job allocated GPU X can still try to run on a *different* GPU — e.g. a script
that hard-codes `--gpu 2` or resets `CUDA_VISIBLE_DEVICES`, overriding what gpuq
set. That strands GPU X reserved-but-idle. With `notify_rebind` enabled in the
`audit` block, `gpuq audit` flags any tracked job whose GPU process is on a GPU
outside its allocation, emails the owner (`[gpuq] <user>: GPU rebind on <host>`),
and — with `--enforce` — kills the process group past its grace deadline, exactly
like the untracked detector:

```json
{
  "audit": {
    "notify_rebind": true,
    "rebind_min_memory_mb": 512,
    "rebind_grace_seconds": 120,
    "rebind_grace_hours": 0.25,
    "rebind_reminder_hours": 2
  }
}
```

It only inspects processes that *belong* to a gpuq job (matched by the job's
cgroup scope / process group / ancestry), so stacking several of your own jobs on
one owned GPU is never flagged — each job's processes sit on a GPU within that
job's own allocation. As with untracked enforcement, killing another user's
process needs root (run the `--enforce` form from root's crontab).

## Local development

The userspace script (`gpuq/userspace.py`) is testable on any laptop without
a real GPU, via these env vars:

| Variable                        | Default                                | Purpose                                       |
| ------------------------------- | -------------------------------------- | --------------------------------------------- |
| `GPUQ_QUEUE_DIR`                | `/var/lib/gpu_queue`                   | Shared coordination dir                       |
| `GPUQ_CONFIG_FILE`              | `/usr/local/bin/gpu_queue_config.json` | Admin config json                             |
| `GPUQ_NVSMI`                    | `nvidia-smi`                           | nvidia-smi binary path                        |
| `GPUQ_DEPRIORITIZED_POLL_SEC`   | `120`                                  | Poll cadence for deprioritized waiters        |
| `GPUQ_SCOPE`                    | `auto`                                 | systemd-scope launch: `auto`/`on`/`off`       |

(Each job also receives `GPUQ_JOB_ID` in its environment, and gpuq sets
`CUDA_VISIBLE_DEVICES` to the allocated GPUs.)

A fake `nvidia-smi` (`gpuq/tests/fake_nvidia_smi.py`) emits the same CSV as
the real one from a JSON state file pointed to by `FAKE_NVSMI_STATE`:

```json
{
  "gpus": [
    {"index": 0, "name": "Fake-GPU", "uuid": "GPU-aaaa",
     "memory_used_mb": 0, "memory_total_mb": 81920, "utilization": 0}
  ],
  "compute_apps": []
}
```

### Run the test suite

```bash
cd gpuq
python3 -m venv .venv && . .venv/bin/activate
pip install -r tests/requirements.txt
pytest -q tests/
```

181 tests; ~140s wall time. No GPU, no root, no network. (The untracked-job and
rebind tests spawn a real same-user `sleep` to exercise the kill path, and skip
if the test user is in the system allowlist, e.g. `root`.)

### Manual smoke test against the fake

```bash
export GPUQ_QUEUE_DIR=$(mktemp -d)
export GPUQ_NVSMI=$PWD/tests/fake_nvidia_smi.py
export FAKE_NVSMI_STATE=$PWD/tests/states/two_idle_gpus.json
export GPUQ_CONFIG_FILE=$GPUQ_QUEUE_DIR/config.json
./userspace.py config init       # writes default
./userspace.py status            # shows two fake GPUs
./userspace.py submit -- /bin/true
./userspace.py history           # the job's ledger record
./userspace.py quota             # usage vs (unset) budget
./userspace.py audit
```
