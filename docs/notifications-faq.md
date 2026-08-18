# GPU Queue Management System – Notifications FAQ

This FAQ covers the notifications produced by the tools running on the Mjolnir
server (host `wsserver1`, 4 × NVIDIA H200 NVL GPUs):

- **gpuq** — the cooperative, daemonless GPU job queue (`userspace.py`). It emails
  about job completion, over-quota deprioritization, untracked GPU processes, and
  GPU rebinds, and posts an aggregate breach summary from `gpuq audit`.
- **Disk-quota check** (`check_quotas.sh`) — emails users over their home-directory
  quota.
- **Scratch cleanup** (`scratch-cleanup.sh`) — warns about, and reports deletion
  of, idle files under `/scratch`.

## ❓ What kinds of notifications does the system send?

The system sends these kinds of notifications:

- **Job completion notifications (gpuq, email)**

  Sent by the foreground `gpuq submit` process when your job ends. The reason is
  one of `completed` (exit code 0), `timed_out` (it exceeded its time limit),
  `killed` (it was ended by a signal, e.g. `gpuq kill`), or `failed` (it exited
  with a non-zero code).

  Example subject: `[gpuq] job 1234 completed`.

- **Over-quota deprioritization notifications (gpuq, email)**

  Sent when you submit a job that would push you over your rolling 7-day GPU-hour
  quota (168 GPU-hours/week on this host). The job is still accepted, but it is
  **held for 15 minutes** from submission (the email states the exact deadline) and
  then queued at low priority, only starting once on-quota submitters have had a
  chance to grab the next free slot.

  Example subject: `[gpuq] alice: GPU-hour quota exceeded - job deprioritized`.

- **Untracked-GPU notifications (gpuq, email — opt-in)**

  Sent by `gpuq audit` when it finds a GPU compute process that was **not** launched
  via `gpuq submit`. The offender is emailed through a warn → remind → overdue →
  killed lifecycle (see below).

  Example subject: `[gpuq] bob: untracked GPU process on wsserver1`.

- **GPU-rebind notifications (gpuq, email — opt-in)**

  Sent by `gpuq audit` when a tracked gpuq job is running on a GPU **other than the
  one it was allocated** (so the allocated card is reserved but idle). Same
  warn → remind → overdue → killed lifecycle.

  Example subject: `[gpuq] bob: GPU rebind on wsserver1 (job 5678)`.

- **Resource-breach summary (gpuq, admin email + Slack)**

  An aggregate summary produced by `gpuq audit` listing resource hogs, over-quota
  users, and any untracked/rebind breaches. This is the **only** notification that
  goes to Slack, and it is sent to the admin, not to job owners.

  Example subject: `[gpuq] resource breaches on wsserver1`.

- **Disk Quota notifications (`check_quotas.sh`, email)**

  Triggered when a user's home directory in `/home/` exceeds its soft quota
  (reported by `repquota`).

  Example:

```text
Hello User,

This is an automated notification from the server.
Your home directory is over its allocated disk quota.

Your current usage: 94GB
Your soft limit:   90GB
Your hard limit:   100GB

Please remove unnecessary files to get back under your quota. If you fail to do so, you may be unable to save new files.

Thank you,
System Administrator
```

  (The 94/90/100 GB figures above are illustrative. The actual soft and hard
  limits are whatever the filesystem quota is configured with — `check_quotas.sh`
  does not hardcode any GB value; it reads them from `repquota -as`.)

- **Scratch Files expiration warning / deletion notification (`scratch-cleanup.sh`, email)**

  Triggered when a file under a scratch folder (except `/scratch/datasets`) has not
  been accessed or modified for at least 23 days (warning period). After 30 days the
  file is deleted and a deletion notification is sent as well. The folders checked
  are `/scratch/shared`, `/scratch/temp`, and `/scratch/users`.

  Example:

~~~
Hello user,

This is an automated notification from the server.
The file /scratch/users/user/test_file.txt has now been automatically deleted due to it not being accessed or modified in the last 30 days.

If the file was not to be deleted: sincere apologies. Do make sure next time to use either the /scratch/datasets folder for permanent files or your own home directory for smaller permanent files.
Any file in any other folder in the /scratch/ directory will be deleted after 30 days of it being unaccessed or unmodified.

Thank you,
System Administrator
~~~

## ❓ Where are notifications sent?

Notifications can go to:

- **Email**

  - To the offending/job-owning user's own address for completion, over-quota,
    untracked, and rebind notices, and for disk-quota and scratch notices.

  - To the admin email for the system-wide `gpuq audit` resource-breach summary.

  The recipient's address is **not** stored in a config map. gpuq reads it from the
  user's account GECOS field via `getent passwd <user>` (see
  [Where is the configuration?](#-where-is-the-configuration-for-notifications)).
  `gpuq submit --notify EMAIL` overrides the address for that one job's completion
  notice. The disk-quota and scratch scripts likewise read the address from the
  GECOS field.

- **Slack**

  - Only the aggregate `gpuq audit` breach summary is posted, to the configured
    Slack webhook (default channel `#gpu-alerts`), and only if Slack is enabled and
    the `requests` library is installed. Per-job completion, over-quota, untracked,
    and rebind notices are **email-only** — they never go to Slack.

- **Console / cron output**

  - gpuq is daemonless: there is no background process printing notifications.
    `gpuq submit` runs your job in the foreground in your terminal and prints a
    one-line summary when the job ends (runtime, GPUs, GPU-hours recorded, exit
    code), plus a "time limit reached" notice if the limit fires; `gpuq audit`
    prints its breach summary to stdout each time it runs (typically from a cron
    job). If you want a persistent record of audit output, redirect it in your
    crontab.

## ❓ Where is the configuration for notifications?

The gpuq config file is located at (default; overridable via the `GPUQ_CONFIG_FILE`
environment variable):

```bash
/usr/local/bin/gpu_queue_config.json
```

It contains settings for:

- `notification_email`: SMTP **sender** credentials (`smtp_server`, `smtp_port`,
  `username`, `password`) plus `admin_email` (where the audit summary is sent).

- `slack`: `enabled`, `webhook_url`, and target `channel`.

- `quotas`: `default_gpu_hours_per_week` (0 = unlimited; 168 on this host), the
  over-quota hold `delay_hours` (`0.25` here — **15 minutes**; it is a float,
  so sub-hour holds are legal), and a per-user `users` map.

- `max_gpus_per_user_hard`: the hard per-user concurrent-card cap enforced at
  submit/claim time (3 on this host; 0 = off).

- `audit`: resource-hog thresholds (`max_gpus_per_user` — the warn-only card
  threshold, 2 here — and `max_total_memory_gb`) and the opt-in detectors
  `notify_untracked` / `notify_rebind` with their grace and reminder thresholds.

There is **no `user_emails` map**. Recipient addresses come from each account's
GECOS field (`getent passwd <user>`); accounts are provisioned with the email there,
so the account is the single source of truth.

> The disk-quota (`check_quotas.sh`) and scratch-cleanup (`scratch-cleanup.sh`)
> scripts are separate shell scripts run from cron. They send mail via `msmtp` and
> read the recipient address from the GECOS field, not from the gpuq config file.

## ❓ When are job completion notifications sent?

`gpuq submit` emails the job owner when the job ends, with one of four reasons:

- **completed**: the command exited with code 0.

- **timed_out**: the job ran longer than its time limit (`-t/--time`, defaulting to
  the config `max_job_time_hours` — 48 hours on this host, 24 in the shipped
  template) and was killed. gpuq also prints a "time limit reached" notice in the
  submitting terminal.

- **killed**: the command was ended by a signal — e.g. you ran `gpuq kill`, or the
  submitting terminal closed and gpuq forwarded the hangup to the job.

- **failed**: the command exited with a non-zero code.

(There is no "killed by another user" notification — you can only kill your own
jobs, so a `killed` notice always stems from your own action.)

## ❓ When are over-quota notifications sent?

When you run `gpuq submit` and the request would push you past your rolling 7-day
GPU-hour budget (`quotas.default_gpu_hours_per_week`, or a per-user override in
`quotas.users`), gpuq:

- prints a notice in your terminal,
- **holds the job for 15 minutes** from submission (`quotas.delay_hours`) — it may
  not start at all before then; the deadline is printed at submit and stated in
  the email,
- queues the job at **low priority** (after the hold it still waits for on-quota
  submitters to grab slots first), and
- emails you once with subject `[gpuq] <user>: GPU-hour quota exceeded - job
  deprioritized`.

If your quota is unlimited (the default `0`), this never fires.

## ❓ When are resource-hog notifications sent?

`gpuq audit` flags a user as a resource hog when they hold:

- **more GPUs than `audit.max_gpus_per_user`** (default 2), or

- **more total requested memory than `audit.max_total_memory_gb`** (default 50 GB).

The total-memory figure is the per-GPU memory requested by the job multiplied by the
number of GPUs it holds — not measured VRAM. There is no percentage rule, no fixed
80 GB rule, and no built-in throttle: `gpuq audit` is stateless per run, so the
cadence is simply whatever your cron schedule is (hourly on this host:
`0 * * * * gpuq audit`).

When breaches are found, gpuq emails the admin (`admin_email`) with subject
`[gpuq] resource breaches on <host>` and posts the same summary to Slack.

## ❓ When are untracked-GPU notifications sent?

This detector is **opt-in** (`audit.notify_untracked: true`). On each `gpuq audit`
run it inspects every GPU compute process and flags any that was not launched via
`gpuq submit` (excluding system accounts, an admin allowlist, and processes smaller
than `untracked_min_memory_mb`). For each offending process group it drives an email
state machine:

- **warn** — first detection; the user is told the process will be killed after its
  grace deadline (`untracked_grace_hours`, **15 min** from first detection on this host).

- **remind** — sent at most every `untracked_reminder_hours` (**2h** here) while the
  process is still untracked and before the deadline. With a 15-minute grace no
  reminder fits inside the window, so in practice you get the warn mail and then
  the kill mail.

- **overdue** — sent once past the grace deadline (subject ends `PAST DEADLINE`).

- **killed** — sent only if `gpuq audit --enforce` actually terminates the process
  group past its deadline. Killing another user's process requires running audit as
  root.

A short grace window (`untracked_grace_seconds`, default 120s) after a submit
suppresses spurious flags during the submit-to-launch race.

## ❓ When are GPU-rebind notifications sent?

This detector is also **opt-in** (`audit.notify_rebind: true`). gpuq allocates each
job specific GPU(s) and sets `CUDA_VISIBLE_DEVICES` to match. A "rebind" is a tracked
gpuq job that is actually running on a GPU **outside** its allocation (so the
allocated card is reserved but idle — typically because the job overrode the device,
e.g. set its own `--gpu N` or reset `CUDA_VISIBLE_DEVICES`).

`gpuq audit` drives the same lifecycle as the untracked detector
(warn → remind → overdue → killed), using `rebind_grace_hours` (**15 min** here),
`rebind_reminder_hours` (**2h** here), and `rebind_grace_seconds` (default 120s).
As with the untracked detector, a 2h reminder cadence cannot fire inside a
15-minute window, so the sequence you actually see is warn → killed.
Processes are killed only under `gpuq audit --enforce`. Both the first detection
and the kill happen on a scheduled audit run, so the wall-clock time to a kill
is the grace plus up to one audit period.

## ❓ When are disk quota notifications sent?

When a user's home directory (e.g. `/home/user`) goes over its **soft** quota as
reported by `repquota`. `check_quotas.sh` reads the soft and hard limits from
`repquota -as` and emails the over-quota user. The 90 GB / 100 GB figures in the
example email are illustrative; the real limits are whatever the filesystem quota is
configured with.

## ❓ When are scratch-file expiration notifications sent?

For any file under a checked scratch directory (`/scratch/shared`, `/scratch/temp`,
`/scratch/users`) — but **not** `/scratch/datasets`:

- **Deletion warning**: when the file has not been accessed or modified in the last
  23 days (`DAYS_TO_NOTIFY=23`).

- **Deletion notification**: when the file has not been accessed or modified in the
  last 30 days (`DAYS_TO_KEEP=30`); the file is deleted and the user is emailed.

## ❓ Where can I find logs of my jobs?

- **Your job's output:**

  `gpuq submit` runs your job in the **foreground**, and its stdout/stderr go
  straight to your terminal — gpuq does **not** capture them to a file. For a
  persistent log, redirect it yourself or run inside a multiplexer so the session
  survives a disconnect:

  ``` bash
  gpuq submit -- python train.py > ~/train.log 2>&1
  # or run inside tmux/screen so the job and its output survive logout
  ```

  > Note: an old `/var/lib/gpu_queue/logs/` directory with `job_<id>_stdout.log` /
  > `job_<id>_stderr.log` files may still exist on the server. Those were written by
  > the **retired** `gpu_queue.py` daemon and are **not** produced for current jobs —
  > do not rely on them.

  For each past job's *metadata* (runtime, GPU-hours, exit code, end reason — not
  its output), run `gpuq history`.

- **`gpuq audit` output:**

  Audit prints its breach summary to stdout on each run. If audit runs from cron,
  capture it by redirecting in the crontab; there is no background daemon logging on
  its own.

- **Email inboxes:**

  Completion, over-quota, untracked, and rebind notices arrive in the relevant
  user's inbox; the audit summary arrives in the admin inbox. Disk-quota and scratch
  notices arrive in the user's inbox.

- **Slack channel:**

  If Slack is enabled, the aggregate audit summary is in the configured channel
  (e.g. `#gpu-alerts`).

- **Scratch cleanup log:**

  `scratch-cleanup.sh` writes a detailed log to `/var/log/scratch-cleanup.log`.

## ❓ Where is gpuq's shared state kept?

gpuq coordinates through files under `/var/lib/gpu_queue/` (group-writable by the
`gpuqueue` group):

- `jobs.json` — queued jobs
- `running.json` — running jobs
- `usage.jsonl` — the per-job usage ledger (runtime, GPU-hours, command, queue
  wait, exit code, end reason — including synthetic `lost` records when a
  supervisor dies, and `cancelled`/`rejected` records for abandoned or refused
  submits); used for quotas and browsable with `gpuq history` / `gpuq quota`
- `untracked_state.json` / `rebind_state.json` — the detectors' notification state
- `.lock` — the coordination lock

These are coordination/state files, not per-job logs. (As noted above, a stale
`logs/` directory from the retired daemon may still be present but is unused.)

## ❓ How do I disable or enable notifications?

Edit the gpuq config file:

```bash
"notification_email": {
  "enabled": true,
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "username": "your.email@gmail.com",
  "password": "app_password",
  "admin_email": "admin@yourlab.com"
},
"slack": {
  "enabled": true,
  "webhook_url": "https://hooks.slack.com/services/...",
  "channel": "#gpu-alerts"
}
```

Set `"enabled": false` to turn off email or Slack notifications. To toggle the
untracked / rebind detectors, set `notify_untracked` / `notify_rebind` to
`true`/`false` in the `audit` block. To verify what is active, run:

```bash
gpuq config
```

(`gpuq config` is read-only — it shows the active settings. Writing a starter
config file is the admin's `gpuq config init`.)

(The disk-quota and scratch notifications are governed by their respective cron jobs
and shell scripts, not by this config file.)

## ❓ Can I check notification history?

gpuq does not log notifications persistently. You can reconstruct history from:

- Your own redirected job output / audit output (if you set up redirection).

- The gpuq usage ledger `/var/lib/gpu_queue/usage.jsonl` — view it with
  `gpuq history` (per-job runtime, GPU-hours, exit code, end reason) and
  `gpuq quota` (rolling 7-day usage vs budget), useful for understanding
  quota/audit behavior.

- The scratch-cleanup log at `/var/log/scratch-cleanup.log`.

- Email / Slack archives, if configured.
