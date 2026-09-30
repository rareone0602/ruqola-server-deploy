# Notifications

The server emails you about your GPU jobs, your `/scratch` files, and your home
directory quota. Every email goes to the address stored on your account. To
change that address, ask the admin.

## gpuq emails

| Subject | Why | What to do |
|---|---|---|
| `[gpuq] job <id> completed` | A job ended. Sent for every job. The last word is the result: `completed`, `failed`, `timed_out` or `killed`. | Nothing, or run `gpuq history`. |
| `[gpuq] <you>: GPU-hour quota exceeded - job deprioritized` | A submit would take you past your budget of 168 GPU-hours per rolling 7-day window. | Nothing. The job waits 15 minutes, then runs at low priority. |
| `[gpuq] <you>: untracked GPU process on wsserver1` | A GPU process you did not start with `gpuq submit`. | Stop it now and restart it with `gpuq submit`. |
| `[gpuq] <you>: GPU rebind on wsserver1 (job <id>)` | Your gpuq job runs on a card it was not given, usually because it set `CUDA_VISIBLE_DEVICES` itself. | Stop it, remove the override, resubmit. |
| `... was KILLED` | The process was still there at its deadline, so it was killed. | Restart it with `gpuq submit`. |
| `... PAST DEADLINE` | The deadline passed but the kill has not happened yet. | Stop it yourself. |

**Job-end email.** It lists the host, command, GPUs, start time, result and exit
code. `gpuq submit --notify EMAIL` sends it to another address for that job.
There is no option to turn it off. A `lost` job (its `gpuq submit` process died)
gets no email.

**Over-quota email.** It shows your usage, what the job asked for, the 168
GPU-hour budget, and the time until which the job is held. See
[GPU-Hour Quotas](gpu-queue-guide.md#gpu-hour-quotas).

**Untracked and rebind emails.** The audit runs every 15 minutes:

1. The first audit that sees the process sends the warning, with a deadline
   15 minutes later.
2. The first audit at or after the deadline kills it and sends the `KILLED`
   email.

Reminders come every 2 hours, so you normally get only the warning and the
kill. Act on the warning at once.

## Scratch cleanup emails

| Subject | Why |
|---|---|
| `Imminent removal: <n> file(s) under /scratch/users` | Files that have gone 166 days with no read and no write. They are deleted when they reach 180 days, about 14 days later. The email lists each file and the days left. |
| `Scratch cleanup: <n> file(s) removed` | Files deleted after 180 days with no read and no write, with their sizes. |

- You get at most one email of each kind per directory (`/scratch/users`,
  `/scratch/shared`, `/scratch/temp`) per night. The warning repeats every night
  until you use the files or they are deleted.
- To keep a file, read it, change it, or `touch` it. See
  [Scratch Storage](scratch-folder.md).
- A failed warning email does not stop the deletion. Run `scratch-status`
  instead of waiting for mail.

## Disk quota email

Subject `Disk Quota Warning`: your home directory is over its 90 GiB soft
quota. The email shows your usage and your soft and hard limits. A check runs
daily at 02:00, but its delivery is not confirmed, so watch `quota -s` yourself.

## Sent to the admin, not to you

Each audit that finds a problem sends a summary to the admin email and Slack
(`[gpuq] resource breaches on wsserver1`). It lists every breach it finds, such
as a user holding more than 2 cards, a user over quota, or an untracked or
rebound process.

## Logs

- **Your job's output:** only in your terminal, unless you saved it. gpuq writes
  no log files.
- **How a job ended:** `gpuq history`.
- **Deleted scratch files:** `/var/log/scratch-cleanup/deleted-YYYY-MM.tsv`,
  readable by the admin only. Ask the admin to look up a missing file.

Admins: notification settings live in the gpuq config file; see the
[gpuq Reference](../gpuq/README.md).
