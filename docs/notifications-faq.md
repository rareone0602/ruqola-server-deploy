# Notifications

The server emails you about your GPU jobs, your `/scratch` files, and your home
directory quota. Every email goes to the address stored on your account. To
change that address, ask the admin.

## gpuq emails

| Subject | Why | What to do |
|---|---|---|
| `[gpuq] job <id> started on GPU(s) <n>` | Your job started after waiting 10 minutes or more. Jobs that start sooner get no email. | Nothing. |
| `[gpuq] job <id>: 1 hour left` | The job is stopped in 1 hour, at its 48-hour limit. A `--devices` job stops when your job(s) on that GPU reach theirs. | Save a checkpoint now. Resubmit later to resume. |
| `[gpuq] job <id> completed` | A job submitted with `--notify` ended. The last word is the result: `completed`, `failed`, `timed_out`, `killed` or `lost`. | Nothing, or run `gpuq history`. |
| `[gpuq] job <id> cancelled` | A `--detach` job with `--devices N` was waiting for room on your GPU N, but your job(s) there ended first. | Submit it again without `--devices`. |

**Job-end email.** Sent only for a job submitted with `gpuq submit --notify`.
It lists the host, command, GPUs, start and end time, result and exit code.

**GPU work outside gpuq.** Start all GPU work through gpuq. gpuq records GPU
use outside it, and will stop it automatically in a later step. It sends no
email about it.

## Scratch cleanup emails

| Subject | Why |
|---|---|
| `Imminent removal: <n> file(s) under <dir>` | Files that have gone 166 days with no read and no write. They are deleted when they reach 180 days, about 14 days later. The email lists up to 100 files, soonest first, with the days left, then gives a `find` command that lists the rest. |
| `Scratch cleanup: <n> file(s) removed` | Files deleted after 180 days with no read and no write: up to 100 of them with their sizes, and the total count. |

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

## Logs

- **Your job's output:** for a job run with `--detach`, in
  `~/gpuq-logs/<id>.log`. Otherwise only in your terminal, unless you saved it.
- **How a job ended:** `gpuq history`.
- **Deleted scratch files:** `/var/log/scratch-cleanup/deleted-YYYY-MM.tsv`,
  readable by the admin only. Ask the admin to look up a missing file.

Admins: the mail settings are described in the
[gpuq Reference](../gpuq/README.md).
