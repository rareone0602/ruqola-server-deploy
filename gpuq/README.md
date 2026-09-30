# gpuq Reference

How gpuq works, how to deploy and configure it, and what its audit does. For
everyday use (submit, status, kill, flags, waiting), read the
[gpuq User Guide](https://rareone0602.github.io/ruqola-server-deploy/#gpuq/gpu-queue-guide).
Values are the live settings on Mjölnir (`wsserver1`, GPUs `0`–`3`; see
[H200 Specs](https://rareone0602.github.io/ruqola-server-deploy/#hardware/h200-specs)).

## How gpuq works

- **No daemon.** `gpuq submit` claims GPUs, sets `CUDA_VISIBLE_DEVICES` and
  `GPUQ_JOB_ID`, runs the command, and stays in the foreground as the job's
  supervisor: SIGTERM at the `-t` limit, SIGKILL 10 s later, a ledger record at
  the end.
- **Shared state** is in `/var/lib/gpu_queue/` (`root:gpuqueue`, `2775`), so
  every gpuq user must be in `gpuqueue`. Changes take an exclusive `flock` on its
  `.lock`; JSON files are replaced by atomic rename. `nvidia-smi` calls time out
  after 15 s, so a wedged driver cannot hold the lock.
- **Polling.** A free slot goes to whichever waiter polls first: every 30 s, or
  120 s when over quota. The queue order in `gpuq status` is informational.
- **Tracking.** A job runs in its own `systemd --user` scope (`gpuq-<id>.scope`)
  if its user lingers (`sudo loginctl enable-linger <user>`; enable it for every
  GPU user), else in its own process group. The audit uses both, plus the
  parent chain. Linger does not save a job from a closed terminal: gpuq forwards
  SIGHUP, SIGINT and SIGTERM to the job.

| File in `/var/lib/gpu_queue/` | Contents |
|---|---|
| `running.json` | running jobs: user, GPUs, `-m`, `-t`, supervisor `pid`, child process group, scope, command |
| `jobs.json` | queued submits, with `priority` and `hold_until` |
| `usage.jsonl` | the [job ledger](#job-ledger) |
| `untracked_state.json`, `rebind_state.json` | audit offenders: first seen, last email |
| `gpuq.py` | shared copy of the script |
| `kill.json`, `last_resource_notification.json`, `logs/` | leftovers of the retired daemon; unused |

**Reaping.** Every `submit`, `status`, `kill` and `audit` first drops dead
entries. A running entry lives while its supervisor or its child process group
does (an orphan still holds its GPU); start times guard against recycled PIDs.
A dead job gets a `lost` record, a dead waiter a `cancelled` one.

**Stopping another user's job.** `gpuq kill` works only on your own jobs. As
root, `kill <pid>` the job's supervisor (`pid` in `running.json`); it forwards
the SIGTERM and records the end.

## Install and deploy

Both installers read `../gpuq/userspace.py` relative to their own folder, so
run them from a gpuq checkout whose folder is named `gpuq`.

```bash
sudo ./install_system.sh             # userspace.py -> /usr/local/bin/gpuq
./install_user.sh --publish-shared   # userspace.py -> /var/lib/gpu_queue/gpuq.py, then link ~/.local/bin/gpuq to it
```

`install_system.sh [--source PATH]` re-runs itself under `sudo` if needed. It
refuses a source that does not compile, backs up the old binary to
`/usr/local/bin/gpuq.bak-<YYYYmmdd-HHMMSS>`, installs atomically
(`0755 root:root`), runs `gpuq --help`, creates `/var/lib/gpu_queue` if missing
(the `gpuqueue` group must exist), and writes a starter config only if none
exists. `./install.sh --check` in `ruqola-server-deploy/scripts` then reports the
backup as drift; `sudo ./install.sh` there retires it.

`install_user.sh` needs no root. It installs `~/.local/bin/gpuq` and warns if
`gpuq` on `PATH` resolves elsewhere.

| Flag | Effect |
|---|---|
| none, or `--symlink-shared` | symlink to `/var/lib/gpu_queue/gpuq.py` |
| `--copy-shared` | private copy of `/var/lib/gpu_queue/gpuq.py` |
| `--copy-from-repo` | private copy of the repo's `userspace.py` |
| `--publish-shared` | first copy `userspace.py` to `/var/lib/gpu_queue/gpuq.py` (mode `0775`); combines with the others |

**Upgrade every copy at once**, since all share the state files. Symlinks
follow the shared copy; `--copy-*` installs do not. Check with
`cmp /usr/local/bin/gpuq /var/lib/gpu_queue/gpuq.py`.

**The deployed binary is older than the repo.** Only built-in defaults differ,
and the live config overrides each of them. But its `gpuq config init` template
has an 8 h over-quota hold and a 4 h kill grace. **Never run
`gpuq config init --force` on the live host:** it replaces the config with that
template, with no credentials, no quota and both detectors off. Redeploy from
the repo to update the defaults.

**Retired daemon.** Never deploy the repo's `gpu_queue.py`.
`/etc/systemd/system/gpu-queue.service` still calls the removed `gpuq daemon`,
as root with `Restart=always`. It is disabled; keep it so.

## Config

`/usr/local/bin/gpu_queue_config.json`, `0640 root:gpuqueue`. It holds the SMTP
password and the Slack webhook: never commit it or make it world-readable (a
starter config is `0644`). gpuq reads it on every command, so edits apply at
once; `gpuq config` shows the values in force. If a user cannot read it or it
is invalid JSON, gpuq warns and falls back to no quota, no card cap, `-t` 24 h
and `-m` 70 GB.

| Key | Live | If missing | Effect |
|---|---|---|---|
| `max_job_time_hours` | 48 | 24 | default `-t`, hours |
| `max_job_time_hours_cap` | 48 | 48 | largest `-t` (a larger default is lowered to it); `0` = no cap |
| `default_min_free_gb` | not set | | default `-m`: GB of free VRAM a card needs |
| `max_memory_per_gpu_gb` | 120 | 70 | default `-m` when `default_min_free_gb` is not set (live: 120 GB) |
| `max_gpus_per_user_hard` | 3 | 0 (off) | most distinct cards per user; at the cap, jobs can only stack |
| `quotas.default_gpu_hours_per_week` | 168 | 0 | budget per rolling 7 days; `0` = unlimited |
| `quotas.users` | `{}` | | per-user budgets, e.g. `{"alice": 250}` |
| `quotas.delay_hours` | 0.25 | 0 | over-quota hold, hours |
| `audit.max_gpus_per_user` | 2 | 2 | breach above this many cards |
| `audit.max_total_memory_gb` | 320 | 50 | breach above this sum of `-m` × cards over a user's jobs |
| `audit.notify_untracked`, `audit.notify_rebind` | true | false | detectors on |
| `audit.untracked_min_memory_mb`, `audit.rebind_min_memory_mb` | 512 | 512 | ignore smaller processes |
| `audit.untracked_grace_seconds`, `audit.rebind_grace_seconds` | 120 | 120 | ignore processes this long after a job starts |
| `audit.untracked_grace_hours`, `audit.rebind_grace_hours` | 0.25 | 24 | warning-to-kill deadline |
| `audit.untracked_reminder_hours`, `audit.rebind_reminder_hours` | 2 | 6 | least time between emails to one offender |
| `audit.untracked_allowlist` | `[]` | `[]` | full login names never flagged |
| `notification_email.enabled` | true | false | send email |
| `notification_email.smtp_server`, `.smtp_port`, `.username`, `.password` | set | port 587 | SMTP login; `username` is also the sender |
| `notification_email.admin_email` | set | | gets audit summaries |
| `slack.enabled`, `slack.webhook_url` | true, set | false | post audit summaries to this Incoming Webhook |
| `slack.channel` | | | a note; never read |

"If missing" gives the deployed binary's fallbacks. A non-number (`true`
included) in `max_gpus_per_user_hard` or `quotas.delay_hours` turns that
feature off, with a warning.

## Job ledger

`/var/lib/gpu_queue/usage.jsonl` has one JSON record per line, appended under
the lock. `gpuq history` and `gpuq quota` read it and skip bad lines. To rotate,
move it to `usage-YYYY-MM.jsonl` in the same folder; readers take every
`usage-*.jsonl`, sorted, before `usage.jsonl`.

| `event` | Written when | Notes |
|---|---|---|
| `end` | a job ended | the only type charged to quota; fields below |
| `cancelled` | a queued submit never ran | adds `at`, `wait_sec`, `reason`: `user` (Ctrl-C or `gpuq kill`), `signal` (SIGTERM or SIGHUP), `lost` (the waiter died) or `stale` (`gpuq kill` found it dead) |
| `rejected` | a submit was refused | no `id`; adds `at`, `reason`: `no_free_gpu`, `devices_unavailable` or `user_card_cap` |

A line with no `event` (from before ledger v2, no `v`) counts as `end`. Many
rejections with little usage mean a starved user, not an idle one.

```json
{"v":2,"event":"end","id":441840678,"user":"alice","host":"wsserver1","command":"python train.py","name":"run1","gpus_requested":1,"gpus":[3],"devices":[3],"memory_gb":20,"max_time_hours":6.0,"priority":"low","over_quota_at_submit":true,"submitted_at":"2026-09-30T17:25:06","started_at":"2026-09-30T17:41:10","ended_at":"2026-09-30T17:45:50","queue_wait_sec":964,"elapsed_hours":0.0779,"gpu_hours":0.0779,"exit_code":0,"end_reason":"completed"}
```

| `end` field | Meaning |
|---|---|
| `command` | first 300 characters |
| `gpus_requested`, `gpus`, `devices` | cards asked for, granted, and pinned with `--devices` |
| `memory_gb`, `max_time_hours` | the `-m` and `-t` in force |
| `*_at` | local time, to the second |
| `gpu_hours` | `elapsed_hours` × number of `gpus` |
| `exit_code` | the job's return code; `-N` = killed by signal N; null for `lost` |
| `end_reason` | `completed` (exit 0), `failed`, `killed` (signal), `timed_out` (hit `-t`), `lost` |
| `synthetic` | `true` on `lost` records: supervisor and job both died unaccounted (a SIGKILLed supervisor, a reboot); charged from start to reap, capped at `-t`; no email |

## Quotas

- **Budget:** `quotas.users[<user>]`, else `quotas.default_gpu_hours_per_week`
  (live: 168 for everyone).
- **Usage:** `gpu_hours` of `end` records in the last 168 h (a job crossing the
  cutoff counts only its share inside), plus running jobs at elapsed × cards.
- **At submit:** over when usage + cards × `-t` > budget, `--devices` or not.
- **Over budget:** never refused. gpuq prints `[gpuq] over quota: used ...`,
  emails the user once, and queues the job at `priority: low` with `hold_until`
  = now + `quotas.delay_hours` (15 min). After the hold it polls every 120 s and
  yields while any normal-priority waiter could claim a slot.

```bash
gpuq quota --all                # every user (or --user NAME); host use against 4 × 168 = 672 GPU-h
gpuq quota --report --weeks 8   # per user per ISO week: P50/P95/max/mean GPU-h, waits, timeout/lost/pinned %
```

`--report` counts a multi-day job in the week it ended.

## Audit

`/dev/nvidia*` is world-accessible, so gpuq cannot block GPU use outside it;
the audit finds and kills it. Root's crontab runs it every 15 minutes:

```
*/15 * * * * /usr/local/bin/gpuq audit --enforce --quiet
```

Each run drops dead entries, then flags:

- users on more than `audit.max_gpus_per_user` cards (live: 3 or more);
- users whose `-m` × cards over all jobs exceeds `audit.max_total_memory_gb`
  (live: 320 GB, which three jobs at the default `-m` 120 exceed);
- users over budget, including users with no running job;
- untracked processes and rebinds.

On any breach it emails the list to `admin_email` as
`[gpuq] resource breaches on <host>`, posts it to Slack, and exits `1`. A clean
run exits `0`, silently with `--quiet`. Nothing is de-duplicated: a standing
breach, such as a user on 3 cards, is re-sent every run.

`--enforce` adds the kills. Without root it kills only your own processes and
prints `[gpuq audit] cannot signal process group N (needs sudo/root); leaving it
for the admin.` for others. If `nvidia-smi` fails, the detectors skip the run.

**Untracked:** a GPU process (owner from `ps -o user:32=`) in no running job's
scope, process group or process tree. Skipped: system accounts (`root`,
`nobody`, `gdm`, `lightdm`, `sddm`, `systemd+`, `_apt`, `nvidia-persistenced`),
`untracked_allowlist`, processes under `untracked_min_memory_mb`, users whose
newest job started under `untracked_grace_seconds` ago, and MIG instances. One
offender is one (user, process group).

**Rebind:** a job's process on a GPU outside the job's `gpus`, usually because
the script set `CUDA_VISIBLE_DEVICES` or `--gpu N` itself. Stacking on your own
card is not a rebind. Skipped: processes under `rebind_min_memory_mb` and jobs
younger than `rebind_grace_seconds`; no allowlist. One offender is one (job,
process group).

| Audit run | Action, email to the offender | Admin summary |
|---|---|---|
| first sighting | warning; deadline = now + `*_grace_hours` | listed |
| before the deadline | reminder every `*_reminder_hours` | listed |
| deadline passed, `--enforce` | SIGTERM to the process group, SIGKILL 10 s later if needed; `... was KILLED` | `PAST DEADLINE - enforcing` |
| deadline passed, no kill | `... PAST DEADLINE` every `*_reminder_hours` | `PAST DEADLINE - escalated to admin` |

A kill first re-checks the sampled PID's owner and process group. A live group
missed for one run keeps its deadline; a recycled group number gets a fresh one.

**On this host** (15-minute cron, 15-minute grace) the kill comes at the next
run or the one after, 15 to 30 minutes after the warning; no reminder is sent.

## Notifications

The user side is in the
[Notifications FAQ](https://rareone0602.github.io/ruqola-server-deploy/#gpuq/notifications-faq).

| Email subject | Sent by | To |
|---|---|---|
| `[gpuq] job <id> <end_reason>` (every end except `lost`) | the user's `gpuq submit` | `--notify`, else account |
| `[gpuq] <user>: GPU-hour quota exceeded - job deprioritized` | the user's `gpuq submit` | `--notify`, else account |
| `[gpuq] <user>: untracked GPU process on <host>`, `[gpuq] <user>: GPU rebind on <host> (job <id>)`, with `reminder —`, `PAST DEADLINE` and `was KILLED` variants | root's audit | account |
| `[gpuq] resource breaches on <host>` | root's audit | `admin_email` |

- **Sending:** SMTP with STARTTLS. Users' own gpuq processes send mail, so they
  must read the config. The sender, `mjolnirruqola@gmail.com`, uses a Gmail app
  password (credentials in the lab OneDrive folder `mjolnir`, unverified). The
  scratch and disk-quota scripts share the account and its daily limit.
- **Account address:** the first email address in the user's GECOS field
  (`getent passwd <user>`). `create_users` sets it with `chfn -o`; change it with
  `sudo chfn -o <address> <user>`. No address, no email; the admin summary
  still lists the breach.
- **Slack:** only the audit summary posts. It needs Python `requests`
  (`python3-requests`, installed), or Slack is skipped silently.
- A failed send prints `warning: email notification failed: ...` or
  `warning: slack notification failed: ...`; gpuq carries on.

## Local development

The repo's `userspace.py` is what gets deployed. These variables let it run
without a GPU or root:

| Variable | Default | Purpose |
|---|---|---|
| `GPUQ_QUEUE_DIR` | `/var/lib/gpu_queue` | state folder |
| `GPUQ_CONFIG_FILE` | `/usr/local/bin/gpu_queue_config.json` | config |
| `GPUQ_NVSMI` | `nvidia-smi` | the `nvidia-smi` to run |
| `GPUQ_SCOPE` | `auto` | `off`: process group only; `on`: scope even without linger |
| `GPUQ_DEPRIORITIZED_POLL_SEC` | `120` | over-quota poll interval |

`tests/fake_nvidia_smi.py` answers like `nvidia-smi` from the JSON file in
`FAKE_NVSMI_STATE` (example: `tests/states/two_idle_gpus.json`).

Run the tests from the gpuq repo root. The host has no conda, and pip works only
in a venv:

```bash
python3 -m venv .venv
.venv/bin/pip install -r tests/requirements.txt
.venv/bin/python -m pytest tests/ -q
```

`tests/conftest.py` sets `GPUQ_QUEUE_DIR`, `GPUQ_CONFIG_FILE`, `GPUQ_NVSMI` and
`GPUQ_SCOPE=off`, so the suite never touches live state. Kill tests skip when
run as `root`. To run `./userspace.py` by hand, export the same four variables
and `FAKE_NVSMI_STATE` first, or it acts on live state.
