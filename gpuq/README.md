# gpuq Reference

How gpuq works, how to install and update it, and what it records. For everyday
use (submit, status, sharing your own GPU, waiting), read the
[gpuq User Guide](https://rareone0602.github.io/ruqola-server-deploy/#gpuq/gpu-queue-guide).
Values are the live settings on Mjölnir (`wsserver1`, GPUs `0`–`3`; see
[H200 Specs](https://rareone0602.github.io/ruqola-server-deploy/#hardware/h200-specs)).
The rules, and why each was chosen, are in `docs/v3-design.md` in the gpuq
repo. This version (v3) has been live since 2026-10-01 20:33.

## How gpuq works

- **One root service runs the queue.** `gpuqd.service` is the only writer of the
  queue and the ledger. Every 30 s, at once on a submit, and whenever a job ends,
  it reads the cards with `nvidia-smi`, asks the planner (`scheduler/planner.py`)
  what to do, and starts what the plan says.
- **Clients talk to it over a socket.** `/usr/local/bin/gpuq` connects to
  `/run/gpuq/gpuqd.sock` (mode `0666`: every local account, no group). Who is
  asking comes from the kernel (`SO_PEERCRED`), never from the request.
- **Each job is its own systemd unit,** `gpuq-job-<id>.service` in `gpuq.slice`,
  run as its submitter (`systemd-run --uid`):
  - it may open only its own `/dev/nvidiaN`, plus the shared control devices
    (`DevicePolicy=closed`, `DeviceAllow=`). Device files are found by UUID,
    because on this host `/dev/nvidiaN` does not follow `nvidia-smi`'s numbering;
  - systemd stops it at its deadline even if gpuqd is down (`RuntimeMaxSec`,
    SIGTERM, then SIGKILL after `TimeoutStopSec=10`);
  - it gets the submitting shell's umask and limits (systemd's defaults are far
    lower: 1,024 open files, 8 MiB locked memory);
  - a root exit hook records how it ended, so a gpuqd restart loses nothing.
- **Attached, shell, detached.** An attached job writes to the submitter's own
  terminal (`--pipe`); `gpuq shell` gets a terminal forwarded to the client's
  (`--pty`); a detached job writes to `~/gpuq-logs/<id>.log`.
- **Restarting gpuqd touches no job** (`KillMode=process`). Waiting and attached
  clients reconnect by themselves within 120 s.

| Path | Contents |
|---|---|
| `/var/lib/gpuq/state.json` | running and waiting jobs, and the promises (root only) |
| `/var/lib/gpuq/usage.jsonl` | the [job ledger](#job-ledger) (`0644`) |
| `/var/lib/gpuq/exits/` | exit reports from the exit hook (root only) |
| `/var/lib/gpuq/launch/` | each job's command, environment and folder, readable only by its user |
| `/var/lib/gpuq/legacy.json` | the previous gpuq's jobs at cutover (root only) |
| `/etc/gpuq/mail.json` | SMTP settings, with the password (root only) |
| `/usr/local/lib/gpuq-v3/` | the code: `scheduler/`, `gpuqd/`, `gpuqcli/` |
| `/usr/local/lib/gpuq-v2/gpuq` | the previous client, for `--rollback` (removed by `--retire`) |

**Logs:** `journalctl -u gpuqd` has every submit, start, end and cancel, and
every GPU process stopped outside gpuq. It needs `sudo` or the `systemd-journal`
group.

**Stopping another user's job:** `sudo gpuq kill <id>`. Root may kill any job.

## The rules

1. **Fair-share.** When more people want GPUs than there are, whoever has held
   the fewest GPU-hours lately goes first. Use fades by half every 7 days.
   Within one pass, everyone waiting gets one turn before anyone gets a second.
2. **A job that fits now starts now,** unless it would still be running when a
   GPU it wants is promised to someone ahead of it.
3. **Promises.** The first job that does not fit is promised the earliest moment
   enough GPUs free up, and those GPUs are held for it. A job needing 2+ GPUs
   keeps its promise: later arrivals plan around it.
4. **48 hours per job.** The one policy number (`MAX_RUNTIME_H` in
   `scheduler/model.py`). There is no `-t`, quota, hold or card cap.
5. **Your GPU is yours.** Nobody else's job is put on a GPU your job runs on.
   `-m X` lets your job join your own GPU when X GB is measured free there.
   `--devices N` adds a job to your GPU N without waiting in line, and it ends
   when your jobs on N reach their 48 h. Fair-share charges each GPU once,
   however many of your jobs share it.

`gpuq status` shows the queue in the order GPUs go out, with promised and
estimated start times. `gpuq why <id>` explains one job's wait.

## Install, update, roll back

Run from a gpuq checkout:

```bash
sudo ./install_v3.sh              # first run: cut over; later runs: update
sudo ./install_v3.sh --rollback   # back to the previous gpuq; refused while gpuqd has jobs
sudo ./install_v3.sh --retire     # after the previous gpuq's last jobs end (see below)
```

- **Update** installs the code and the client and restarts gpuqd. Running jobs
  carry on.
- **The cutover** (done 2026-10-01) refused while anyone waited in the previous
  queue, wrote `/etc/gpuq/mail.json` from the previous config, swapped
  `/usr/local/bin/gpuq`, and copied the previous ledger. The previous gpuq's
  running jobs kept their GPUs until they ended. Their end records were copied
  over.
- **`--retire`** refuses while any of those jobs runs. Then it archives
  `/var/lib/gpu_queue`, `/usr/local/bin/gpu_queue_config.json` (which holds the
  mail password and is readable by members) and `/usr/local/lib/gpuq-v2` into a
  root-only tarball in `/var/backups`, and removes them. After it, `--rollback`
  is no longer possible.
- **Leftovers to remove after `--retire`:** root's cron line
  `*/15 * * * * /usr/local/bin/gpuq audit --enforce --quiet`, which does nothing
  now, and the disabled `gpu-queue.service` of the retired root daemon.

**Before a risky change, run the root smoke check** from the gpuq checkout. It
runs real units through gpuqd's own runner on two idle GPUs, with its state in a
temporary folder:

```bash
sudo env PYTHONPATH=$PWD python3 -m gpuqd.smoke --cards 0,1 --python ~/venvs/nlgen/bin/python
```

## Config

There is no config file. The one policy number, 48 h, is in the code;
`gpuq config` shows it and where things are. The only settings are the mail
settings, `/etc/gpuq/mail.json` (`0600 root`):

```json
{"enabled": true, "smtp_server": "...", "smtp_port": 587, "username": "...", "password": "..."}
```

The installer writes it once, at cutover, and keeps it on updates. Edit it as
root, then `sudo systemctl restart gpuqd`.

## Job ledger

`/var/lib/gpuq/usage.jsonl`, one JSON record per line. It uses the same format
(ledger v2) as before the cutover, which copied the old ledger in, so
`gpuq history` and fair-share see the whole history. Only gpuqd writes it.
Readers skip bad lines. To rotate, move it to `usage-YYYY-MM.jsonl` in the same
folder; readers take every `usage-*.jsonl`, sorted, before `usage.jsonl`.

| `event` | Written when | Notes |
|---|---|---|
| `end` | a job ended | the record fair-share charges; fields below |
| `cancelled` | a job left the queue without running | adds `at`, `wait_sec`, `reason`: `user` (Ctrl-C or `gpuq kill`), `signal`, `lost` (its client never came back), `released` (a `--devices` job whose GPU's jobs all ended first) |
| `rejected` | before the cutover only | gpuqd refuses bad requests at once and records nothing |

```json
{"v":2,"id":945363310,"user":"alice","host":"wsserver1","command":"nvidia-smi -L","name":null,"gpus_requested":1,"devices":null,"memory_gb":null,"max_time_hours":48.0,"priority":"normal","submitted_at":"2026-10-01T20:47:05","event":"end","gpus":[1],"over_quota_at_submit":false,"started_at":"2026-10-01T20:47:05","ended_at":"2026-10-01T20:47:06","queue_wait_sec":0,"elapsed_hours":0.0002,"gpu_hours":0.0002,"exit_code":0,"end_reason":"completed"}
```

| `end` field | Meaning |
|---|---|
| `command` | first 300 characters |
| `gpus_requested`, `gpus` | GPUs asked for, and the host's numbers of those it got |
| `devices` | the GPUs a `--devices` job joined; else null |
| `memory_gb` | the `-m`; null = a GPU to itself |
| `max_time_hours` | the job's limit: 48, or less for a `--devices` job |
| `*_at` | local time, to the second |
| `gpu_hours` | `elapsed_hours` × number of `gpus` |
| `exit_code` | as a shell reports it: 143 for SIGTERM (128 + signal); null for `lost` |
| `end_reason` | `completed` (exit 0), `failed`, `killed` (`gpuq kill`, a closed terminal, a signal), `timed_out` (hit 48 h), `lost` |
| `synthetic` | `true` when gpuqd had to write the record itself: a job whose unit vanished with no exit report (`lost`), or a previous-gpuq job with no end record of its own |
| `priority`, `over_quota_at_submit` | always `normal` and `false` now; kept so older readers work |

**Fair-share** reads `end` records and running jobs as GPU-hours **held**:
overlapping jobs of one user on one GPU count once. Use fades by half every 7
days. `gpuq share --all` shows everyone's.

## GPU use outside gpuq

`/dev/nvidia*` is world-accessible, so a process can use a GPU without gpuq. A
job's processes are exactly its unit's cgroup
(`/gpuq.slice/gpuq-job-<id>.service`), so gpuqd knows for sure which GPU
processes are not part of any job. Accounts below uid 1000 are skipped.

Such a process is **stopped, with no warning period** (`gpuqd/stopper.py`).
If people see others skip the queue and get away with it, soon nobody queues.

- **When:** on its second sighting, at least 60 s after the first. Passes run
  every 30 s, so that is 60 to 90 s after it first shows up. One sighting is
  never enough, so a single bad `nvidia-smi` sample cannot cost anyone work.
- **How:** gpuqd pins the process with a pidfd and checks that it is still the
  process it saw (same kernel start time, same owner, uid 1000 or above). Then
  SIGTERM, and SIGKILL 10 s later if it is still there. Only that process is
  signalled, never its parent or its group, and a recycled pid is never hit.
- **Its owner gets an email** listing what was stopped and the `gpuq submit`
  command that runs it properly. At most one an hour per person; every stop is
  in the journal (`rule 4: stopped pid ...`, then `... SIGKILL` if needed).
- Until it is gone, a GPU held outside gpuq is never given out, and
  `gpuq status` lists it under "GPU use outside gpuq".
- **Emergency off switch:** `sudo touch /etc/gpuq/report-only`. From the next
  pass, such processes are only logged (`rule 4 would stop pid ...`); no restart
  needed. `sudo rm /etc/gpuq/report-only` switches stopping back on.
  `gpuq config` shows which is in force.

`gpuq audit` is accepted and does nothing, so old cron lines stay quiet.

## Notifications

gpuqd sends every email itself, as root, on a background thread, so a slow mail
server never stalls the queue.

| Email subject | When |
|---|---|
| `[gpuq] job <id> started on GPU(s) <list>` | a job that waited 10 minutes or more starts |
| `[gpuq] job <id>: 1 hour left` | 1 hour before the job's deadline |
| `[gpuq] job <id> <end_reason>` | the job ended, only if submitted with `--notify` |
| `[gpuq] job <id> cancelled` | a detached `--devices` job whose GPU's jobs all ended first |
| `[gpuq] stopped your GPU process on <host>` | a process of the user's that used a GPU outside gpuq was stopped (at most one an hour per person) |

- **Sending:** SMTP with STARTTLS. The sender is `username` in
  `/etc/gpuq/mail.json`, `mjolnirruqola@gmail.com`, with a Gmail app password.
  The scratch and disk-quota scripts share the account and its daily limit.
- **Address:** only the account's own address, the first email address in its
  GECOS field (`getent passwd <user>`). `create_users` sets it with `chfn -o`;
  change it with `sudo chfn -o <address> <user>`. No address, no email. An
  address given to `--notify` is ignored.
- There are no quota, audit or Slack messages any more.

## Local development

```bash
python3 -m pytest tests/ -q       # from the gpuq repo root; about 3 minutes
```

The suite is hermetic. `tests/daemon_world.py` fakes the host for gpuqd: 4 H200s
through `tests/fake_nvidia_smi.py`, a fake `/proc`, and a fake systemd that
records what it was asked to do. `tests/test_gpuq_client.py` runs the real
client against a real gpuqd socket loop, with jobs as plain processes through
the real `launch.py` and `exithook.py`. These variables point a client or gpuqd
at a test setup:

| Variable | Default | Purpose |
|---|---|---|
| `GPUQ_SOCKET` | `/run/gpuq/gpuqd.sock` | the socket |
| `GPUQD_STATE_DIR` | `/var/lib/gpuq` | state and ledger |
| `GPUQ_NVSMI` | `nvidia-smi` | the `nvidia-smi` to run |
| `GPUQD_LIB` | `/usr/local/lib/gpuq-v3` | where a job's launcher and exit hook live |

The previous gpuq's code (`userspace.py`, `install_system.sh`,
`install_user.sh`) and its tests are still in the repo, but no longer installed.
`--rollback` uses the copy in `/usr/local/lib/gpuq-v2`.
