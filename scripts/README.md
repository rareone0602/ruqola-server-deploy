# ruqola-admin: the sudoer's toolkit for wsserver1

Every custom script that runs on the host as root or by a sudoer, in one place:
one shared library, one installer, one test suite, one manifest that says what
belongs where. If a file sits in `/usr/local/bin` and this manifest does not
describe it, that is a finding, not a feature.

```
scripts/
├── MANIFEST          what goes where, with mode and owner  (install.sh reads only this)
├── install.sh        --check | --diff | sudo install      (backups + rollback built in)
├── lib/              shared library, one concern per file
│   ├── init.sh       loads the rest; scripts source this one file
│   ├── log.sh        log_message, acquire_lock/release_lock, require_root/refuse_root
│   ├── mail.sh       email_for_user, full_name_for_user, send_mail
│   └── fs.sh         format_bytes, file_meta, manifest_open/manifest_record
├── bin/              installed to /usr/local/bin
├── systemd/          the reaper's .service and .timer
├── logrotate.d/      rotation for the reaper's chatty log
├── tools/            generators run by the installer (not installed)
└── tests/            run_tests.sh, lib.sh, stubs/, one test_*.sh per concern
```

## The map

| installed path | source | who runs it | when | as whom |
|---|---|---|---|---|
| `/usr/local/lib/ruqola-admin/*.sh` | `lib/` | sourced by the scripts below | | |
| `/usr/local/bin/scratch-cleanup.sh` | `bin/` | `scratch-cleanup.timer` | nightly, 02:00 + up to 30 min | root |
| `/etc/systemd/system/scratch-cleanup.{service,timer}` | `systemd/` | systemd | | |
| `/etc/logrotate.d/scratch-cleanup` | `logrotate.d/` | `logrotate.timer` | daily 00:00 | root |
| `/usr/local/bin/check_quotas.sh` | `bin/` | an administrator, by hand | not scheduled | root via sudo |
| `/usr/local/bin/add_users.sh` (+ `create_users` link) | `bin/` | an administrator, by hand | | a sudoer, not root |
| `/usr/local/bin/delete_users.sh` (+ `delete_users` link) | `bin/` | an administrator, by hand | | a sudoer, not root |
| `/usr/local/bin/scratch-usage.sh`, `scratch-status` | `bin/` | anyone | | any user |
| `/scratch/README.txt` | `tools/render-readme.sh` | generated at install | | |

**Not managed here, on purpose:** `gpuq` (its own repo and `install_system.sh`),
`ollama` (upstream binary and unit), `nsys` (NVIDIA alternatives), `uv` (root's own).
The manifest also names what must *not* exist: `scratch-backup.sh`, which was
never a backup but a stale copy of the 30-day cleaner with `/scratch/datasets` in
its list, and any `*.bak-*` on `PATH`.

## Three commands

```bash
tests/run_tests.sh          # every test; no root; touches nothing real
./install.sh --check        # how far the host has drifted from this repo; no root
sudo ./install.sh           # tests -> check -> confirm -> install -> verify
```

`--check` prints one line per manifest entry: `same`, `MISSING`, `DIFFERS`, `MODE`,
`OWNER`, or `PRESENT` (something that should be gone). `--diff` adds unified diffs.
Install refuses to start if any test fails, shows you exactly what will change,
and if the reaper or its library is among the changes, dry-runs the new reaper
against the real `/scratch` first so you see who would be emailed. Nothing is
written before you type `yes`.

## Backups and rollback

Nothing is ever backed up next to its target. Every install that changes
something creates `/var/backups/ruqola-admin/<stamp>/` holding:

- `files/` — a mirror of every path replaced, at its full path
- `retired/` — files swept off the host by `retire` entries
- `rollback.sh` — run as root to undo that install, exactly
- `MANIFEST`, `PROVENANCE` — what was installed, from which git revision

## The reaper's two records

| | path | rotated? | who reads it |
|---|---|---|---|
| chatty log | `/var/log/scratch-cleanup.log` | weekly or at 50 MB, 12 kept, compressed | debugging a run |
| deletion manifest | `/var/log/scratch-cleanup/deleted-YYYY-MM.tsv` | never | "what was deleted from my dir on that date?" |

The manifest is tab-separated with a header: `deleted_at kind owner bytes
last_access last_modified path`. Before deleting anything in a directory the
reaper checks it can write the manifest; if it cannot, nothing in that directory
is deleted and the unit exits non-zero. A dry run writes no manifest.

```bash
# everything removed from alice's space in December
sudo awk -F'\t' '$3=="alice"' /var/log/scratch-cleanup/deleted-2026-12.tsv
# bytes reclaimed that month
sudo awk -F'\t' 'NR>1{s+=$4} END{print s}' /var/log/scratch-cleanup/deleted-2026-12.tsv | numfmt --to=iec
```

## The single source of truth

`scratch-cleanup.sh` holds the retention numbers. Everything else asks it:

```
$ bin/scratch-cleanup.sh --show-config
DAYS_TO_KEEP=180
DAYS_TO_NOTIFY=166
SCRATCH_DIRS=/scratch/shared /scratch/temp /scratch/users
LOG_FILE=/var/log/scratch-cleanup.log
MANIFEST_DIR=/var/log/scratch-cleanup
```

`scratch-usage.sh`, `tools/render-readme.sh`, and the two docs pages that quote
the policy are tested against that output. Change the number in one place and
the tests tell you every other place that has to follow.

## Adding a script

Three files, and `tests/test_project_layout.sh` fails until all three exist:

1. **`bin/<name>`** — `#!/bin/bash`, executable. If it needs the library, copy the
   loader block from `bin/check_quotas.sh` verbatim (the layout test checks it
   matches). Give every path it touches an environment seam with a default, so
   the tests can point it at a sandbox. Put `main` behind
   `if [[ "${BASH_SOURCE[0]}" == "$0" ]]` if tests need to source its functions.
2. **`MANIFEST`** — one `file` line: source, destination, mode, owner.
3. **`tests/test_<name>.sh`** — `source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"`,
   build a sandbox with `new_sandbox`, run the script with `PATH="$STUBS:$PATH"`
   and its seams set, assert with `check`, end with `finish`. The runner finds it
   by name.

Adding a library file: `lib/<concern>.sh`, list it in `lib/init.sh`, test it in
`tests/test_lib_<concern>.sh`.

Stubs in `tests/stubs/` stand in for `getent`, `stat`, `msmtp`, `sudo`,
`repquota`, and `systemctl`. A test never sends mail, never needs root, never
reads `/scratch` or `/var/log`.

## Rules the tests enforce

- The reaper never deletes without recording, never records on a dry run, never
  emails an empty address, sends one digest per user per run, and never warns
  about a file it is deleting in the same run.
- Empty directories age out on the same clock as files; a user's top-level
  directory is never removed.
- `--show-config` is the only source for the retention numbers.
- Every `bin/` script is executable, parses, is in the manifest, and has a test.
- Every script that sources the library uses the same loader block.
- The install is idempotent: a second run changes nothing and reloads nothing.
