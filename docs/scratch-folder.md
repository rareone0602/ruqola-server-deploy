# Scratch Storage

`/scratch` is the large shared disk for datasets and working files. It is not a
backup. Keep anything you cannot recreate somewhere else as well.

## The deletion rule

A file under `/scratch/users`, `/scratch/shared` or `/scratch/temp` is deleted
once it has gone **180 days with no read and no write**. Reading it resets the
clock, and so does changing it. `/scratch/datasets` is never cleaned.

- **Warning:** after 166 days with no read and no write, you get an email about
  14 days before deletion. It repeats each night until you use the file or it is
  deleted. See [Notifications](notifications-faq.md#scratch-cleanup-emails).
- **Do not rely on the email.** A failed warning email does not stop the
  deletion. Run `scratch-status` to see what is at risk.
- **Empty directories** are removed on the same 180-day clock. Your own
  `/scratch/users/<you>` directory is never removed.
- **When:** the cleanup runs once a day, starting between 02:00 and 02:30, and
  also shortly after a reboot.
- **Every deletion is recorded.** If a file is missing, ask the admin to look it
  up in `/var/log/scratch-cleanup/deleted-YYYY-MM.tsv`.

The day numbers come from the cleanup script itself
(`scratch-cleanup.sh --show-config`), and a test keeps this page in step with it.

## Where to put things

| Path | Use it for | Who can write | Cleaned? |
|---|---|---|---|
| `/scratch/users/<you>/` | your working files, caches, checkpoints | you | yes |
| `/scratch/datasets/` | large data the group shares | the `scratch-users` group | never |
| `/scratch/shared/` | files you share with the group | the `scratch-users` group | yes |
| `/scratch/temp/` | temporary files | everyone; only the owner can delete (like `/tmp`) | yes |
| your home `~` | code, venvs, small files | you | no, but 90 GiB soft / 100 GiB hard quota |

Put a venv in your home, not in `/scratch`: the cleanup deletes the venv files
you never read, which can break it.

## Check what is at risk

```bash
scratch-status
```

It lists files under the cleaned directories that have gone 166 days with no
read and no write (the first 20), and shows disk use. To check by hand:

```bash
find /scratch/users/$USER -type f -atime +166 -mtime +166   # in the warning window
find /scratch/users/$USER -type f -atime +180 -mtime +180   # deleted on the next run
stat /scratch/users/$USER/some_file                         # "Access:" and "Modify:" times
du -sh /scratch/users/$USER                                 # how much you use
```

## Keep a file

- Use it: reading or changing it resets its clock.
- Refresh it without using it: `touch file`, or for a whole tree
  `find /scratch/users/$USER/project -type f -exec touch {} +`.
- Move it somewhere permanent: `/scratch/datasets/` for large shared data, your
  home for small files.

## Examples

```bash
cd /scratch/users/$USER && mkdir -p my_project          # your working area
ln -s /scratch/datasets/some_dataset ./my_project/data  # link shared data, don't copy it
export TMPDIR=/scratch/temp/$USER && mkdir -p "$TMPDIR" # big temporary files
```
