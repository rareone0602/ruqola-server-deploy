# Home Directory Quota

Your home directory is for code, venvs, scripts and small files. Large data
belongs in [Scratch Storage](scratch-folder.md).

Each account has a disk quota of **90 GiB soft** and **100 GiB hard**:

- **Over the soft limit**, you can still write for a grace period. `quota -s`
  shows the time left in the `grace` column. When it runs out, writes fail until
  you are back under 90 GiB. The Linux default grace is 7 days; the setting on
  this host has not been checked.
- **At the hard limit**, writes fail at once.
- The number of files is not limited.

## Check your usage

```bash
quota -s
```

```
Disk quotas for user alice (uid 1001):
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
 /dev/nvme1n1p3  1024M  92160M    100G            1234       0       0
```

`space` is what you use, `quota` is the soft limit, `limit` is the hard limit.
The quota lives on the root filesystem, which holds `/home`, so the device name
appears instead of `/home`. `0` means no limit.

## Free up space

```bash
du -h --max-depth=1 ~ | sort -hr | head -20
```

Common culprits:

- `~/.cache/pip`: clear it with `pip cache purge`.
- `~/.cache/huggingface`: set `export HF_HOME=/scratch/users/$USER/hf` in
  `~/.bashrc`, then move the old folder's contents there.
- Checkpoints and datasets: move them to `/scratch/users/$USER/`.

## Email warning

Root runs `check_quotas.sh` daily at 02:00. It emails every user whose home is
over the soft limit (`Disk Quota Warning`). Its delivery has not been confirmed,
so check `quota -s` yourself.

## For admins

```bash
sudo repquota -as                 # every account's usage and limits
sudo check_quotas.sh --dry-run    # who would be emailed; sends nothing
```

In `repquota`, the status column has two characters: disk space, then file
count. `+` means over the soft limit, `-` means within it. `check_quotas.sh`
acts on the first character only. New accounts get their quota from
`add_users.sh` (`QUOTA_SOFT="90G"`, `QUOTA_HARD="100G"`). The stock `warnquota`
cron job is off (`run_warnquota=` is empty in `/etc/default/quota`).
