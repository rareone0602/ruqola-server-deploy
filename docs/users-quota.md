# Quotas and Personal Folder

Each user is allocated a personal folder within the /home/ directory that can be used for collecting softwares, virtual environments, scripts and light datasets (refer to the [scratch folder](scratch-folder.md) documentation to save big datasets).

Specifically, each user folder has a soft limit of 90 GiB and a hard limit of 100 GiB. If you exceed the **soft** limit, a grace period of 7 days (the Linux quota system default) applies before the limit is enforced strictly.

## User Guide: Checking Disk Quotas on Ubuntu

This guide explains how to monitor your disk usage, understand quotas, and handle situations where you exceed your limits.

Below is some more information about disk quotas and useful commands related to them.

## 1. What is a Disk Quota?

A disk quota limits the amount of storage space a user can consume on a server or shared filesystem. Quotas are typically enforced to prevent a single user from using all available disk space.

Soft limit: A warning threshold. You can temporarily exceed this limit during the grace period, but you should bring your usage back down before the grace period expires.

Hard limit: Absolute maximum storage. You cannot exceed this limit.

Grace period: A time window during which you can exceed the **soft** limit without being blocked. After this period (7 days by default), the system enforces the quota strictly and you will be unable to write new data until you are back under the soft limit. The grace period applies to the soft limit only — there is no grace for the hard limit.

> Note: On this server, quotas are enforced on the `/` filesystem (which contains `/home`). There is no separate `/home` mount, so your quota usage is measured against the root filesystem device.

## 2. Checking Your Quota
Using the quota command

To check your personal disk usage and limits:

```bash
quota -s
```

Example output (the `-s` flag prints sizes in human-readable form):

```
Disk quotas for user iacopo (uid 1001):
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
  /dev/nvme1n1p3  1024M  92160M    100G            1234       0       0
```

space → your current block (disk space) usage

quota → soft limit

limit → hard limit

grace → remaining time before enforcement once you are over the soft limit (blank when you are within limits)

files → your current inode (file count) usage; the following quota/limit/grace columns are the inode limits (0 = unlimited here)

The `-s` option prints sizes in human-readable form (M, G, etc.). Without `-s`, the first column header is `blocks` and sizes are shown in 1 KiB blocks. Note that the `Filesystem` column shows the device path (e.g. `/dev/nvme1n1p3`), not `/home`, because the quota lives on the root filesystem.

Using repquota (for a more detailed report)

If you want to see all users on a filesystem (requires permission):

```bash
sudo repquota -as
```

This shows all users, their usage, soft/hard limits, and a two-character status field indicating whether each user is within their block and inode limits. Each character is `+` (over the soft limit / in grace) or `-` (within limits). See the summary table below for the exact meaning of these characters.

## 3. Notifications

There is **no fully automatic** quota-notification daemon enabled on this host by default. The standard `warnquota` daily cron job is disabled (`run_warnquota=` is empty in `/etc/default/quota`), and the project's helper script `check_quotas.sh` is not scheduled in cron out of the box.

When run by an administrator, `check_quotas.sh` parses `repquota -as`, finds users whose **block** usage is over the soft limit (the `+` status), and emails each affected user (the address is read from the GECOS field of their account via `getent passwd`). It does **not** warn you for merely "approaching" the hard limit — it only fires once you are already over the soft limit.

In short: the administrator **may** send you an email when you exceed your soft limit, but you should not rely on receiving one. Check your own usage proactively with `quota -s`.

Tip: If you do receive a "quota exceeded" warning, act on it promptly.

## 4. What to Keep an Eye On

Current usage vs. soft limit

Try to stay below the soft limit to avoid grace period warnings.

Hard limit

You cannot exceed this. If you hit it, new files may fail to save.

Grace period

If you exceed the soft limit, you have a limited time (7 days by default) to reduce usage before write restrictions are applied.

Use commands like du or file manager tools to identify and remove large/unnecessary files.

Email notifications

If the administrator's quota check is run and you are over your soft limit, the notification will indicate your current usage and soft/hard limits. Follow the instructions promptly to avoid being blocked from writing more files. Because these emails are not guaranteed, do not depend on them — monitor your usage yourself.

## 5. Checking Disk Usage for Cleanup

To see which files are using the most space:

```bash
du -h ~ | sort -hr | head -n 20
```

du -h ~ → human-readable sizes in your home directory

sort -hr → sort largest to smallest

head -n 20 → show top 20 largest items

Remove unnecessary files to stay within limits.

## 6. Tips to Avoid Quota Problems

Regularly monitor your usage with quota -s.

Clean up temporary or old files in your home directory.

Use compression (gzip, tar) for large files you want to keep.

Move rarely used files to external storage if possible.

## Summary Table
| Term | What It Means | Action for Users |
| :--- | :--- | :--- |
| **Soft limit** | Warning threshold (90 GiB) | Reduce usage if approaching/exceeding |
| **Hard limit** | Absolute limit (100 GiB) | Cannot exceed; must delete files if reached |
| **Grace period** | Time to fix over-soft-limit usage (7 days default) | Free up space before restrictions are applied |
| **`+` status (in repquota)** | Over the soft limit / in the grace period | Reduce usage before the grace period expires |
| **`-` status (in repquota)** | Within limits (OK) | No action needed |

> Note on repquota status: the per-user status in `repquota` is a two-character field — the first character is the **block** (disk space) status and the second is the **inode** (file count) status. Each is `+` (over soft limit / in grace) or `-` (within limits). There is no separate flag for "over the hard limit"; hitting the hard limit (or the grace period expiring) is reported by write failures and an expired entry in the `grace` column, not by a `-`.
