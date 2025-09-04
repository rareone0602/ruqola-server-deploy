# Quotas and Personal Folder

Each user is allocated a personal folder within the /home/ directory that can be used for collecting softwares, virtual environments, scripts and light datasets (refer to the [scratch folder](scratch-folder.md) documentation to save big datasets).

Specifically, each user folder has a soft limit of 90GB and a hard limit of 100GB with a grace period of 6 days if the limit is exceeded.

## User Guide: Checking Disk Quotas on Ubuntu

This guide explains how to monitor your disk usage, understand quotas, and handle notifications about exceeding limits.

Below is some more information about disk quotas and useful commands related to them.

## 1. What is a Disk Quota?

A disk quota limits the amount of storage space a user can consume on a server or shared filesystem. Quotas are typically enforced to prevent a single user from using all available disk space.

Soft limit: A warning threshold. You can temporarily exceed this limit, but you will receive notifications.

Hard limit: Absolute maximum storage. You cannot exceed this limit.

Grace period: A time window during which you can exceed the soft limit without being blocked. After this period, the system enforces the quota strictly.

## 2. Checking Your Quota
Using the quota command

To check your personal disk usage and limits:

```bash
quota -s
```

Example output:

```
Disk quotas for user iacopo (uid 1001):
     Filesystem  blocks   quota   limit   grace
     /home      1024K    2000K   2500K
```

blocks → your current usage

quota → soft limit

limit → hard limit

grace → remaining time before enforcement if over soft limit

The -s option prints sizes in human-readable form (KB, MB, etc.).

Using repquota (for a more detailed report)

If you want to see all users on a filesystem (requires permission):

```bash
sudo repquota -as
```

This shows all users, their usage, soft/hard limits, and a + or - if over quota.

## 3. Automatic Notifications

The system will automatically send email notifications if:

You exceed your soft limit (+ in repquota)

You are approaching your hard limit

Tip: Check your inbox regularly, especially if you receive “quota exceeded” warnings.

## 4. What to Keep an Eye On

Current usage vs. soft limit

Try to stay below the soft limit to avoid grace period warnings.

Hard limit

You cannot exceed this. If you hit it, new files may fail to save.

Grace period

If you exceed the soft limit, you have a limited time to reduce usage before restrictions are applied.

Use commands like du or file manager tools to identify and remove large/unnecessary files.

Email notifications

Notifications will indicate your current usage and soft/hard limits.

Follow instructions promptly to avoid being blocked from writing more files.

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
| **Soft limit** | Warning threshold | Reduce usage if approaching/exceeding |
| **Hard limit** | Absolute limit | Cannot exceed; must delete files if reached |
| **Grace period** | Time to fix over-usage | Free up space before restrictions are applied |
| **+ flag** | Over soft limit | Check email notification |
| **- flag** | Over hard limit | Must delete files immediately |

