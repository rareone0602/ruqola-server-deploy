# GPU Queue Management System – Notifications FAQ

## ❓ What kinds of notifications does the system send?

The system sends five main types of notifications:

- Job lifecycle notifications

  Triggered when a job is completed, timed out, or killed.

  Example: “Job 1234 by alice has completed”.

- Resource hog notifications

  Triggered when one or more users are consuming too many GPUs or too much GPU memory.

  Example: “🚨 GPU Resource Usage Alert – 2 users are consuming excessive GPU resources”.

- Kill-job notifications

  Triggered when one user requests to kill another’s job.

  Example: “Job 5678 has been killed by user bob”.

- Disk Quota notificatins

  Triggered when the disk quota is exceeded by a user (i.e. the content of a user's personal directory in /home/ surpasses 90GB)

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

- Scratch Files expiration warning/notification

  Triggered when a file created by the user in any location of the scratch folder (except for /scratch/datasets) has not been accessed or modified for at least 23 days (warning period, after 30 days the file will be deleted and a deletion notification will be sent as well).

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

- Email

  - To the job owner’s email (if configured in user_emails inside the config file).

  - To the admin email (for system-wide alerts about resource hogging).

- Slack

  - To the configured Slack channel (default: #gpu-alerts) if Slack notifications are enabled.

- Console logs

  - Notifications also appear in the daemon output (printed in the terminal where the daemon runs).

## ❓ Where is the configuration for notifications?

The config file is located at:
```bash
/usr/local/bin/gpu_queue_config.json
```

It contains settings for:

- notification_email: SMTP details and admin_email.

- slack: Webhook URL and target channel.

- user_emails: Mapping from Linux usernames to their email addresses.

## ❓ When are job notifications sent?

- Completion: When a job finishes normally.

- Timeout: If a job runs longer than its configured max_time_hours (default: 24 hours).

- Killed: When another user explicitly terminates it via the kill action.

## ❓ When are resource hog notifications sent?

When a user:

- Uses ≥50% of available GPUs or at least 3 GPUs.

- Consumes >80GB total GPU memory.

- Uses >30GB memory on multiple GPUs simultaneously.

These notifications are throttled:

- At most once every 15 minutes.

- Only if the set of “hog” users has changed since the last alert.

## ❓ When are disk quota notifications sent?

When a user:

- surpasses the soft limit: >90GB of disk space on their own home directory (e.g. /home/user).

- surpasses the hard limit: >100GB of disk space on their own home directory (e.g. /home/user).

## ❓ When are disk quota notifications sent?

When a file in any /scratch/ directory but /scratch/datasets:

- Has not been accessed or modified in the last 23 days (deletion warning).

- Has not been accessed or modified in the last 30 days (deletion notification).

## ❓ Where can I find logs of notifications?

- Job logs:

Located in the queue directory under logs/.

Example:
``` bash
/var/lib/gpu_queue/logs/job_1234_stdout.log
/var/lib/gpu_queue/logs/job_1234_stderr.log
```

- Daemon console output:

If you’re running the daemon (python gpu_queue.py daemon), messages are printed every 30 seconds.

- Email inboxes:

If email is enabled, check the job owner’s email and the admin email.

- Slack channel:

If Slack is enabled, check the configured Slack channel (e.g., #gpu-alerts).

## ❓ How do I disable or enable notifications?

Edit the config file:

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

Set "enabled": false to turn off email or Slack notifications.

## ❓ Can I check notifications history?

Notifications themselves are not logged persistently, but you can reconstruct history from:

- Job logs (logs/job_*_stdout.log / logs/job_*_stderr.log).

- Daemon console output (redirect it to a file if you want persistent logs).

- Email / Slack archives if configured.