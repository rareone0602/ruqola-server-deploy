# Users Creation

Create accounts with [**add_users.sh**](/scripts/bin/add_users.sh) (also on the
host as `create_users`). Run it as a sudoer, not as root; it refuses root.

```bash
add_users.sh users.csv                                           # many users
add_users.sh --single testuser some_password "A Test User" email@gmail.com   # one user
```

The CSV needs a header line, which is skipped, then one user per line with the
columns in this order: `username,password,full_name,email`. Usernames must start
with a lowercase letter and contain only lowercase letters, digits, `_` and `-`.

**Keep CSV fields plain.** The parser breaks on quotes, apostrophes, backslashes
and commas inside a field: an apostrophe in a password leaves the password
empty, `Sean O'Brien` becomes `Sean`, and `"Smith, John"` shifts into the email
column. Save the file with Unix line endings. With `--single`, the password is
visible to other users in `ps` while the script runs.

For each user, the script:

- creates the account with `bash` as its shell, in the groups `users`,
  `scratch-users` and `gpuqueue`
- sets the password and forces a change at first login
- stores the email address in the account's GECOS field, where gpuq and the
  cleanup scripts look for it
- creates `/home/<user>` (mode 750) with `projects/`, `data/`, `scripts/` and
  `venvs/`, and `/scratch/users/<user>` (mode 750)
- sets the disk quota to 90 GiB soft, 100 GiB hard
- emails the user the username, the password and a link to this site

Check the result:

```bash
getent passwd <user>                           # the email appears in the 5th field
id <user>                                      # groups
ls -ld /home/<user> /scratch/users/<user>
sudo quota -s -u <user>
```

# Users Deletion

Delete accounts with [**delete_users.sh**](/scripts/bin/delete_users.sh) (also
`delete_users`), again as a sudoer:

```bash
delete_users.sh users.csv                  # or: delete_users.sh --csv users.csv
delete_users.sh --single <user>
delete_users.sh --single <user> --no-backup
delete_users.sh users.csv --no-backup      # also: --csv users.csv --no-backup
```

Only the first CSV column (`username`) is read; the header line is skipped. The
script asks once, `y` to go on. Then, for each user, it kills their processes,
removes their quota, deletes `/scratch/users/<user>`, and removes the account
and home directory.

Unless you pass `--no-backup`, it first copies the home and scratch directories
to `/var/backups/deleted_users/<user>_<timestamp>/`.

**Check the list before you confirm.** The script has no guard against system
or admin accounts: it will delete any account named in the list, and one `y`
covers the whole CSV. Also:

- The backup lands on the root filesystem. Check the size first:
  `sudo du -sh /scratch/users/<user>`.
- A failed backup copy is logged as a warning, and the deletion still goes on.
  Check the backup folder before you rely on it.
