# Users Creation

In order to create a user, use the provided [**add_users.sh**](/scripts/add_users.sh) script.
This script can be used in two distinct ways:

1) For bulk users addition, run the following:
```bash
add_users.sh users.csv
```
where users.csv is a csv file containing with columns named as follow: 
- username: the username of each user to be created
- password: the provisional password for each user (the password will be changed at the first login as explained below)
- full_name: the full name of each user
- email: the email address of each user

2) For single user addition, run the following:
```bash
add_users.sh --single username password full_name email
```
where the various arguments are defined as of point 1. Be careful at this stage to include the full name within "" if the name includes white spaces.
For example:
```bash
add_users.sh --single testuser some_password "A Test User" email@gmail.com
```

After having set up the user(s), all the information associated with the user can be assessed with:
```bash
getent passwd username
```

Also, at this stage it is important to ensure that all the relevant folders to be used by the user both in /home/ and in /scratch/users have been created.

```bash
ls /home/
```

```bash
ls /scratch/users
```

Where in both cases you should see new folder(s) having the same name of the user(s) just created.

# Users Deletion
In order to delete a user, use the provided [**delete_users.sh**](/scripts/delete_users.sh) script.
This script can be used in two distinct ways:

1) For bulk users deletions, run the following:
```bash
delete_users.sh users.csv
```
where users.csv is a csv file containing with columns named as follow: 
- username: the username of each user to be created

2) For single user addition, run the following:
```bash
delete_users.sh --single username
```
where username is also here the username of the user to be deleted.

Once run, the script will ask to confirm and, if so, press and enter "y" to confirm and "n" to cancel the operation.

By default, the script will save a backup copy of the users' files that have been deleted. In order not to save such a backup, add the following argument:

```bash
delete_users.sh users.csv --no-backup
```

And, equally, in the single user case:

```bash
delete_users.sh --single username --no-backup
```