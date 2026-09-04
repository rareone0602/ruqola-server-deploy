# SCRATCH FOLDER USAGE GUIDELINES

The scratch space is designed for temporary storage of datasets and computational work. A file is automatically deleted once it has gone **180 days with no read and no write**. Both count: opening a file resets its clock (the filesystem is mounted `relatime`, so reads are tracked), and so does modifying it. You are emailed a warning after 166 days, about 14 days before removal. To preserve a file, read or modify it within the window, or move it to a permanent location (`/scratch/datasets/` for large shared data, or your home directory for small files).

The exact numbers are read from the cleanup script itself (`scratch-cleanup.sh --show-config`); this page is checked against it by the test suite.

## DIRECTORY STRUCTURE:
The main directory can be found at /scratch/, with all the subfolders described as follow:

```
/scratch/
├── shared/     - Shared space for all users (group writable)
├── temp/       - Temporary files (world-writable with sticky bit, like /tmp); subject to the same 180-day cleanup as other scratch areas
├── datasets/   - Shared datasets (group readable/writable, no expiration)
└── users/      - Individual user directories
    ├── user1/  - Personal scratch space for user1
    ├── user2/  - Personal scratch space for user2
    └── ...
```

Each user, when created, should be automatically added to the scratch-users permission group, which grants write/read control over various locations in the scratch folder (see below). At the same time, a new user folder will be created in /scratch/users/ named after the username of the new user.

## USAGE EXAMPLES:

### Accessing Your Personal Scratch Space
```bash
# Navigate to your personal scratch directory
cd /scratch/users/$USER

# Create a project directory
mkdir my_project
cd my_project

# Copy large input files from your home directory
cp ~/large_dataset.csv ./

# Symbolic link to avoid duplicating data
ln -s /scratch/datasets/reference_genome/ ./ref_genome
```
### Using Shared Datasets
```bash
# List available shared datasets
ls /scratch/datasets/

# Copy a dataset to your working directory (if you need to modify it).
# Use -r because datasets are directories; plain cp without -r fails on directories.
# Better still: symlink instead of copying, to avoid wasting scratch space.
cp -r /scratch/datasets/common_crawl/ ./my_copy/

# Or work directly with the shared data (read-only recommended)
analyze_tool --input /scratch/datasets/imaging_data/
```
### Temporary File Operations
```bash
# Use temp space for intermediate processing files
export TMPDIR=/scratch/temp/$USER
mkdir -p $TMPDIR

# Process large files temporarily
sort large_file.txt > $TMPDIR/sorted_output.txt
```

## CHECKING TIMESTAMPS AND CLEANUP STATUS

A file is removed only when **both** its last-read time (atime) and its last-write time (mtime) are more than 180 days old. Either one being recent keeps it.

### Find Files Approaching Deletion
The simplest way is the report every user can run:
```bash
scratch-status
```
It reads the current policy from the cleanup script and lists only files that are genuinely at risk. To check by hand:
```bash
# Files in your personal scratch that have had no read AND no write for 166+ days
# (166 days is when the email warning is sent; 180 days triggers deletion)
find /scratch/users/$USER/ -type f -atime +166 -mtime +166 -ls

# Files eligible for deletion now
find /scratch/users/$USER/ -type f -atime +180 -mtime +180
```
### View Detailed Timestamp Information
```bash
# Check exactly when a specific file was last read and last modified
stat /scratch/users/$USER/my_large_file.dat   # look at the "Access:" and "Modify:" lines

# Sort by modification time, newest last
ls -ltr /scratch/users/$USER/
```
### Keeping Files Active

Reading a file resets its clock, so a file you actually use will not be deleted. To keep a file you are not using, `touch` it (this resets both timestamps):

```bash
touch /scratch/users/$USER/important_dataset.h5

# Recursively refresh a directory tree
find /scratch/users/$USER/project_x/ -type f -exec touch {} \;
```

The most reliable way to preserve important data is to move or copy it to a permanent location (your home directory for small files, or `/scratch/datasets/` for large shared files).

# BEST PRACTICES
Organize by project: 
```bash
mkdir /scratch/users/$USER/project_{name}
```

Clean up regularly: Remove files you no longer need

Use symbolic links: Point to shared datasets instead of copying

Monitor usage regularly: 
```
du -sh /scratch/users/$USER/
```

Set reminders: For important files approaching 180 days (you will also get an automated email warning once a file has gone 166 days with no read and no write)

# IMPORTANT RULES:

1. Files with no read and no write for 180 days are automatically deleted (to keep a file you are not using, `touch` it within the window, or move it somewhere permanent)
2. This is NOT a backup location - keep important files elsewhere
3. Use appropriate subdirectories for your work
4. Be respectful of shared space
5. Large datasets should go in /scratch/datasets/ for sharing (that directory is exempt from automatic cleanup)

# ACCESS PERMISSIONS:

- Your personal directory (/scratch/users/yourname/): Full control
- Shared directories: Read/write for all scratch-users group members
- Temp directory: World writable with sticky bit (your files protected)

For questions or issues, contact your system administrator.
