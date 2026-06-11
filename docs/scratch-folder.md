# SCRATCH FOLDER USAGE GUIDELINES

The scratch space is designed for temporary storage of datasets and computational work. Files are automatically cleaned up **30 days after they were last modified**. The `/scratch` filesystem is mounted `noatime`, so simply *reading/opening* a file is **not** tracked and does **not** keep it alive — only the modification time (mtime) counts. To preserve a file, modify it (or run `touch` on it) within the window, or move it to a permanent location (`/scratch/datasets/` for large shared data, or your home directory for small files).

## DIRECTORY STRUCTURE:
The main directory can be found at /scratch/, with all the subfolders described as follow:

```
/scratch/
├── shared/     - Shared space for all users (group writable)
├── temp/       - Temporary files (world-writable with sticky bit, like /tmp); subject to the same 30-day cleanup as other scratch areas
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

## CHECKING MODIFICATION TIMES AND CLEANUP STATUS

Cleanup is driven by **modification time (mtime)** only: a file is removed once 30 days have passed since it was last modified. (Because `/scratch` is mounted `noatime`, access time is not tracked and is ignored.) The checks below therefore test mtime.

### Find Files Approaching Deletion
```bash
# Check your personal scratch for files not modified in 23+ days
# (23 days is when an email warning is sent; 30 days triggers deletion)
find /scratch/users/$USER/ -type f -mtime +23 -ls

# Find files not modified in 30+ days (eligible for deletion now)
find /scratch/users/$USER/ -type f -mtime +30

# Check specific shared directories (note: /scratch/datasets is exempt from cleanup)
find /scratch/shared/ -type f -mtime +23
```
### View Detailed Modification Information
```bash
# Detailed listing showing modification times (the default for ls -l)
ls -la /scratch/users/$USER/

# Sort by modification time, newest last
ls -ltr /scratch/users/$USER/

# Check exactly when a specific file was last modified
stat /scratch/users/$USER/my_large_file.dat   # look at the "Modify:" line
```
### Keeping Files Active (Resetting Modification Time)

To keep a file, its modification time must be within the last 30 days. Plain `touch` updates the modification time:

```bash
# Touch a file to reset its modification time to now
touch /scratch/users/$USER/important_dataset.h5

# Recursively refresh modification times for a directory tree
find /scratch/users/$USER/project_x/ -type f -exec touch {} \;
```

Note: opening or reading a file (`cat file > /dev/null`, `touch -a`, etc.) does **not** help — only the modification time matters, and reads aren't tracked under `noatime`. The most reliable way to preserve important data is to move or copy it to a permanent location (your home directory for small files, or `/scratch/datasets/` for large shared files).

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

Set reminders: For important files approaching 30 days (you will also get an automated email warning once a file is 23 days stale)

# IMPORTANT RULES:

1. Files not **modified** for 30 days will be automatically deleted (reads are not tracked — to keep a file, run plain `touch` on it within the window, or move it somewhere permanent)
2. This is NOT a backup location - keep important files elsewhere
3. Use appropriate subdirectories for your work
4. Be respectful of shared space
5. Large datasets should go in /scratch/datasets/ for sharing (that directory is exempt from automatic cleanup)

# ACCESS PERMISSIONS:

- Your personal directory (/scratch/users/yourname/): Full control
- Shared directories: Read/write for all scratch-users group members
- Temp directory: World writable with sticky bit (your files protected)

For questions or issues, contact your system administrator.
