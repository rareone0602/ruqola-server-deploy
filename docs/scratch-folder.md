# SCRATCH FOLDER USAGE GUIDELINES

The scratch space is designed for temporary storage of datasets and computational work. Files are automatically cleaned up after 30 days of no access.

## DIRECTORY STRUCTURE:
The main directory can be found at /scratch/, with all the subfolders described as follow:

/scratch/
├── shared/     - Shared space for all users (group writable)
├── temp/       - Temporary files (like /tmp, auto-cleaned frequently)
├── datasets/   - Shared datasets (group readable/writable, no expiration)
└── users/      - Individual user directories
    ├── user1/  - Personal scratch space for user1
    ├── user2/  - Personal scratch space for user2
    └── ...

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

# Copy a dataset to your working directory (if you need to modify it)
cp /scratch/datasets/common_crawl/ ./my_copy/

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

## CHECKING FILE ACCESS TIMES AND CLEANUP STATUS
### Find Files Approaching Deletion (30+ days old)
```bash
# Check your personal scratch for files not accessed in 25+ days (warning)
find /scratch/users/$USER/ -atime +25 -type f -ls

# Find files not accessed for 29 days (imminent deletion)
find /scratch/users/$USER/ -atime +29 -type f

# Check specific shared directories
find /scratch/datasets/ -atime +25 -type f
```
### View Detailed File Access Information
```bash
# List files with last access time
ls -lu /scratch/users/$USER/*

# Detailed listing with access times
ls -la --time=atime /scratch/users/$USER/

# Check when a specific file was last accessed
stat /scratch/users/$USER/my_large_file.dat
```
### Keeping Files Active (Resetting Access Time)
```bash
# Touch files to update access time without modifying content
touch -a /scratch/users/$USER/important_dataset.h5

# Recursively update access times for a directory
find /scratch/users/$USER/project_x/ -exec touch -a {} \;

# Read files to reset access time (alternative method)
cat /scratch/users/$USER/datafile > /dev/null
```

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

Set reminders: For important files approaching 30 days

# IMPORTANT RULES:

1. Files not accessed for 30 days will be automatically deleted
2. This is NOT a backup location - keep important files elsewhere
3. Use appropriate subdirectories for your work
4. Be respectful of shared space
5. Large datasets should go in /scratch/datasets/ for sharing

# ACCESS PERMISSIONS:

- Your personal directory (/scratch/users/yourname/): Full control
- Shared directories: Read/write for all scratch-users group members
- Temp directory: World writable with sticky bit (your files protected)

For questions or issues, contact your system administrator.