# Bash Basics for Server Users

This guide covers essential bash commands and concepts for new users of the Ruqola server.

## Essential Commands

### Navigation and Basic Operations

```bash
# Print current directory
pwd

# List files and directories
ls                  # basic listing
ls -la              # detailed listing with hidden files
ls -lh              # human-readable file sizes

# Change directory
cd /path/to/directory
cd ~                # go to home directory
cd ..               # go up one directory
cd -                # go to previous directory

# Create and remove
mkdir myproject     # create directory
mkdir -p deep/nested/dirs  # create nested directories
rmdir myproject     # remove empty directory
rm file.txt         # remove file
rm -rf directory/   # remove directory and all contents (BE CAREFUL!)
```

### File Operations

```bash
# Copy and move
cp file.txt backup.txt          # copy file
cp -r directory/ backup/        # copy directory recursively
mv oldname.txt newname.txt      # rename/move file
mv file.txt /path/to/directory/ # move file to directory

# View file contents
cat file.txt        # display entire file
head file.txt       # first 10 lines
tail file.txt       # last 10 lines
tail -f logfile.txt # follow file changes (useful for logs)
less file.txt       # paginated view (press q to quit)

# File information
file somefile       # determine file type
du -sh directory/   # directory size
df -h               # disk space usage
```

## File and Directory Operations

### Finding Files

```bash
# Find files by name
find /path/to/search -name "*.py"
find . -name "train.py"

# Find files by type
find . -type f -name "*.log"    # files only
find . -type d -name "models"   # directories only

# Find recent files
find . -mtime -1                # files modified in last day
find . -mtime +7                # files older than 7 days
```

> **Note:** `locate`/`updatedb` are **not** installed on this server by default —
> running them returns `command not found`. Use `find` (shown above) instead. If
> you really want a `locate` database you can install one yourself with
> `sudo apt install plocate` (requires sudo).

### Permissions

```bash
# View permissions
ls -l file.txt

# Change permissions
chmod +x script.py              # make executable
chmod 644 file.txt              # rw-r--r--
chmod 755 script.py             # rwxr-xr-x
chmod -R 755 directory/         # recursive

# Change ownership (usually requires sudo)
chown user:group file.txt
chown -R user:group directory/
```

## Text Processing

### Viewing and Editing

```bash
# Quick text editors
nano file.txt       # beginner-friendly editor
vim file.txt        # powerful editor (learning curve)

# Text processing
grep "pattern" file.txt         # search for pattern
grep -r "function" .            # recursive search in current directory
grep -i "error" logfile.txt     # case-insensitive search
grep -n "TODO" *.py             # show line numbers

# Sorting and filtering
sort file.txt                   # sort lines
sort -n numbers.txt             # numeric sort
uniq file.txt                   # remove duplicate lines
sort file.txt | uniq           # sort and remove duplicates

# Counting
wc file.txt                     # word count (lines, words, characters)
wc -l file.txt                  # line count only
```

### Text Manipulation

```bash
# Cut and paste
cut -d',' -f1 data.csv          # extract first column from CSV
cut -c1-10 file.txt             # extract characters 1-10

# Sed (stream editor)
sed 's/old/new/g' file.txt      # replace all occurrences
sed -i 's/old/new/g' file.txt   # edit file in-place

# Awk (pattern processing)
awk '{print $1}' file.txt       # print first column
awk -F',' '{print $2}' data.csv # use comma as delimiter
```

## Process Management

### Running and Monitoring Processes

```bash
# Run commands
command                         # run in foreground
command &                       # run in background
nohup command &                 # run in background, ignore hangup

# Process monitoring
ps aux                          # list all processes
ps aux | grep python            # find python processes
top                             # real-time process monitor
htop                            # better version of top (if available)

# Process control
jobs                            # list background jobs
fg %1                           # bring job 1 to foreground
bg %1                           # send job 1 to background
kill PID                        # terminate process by PID
kill -9 PID                     # force kill process
killall python                 # kill all python processes
```

### Screen and Tmux (Session Management)

`gpuq submit` runs your job **attached** to the terminal you launched it from,
so closing that terminal stops the job. A terminal multiplexer like `screen` or
`tmux` is the recommended way to keep a long job alive after you disconnect.
Start the job inside a session, detach, and reattach later. (`gpuq submit
--detach` is the other way: the job keeps running after you log out.)

```bash
# Screen
screen                          # start new screen session
screen -S session_name          # start named session
screen -ls                      # list sessions
screen -r session_name          # attach to session
# Inside screen: Ctrl+A, D to detach

# Tmux (if available)
tmux                           # start new session
tmux new -s session_name       # start named session
tmux ls                        # list sessions
tmux attach -t session_name    # attach to session
# Inside tmux: Ctrl+B, D to detach
```

## Environment Variables

### Viewing and Setting Variables

```bash
# View environment variables
env                             # all environment variables
echo $HOME                      # home directory
echo $USER                      # current user
echo $PATH                      # executable search path

# Set variables
export MY_VAR="value"           # set environment variable
export PATH=$PATH:/new/path     # add to PATH

# Persistent variables (add to ~/.bashrc or ~/.bash_profile)
echo 'export HF_HOME=/scratch/users/$USER/hf' >> ~/.bashrc
source ~/.bashrc                # reload configuration
```

### Common Environment Variables

```bash
# Important paths
echo $HOME                      # /home/username
echo $PWD                       # current directory
echo $OLDPWD                    # previous directory

echo $CUDA_VISIBLE_DEVICES       # inside a gpuq job: 0 (0,1 for 2 GPUs); cuda:0 is your first GPU
echo $GPUQ_GPUS                  # inside a gpuq job: the GPU numbers nvidia-smi shows
```

> Do not set `CUDA_VISIBLE_DEVICES` yourself. gpuq sets it for each job, and a
> job can reach only the GPUs gpuq gave it: other numbers find no GPU (see the
> [GPU Queue Guide](gpu-queue-guide.md)).

## SSH and Remote Access

### SSH Basics

```bash
# Connect to server
ssh username@server.com
ssh -p 2222 username@server.com # custom port

# SSH with key
ssh-keygen -t ed25519           # generate SSH key pair (ed25519 recommended)
ssh-copy-id username@server.com # copy public key to server

# File transfer
scp file.txt username@server.com:/path/to/destination/
scp -r directory/ username@server.com:/path/to/destination/
rsync -avz directory/ username@server.com:/path/to/destination/
```

### Port Forwarding (for Jupyter Notebooks)

```bash
# Forward remote port to local
ssh -L 8888:localhost:8888 username@server.com

# Background tunnel
ssh -f -N -L 8888:localhost:8888 username@server.com
```

## Useful Shortcuts

### Command Line Shortcuts

```bash
# Navigation
Ctrl+A              # beginning of line
Ctrl+E              # end of line
Ctrl+U              # clear line before cursor
Ctrl+K              # clear line after cursor

# History
Ctrl+R              # search command history
!!                  # repeat last command
!string             # repeat last command starting with 'string'
history             # show command history

# Completion
Tab                 # auto-complete
Tab Tab             # show all possibilities
```

### Aliases and Functions

Add to `~/.bashrc` for convenience:

```bash
# Useful aliases
alias ll='ls -la'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'

# GPU-related aliases
alias gpu='nvidia-smi'
alias gpuwatch='watch -n 1 nvidia-smi'
alias q='gpuq status'

# Quick functions
mkcd() { mkdir -p "$1" && cd "$1"; }  # make directory and cd into it
```

## Server-Specific Commands

### GPU Monitoring

```bash
# NVIDIA commands
nvidia-smi                      # GPU status (shows all 4 H200 GPUs)
nvidia-smi -l 1                 # continuous monitoring (1-second intervals)
nvidia-smi -q                   # detailed GPU info

# Our custom queue system (gpuq)
gpuq status                     # check queue and GPU ownership
gpuq submit -- python train.py  # 1 GPU to yourself
gpuq submit -g 2 -- python train.py  # 2 GPUs
gpuq submit --command "python train.py"   # 1 GPU, the command as one string
gpuq submit --detach -- python train.py   # return at once; output to ~/gpuq-logs/<id>.log
gpuq shell                      # an interactive shell on a GPU
gpuq why 12345                  # why a job waits, and when it should start
gpuq history                    # your recent jobs (runtime, exit code, end reason)
gpuq share                      # your recent GPU-hours and your place in line
gpuq kill 12345                 # stop a running job or cancel a queued one (ids can be listed)
gpuq kill --mine                # stop all your running jobs, cancel all your queued ones
```

By default your job runs **attached** to your terminal, so its output goes
straight to your screen. Redirect it yourself if you want a log
(`gpuq submit -- python train.py > train.log 2>&1`), and run inside
`screen`/`tmux` for long jobs. With `--detach`, the job keeps running after you
log out, and its output goes to `~/gpuq-logs/<id>.log`. See the
[GPU Queue Guide](gpu-queue-guide.md#background-jobs-and-shells) for full details.

### System Information

```bash
# Hardware info
lscpu                           # CPU information (256 logical CPUs on this host)
lsmem                           # memory information (~755 GiB RAM)
lspci | grep -i nvidia          # GPU information (4x H200 NVL)
df -h                           # disk usage
free -h                         # memory usage
uptime                          # system uptime and load
```

### Python Packages

The system has `python3` but no `python`, and `pip install` only works inside a
virtual environment. System packages (`apt`) need `sudo`; ask the admin.

```bash
python3 -m venv ~/venvs/myenv   # create a venv (once)
source ~/venvs/myenv/bin/activate   # use it; now `python` and `pip` work
pip install package-name        # install Python package
pip install -r requirements.txt # install from requirements file
pip list                        # list installed packages
deactivate                      # leave the venv
```

## Best Practices

### File Management

1. **Organize your work**: Create separate directories for different projects
2. **Use descriptive names**: `experiment_2024_transformer.py` not `exp1.py`
3. **Backup important work**: Copy to multiple locations
4. **Clean up**: Remove unnecessary files regularly

### Command Line Habits

1. **Use tab completion**: Saves typing and prevents errors
2. **Check before destructive operations**: `ls` before `rm`
3. **Use version control**: Learn basic `git` commands
4. **Document your commands**: Keep notes of useful command sequences

### GPU Usage Etiquette

1. **Check status first**: `gpuq status` before submitting jobs
2. **Specify resource requirements**: Don't request more GPUs than you need
3. **A GPU your job runs on is yours**: nobody else's job is put on it, and you
   can add more of your own jobs to it (`gpuq submit --devices N`). When GPUs are
   short, whoever has used the fewest GPU-hours lately goes first (check yours
   with `gpuq share`) — see the [GPU Queue Guide](gpu-queue-guide.md#rules).
4. **Keep long jobs in `screen`/`tmux`**: gpuq jobs run attached to your
   terminal, so a multiplexer keeps them alive if you disconnect. Or use
   `gpuq submit --detach`.
5. **Clean up abandoned jobs**: `gpuq kill --mine` stops all your running jobs
   and cancels all your queued ones.

## Quick Reference Card

```bash
# Most used commands for server work
pwd                             # where am I?
ls -la                          # what's here?
cd directory                    # go somewhere
cp file.txt backup.txt          # make a copy
nvidia-smi                      # check GPUs (4x H200)
gpuq status                     # check queue
gpuq submit -- python train.py  # run a job on 1 GPU
top                             # what's running?
kill PID                        # stop a process
```

---

**Next Steps**: Once comfortable with these basics, check out the [GPU Queue Guide](gpu-queue-guide.md) to start running your first jobs on the server.
