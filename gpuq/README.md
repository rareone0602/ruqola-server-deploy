# Custom GPU Queue Management
This subfolder contains the custom script implementing the GPU queuing and monitoring system set up on the Mjölnir server.
The script is contained in gpu_queue.py. Below instructions show how the script was set up on the server (and how it can be set up again, in case), as well as some basic usage examples and other useful information for GPU monitoring on the server.

# GPU Queue Management System Setup

## Installation

1. Save the Python script as `gpu_queue.py` and make it executable:
```bash
chmod +x gpu_queue.py
sudo cp gpu_queue.py /usr/local/bin/gpuq
```

2. Install required packages:
```bash
pip install requests  # for Slack notifications
```

3. Create configuration file (an example of such file is already provided in this directory):
```bash
gpuq config
```

4. Edit the configuration file to enable notifications:
```json
{
  "max_job_time_hours": 24,
  "max_memory_per_gpu_gb": 70,
  "notification_email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "your-email@gmail.com",
    "password": "your-app-password",
    "admin_email": "admin@yourlab.com"
  },
  "slack": {
    "enabled": true,
    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    "channel": "#gpu-alerts"
  },
  "user_emails": {
    "alice": "alice@yourlab.com",
    "bob": "bob@yourlab.com"
  }
}
```

## Setting up the Daemon

Create a systemd service to run the queue daemon:

```bash
sudo tee /etc/systemd/system/gpu-queue.service << EOF
[Unit]
Description=GPU Queue Manager
After=network.target

[Service]
Type=simple
User=nobody
ExecStart=/usr/local/bin/gpuq daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable gpu-queue
sudo systemctl start gpu-queue
```

## Usage Examples

### Virtual Environment
When using the gpuq tool, always remember to activate your virtual environment first, so that the required libraries for your experiments will be available to the script.

#### Activate the environment with conda
```bash
conda activate $your_environment
```

#### Activate the environment with venv
```bash
source venvs/$your_environment/bin/activate
```

### Basic Job Submission
```bash
# Submit a simple training job
gpuq submit --command "python train.py --epochs 100"

# Request 2 GPUs with 40GB memory each for 12 hours
gpuq submit --command "python big_model.py" --gpus 2 --memory 40 --time 12

# Submit with email notification
gpuq submit --command "python experiment.py" --email "user@lab.com"
```

### Monitoring
```bash
# Check current status
gpuq status

# Monitor in real-time
watch -n 5 gpuq status
```

### Advanced Usage
```bash
# Kill a specific job
gpuq kill --job-id 12345

# Submit interactive job (for Jupyter notebooks)
gpuq submit --command "jupyter notebook --ip=0.0.0.0 --port=8888" --time 4
```

## Creating Wrapper Scripts

Create convenient aliases for common tasks:

### `/usr/local/bin/gpu-train` - Training wrapper
```bash
#!/bin/bash
# Wrapper for training jobs
if [ $# -eq 0 ]; then
    echo "Usage: gpu-train <script.py> [--gpus N] [--time H]"
    exit 1
fi

SCRIPT=$1
shift
gpuq submit --command "python $SCRIPT" "$@"
```

### `/usr/local/bin/gpu-jupyter` - Jupyter wrapper
```bash
#!/bin/bash
# Start Jupyter with GPU access
PORT=${1:-8888}
gpuq submit --command "jupyter notebook --ip=0.0.0.0 --port=$PORT --no-browser" --time 8
echo "Jupyter will start soon. Check 'gpuq status' for details."
```

## User Guidelines

### For Users
1. **Always specify resource requirements** - don't hog more than you need
2. **Use time limits** - helps others plan their work
3. **Monitor your jobs** - check `gpuq status` regularly
4. **Kill finished jobs** - if something hangs, use `gpuq kill`

### For Admins
1. **Monitor the Slack channel** - on the #gpu-alerts channel you'll get alerts about resource hogs and jobs' status
2. **Check logs** in `/tmp/gpu_queue/logs/`
3. **Adjust limits** in the config file as needed
4. **Set up log rotation** to prevent disk filling

## Slack Integration

To set up Slack notifications:

1. Go to your Slack workspace
2. Create a new app at https://api.slack.com/apps
3. Add "Incoming Webhooks" feature
4. Create a webhook for your channel
5. Copy the webhook URL to your config file

The system will send notifications for:
- Jobs timing out
- Users consuming excessive resources
- System alerts

## Email Notifications

For Gmail, you'll need to:
1. Enable 2-factor authentication
2. Generate an "App Password"
3. Use the app password in the config (not your regular password)

At the moment we use a new gmail account **mjolnirruqola@gmail.com** to send email notifications via smtp.
All information related to this account (passwords/app password, etc.) are securely stored on the OneDrive folder named mjolnir.

## Resource Limits

The system automatically:
- **Prevents jobs from starting** if insufficient GPU memory
- **Kills jobs after timeout** (default 24 hours)
- **Alerts about resource hogs** (users with 2+ GPUs or 50GB+ memory)
- **Queues jobs fairly** (first-come, first-served)

## Monitoring Commands

```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Check who's using what
gpuq status | grep -A 10 "Running Jobs"

# View job logs
tail -f /tmp/gpu_queue/logs/job_12345_stdout.log
```