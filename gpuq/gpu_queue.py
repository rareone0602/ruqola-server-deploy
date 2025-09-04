#!/usr/bin/env python3
"""
Lightweight GPU Queue Management System
For small research teams sharing GPU resources
"""

import os
import signal
import sys
import time
import json
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests  # for Slack notifications
import shlex

class GPUQueueManager:
    def __init__(self, config_file="/usr/local/bin/gpu_queue_config.json"):
        self.config_file = config_file
        
        # Try multiple locations for queue directory with proper permissions
        possible_dirs = [
            Path("/var/lib/gpu_queue"),  # System-wide (preferred for production)
            Path("/tmp/gpu_queue_shared"),  # Shared temp (fallback)
            Path(f"/tmp/gpu_queue_{os.getenv('USER', 'default')}")  # User-specific
        ]
        
        self.queue_dir = None
        for queue_dir in possible_dirs:
            try:
                queue_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
                # Test write permissions
                test_file = queue_dir / "test_write"
                test_file.write_text("test")
                test_file.unlink()
                self.queue_dir = queue_dir
                break
            except (PermissionError, OSError):
                continue
        
        if self.queue_dir is None:
            # Last resort: use user's home directory
            self.queue_dir = Path.home() / ".gpu_queue"
            self.queue_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        
        self.job_file = self.queue_dir / "jobs.json"
        self.running_file = self.queue_dir / "running.json"
        self.kill_file = self.queue_dir / "kill.json"
        self.last_notification_file = self.queue_dir / "last_resource_notification.json"
        
        # Resource monitoring settings
        self.notification_cooldown_minutes = 15  # Don't spam notifications
        self.memory_threshold_gb = 5  # Minimum memory to consider "using" a GPU
        self.utilization_threshold = 15  # Minimum GPU utilization to consider "using"
        
        self.load_config()
        
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "max_job_time_hours": 24,
            "max_memory_per_gpu_gb": 70,  # H200 has ~80GB
            "notification_email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "admin_email": "admin@yourlab.com"
            },
            "slack": {
                "enabled": False,
                "webhook_url": "",
                "channel": "#gpu-alerts"
            },
            "user_emails": {
                # "username": "user@email.com"
            }
        }
        
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
                # self.config = {**default_config, **json.load(f)}
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_gpu_info(self):
        """Get current GPU usage information"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_used': int(parts[2]),
                        'memory_total': int(parts[3]),
                        'utilization': int(parts[4])
                    })
            return gpus
        except subprocess.CalledProcessError:
            return []
    
    def get_gpu_processes(self):
        """Get processes running on GPUs"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line and line != 'No running processes found':
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        processes.append({
                            'pid': int(parts[0]),
                            'name': parts[1],
                            'gpu_uuid': parts[2],
                            'memory': int(parts[3])
                        })
            return processes
        except subprocess.CalledProcessError:
            return []
    
    def get_gpu_to_uuid_mapping(self):
        """Get mapping of GPU UUID to index"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,uuid',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            gpu_mapping = {}
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    gpu_mapping[parts[1]] = int(parts[0])  # UUID -> index
            return gpu_mapping
        except subprocess.CalledProcessError:
            return {}
    
    def get_process_user(self, pid):
        """Get the user who owns a process"""
        try:
            result = subprocess.run(['ps', '-o', 'user=', '-p', str(pid)], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def should_send_resource_notification(self):
        """Check if enough time has passed since last resource notification"""
        try:
            if self.last_notification_file.exists():
                with open(self.last_notification_file, 'r') as f:
                    last_data = json.load(f)
                
                last_time = datetime.fromisoformat(last_data['timestamp'])
                if datetime.now() - last_time < timedelta(minutes=self.notification_cooldown_minutes):
                    return False, last_data.get('last_reported_users', [])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        return True, []
    
    def update_last_resource_notification(self, reported_users):
        """Update the last resource notification timestamp and users"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'last_reported_users': reported_users
        }
        
        try:
            with open(self.last_notification_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save notification timestamp: {e}")
    
    def load_jobs(self):
        """Load job queue from file"""
        try:
            if self.job_file.exists():
                with open(self.job_file, 'r') as f:
                    return json.load(f)
        except (PermissionError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load jobs file: {e}")
        return []
    
    def save_jobs(self, jobs):
        """Save job queue to file"""
        try:
            # Ensure directory exists and is writable
            self.queue_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
            with open(self.job_file, 'w') as f:
                json.dump(jobs, f, indent=2)
            # Make file readable by all users
            # os.chmod(self.job_file, 0o666)
        except PermissionError as e:
            print(f"Error: Cannot write to {self.job_file}: {e}")
            print("Try running: sudo chmod 777 /tmp/gpu_queue")
            raise
    
    def load_running_jobs(self):
        """Load running jobs from file"""
        try:
            if self.running_file.exists():
                with open(self.running_file, 'r') as f:
                    return json.load(f)
        except (PermissionError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load running jobs file: {e}")
        return []
    
    def save_running_jobs(self, jobs):
        """Save running jobs to file"""
        try:
            self.queue_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
            with open(self.running_file, 'w') as f:
                json.dump(jobs, f, indent=2)
            # os.chmod(self.running_file, 0o666)
        except PermissionError as e:
            print(f"Error: Cannot write to {self.running_file}: {e}")
            print("Try running: sudo chmod 777 /tmp/gpu_queue")
            raise

    def load_kill_jobs(self):
        """Load kill jobs from file"""
        try:
            if self.kill_file.exists():
                with open(self.kill_file, 'r') as f:
                    return json.load(f)
        except (PermissionError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load running jobs file: {e}")
        return []
    
    def save_kill_jobs(self, jobs):
        """Save kill jobs to file"""
        try:
            with open(self.kill_file, 'w') as f:
                json.dump(jobs, f, indent=2)
            # os.chmod(self.running_file, 0o666)
        except PermissionError as e:
            print(f"Error: Cannot write to {self.running_file}: {e}")
            print("Try running: sudo chmod 777 /tmp/gpu_queue")
            raise
    
    def submit_job(self, command, gpu_count=1, memory_gb=None, max_time_hours=None, email=None):
        """Submit a job to the queue"""
        jobs = self.load_jobs()
        
        # Detect and capture virtual environment
        venv_info = self.detect_virtual_environment()
        
        job = {
            'id': len(jobs) + int(time.time()),
            'user': os.getenv('USER', 'unknown'),
            'command': command,
            'working_directory': os.getcwd(),  # Capture current working directory
            'virtual_env': venv_info,  # Store venv information
            'gpu_count': gpu_count,
            'memory_gb': memory_gb or self.config['max_memory_per_gpu_gb'],
            'max_time_hours': max_time_hours or self.config['max_job_time_hours'],
            'submitted_at': datetime.now().isoformat(),
            'email': email or self.config['user_emails'].get(os.getenv('USER', ''), ''),
            'status': 'queued'
        }
        
        jobs.append(job)
        self.save_jobs(jobs)
        
        print(f"Job {job['id']} submitted to queue")
        print(f"Working directory: {job['working_directory']}")
        if venv_info:
            print(f"Virtual environment: {venv_info['name']} ({venv_info['path']})")
        else:
            print("Virtual environment: None (using system Python)")
        return job['id']
    
    def detect_virtual_environment(self):
        """Detect the currently active virtual environment"""
        venv_info = None
        
        # Check for conda environment
        conda_env = os.getenv('CONDA_DEFAULT_ENV')
        conda_prefix = os.getenv('CONDA_PREFIX')
        if conda_env and conda_prefix:
            venv_info = {
                'type': 'conda',
                'name': conda_env,
                'path': conda_prefix,
                'python': os.path.join(conda_prefix, 'bin', 'python'),
                'activate_cmd': f'source {conda_prefix}/etc/profile.d/conda.sh && conda activate {conda_env}'
            }
        
        # Check for virtualenv/venv
        elif os.getenv('VIRTUAL_ENV'):
            venv_path = os.getenv('VIRTUAL_ENV')
            venv_name = os.path.basename(venv_path)
            venv_info = {
                'type': 'venv',
                'name': venv_name,
                'path': venv_path,
                'python': os.path.join(venv_path, 'bin', 'python'),
                'activate_cmd': f'source {venv_path}/bin/activate'
            }
        
        # Check for pyenv
        elif os.getenv('PYENV_VERSION') or os.getenv('PYENV_ROOT'):
            pyenv_version = os.getenv('PYENV_VERSION', 'system')
            pyenv_root = os.getenv('PYENV_ROOT', os.path.expanduser('~/.pyenv'))
            venv_info = {
                'type': 'pyenv',
                'name': pyenv_version,
                'path': pyenv_root,
                'python': f'{pyenv_root}/versions/{pyenv_version}/bin/python',
                'activate_cmd': f'export PYENV_VERSION={pyenv_version}'
            }
        
        # Check for pipenv
        elif os.getenv('PIPENV_ACTIVE'):
            # Find pipenv virtual env
            try:
                result = subprocess.run(['pipenv', '--venv'], capture_output=True, text=True, check=True)
                venv_path = result.stdout.strip()
                venv_info = {
                    'type': 'pipenv',
                    'name': 'pipenv',
                    'path': venv_path,
                    'python': os.path.join(venv_path, 'bin', 'python'),
                    'activate_cmd': 'pipenv shell'
                }
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        return venv_info
    
    def can_run_job(self, job, gpus):
        """Check if a job can run given current GPU state"""
        available_gpus = []
        for gpu in gpus:
            memory_free = gpu['memory_total'] - gpu['memory_used']
            if (memory_free >= job['memory_gb'] * 1024 and 
                gpu['utilization'] < 10):  # Less than 10% utilization
                available_gpus.append(gpu['index'])
        
        return len(available_gpus) >= job['gpu_count']
    
    def run_job(self, job):
        """Start running a job as the submitting user."""
        gpus = self.get_gpu_info()
        if not self.can_run_job(job, gpus):
            return False

        # Find available GPUs
        available_gpus = []
        for gpu in gpus:
            memory_free = gpu['memory_total'] - gpu['memory_used']
            if (memory_free >= job['memory_gb'] * 1024 and 
                gpu['utilization'] < 10):
                available_gpus.append(str(gpu['index']))
                if len(available_gpus) >= job['gpu_count']:
                    break
        
        gpu_list = ','.join(available_gpus[:job['gpu_count']])
        
        # Get the user who submitted the job
        user = job.get('user')
        if not user or user == 'root':
            print(f"Error: Job {job['id']} submitted by invalid user '{user}'. Cannot run.")
            return False

        # Create log directory and ensure it's writable by all users
        log_dir = self.queue_dir / "logs"
        log_dir.mkdir(mode=0o777, exist_ok=True)
        # Explicitly set permissions in case it already existed with different ones
        # os.chmod(log_dir, 0o777)
        
        stdout_log = log_dir / f"job_{job['id']}_stdout.log"
        stderr_log = log_dir / f"job_{job['id']}_stderr.log"

        # Use an empty environment dict, as we'll inject variables into the shell command
        base_env = {} 
        job_command, _ = self.setup_job_environment(job, job["command"], base_env)
        
        # Safely quote paths for shell execution
        quoted_wd = shlex.quote(job.get('working_directory', f'/home/{user}'))
        quoted_stdout_log = shlex.quote(str(stdout_log))
        quoted_stderr_log = shlex.quote(str(stderr_log))

        out = open(quoted_stdout_log, "ab")
        err = open(quoted_stderr_log, "ab")
        
        # Construct the final command to be run inside the user's login shell.
        # This sets the environment, changes directory, and executes the job with logging.
        # The user's shell will create the log files, ensuring correct ownership.
        # final_shell_command = (
        #     f"export CUDA_VISIBLE_DEVICES={gpu_list}; "
        #     f"cd {quoted_wd} && exec "
        #     f"({job_command}) > {quoted_stdout_log} 2> {quoted_stderr_log}"
        # )
        final_shell_command = [f"CUDA_VISIBLE_DEVICES={gpu_list}"] + job_command.split()

        # The command for Popen uses `su` to switch to the correct user.
        # popen_command = ['su', '-', user, '-c', final_shell_command]
        popen_command = ['sudo', '-u', user] + final_shell_command
        
        try:
            # Start the job as the specified user.
            # The process is detached from the daemon's stdio.
            process = subprocess.Popen(
                popen_command,
                shell=False, # We are passing a list of args, not a shell string
                cwd=quoted_wd,
                #stdout=subprocess.DEVNULL,
                #stderr=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                start_new_session=True  # 👈 each Popen gets its own PGID
            )
            
            # Update job status
            job['status'] = 'running'
            job['started_at'] = datetime.now().isoformat()
            job['pid'] = process.pid
            job['gpus'] = gpu_list
            job['stdout_log'] = str(stdout_log)
            job['stderr_log'] = str(stderr_log)
            job['actual_command'] = final_shell_command # Store the full command for debugging
            
            # Add to running jobs
            running_jobs = self.load_running_jobs()
            running_jobs.append(job)
            self.save_running_jobs(running_jobs)
            
            print(f"Started job {job['id']} as user '{user}' on GPUs {gpu_list}")
            return True
            
        except Exception as e:
            print(f"Failed to start job {job['id']} as user '{user}': {e}")
            return False
    
    def setup_job_environment(self, job, command, env):
        """Setup the job environment including virtual environment"""
        venv_info = job.get('virtual_env')
        
        if venv_info:
            venv_type = venv_info['type']
            
            if venv_type == 'conda':
                # For conda environments
                if 'CONDA_PREFIX' in env:
                    del env['CONDA_PREFIX']  # Clear existing conda env
                if 'CONDA_DEFAULT_ENV' in env:
                    del env['CONDA_DEFAULT_ENV']
                
                # Use conda's python directly
                python_path = venv_info['python']
                if os.path.exists(python_path):
                    # Replace python/python3 in command with full conda python path
                    command = self.replace_python_command(command, python_path)
                else:
                    # Fallback: create activation command
                    command = f"{venv_info['activate_cmd']} && {command}"
                
                # Set conda environment variables
                env['CONDA_PREFIX'] = venv_info['path']
                env['CONDA_DEFAULT_ENV'] = venv_info['name']
                env['PATH'] = f"{venv_info['path']}/bin:" + env.get('PATH', '')
            
            elif venv_type in ['venv', 'virtualenv']:
                # For virtualenv/venv environments
                python_path = venv_info['python']
                if os.path.exists(python_path):
                    command = self.replace_python_command(command, python_path)
                    env['VIRTUAL_ENV'] = venv_info['path']
                    env['PATH'] = f"{venv_info['path']}/bin:" + env.get('PATH', '')
                    # Remove PYTHONHOME if set (can interfere with venv)
                    env.pop('PYTHONHOME', None)
                else:
                    command = f"{venv_info['activate_cmd']} && {command}"
            
            elif venv_type == 'pyenv':
                # For pyenv environments
                env['PYENV_VERSION'] = venv_info['name']
                if 'PYENV_ROOT' not in env:
                    env['PYENV_ROOT'] = venv_info['path']
                python_path = venv_info['python']
                if os.path.exists(python_path):
                    command = self.replace_python_command(command, python_path)
            
            elif venv_type == 'pipenv':
                # For pipenv environments
                python_path = venv_info['python']
                if os.path.exists(python_path):
                    command = self.replace_python_command(command, python_path)
                    env['VIRTUAL_ENV'] = venv_info['path']
                    env['PATH'] = f"{venv_info['path']}/bin:" + env.get('PATH', '')
                else:
                    # Use pipenv run instead
                    command = f"pipenv run {command}"
        
        else:
            # No virtual environment - ensure we use python3
            command = self.replace_python_command(command, 'python3')
        
        return command, env
    
    def replace_python_command(self, command, python_executable):
        """Replace python/python3 in command with specified executable"""
        # Handle different python command patterns
        if command.startswith('python3 '):
            return command.replace('python3 ', f'{python_executable} ', 1)
        elif command.startswith('python '):
            return command.replace('python ', f'{python_executable} ', 1)
        elif command.startswith('python'):
            parts = command.split()
            if len(parts) > 0 and parts[0] in ['python', 'python3']:
                parts[0] = python_executable
                return ' '.join(parts)
        
        return command
    
    def check_running_jobs(self):
        """Check status of running jobs and clean up finished ones"""
        running_jobs = self.load_running_jobs()
        still_running = []
        
        for job in running_jobs:
            try:
                # Check if process is still running
                subprocess.run(['kill', '-0', str(job['pid'])], 
                             check=True, capture_output=True)
                
                # Check for timeout
                started = datetime.fromisoformat(job['started_at'])
                if datetime.now() - started > timedelta(hours=job['max_time_hours']):
                    print(f"Job {job['id']} timed out, killing...")
                    subprocess.run(['kill', str(job['pid'])], capture_output=True)
                    self.send_notification(job, "timeout")
                else:
                    still_running.append(job)
                    
            except subprocess.CalledProcessError:
                # Process finished
                print(f"Job {job['id']} finished")
                self.send_notification(job, "completed")
        
        self.save_running_jobs(still_running)
        return still_running
    
    def process_queue(self):
        """Process the job queue"""
        jobs = self.load_jobs()
        remaining_jobs = []
        
        for job in jobs:
            if job['status'] == 'queued':
                if self.run_job(job):
                    job['status'] = 'running'
                else:
                    remaining_jobs.append(job)
            else:
                remaining_jobs.append(job)
        
        self.save_jobs(remaining_jobs)

    def process_kill_jobs(self, job=None, request="add"):
        jobs = self.load_kill_jobs()
        assert isinstance(jobs, list), "jobs should be a list!"
        if request=="add":
            assert job is not None, "If using process_kill_jobs with request=add, you need to pass a job!"
            jobs.append(job)
            self.save_kill_jobs(jobs)
        elif request=="kill":
            for job in jobs:
                job_id = str(job["job_id"])
                try:
                    # subprocess.run(["kill", str(job["pid"])])
                    pgid = os.getpgid(job["pid"])
                    # print(f"Killing process group: {pgid}")
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, FileNotFoundError) as e:
                    print(f"Problems in paradise for job {job_id}")
                    print(e)
                    continue
                    
                to_email = self.config["user_emails"][job["requested_by"]]
                subject = f"Killed job {job_id}"

                message = f"Job {job_id} has been killed by user {job['requested_by']}"
                
                # print(to_email)
                print(subject)
                print(message)
                
                self.send_email(to_email, subject, message)
                self.send_slack(message)
            self.save_kill_jobs([])
        else:
            raise NotImplementedError()
    
    def send_notification(self, job, event_type):
        """Send notification about job events"""
        message = f"Job {job['id']} by {job['user']} has {event_type}"
        
        # Email notification
        # print(job['email'])
        # print(self.config['notification_email']['enabled'])
        if self.config['notification_email']['enabled'] and job.get('email'):
            self.send_email(job['email'], f"GPU Job {event_type.title()}", message)
        
        # Slack notification
        if self.config['slack']['enabled']:
            self.send_slack(message)
    
    def send_email(self, to_email, subject, message):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['notification_email']['username']
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.config['notification_email']['smtp_server'], 
                                self.config['notification_email']['smtp_port'])
            server.starttls()
            server.login(self.config['notification_email']['username'], 
                        self.config['notification_email']['password'])
            server.sendmail(msg['From'], msg['To'], msg.as_string())
            server.quit()
        except Exception as e:
            print(f"Failed to send email: {e} {msg['From']} {msg['To']} {self.config['notification_email']['smtp_server']} {self.config['notification_email']['smtp_port']}")
    
    def send_slack(self, message):
        """Send Slack notification"""
        try:
            payload = {
                'text': message,
                'channel': self.config['slack']['channel']
            }
            requests.post(self.config['slack']['webhook_url'], json=payload)
        except Exception as e:
            print(f"Failed to send Slack message: {e}")
    
    def check_resource_hogs(self):
        """Improved resource hog detection with accurate GPU mapping and throttling"""
        gpus = self.get_gpu_info()
        processes = self.get_gpu_processes()
        gpu_uuid_mapping = self.get_gpu_to_uuid_mapping()
        
        # Track resource usage by user with accurate GPU mapping
        user_usage = {}
        
        # First pass: map processes to GPUs and users accurately
        for proc in processes:
            user = self.get_process_user(proc['pid'])
            if not user:
                continue
            
            # Get actual GPU index from UUID
            gpu_index = gpu_uuid_mapping.get(proc['gpu_uuid'])
            if gpu_index is None:
                continue
            
            # Initialize user data
            if user not in user_usage:
                user_usage[user] = {
                    'gpus': set(),
                    'total_memory_gb': 0,
                    'gpu_details': {},
                    'process_count': 0
                }
            
            # Only count as "using" GPU if significant memory usage
            memory_gb = proc['memory'] / 1024
            if memory_gb >= self.memory_threshold_gb:
                user_usage[user]['gpus'].add(gpu_index)
                
                if gpu_index not in user_usage[user]['gpu_details']:
                    user_usage[user]['gpu_details'][gpu_index] = {
                        'memory_gb': 0,
                        'processes': []
                    }
                
                user_usage[user]['gpu_details'][gpu_index]['memory_gb'] += memory_gb
                user_usage[user]['gpu_details'][gpu_index]['processes'].append({
                    'pid': proc['pid'],
                    'name': proc['name'],
                    'memory_gb': memory_gb
                })
            
            user_usage[user]['total_memory_gb'] += memory_gb
            user_usage[user]['process_count'] += 1
        
        # Second pass: check GPU utilization to catch compute-heavy users
        for gpu in gpus:
            if gpu['utilization'] >= self.utilization_threshold:
                # Find processes on this GPU
                gpu_processes = [p for p in processes 
                               if gpu_uuid_mapping.get(p['gpu_uuid']) == gpu['index']]
                
                for proc in gpu_processes:
                    user = self.get_process_user(proc['pid'])
                    if user:
                        if user not in user_usage:
                            user_usage[user] = {
                                'gpus': set(),
                                'total_memory_gb': 0,
                                'gpu_details': {},
                                'process_count': 0
                            }
                        
                        user_usage[user]['gpus'].add(gpu['index'])
                        
                        if gpu['index'] not in user_usage[user]['gpu_details']:
                            user_usage[user]['gpu_details'][gpu['index']] = {
                                'memory_gb': proc['memory'] / 1024,
                                'utilization': gpu['utilization'],
                                'processes': []
                            }
                        else:
                            user_usage[user]['gpu_details'][gpu['index']]['utilization'] = gpu['utilization']
        
        # Identify resource hogs
        resource_hogs = []
        total_gpus = len(gpus)
        
        for user, usage in user_usage.items():
            gpu_count = len(usage['gpus'])
            total_memory = usage['total_memory_gb']
            
            # Define resource hog criteria
            is_hog = False
            reasons = []
            
            # Using too many GPUs (≥50% of available or ≥3 GPUs)
            if gpu_count >= max(2, total_gpus // 2) or gpu_count >= 3:
                is_hog = True
                reasons.append(f"using {gpu_count}/{total_gpus} GPUs")
            
            # Using excessive total memory (>80GB across all GPUs)
            if total_memory > 80:
                is_hog = True
                reasons.append(f"using {total_memory:.1f}GB total memory")
            
            # Using high memory on multiple GPUs simultaneously
            high_memory_gpus = sum(1 for details in usage['gpu_details'].values() 
                                 if details['memory_gb'] > 30)
            if high_memory_gpus >= 2:
                is_hog = True
                reasons.append(f"using >30GB on {high_memory_gpus} GPUs")
            
            if is_hog:
                # Build detailed GPU info
                gpu_details = []
                for gpu_idx in sorted(usage['gpus']):
                    details = usage['gpu_details'][gpu_idx]
                    memory = details['memory_gb']
                    util = details.get('utilization', 0)
                    proc_count = len(details['processes'])
                    gpu_details.append(f"GPU{gpu_idx}({memory:.1f}GB,{util}%,{proc_count}proc)")
                
                resource_hogs.append({
                    'user': user,
                    'gpu_count': gpu_count,
                    'gpu_list': sorted(usage['gpus']),
                    'gpu_details': gpu_details,
                    'total_memory_gb': total_memory,
                    'reasons': reasons
                })
        
        # Check if we should send notifications (throttling)
        should_notify, last_reported = self.should_send_resource_notification()
        current_hog_users = [hog['user'] for hog in resource_hogs]
        
        # Only notify if enough time has passed AND there are changes
        if (should_notify and resource_hogs and 
            set(current_hog_users) != set(last_reported)):
            
            self.send_resource_hog_notification(resource_hogs, total_gpus)
            self.update_last_resource_notification(current_hog_users)
        
        # Always print current status for console monitoring
        if resource_hogs:
            print(f"\n⚠️ Resource usage alert - {len(resource_hogs)} users using excessive resources:")
            for hog in resource_hogs:
                gpu_info = ", ".join(hog['gpu_details'])
                reasons = " and ".join(hog['reasons'])
                print(f"  👤 {hog['user']}: {reasons}")
                print(f"     Details: {gpu_info}")
        
        return resource_hogs
    
    def send_resource_hog_notification(self, resource_hogs, total_gpus):
        """Send consolidated notification about all resource hogs"""
        if not resource_hogs:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build comprehensive Slack message
        message_lines = [
            f"🚨 **GPU Resource Usage Alert** - {timestamp}",
            f"📊 **{len(resource_hogs)} users** are consuming excessive GPU resources:",
            ""
        ]
        
        for hog in resource_hogs:
            gpu_list = ", ".join([f"GPU{i}" for i in hog['gpu_list']])
            message_lines.extend([
                f"👤 **{hog['user']}**:",
                f"   • GPUs: {gpu_list} ({hog['gpu_count']}/{total_gpus} total)",
                f"   • Memory: {hog['total_memory_gb']:.1f}GB total",
                f"   • Issue: {' and '.join(hog['reasons'])}",
                ""
            ])
        
        # Add summary and recommendations
        total_hogged_gpus = len(set([gpu for hog in resource_hogs for gpu in hog['gpu_list']]))
        message_lines.extend([
            f"📈 **Impact**: {total_hogged_gpus}/{total_gpus} GPUs in heavy use",
            "💡 **Action needed**: Please check if long-running jobs can be optimized or paused",
            f"🔄 Next check in {self.notification_cooldown_minutes} minutes"
        ])
        
        message = "\n".join(message_lines)
        
        # Send to Slack
        if self.config.get('slack', {}).get('enabled'):
            try:
                self.send_slack(message)
                print(f"📤 Sent resource alert to Slack for {len(resource_hogs)} users")
            except Exception as e:
                print(f"Failed to send Slack notification: {e}")
        
        # Send to admin email
        email_config = self.config.get('notification_email', {})
        if email_config.get('enabled') and email_config.get('admin_email'):
            try:
                subject = f"GPU Resource Alert - {len(resource_hogs)} users using excessive resources"
                self.send_email(email_config['admin_email'], subject, message)
                print(f"📤 Sent resource alert email to admin")
            except Exception as e:
                print(f"Failed to send admin email: {e}")
    
    def status(self):
        """Show current system status with enhanced user breakdown"""
        print("=== GPU Queue Status ===")
        
        # GPU status
        gpus = self.get_gpu_info()
        print(f"\nGPUs ({len(gpus)} total):")
        for gpu in gpus:
            memory_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
            status = "BUSY" if gpu['utilization'] > 10 or memory_pct > 10 else "FREE"
            print(f"  GPU {gpu['index']}: {gpu['name']} - {status}")
            print(f"    Memory: {gpu['memory_used']}/{gpu['memory_total']}MB ({memory_pct:.1f}%)")
            print(f"    Utilization: {gpu['utilization']}%")
        
        # Running jobs
        running_jobs = self.load_running_jobs()
        print(f"\nRunning Jobs ({len(running_jobs)}):")
        for job in running_jobs:
            runtime = datetime.now() - datetime.fromisoformat(job['started_at'])
            print(f"  Job {job['id']} by {job['user']} - GPUs: {job['gpus']} - Runtime: {runtime}")
        
        # Queued jobs
        jobs = [j for j in self.load_jobs() if j['status'] == 'queued']
        print(f"\nQueued Jobs ({len(jobs)}):")
        for job in jobs:
            wait_time = datetime.now() - datetime.fromisoformat(job['submitted_at'])
            print(f"  Job {job['id']} by {job['user']} - Waiting: {wait_time}")
        
        # Enhanced user breakdown
        self.show_user_gpu_breakdown()
    
    def show_user_gpu_breakdown(self):
        """Show detailed breakdown of GPU usage by user"""
        processes = self.get_gpu_processes()
        if not processes:
            print(f"\nUser GPU Usage: No active processes")
            return
        
        gpus = self.get_gpu_info()
        gpu_uuid_mapping = self.get_gpu_to_uuid_mapping()
        
        # Build user usage summary
        user_gpu_usage = {}
        
        for proc in processes:
            user = self.get_process_user(proc['pid'])
            if not user:
                continue
                
            gpu_index = gpu_uuid_mapping.get(proc['gpu_uuid'])
            if gpu_index is None:
                continue
            
            if user not in user_gpu_usage:
                user_gpu_usage[user] = {}
            
            if gpu_index not in user_gpu_usage[user]:
                user_gpu_usage[user][gpu_index] = {
                    'memory_gb': 0,
                    'processes': []
                }
            
            memory_gb = proc['memory'] / 1024
            user_gpu_usage[user][gpu_index]['memory_gb'] += memory_gb
            user_gpu_usage[user][gpu_index]['processes'].append({
                'name': proc['name'],
                'memory_gb': memory_gb
            })
        
        # Display user breakdown
        print(f"\nUser GPU Usage:")
        if not user_gpu_usage:
            print("  No significant GPU usage detected")
            return
        
        for user in sorted(user_gpu_usage.keys()):
            gpus_used = user_gpu_usage[user]
            gpu_list = ", ".join([f"GPU{idx}" for idx in sorted(gpus_used.keys())])
            total_memory = sum(details['memory_gb'] for details in gpus_used.values())
            
            print(f"  👤 {user}: {gpu_list} ({total_memory:.1f}GB total)")
            
            for gpu_idx in sorted(gpus_used.keys()):
                details = gpus_used[gpu_idx]
                if details['memory_gb'] >= self.memory_threshold_gb:
                    proc_names = [p['name'] for p in details['processes'][:3]]  # Show first 3
                    more_text = f" +{len(details['processes'])-3} more" if len(details['processes']) > 3 else ""
                    print(f"     GPU{gpu_idx}: {details['memory_gb']:.1f}GB - {', '.join(proc_names)}{more_text}")

def main():
    parser = argparse.ArgumentParser(description='GPU Queue Management System')
    parser.add_argument('action', choices=['submit', 'status', 'daemon', 'config', 'kill'],
                       help='Action to perform')
    parser.add_argument('--command', help='Command to run (for submit)')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs needed')
    parser.add_argument('--memory', type=int, help='Memory per GPU in GB')
    parser.add_argument('--time', type=int, help='Max time in hours')
    parser.add_argument('--email', help='Email for notifications')
    parser.add_argument('--job-id', type=int, help='Job ID to kill')
    
    args = parser.parse_args()
    manager = GPUQueueManager()
    
    if args.action == 'submit':
        if not args.command:
            print("Error: --command is required for submit")
            sys.exit(1)
        job_id = manager.submit_job(args.command, args.gpus, args.memory, args.time, args.email)
        print(f"Job submitted with ID: {job_id}")
        
    elif args.action == 'status':
        manager.status()
        
    elif args.action == 'daemon':
        print("Starting GPU queue daemon...")
        while True:
            manager.process_kill_jobs(request="kill")
            manager.check_running_jobs()
            manager.process_queue()
            manager.check_resource_hogs()
            time.sleep(30)  # Check every 30 seconds
            
    elif args.action == 'config':
        print("Configuration file:", manager.config_file)
        print("Edit this file to configure email/Slack notifications")
        
    elif args.action == 'kill':
        if not args.job_id:
            print("Error: --job-id is required for kill")
            sys.exit(1)

        # Delete job from running queue
        running_jobs = manager.load_running_jobs()
        new_running_jobs = []
        for job in running_jobs:
            if job['id'] == args.job_id:
                # Write kill request to a file that the daemon monitors
                kill_request = {
                    'job_id': args.job_id,
                    'pid': job["pid"],
                    'requested_by': os.getenv('USER'),
                    'timestamp': time.time()
                }
                
                manager.process_kill_jobs(job=kill_request, request="add")
                print(f"Killed job {args.job_id}")
            else:
                new_running_jobs.append(job)
        manager.save_running_jobs(new_running_jobs)


        # Delete job from queue, no need to kill any process
        queue_jobs = manager.load_jobs()
        new_jobs = []
        for job in queue_jobs:
            if job['id'] == args.job_id:
                print(f"Killed job {args.job_id}")
            else:
                new_jobs.append(job)
        manager.save_jobs(new_jobs)

if __name__ == '__main__':
    main()
