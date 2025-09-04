# GPU Queue System User Guide

Comprehensive guide to using the Ruqola server's custom GPU queue management system for efficient resource sharing.

## 📖 Table of Contents

1. [Overview](#overview)
2. [Basic Usage](#basic-usage)
3. [Job Submission](#job-submission)
4. [Monitoring and Management](#monitoring-and-management)
5. [Advanced Usage](#advanced-usage)
6. [Best Practices](#best-practices)
7. [Common Workflows](#common-workflows)
8. [Troubleshooting](#troubleshooting)

## Overview

### What is the GPU Queue System?

Our custom GPU queue system (`gpuq`) manages fair access to the server's 3 H200 GPUs, ensuring:

- **Fair resource allocation** - First-come, first-served job scheduling
- **Resource limits** - Prevents users from monopolizing GPUs
- **Automatic cleanup** - Jobs are terminated after time limits
- **Usage monitoring** - Track who's using what resources
- **Notifications** - Email/Slack alerts for job completion and issues

### Key Features

- **80GB memory per H200 GPU** - Massive memory for large models
- **Time-limited jobs** - Default 24-hour maximum runtime
- **Queue management** - Jobs wait for available resources
- **Resource monitoring** - Real-time GPU usage tracking
- **Flexible submission** - Support for various workload types

## Basic Usage

### Quick Start Commands

```bash
# Check current GPU status and queue
gpuq status

# Submit a simple training job
gpuq submit --command "python train.py"

# Submit with specific requirements
gpuq submit --command "python large_model.py" --gpus 2 --memory 40 --time 12

# Check your jobs
gpuq status | grep $USER

# Kill a specific job
gpuq kill --job-id 12345
```

### First Time Setup

1. **Check system access**:
```bash
gpuq status
nvidia-smi
```

2. **Test job submission**:
```bash
gpuq submit --command "python -c 'print(\"Hello GPU!\")'" --time 1
```

3. **Monitor job progress**:
```bash
watch -n 5 gpuq status
```

## Job Submission

### Basic Job Submission

```bash
# Minimal submission (uses defaults)
gpuq submit --command "python train.py"

# Specify all parameters
gpuq submit \
  --command "python train.py --epochs 100 --batch-size 32" \
  --gpus 1 \
  --memory 40 \
  --time 8 \
  --email "your-email@example.com"
```

### Resource Specification

#### GPU Requirements

```bash
# Single GPU (default)
gpuq submit --command "python train.py" --gpus 1

# Multi-GPU training
gpuq submit --command "python -m torch.distributed.launch train.py" --gpus 2

# All available GPUs
gpuq submit --command "python multi_gpu_train.py" --gpus 3
```

#### Memory Requirements

```bash
# Specify memory per GPU (in GB)
gpuq submit --command "python big_model.py" --memory 60

# For memory-intensive models
gpuq submit --command "python huge_model.py" --memory 75

# Conservative memory usage
gpuq submit --command "python small_model.py" --memory 20
```

#### Time Limits

```bash
# Short experiments (1 hour)
gpuq submit --command "python quick_test.py" --time 1

# Medium training (8 hours)
gpuq submit --command "python train.py" --time 8

# Long training (24 hours - maximum)
gpuq submit --command "python long_train.py" --time 24
```

### Command Examples

#### Training Scripts

```bash
# PyTorch training
gpuq submit --command "python train.py --model resnet50 --epochs 100" --gpus 1 --memory 30 --time 12

# TensorFlow training
gpuq submit --command "python tf_train.py --model_dir ./models" --gpus 1 --memory 25 --time 8

# Distributed training
gpuq submit --command "torchrun --nproc_per_node=2 distributed_train.py" --gpus 2 --memory 40 --time 16
```

#### Jupyter Notebooks

```bash
# Start Jupyter on port 8888
gpuq submit --command "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser" --gpus 1 --time 8

# JupyterLab with custom port
gpuq submit --command "jupyter lab --ip=0.0.0.0 --port=9999 --no-browser" --gpus 1 --memory 30 --time 4

# Jupyter with specific working directory
gpuq submit --command "cd /path/to/project && jupyter notebook --ip=0.0.0.0 --port=8888" --gpus 1 --time 6
```

#### Data Processing

```bash
# Large dataset preprocessing
gpuq submit --command "python preprocess_data.py --dataset imagenet" --gpus 1 --memory 50 --time 4

# Feature extraction
gpuq submit --command "python extract_features.py --model vit_large" --gpus 1 --memory 35 --time 3
```

## Monitoring and Management

### Checking Job Status

```bash
# Overall system status
gpuq status

# Detailed status with job information
gpuq status --detailed

# Monitor in real-time
watch -n 5 gpuq status

# Check only your jobs
gpuq status | grep $USER
```

### GPU Monitoring

```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# GPU utilization and memory
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

# Continuous monitoring with better formatting
watch -n 2 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

### Job Logs

```bash
# View job output logs
tail -f /tmp/gpu_queue/logs/job_12345_stdout.log

# View error logs
tail -f /tmp/gpu_queue/logs/job_12345_stderr.log

# Search for specific patterns in logs
grep -i "error\|warning" /tmp/gpu_queue/logs/job_12345_stderr.log
```

### Managing Jobs

```bash
# Kill a specific job
gpuq kill --job-id 12345

# Kill all your jobs (be careful!)
gpuq status | grep $USER | awk '{print $1}' | xargs -I {} gpuq kill --job-id {}

# Check if job completed successfully
echo $?  # after job completion, 0 = success
```

## Advanced Usage

### Environment Variables

```bash
# Set CUDA devices within job
gpuq submit --command "CUDA_VISIBLE_DEVICES=0 python train.py"

# Use specific conda environment
gpuq submit --command "conda activate myenv && python train.py"

# Set multiple environment variables
gpuq submit --command "export PYTHONPATH=/path/to/modules && python train.py"
```

### Complex Commands

```bash
# Chain multiple commands
gpuq submit --command "cd /path/to/project && python preprocess.py && python train.py"

# Conditional execution
gpuq submit --command "python train.py && python evaluate.py || echo 'Training failed'"

# Background processes within job
gpuq submit --command "python train.py > output.log 2>&1 &"
```

### Resource Optimization

```bash
# Memory-efficient training with gradient checkpointing
gpuq submit --command "python train.py --gradient-checkpointing --batch-size 16" --memory 35

# Mixed precision training
gpuq submit --command "python train.py --fp16 --batch-size 64" --memory 25

# Model parallelism
gpuq submit --command "python train.py --model-parallel" --gpus 2 --memory 60
```

### Interactive Jobs

```bash
# Interactive Python session
gpuq submit --command "python -i" --gpus 1 --time 2

# Interactive shell with GPU access
gpuq submit --command "bash" --gpus 1 --time 1

# Remote development session
gpuq submit --command "code-server --bind-addr 0.0.0.0:8080" --gpus 1 --time 8
```

## Best Practices

### Resource Management

1. **Request only what you need**:
   ```bash
   # Good: Specific requirements
   gpuq submit --command "python train.py" --gpus 1 --memory 30 --time 8
   
   # Bad: Excessive resources
   gpuq submit --command "python train.py" --gpus 3 --memory 75 --time 24
   ```

2. **Use appropriate time limits**:
   - Quick tests: 1-2 hours
   - Medium experiments: 4-8 hours
   - Long training: 12-24 hours

3. **Monitor resource usage**:
   ```bash
   # Check if you're using allocated resources efficiently
   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 10
   ```

### Job Submission Best Practices

1. **Test locally first**:
   ```bash
   # Test with small dataset/short training first
   python train.py --epochs 1 --batch-size 8
   ```

2. **Use absolute paths**:
   ```bash
   # Good
   gpuq submit --command "cd /home/user/project && python train.py"
   
   # Bad (relative paths may not work)
   gpuq submit --command "python ../train.py"
   ```

3. **Specify output directories**:
   ```bash
   gpuq submit --command "python train.py --output-dir /home/user/results/exp1"
   ```

### Code Organization

1. **Use configuration files**:
   ```python
   # config.yaml
   model:
     name: "resnet50"
     batch_size: 32
   training:
     epochs: 100
     lr: 0.001
   ```

2. **Enable checkpointing**:
   ```python
   # Save checkpoints regularly
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
   }, f'checkpoint_epoch_{epoch}.pth')
   ```

3. **Log important metrics**:
   ```python
   import wandb  # or tensorboard, mlflow
   wandb.log({"loss": loss, "accuracy": acc, "epoch": epoch})
   ```

## Common Workflows

### Training a Deep Learning Model

```bash
# 1. Prepare data and code
cd /home/user/myproject
ls -la  # check files are ready

# 2. Test locally with small dataset
python train.py --epochs 1 --batch-size 4 --debug

# 3. Submit full training job
gpuq submit \
  --command "python train.py --epochs 100 --batch-size 32 --save-dir ./models" \
  --gpus 1 \
  --memory 40 \
  --time 12 \
  --email "user@example.com"

# 4. Monitor progress
watch -n 10 gpuq status
tail -f /tmp/gpu_queue/logs/job_XXXXX_stdout.log
```

### Hyperparameter Tuning

```bash
# Submit multiple jobs with different hyperparameters
for lr in 0.001 0.01 0.1; do
  for bs in 16 32 64; do
    gpuq submit \
      --command "python train.py --lr $lr --batch-size $bs --name lr${lr}_bs${bs}" \
      --gpus 1 --memory 30 --time 8
  done
done
```

### Model Inference

```bash
# Batch inference on large dataset
gpuq submit \
  --command "python inference.py --model-path ./best_model.pth --data-dir ./test_data" \
  --gpus 1 \
  --memory 25 \
  --time 4
```

### Interactive Development

```bash
# Start Jupyter for development
gpuq submit \
  --command "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser" \
  --gpus 1 \
  --memory 30 \
  --time 8

# Connect via SSH tunnel (from your local machine)
ssh -L 8888:localhost:8888 user@server.com

# Open http://localhost:8888 in your browser
```

## Troubleshooting

### Common Issues

#### Job Won't Start

```bash
# Check queue status
gpuq status

# Common causes:
# 1. All GPUs busy - wait or reduce resource requirements
# 2. Requesting more memory than available (max ~75GB per H200)
# 3. Syntax error in command

# Debugging
gpuq submit --command "echo 'Test job'" --gpus 1 --time 1
```

#### Job Killed Unexpectedly

```bash
# Check job logs for errors
tail -100 /tmp/gpu_queue/logs/job_XXXXX_stderr.log

# Common causes:
# 1. Out of memory - reduce batch size or model size
# 2. Time limit exceeded - increase time limit
# 3. Code error - check stderr logs
```

#### GPU Out of Memory

```bash
# Check current GPU memory usage
nvidia-smi

# Solutions:
# 1. Reduce batch size
# 2. Use gradient accumulation
# 3. Enable gradient checkpointing
# 4. Use mixed precision (fp16)

# Example with memory optimization
gpuq submit \
  --command "python train.py --batch-size 16 --gradient-checkpointing --fp16" \
  --gpus 1 --memory 30
```

#### Can't Access Job Output

```bash
# Check if job is still running
gpuq status | grep job_id

# Find log files
ls -la /tmp/gpu_queue/logs/job_*

# Check permissions
ls -la /tmp/gpu_queue/logs/job_XXXXX_*.log
```

### Performance Issues

#### Slow Training

```bash
# Check GPU utilization
nvidia-smi -l 1

# If utilization is low:
# 1. Increase batch size
# 2. Check data loading (use more workers)
# 3. Profile your code

# Example with optimized data loading
gpuq submit \
  --command "python train.py --num-workers 8 --batch-size 64" \
  --gpus 1
```

#### Memory Leaks

```bash
# Monitor memory usage over time
nvidia-smi --query-gpu=memory.used --format=csv -l 60 > memory_usage.log

# In Python code, use memory profiling
pip install memory-profiler
python -m memory_profiler train.py
```

### Getting Help

1. **Check system status**: `gpuq status`
2. **Review logs**: `/tmp/gpu_queue/logs/`
3. **Test with simple commands**: `gpuq submit --command "nvidia-smi"`
4. **Contact administrator** with:
   - Job ID
   - Command used
   - Error logs
   - Expected vs actual behavior

---

**Next Steps**: Once you're comfortable with the queue system, check out the framework-specific guides:
- [PyTorch with H200](pytorch-guide.md)
- [TensorFlow with H200](tensorflow-guide.md)
- [JAX with H200](jax-guide.md)