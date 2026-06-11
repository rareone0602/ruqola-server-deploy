# Best Practices and Troubleshooting Guide

Comprehensive guide for efficient and respectful use of the Ruqola server's H200 NVL GPUs, including common issues and solutions.

## 📖 Table of Contents

1. [Resource Management Best Practices](#resource-management-best-practices)
2. [Server Etiquette](#server-etiquette)
3. [Performance Optimization](#performance-optimization)
4. [Memory Management](#memory-management)
5. [Common Issues and Solutions](#common-issues-and-solutions)
6. [Framework-Specific Tips](#framework-specific-tips)
7. [Monitoring and Debugging](#monitoring-and-debugging)
8. [Emergency Procedures](#emergency-procedures)

## Resource Management Best Practices

### GPU Allocation Strategy

The server has **4 NVIDIA H200 NVL GPUs** (indices 0,1,2,3), each with ~141 GB of VRAM.
Under gpuq's "you own your allocated GPU" model, request the number of GPUs your job
actually needs — `gpuq` prefers handing you free cards and will stack onto cards you
already hold. Just don't camp on GPUs you aren't using.

```bash
# ✅ GOOD: Request what your job needs
gpuq submit --command "python train.py" --gpus 1 --memory 40 --time 8

# ✅ ALSO FINE: A multi-GPU job that genuinely uses all 4 cards
gpuq submit --command "torchrun --nproc_per_node=4 train.py" --gpus 4 --time 12

# ❌ BAD: Holding GPUs you don't need, then leaving them idle
#         (or queueing duplicate jobs you'll never look at)
```

> **What `--memory N` means:** it is the *minimum free VRAM* a candidate GPU must
> have for gpuq to select it (an admission gate), **not** a reservation the job is
> held to. On a ~141 GB H200 NVL, even `--memory 120` is satisfiable on a free card —
> it just narrows which cards qualify; it does not cap your job at 120 GB.

### Time Management

1. **Estimate conservatively, then add buffer**:
   ```bash
   # If your model takes ~6 hours, request 8 hours
   gpuq submit --command "python train.py" --time 8
   ```

2. **Use checkpoints for long training**:
   ```python
   # Save checkpoints every epoch or few hours
   if epoch % save_interval == 0:
       torch.save({
           'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           'loss': loss,
       }, f'checkpoint_epoch_{epoch}.pth')
   ```

3. **Test with small datasets first**:
   ```bash
   # Test with subset before full training
   gpuq submit --command "python train.py --debug --epochs 1 --batch-size 4" --time 1
   ```

### Memory Planning

Each H200 NVL has ~141 GB of VRAM, so a single card can hold substantially larger
models than an 80 GB H100/A100. Use the table below as a rough starting point, then
verify with the estimation formula and a real profiling run.

| Model Size | Estimated H200 NVL Memory | Batch Size Recommendation |
|------------|---------------------------|---------------------------|
| **Small (< 100M params)** | 5-15 GB | 64-256 |
| **Medium (100M-1B params)** | 15-40 GB | 16-64 |
| **Large (1B-10B params)** | 40-90 GB | 4-16 |
| **Very Large (10B+ params)** | 90-140 GB | 1-4 |

```python
# Memory estimation formula (rough)
def estimate_memory_usage(params, batch_size, sequence_length=512):
    """Estimate GPU memory usage in GB"""
    # Model parameters (FP16)
    model_memory = params * 2 / 1024**3
    
    # Gradients (FP32)
    gradient_memory = params * 4 / 1024**3
    
    # Optimizer states (Adam: 8 bytes per param)
    optimizer_memory = params * 8 / 1024**3
    
    # Activations (estimated)
    activation_memory = batch_size * sequence_length * 1024 * 4 / 1024**3
    
    total = model_memory + gradient_memory + optimizer_memory + activation_memory
    return total * 1.2  # Add 20% buffer

# Example usage
memory_needed = estimate_memory_usage(1.5e9, batch_size=16)  # 1.5B parameter model
print(f"Estimated memory: {memory_needed:.1f} GB")
```

## Server Etiquette

### Do's and Don'ts

#### ✅ DO:
- **Check current state when you're curious**: `gpuq status` (informational; not a
  required pre-step — `gpuq submit` claims a slot atomically under a lock)
- **Let gpuq wait for you instead of polling**: `gpuq submit --queue ...` enqueues
  your job and starts it as soon as a slot (or your pinned `--devices`) frees up
- **Monitor your jobs**: `watch -n 10 gpuq status`
- **Kill finished/failed jobs**: `gpuq kill 12345` (or `gpuq kill --job-id 12345`)
- **Use appropriate batch sizes**: Start small, increase gradually
- **Clean up temporary files**: Remove large datasets after use
- **Communicate with team**: Let others know about long-running jobs

#### ❌ DON'T:
- Submit multiple identical jobs simultaneously
- Hold GPUs you aren't actively using (camping on a card someone else could use)
- Leave crashed/finished jobs running
- Run CPU-intensive tasks on the login shell without need
- Store large files in home directory without cleanup
- Ignore memory warnings or errors

> **The allocation model:** gpuq enforces "you own your allocated GPU." You may
> legitimately **stack** multiple jobs on a GPU you already hold, but GPUs held by
> *other* users are off-limits until they free them. Requesting several GPUs is fine
> when your job uses them; the thing to avoid is reserving cards you then leave idle.

### Communication Templates

**Slack/Email notification for long jobs**:
```
🔴 Starting long training job
Job ID: 12345
Estimated duration: 18 hours
Resources: 2x H200 NVL, model needs ~110 GB/GPU
Purpose: Fine-tuning LLaMA-70B for [project name]
Expected completion: [date/time]
```

**When encountering issues**:
```
⚠️ Job experiencing issues
Job ID: 12345
Issue: OOM errors despite a 141 GB card
Current approach: Reducing batch size and using gradient checkpointing
ETA for resolution: 2 hours
```

## Performance Optimization

### Universal Optimization Principles

1. **Use Mixed Precision Training**:
   ```python
   # PyTorch (use torch.amp; torch.cuda.amp.* is deprecated)
   from torch.amp import autocast, GradScaler
   scaler = GradScaler('cuda')
   
   with autocast('cuda'):
       output = model(input)
       loss = criterion(output, target)
   
   # TensorFlow
   tf.keras.mixed_precision.set_global_policy('mixed_float16')
   
   # JAX
   # Convert to half precision in forward pass
   x = x.astype(jnp.float16)
   ```

   On the H200 NVL (Hopper, compute capability 9.0), prefer **BF16** for training —
   it has the dynamic range of FP32, so you often don't need a GradScaler at all:
   ```python
   with autocast('cuda', dtype=torch.bfloat16):
       output = model(input)
       loss = criterion(output, target)
   ```

2. **Optimize Tensor Shapes for H200**:
   ```python
   # Ensure dimensions are multiples of 8 for Tensor Core usage
   def make_tensor_core_friendly(size):
       return ((size + 7) // 8) * 8
   
   batch_size = make_tensor_core_friendly(batch_size)
   hidden_dim = make_tensor_core_friendly(hidden_dim)
   sequence_length = make_tensor_core_friendly(sequence_length)
   ```

3. **Use Efficient Data Loading**:
   ```python
   # Optimized DataLoader settings for H200
   # The host has 256 logical CPUs, so num_workers can be generous —
   # but match it to your job's needs, not the whole machine.
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,
       num_workers=8,  # Scale to available cores; don't grab all 256
       pin_memory=True,
       persistent_workers=True,
       prefetch_factor=2,
   )
   ```

### Framework-Agnostic Performance Tips

#### Data Pipeline Optimization
```python
# Preprocess data offline when possible
# Use memory mapping for large datasets
# Implement data augmentation on GPU if possible
# Cache preprocessed data to SSD/RAM

# Example: Preprocessing pipeline
def create_efficient_pipeline(data_path, batch_size):
    # Load data with memory mapping
    data = np.memmap(data_path, dtype='float32', mode='r')
    
    # Create batches with optimal sizes
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batches.append(torch.from_numpy(batch.copy()))
    
    return batches
```

#### Model Architecture Optimization
```python
# Use layer normalization instead of batch normalization
# Implement gradient checkpointing for memory efficiency
# Use efficient attention mechanisms (Flash Attention, etc.)
# Consider model pruning and quantization for inference

# Example: Efficient transformer block
class EfficientTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Use SwiGLU instead of standard FFN for better performance
        self.ffn = SwiGLUFFN(d_model)
    
    def forward(self, x):
        # Pre-norm architecture for better gradient flow
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

## Memory Management

### Memory Optimization Strategies

#### 1. Gradient Checkpointing
```python
# PyTorch
import torch.utils.checkpoint as checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        return checkpoint.checkpoint(self.layer, x, use_reentrant=False)

# Hugging Face models: enable it on the model (or via TrainingArguments)
model.gradient_checkpointing_enable()

# TensorFlow
@tf.function
def checkpointed_layer(x):
    return tf.recompute_grad(layer)(x)

# JAX (remat lives at the top level: jax.checkpoint / jax.remat)
import jax
@jax.checkpoint
def checkpointed_layer(x, params):
    return layer_fn(x, params)
```

#### 2. Gradient Accumulation
```python
def train_with_gradient_accumulation(model, dataloader, accumulation_steps=4):
    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        # Forward pass
        output = model(batch)
        loss = criterion(output, target) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

#### 3. Memory Profiling
```python
def profile_memory_usage():
    # PyTorch
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    
    # TensorFlow
    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
    print(f"TF Memory: {gpu_info['current'] / 1024**3:.1f} GB")
    
    # System memory
    import psutil
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.percent:.1f}% used")
```

### Memory-Efficient Techniques by Model Size

#### Small Models (< 1B parameters)
- Use standard training with BF16/FP16
- Batch size: 32-128
- No special memory optimization needed — these fit comfortably on one ~141 GB card

#### Medium Models (1B-10B parameters)
- Enable gradient checkpointing
- Use gradient accumulation
- Consider DeepSpeed Stage 1

#### Large Models (10B+ parameters)
- Use DeepSpeed ZeRO Stage 2/3
- Model parallelism across multiple GPUs (up to all 4 on this host)
- Offload optimizer states to CPU (the host has 755 GiB RAM)

## Common Issues and Solutions

### Issue 1: Out of Memory (OOM) Errors

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GB (GPU 0; 140.XX GB total capacity)
```

**Solutions**:
```python
# 1. Reduce batch size
batch_size = batch_size // 2

# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()  # Hugging Face models
# or
import torch.utils.checkpoint as checkpoint

# 3. Use gradient accumulation
effective_batch_size = small_batch_size * accumulation_steps

# 4. Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()

# 5. Use mixed precision (torch.amp, not the deprecated torch.cuda.amp)
from torch.amp import autocast
with autocast('cuda', dtype=torch.bfloat16):
    output = model(input)
```

### Issue 2: Slow Training Speed

**Symptoms**:
- Low GPU utilization (< 80% in `nvidia-smi`)
- High CPU usage during training
- Slow data loading

> **Caveat on "utilization":** `nvidia-smi` `utilization.gpu` only reports whether a
> kernel was active during the sample window — it says nothing about arithmetic
> efficiency. On an H200 NVL (~836 TFLOP/s peak BF16 dense, measured on this server)
> utilization can read 90-100% while your *model FLOP utilization* (MFU) is far lower.
> For real efficiency, compare your achieved TFLOP/s against that ~836 TFLOP/s peak
> using a profiler, not just the utilization gauge.

**Solutions**:
```python
# 1. Optimize data loading
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=min(8, os.cpu_count()),  # Scale to cores; the host has 256
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

# 2. Use non-blocking transfers
data = data.cuda(non_blocking=True)
target = target.cuda(non_blocking=True)

# 3. Enable compiler optimizations
torch.backends.cudnn.benchmark = True  # PyTorch
tf.config.optimizer.set_jit(True)      # TensorFlow
model = jax.jit(model)                  # JAX

# 4. Profile your code
# Use appropriate profilers for each framework
```

### Issue 3: Job Gets Killed Unexpectedly

**Symptoms**:
- Job exits and its terminal returns to a prompt
- The process received a kill signal
- Job no longer appears in `gpuq status`

**Common Causes & Solutions**:
```bash
# 1. Time limit exceeded
# gpuq enforces --time: at the deadline it SIGTERMs (then SIGKILLs) the job.
# Solution: Request more time or optimize code
gpuq submit --command "python train.py" --time 24

# 2. You lost your terminal/SSH session
# gpuq runs your job in the FOREGROUND of the terminal you submit from.
# If that session dies, the job can die with it. Solution: use tmux/screen,
# or run on a host where your user has lingering enabled (gpuq then launches
# the job in a systemd --user scope that survives logout).
tmux new -s train
gpuq submit -- python train.py

# 3. System OOM killer
# Check system logs
dmesg | grep -i "killed process"

# Solution: Monitor memory usage more carefully
watch -n 5 'free -h && nvidia-smi'

# 4. Audit enforcement
# `gpuq audit --enforce` can terminate jobs running on the wrong physical GPU
# (a "rebind") or GPU processes not launched via gpuq, once past their grace
# deadline. Always launch through `gpuq submit` and don't override the device
# gpuq picked (don't reset CUDA_VISIBLE_DEVICES); pin a card with --devices.
```

### Issue 4: Poor Multi-GPU Performance

**Symptoms**:
- Linear scaling not achieved
- GPU utilization varies significantly between GPUs
- Communication overhead

**Solutions**:
```python
# 1. Use proper distributed training
# PyTorch DistributedDataParallel instead of DataParallel
model = DDP(model, device_ids=[local_rank])

# 2. Ensure balanced data loading
sampler = DistributedSampler(dataset)

# 3. Optimize communication
# Use NCCL backend for GPU-to-GPU communication
dist.init_process_group("nccl")
```
```bash
# 4. Check the GPU interconnect topology (the 4 cards are NVL-linked)
nvidia-smi topo -m
```

### Issue 5: Inconsistent Results Across Runs

**Symptoms**:
- Different accuracy/loss values with same hyperparameters
- Non-reproducible results

**Solutions**:
```python
# 1. Set all random seeds
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For exact reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 2. Fix data loading order
# Use same seed for data shuffling
DataLoader(dataset, shuffle=True, generator=torch.Generator().manual_seed(42))

# 3. Handle floating point precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

## Framework-Specific Tips

The stack on this host is CUDA 12.9 (driver 575.57.08) on Ubuntu 24.04. Install GPU
wheels built for CUDA 12.x (cu124 or newer), not cu121.

### PyTorch Optimization

```python
# 1. Use torch.compile (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='max-autotune')

# 2. Optimize CUDA settings
torch.backends.cudnn.benchmark = True  # For fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for speed

# 3. Use efficient optimizers
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

# 4. Memory-efficient attention
from torch.nn import functional as F
# Use F.scaled_dot_product_attention for memory efficiency (FlashAttention on Hopper)
```

```bash
# Install a current CUDA 12.x wheel (cu124+), not cu121:
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

### TensorFlow Optimization

```python
# 1. Enable XLA compilation
@tf.function(jit_compile=True)
def train_step(x, y):
    # Training logic here
    pass

# 2. Use mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# 3. Cap or grow VRAM per GPU.
#    Note: there is NO tf.config.experimental.set_memory_limit. Use one of:
gpus = tf.config.list_physical_devices('GPU')
# (a) hard cap a logical device (MB):
tf.config.set_logical_device_configuration(
    gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=40000)]
)
# (b) or grow on demand:
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 4. Optimize data pipeline
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.cache()  # If dataset fits in memory

# 5. Use distribution strategies (use all 4 GPUs)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()
```

> For training 🤗 Transformers, `TFTrainer`/`TFTrainingArguments` have been **removed**.
> Either train a TF/Keras model with `model.fit(...)`, or use the PyTorch `Trainer`.
> In `TrainingArguments`, use `eval_strategy` (the old `evaluation_strategy` is removed)
> and set the cache directory via `HF_HOME` (`TRANSFORMERS_CACHE` is deprecated).

### JAX/Flax Optimization

```python
# 1. Use jit compilation aggressively
import jax
train_step = jax.jit(train_step)

# 2. Use vmap for batch operations
from jax import vmap
batch_fn = vmap(single_example_fn)

# 3. Multi-GPU via sharding (jax.experimental.pjit is removed; use jax.jit
#    with sharding + jax.sharding.Mesh / PartitionSpec)
from jax.sharding import Mesh, PartitionSpec
mesh = Mesh(jax.devices(), axis_names=("data",))
in_shardings = PartitionSpec("data")
parallel_train_step = jax.jit(train_step, in_shardings=in_shardings)

# 4. Inspect array placement (.device() is gone; use .devices()/.sharding)
x = jnp.array(x)        # JAX array
print(x.devices())      # set of devices the array lives on
print(x.sharding)       # its sharding
```

```bash
# Install JAX with CUDA 12 support (the jax[cuda12_pip] + find-links recipe is stale):
pip install -U "jax[cuda12]"
```

## Monitoring and Debugging

### System Monitoring Commands

```bash
# GPU monitoring
nvidia-smi -l 1                    # Real-time GPU stats (4x H200 NVL)
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv -l 5

# System resources
htop                               # CPU and memory usage
iotop                              # Disk I/O
nethogs                            # Network usage per process

# Job monitoring
gpuq status                        # Queue + per-GPU + all GPU procs
watch -n 5 gpuq status             # Real-time queue monitoring
```

> **Where is my job's output?** `gpuq` is daemonless: each `gpuq submit` runs your job
> in the **foreground** of the terminal you launched it from, and the job's stdout/stderr
> stream straight to that terminal. There are no per-job log files (the old
> `/tmp/gpu_queue/logs/...` path belonged to a retired daemon and no longer exists).
> If you want a log file, redirect it yourself, and use `tmux`/`screen` so it survives
> a dropped SSH session:
> ```bash
> tmux new -s train
> gpuq submit -- python train.py > train.log 2>&1
> # ... detach with Ctrl-b d; later: tmux attach -t train; or: tail -f train.log
> ```

### Debug Information Collection

```python
def collect_debug_info():
    """Collect comprehensive debug information"""
    info = {}
    
    # System information
    import platform, psutil
    info['system'] = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / 1024**3,
    }
    
    # GPU information
    if torch.cuda.is_available():
        info['gpu'] = {
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'memory_allocated': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved': torch.cuda.memory_reserved() / 1024**3,
        }
    
    # Framework versions
    info['frameworks'] = {
        'torch': torch.__version__ if 'torch' in globals() else 'Not available',
        'tensorflow': tf.__version__ if 'tf' in globals() else 'Not available',
        'jax': jax.__version__ if 'jax' in globals() else 'Not available',
    }
    
    return info

# Usage
debug_info = collect_debug_info()
print(json.dumps(debug_info, indent=2))
```

### Performance Profiling

```python
# PyTorch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Training code here
    pass

prof.export_chrome_trace("trace.json")

# TensorFlow profiler
tf.profiler.experimental.start('logs')
# Training code
tf.profiler.experimental.stop()

# JAX profiler  
jax.profiler.start_trace("/tmp/jax_trace")
# Training code
jax.profiler.stop_trace()
```

## Emergency Procedures

### When Things Go Wrong

#### 1. System Becomes Unresponsive
```bash
# Check system load
uptime

# Find resource-intensive processes
top -o %CPU
top -o %MEM

# Kill runaway processes (be careful!)
kill -9 PID

# Emergency: Kill all your processes
pkill -u $USER python
```

#### 2. Disk Space Issues
```bash
# Check disk usage
df -h

# Find large files
du -h --max-depth=1 | sort -hr

# Clean up common locations
rm -rf ~/.cache/pip/*
rm -rf /tmp/tmp*
rm -rf checkpoint_*.pth  # Old checkpoints
```

#### 3. GPU Memory Issues
```python
# Emergency memory cleanup
import torch
torch.cuda.empty_cache()

# Reset CUDA context (last resort)
torch.cuda.synchronize()
torch.cuda.empty_cache()

# Check what's using memory
import gc
gc.collect()
```

#### 4. Contact Administrator

**When to contact admin**:
- System-wide issues affecting multiple users
- Hardware failures or unusual GPU behavior
- Network connectivity problems
- Queue system malfunctions

**Information to provide**:
- Your username and job ID
- Exact error messages
- Commands that led to the issue
- System state (output of `nvidia-smi`, `gpuq status`)
- Any log file you redirected your job's output to

### Prevention Checklist

Before submitting large jobs, verify:
- [ ] Code tested with small dataset
- [ ] Memory requirements estimated
- [ ] Checkpointing implemented
- [ ] Time limits appropriate
- [ ] Error handling in place
- [ ] Resource requirements justified
- [ ] Running inside tmux/screen (or as a lingering user) so the job survives disconnects
- [ ] Team informed if using multiple GPUs

---

**Additional Resources**:
- [H200 Specifications](h200-specs.md)
- [Framework-specific guides](pytorch-guide.md): PyTorch, TensorFlow, JAX
- [GPU Queue System](gpu-queue-guide.md)
- [Example Scripts](../examples/)
