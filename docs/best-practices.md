# Best Practices and Troubleshooting Guide

Comprehensive guide for efficient and respectful use of the Ruqola server's H200 GPUs, including common issues and solutions.

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

```bash
# ✅ GOOD: Request only what you need
gpuq submit --command "python train.py" --gpus 1 --memory 40 --time 8

# ❌ BAD: Over-requesting resources
gpuq submit --command "python train.py" --gpus 3 --memory 120 --time 24
```

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

| Model Size | Estimated H200 Memory | Batch Size Recommendation |
|------------|----------------------|---------------------------|
| **Small (< 100M params)** | 5-15 GB | 64-256 |
| **Medium (100M-1B params)** | 15-40 GB | 16-64 |
| **Large (1B-10B params)** | 40-80 GB | 4-16 |
| **Very Large (10B+ params)** | 80-120 GB | 1-4 |

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
- **Check availability before submitting**: `gpuq status`
- **Monitor your jobs**: `watch -n 10 gpuq status`
- **Kill finished/failed jobs**: `gpuq kill --job-id XXXXX`
- **Use appropriate batch sizes**: Start small, increase gradually
- **Clean up temporary files**: Remove large datasets after use
- **Communicate with team**: Let others know about long-running jobs

#### ❌ DON'T:
- Submit multiple identical jobs simultaneously
- Request all GPUs unless absolutely necessary
- Leave crashed/finished jobs running
- Run CPU-intensive tasks on login node
- Store large files in home directory without cleanup
- Ignore memory warnings or errors

### Communication Templates

**Slack/Email notification for long jobs**:
```
🔴 Starting long training job
Job ID: 12345
Estimated duration: 18 hours
Resources: 2x H200, 100GB memory
Purpose: Fine-tuning LLaMA-70B for [project name]
Expected completion: [date/time]
```

**When encountering issues**:
```
⚠️ Job experiencing issues
Job ID: 12345
Issue: OOM errors despite requesting 80GB
Current approach: Reducing batch size and using gradient checkpointing
ETA for resolution: 2 hours
```

## Performance Optimization

### Universal Optimization Principles

1. **Use Mixed Precision Training**:
   ```python
   # PyTorch
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       output = model(input)
       loss = criterion(output, target)
   
   # TensorFlow
   tf.keras.mixed_precision.set_global_policy('mixed_float16')
   
   # JAX
   # Convert to half precision in forward pass
   x = x.astype(jnp.float16)
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
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,
       num_workers=8,  # Match available CPU cores
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
        return checkpoint.checkpoint(self.layer, x)

# TensorFlow
@tf.function
def checkpointed_layer(x):
    return tf.recompute_grad(layer)(x)

# JAX
from jax.experimental import remat
@remat
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
- Use standard training with FP16
- Batch size: 32-128
- No special memory optimization needed

#### Medium Models (1B-10B parameters)
- Enable gradient checkpointing
- Use gradient accumulation
- Consider DeepSpeed Stage 1

#### Large Models (10B+ parameters)
- Use DeepSpeed ZeRO Stage 2/3
- Model parallelism across multiple GPUs
- Offload optimizer states to CPU

## Common Issues and Solutions

### Issue 1: Out of Memory (OOM) Errors

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GB (GPU 0; 141.XX GB total capacity)
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

# 5. Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
```

### Issue 2: Slow Training Speed

**Symptoms**:
- Low GPU utilization (< 80%)
- High CPU usage during training
- Slow data loading

**Solutions**:
```python
# 1. Optimize data loading
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=min(8, os.cpu_count()),  # Don't exceed CPU cores
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
model = jit(model)                     # JAX

# 4. Profile your code
# Use appropriate profilers for each framework
```

### Issue 3: Job Gets Killed Unexpectedly

**Symptoms**:
- Job disappears from queue without error
- No output in log files
- Process killed signal

**Common Causes & Solutions**:
```bash
# 1. Time limit exceeded
# Solution: Request more time or optimize code
gpuq submit --command "python train.py" --time 24

# 2. Memory limit exceeded
# Solution: Request more memory or optimize usage
gpuq submit --command "python train.py" --memory 100

# 3. System OOM killer
# Check system logs
dmesg | grep -i "killed process"

# Solution: Monitor memory usage more carefully
watch -n 5 'free -h && nvidia-smi'
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

# 4. Check network topology
# Ensure GPUs are connected via NVLink for best performance
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
# Use F.scaled_dot_product_attention for memory efficiency
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

# 3. Optimize data pipeline
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.cache()  # If dataset fits in memory

# 4. Use distribution strategies
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()
```

### JAX/Flax Optimization

```python
# 1. Use jit compilation aggressively
from jax import jit
train_step = jit(train_step)

# 2. Use vmap for batch operations
from jax import vmap
batch_fn = vmap(single_example_fn)

# 3. Optimize with pmap for multi-GPU
from jax import pmap
parallel_train_step = pmap(train_step)

# 4. Use efficient data structures
# Use JAX arrays instead of NumPy when possible
x = jnp.array(x)  # JAX array
```

## Monitoring and Debugging

### System Monitoring Commands

```bash
# GPU monitoring
nvidia-smi -l 1                    # Real-time GPU stats
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv -l 5

# System resources
htop                               # CPU and memory usage
iotop                              # Disk I/O
nethogs                            # Network usage per process

# Job monitoring
gpuq status                        # Queue status
watch -n 5 gpuq status            # Real-time queue monitoring
tail -f /tmp/gpu_queue/logs/job_XXXXX_stdout.log  # Job output
```

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
- Log files if available

### Prevention Checklist

Before submitting large jobs, verify:
- [ ] Code tested with small dataset
- [ ] Memory requirements estimated
- [ ] Checkpointing implemented
- [ ] Time limits appropriate
- [ ] Error handling in place
- [ ] Resource requirements justified
- [ ] Team informed if using multiple GPUs

---

**Additional Resources**:
- [H200 Specifications](h200-specs.md)
- [Framework-specific guides](pytorch-guide.md): PyTorch, TensorFlow, JAX
- [GPU Queue System](gpu-queue-guide.md)
- [Example Scripts](../examples/)