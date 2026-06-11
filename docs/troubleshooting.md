# Troubleshooting Guide

Quick reference for resolving common issues on the Ruqola server's H200 GPUs.

> **Server at a glance:** host `wsserver1` ("Mjolnir") has **4 x NVIDIA H200 NVL** GPUs (indices `0,1,2,3`, ~141 GB VRAM each, ~564 GB total), compute capability 9.0 (Hopper), 256 logical CPUs, 755 GiB RAM, Ubuntu 24.04.4 LTS, GPU driver **575.57.08**, CUDA (driver) **12.9**.

## 🚨 Emergency Quick Fixes

### Immediate Actions for Critical Issues

```bash
# If system is unresponsive
sudo reboot  # Last resort - contact admin first

# Kill your runaway Python jobs (scoped: does NOT touch your shell/SSH session)
pkill -u $USER python
# WARNING: a bare `pkill -u $USER` kills EVERY process you own, including your
# login shell, tmux/screen, and SSH session - it will disconnect you. Use the
# scoped form above unless you really mean to nuke everything.

# Clear GPU memory (from inside a Python process that holds it)
python -c "import torch; torch.cuda.empty_cache()"

# Check system status
nvidia-smi
gpuq status
df -h
free -h
```

## 📋 Common Error Messages and Solutions

### CUDA Out of Memory

**Error Message** (illustrative; each H200 NVL reports 143771 MiB ≈ 140.4 GiB total):
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 140.40 GiB total capacity; 138.10 GiB already allocated)
```

**Quick Fixes:**
```python
# 1. Reduce batch size immediately
batch_size = batch_size // 2

# 2. Clear cache
torch.cuda.empty_cache()

# 3. Enable gradient checkpointing
#    - plain PyTorch model:
model.gradient_checkpointing_enable()
#    - Hugging Face Trainer: pass gradient_checkpointing=True via TrainingArguments
#      (it is NOT a from_pretrained() kwarg)

# 4. Use gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Job Won't Start

**Possible Causes:**
- All GPUs busy (held by other users)
- Requesting too much memory (no candidate GPU meets the `-m/--memory` free-VRAM floor)
- Syntax error in command
- Queue system issue

**Diagnostic Commands:**
```bash
gpuq status                    # Check queue state
nvidia-smi                     # Check GPU usage
gpuq submit --command "echo test" --gpus 1 --time 1  # Test submission
```

If you need specific GPUs, pin them with `--devices` (e.g. `--devices 0,1`). Pinning
**rejects immediately** if any requested GPU is unavailable, unless you add `--queue`,
in which case `gpuq` **waits** until those GPUs are free or already owned by you. Under
the "you own your allocated GPU" policy you may stack additional jobs on GPUs you
already hold, but GPUs held by other users stay off-limits until they free them.

### Job Gets Killed

`gpuq` is **daemonless**: each `gpuq submit` runs your job in the **foreground in your
own terminal**. There is no background daemon and **no per-job log files** - your job's
stdout/stderr stream straight to your terminal. If you want a persistent log, redirect
it yourself:

```bash
# Redirect your job's output to files you control
gpuq submit --gpus 1 --command "python train.py" > train.out 2> train.err

# ...or capture both streams to one file
gpuq submit --gpus 1 --command "python train.py" 2>&1 | tee train.log

# For long runs, detach with tmux/screen so the job survives an SSH drop:
tmux new -s train
gpuq submit --gpus 1 --command "python train.py" 2>&1 | tee train.log
# (Ctrl-b d to detach; `tmux attach -t train` to return)
```

If your job died and you did not capture its output, re-run it with a redirect as above
to see the error. To diagnose a kernel-level OOM kill or hardware fault, check the
system logs:

```bash
# Check system logs (e.g. the OOM killer, Xid GPU errors)
dmesg | tail -50
journalctl -f  # Real-time system logs
```

You can also inspect the shared queue state to see what `gpuq` knows about running jobs:

```bash
gpuq status                            # who holds which GPU
cat /var/lib/gpu_queue/running.json    # raw running-jobs state
```

## 🔧 Framework-Specific Troubleshooting

### PyTorch Issues

#### Import Errors
```python
# Test PyTorch installation
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

```bash
# Common fix for a CUDA version mismatch.
# The server runs driver 575.57.08 / CUDA 12.9, so install a current CUDA 12.x
# wheel. Any cu12x wheel works against the 12.9 driver thanks to forward-
# compatible minor versions; prefer a recent index (cu124/cu126/cu128) over the
# older cu121 to get Hopper-tuned kernels.
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Mixed Precision (AMP) Deprecation Warnings
```python
# Use the current device-agnostic AMP API (torch.cuda.amp.* is deprecated):
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')
for batch in dataloader:
    optimizer.zero_grad()
    with autocast('cuda', dtype=torch.bfloat16):  # BF16 is native on Hopper H200
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### DataLoader Hanging
```python
# Reduce num_workers if hanging
dataloader = DataLoader(dataset, num_workers=0)  # Single threaded

# Or try:
dataloader = DataLoader(dataset, num_workers=2, persistent_workers=False)
```

### TensorFlow Issues

#### GPU Not Detected
```python
import tensorflow as tf

# Check GPU detection
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

#### Capping VRAM for a TensorFlow Process
```python
# There is NO tf.config.experimental.set_memory_limit. Cap VRAM with a
# logical-device configuration (memory_limit is in MB):
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=40000)]  # ~40 GB
    )
# ...or just enable memory growth (above) so TF grows on demand.
```

#### cuDNN Errors
```bash
# Check CUDA/cuDNN compatibility
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"

# Reinstall TensorFlow with bundled CUDA (matches the system's CUDA 12.x)
pip install -U "tensorflow[and-cuda]"
```

> **Note:** `TFTrainer` / `TFTrainingArguments` were **removed** from `transformers`.
> For TensorFlow models train with Keras `model.fit(...)`; otherwise use the PyTorch
> `Trainer`.

### JAX Issues

#### Device Not Found
```python
import jax
print("Devices:", jax.devices())  # expect 4 devices on wsserver1
```

```bash
# Install the current CUDA 12 JAX wheels (the old jax[cuda12_pip] + find-links
# recipe is stale):
pip install -U "jax[cuda12]"
```

#### Sharding / Device Placement (API moved)
```python
# Use the stable jax.sharding API; jax.experimental.pjit / PartitionSpec are gone.
from jax.sharding import Mesh, PartitionSpec
import jax

mesh = Mesh(jax.devices(), axis_names=('data',))
# pass sharding into jit via in_shardings=...

# Inspect where an array lives: x.devices() / x.sharding
# (the old x.device() method has been removed).

# Activation checkpointing is jax.checkpoint (a.k.a. jax.remat),
# not jax.experimental.remat.
```

#### Memory Preallocation
```bash
# Disable memory preallocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Or set fraction
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

## 📊 Diagnostic Tools and Commands

### System Health Check

```bash
#!/bin/bash
# system_health_check.sh

echo "=== System Health Check ==="
echo "Date: $(date)"
echo

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

echo "=== Queue Status ==="
gpuq status

echo "=== Disk Usage ==="
df -h

echo "=== Memory Usage ==="
free -h

echo "=== CPU Load ==="
uptime

echo "=== Top Processes ==="
ps aux --sort=-%cpu | head -10
```

### Memory Profiling Script

```python
#!/usr/bin/env python3
"""
memory_profile.py - Monitor GPU and system memory usage
"""

import time
import psutil
import subprocess
import json
from datetime import datetime

def get_gpu_memory():
    """Get GPU memory usage via nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            used, total = map(int, line.split(', '))
            gpu_info.append({
                'used_mb': used,
                'total_mb': total,
                'utilization_percent': (used / total) * 100
            })
        return gpu_info
    except:
        return []

def monitor_memory(interval=5, duration=300):
    """Monitor memory usage for specified duration"""
    start_time = time.time()
    data = []
    
    while time.time() - start_time < duration:
        timestamp = datetime.now().isoformat()
        
        # System memory
        memory = psutil.virtual_memory()
        
        # GPU memory
        gpu_memory = get_gpu_memory()
        
        data_point = {
            'timestamp': timestamp,
            'system_memory': {
                'used_gb': memory.used / 1024**3,
                'total_gb': memory.total / 1024**3,
                'percent': memory.percent
            },
            'gpu_memory': gpu_memory
        }
        
        data.append(data_point)
        print(f"[{timestamp}] System: {memory.percent:.1f}%, GPU: {gpu_memory}")
        
        time.sleep(interval)
    
    # Save data
    with open('memory_profile.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Memory profile saved to memory_profile.json")

if __name__ == '__main__':
    monitor_memory(interval=5, duration=300)  # 5 minutes
```

### Network Connectivity Test

```bash
#!/bin/bash
# network_test.sh

echo "=== Network Connectivity Test ==="

# Test internet connectivity
echo "Testing internet connectivity..."
ping -c 3 google.com

# Test internal network
echo "Testing internal connectivity..."
ping -c 3 localhost

# Check network interfaces
echo "Network interfaces:"
ip addr show

# Check DNS resolution
echo "DNS resolution test:"
nslookup google.com
```

## 🐛 Debug Mode Scripts

### PyTorch Debug Script

```python
#!/usr/bin/env python3
"""
pytorch_debug.py - Comprehensive PyTorch debugging
"""

import torch
import torch.nn as nn
import sys
import traceback
from torch.profiler import profile, ProfilerActivity

def debug_pytorch_setup():
    """Debug PyTorch installation and CUDA setup"""
    print("=== PyTorch Debug Information ===")
    
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"Python version: {sys.version}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")  # expect 4 on wsserver1
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")  # ~140 GB each
                print(f"  Compute capability: {props.major}.{props.minor}")  # 9.0 (Hopper)
        else:
            print("CUDA not available - this will severely limit performance")
            
    except Exception as e:
        print(f"Error during setup check: {e}")
        traceback.print_exc()

def test_basic_operations():
    """Test basic GPU operations"""
    print("\n=== Testing Basic Operations ===")
    
    try:
        # Test tensor creation and movement
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Test computation
        z = torch.mm(x, y)
        print(f"Matrix multiplication successful, result shape: {z.shape}")
        
        # Test memory usage
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        # Clear memory
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error during basic operations: {e}")
        traceback.print_exc()

def test_model_training():
    """Test simple model training"""
    print("\n=== Testing Model Training ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(device)
        
        # Test data
        x = torch.randn(32, 784, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"Training step successful, loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    debug_pytorch_setup()
    test_basic_operations()
    test_model_training()
    print("\n=== Debug Complete ===")
```

## 📈 Performance Debugging

### GPU Utilization Check

```bash
#!/bin/bash
# gpu_utilization_monitor.sh

echo "Monitoring GPU utilization for 60 seconds..."

for i in {1..12}; do
    echo "=== Check $i/12 ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    echo
    sleep 5
done

echo "Interpreting nvidia-smi 'utilization.gpu':"
echo "  - It is the FRACTION OF TIME a kernel was running, NOT compute efficiency."
echo "  - On H200 NVL it often reads 90-100% even at low FLOP efficiency (MFU),"
echo "    so high util alone does NOT mean the GPU is well-utilized."
echo "  - Sustained util well below ~80-90% usually points to a bottleneck:"
echo "      1. Batch size too small"
echo "      2. CPU bottleneck in data loading"
echo "      3. Model too simple for the GPU"
echo "      4. I/O bottleneck"
echo "  - For REAL efficiency, profile with torch.profiler or DCGM (compute MFU),"
echo "    not the single utilization number."
```

### Data Loading Profiler

```python
#!/usr/bin/env python3
"""
data_loading_profiler.py - Profile data loading performance
"""

import time
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, size=10000, feature_dim=784):
        self.size = size
        self.feature_dim = feature_dim
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Simulate data loading with small delay
        time.sleep(0.001)  # 1ms delay
        x = np.random.randn(self.feature_dim).astype(np.float32)
        y = np.random.randint(0, 10)
        return torch.from_numpy(x), y

def profile_dataloader(batch_size=32, num_workers=0):
    """Profile DataLoader performance"""
    print(f"Profiling DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    
    dataset = DummyDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    start_time = time.time()
    batch_count = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        if torch.cuda.is_available():
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        
        batch_count += 1
        if batch_count >= 100:  # Test first 100 batches
            break
    
    elapsed_time = time.time() - start_time
    batches_per_second = batch_count / elapsed_time
    samples_per_second = batches_per_second * batch_size
    
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Batches/sec: {batches_per_second:.2f}")
    print(f"  Samples/sec: {samples_per_second:.2f}")
    print()

if __name__ == '__main__':
    print("=== DataLoader Performance Profiling ===")
    
    # Test different configurations
    configs = [
        (32, 0),   # Single-threaded
        (32, 2),   # 2 workers
        (32, 4),   # 4 workers
        (32, 8),   # 8 workers
        (64, 4),   # Larger batch
        (128, 4),  # Even larger batch
    ]
    
    for batch_size, num_workers in configs:
        profile_dataloader(batch_size, num_workers)
```

## 🚀 Performance Recovery Scripts

### Memory Cleanup Script

```python
#!/usr/bin/env python3
"""
cleanup_memory.py - Aggressive memory cleanup
"""

import gc
import os

def cleanup_python_memory():
    """Clean up Python memory"""
    print("Cleaning Python memory...")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"Collected {collected} objects")
    
    # Clear import cache
    import sys
    if hasattr(sys, 'modules'):
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('torch') or module_name.startswith('tensorflow'):
                print(f"Clearing {module_name}")
                del sys.modules[module_name]

def cleanup_gpu_memory():
    """Clean up GPU memory for all frameworks"""
    print("Cleaning GPU memory...")
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("PyTorch GPU cache cleared")
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        print("TensorFlow session cleared")
    except ImportError:
        pass
    
    try:
        import jax
        # JAX doesn't have explicit memory clearing, but we can try
        print("JAX memory cleanup attempted")
    except ImportError:
        pass

def cleanup_system_cache():
    """Clean up system caches"""
    print("Cleaning system caches...")
    
    # Clear pip cache
    os.system("pip cache purge")
    
    # Clear conda cache if available
    if os.path.exists(os.path.expanduser("~/anaconda3/bin/conda")):
        os.system("conda clean -a -y")

if __name__ == '__main__':
    print("=== Emergency Memory Cleanup ===")
    cleanup_gpu_memory()
    cleanup_python_memory() 
    cleanup_system_cache()
    print("Cleanup complete!")
```

### Process Recovery Script

```bash
#!/bin/bash
# process_recovery.sh

echo "=== Process Recovery Script ==="

echo "Current processes:"
ps aux | grep $USER | grep python

echo
read -p "Kill all Python processes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Killing Python processes..."
    pkill -u $USER python   # scoped to python; leaves your shell/SSH session alive
    sleep 5
    
    echo "Remaining processes:"
    ps aux | grep $USER | grep python
fi

echo
echo "Clearing GPU memory..."
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleared')
else:
    print('No CUDA available')
"

echo
echo "System status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

## 📞 When to Contact Support

### Contact Administrator When:

1. **Hardware Issues**:
   - GPU temperature >85°C
   - GPU memory errors (e.g. Xid errors in `dmesg`)
   - System crashes or freezes

2. **Queue System Issues**:
   - Jobs not starting despite available resources
   - Queue commands not responding
   - Inconsistent resource reporting (e.g. `gpuq status` disagrees with `nvidia-smi`)

3. **Network/Storage Issues**:
   - Cannot access shared storage
   - Network connectivity problems
   - Permission errors on system directories

4. **Multi-User Conflicts**:
   - Resource conflicts between users
   - Queue fairness issues
   - System-wide performance problems

### Information to Include in Support Requests:

```bash
# Collect this information before contacting support
echo "=== Support Information Package ==="
echo "User: $USER"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo

echo "=== System Status ==="
uptime
df -h
free -h

echo "=== GPU Status ==="
nvidia-smi

echo "=== Queue Status ==="
gpuq status

echo "=== Running-jobs state (shared queue dir) ==="
cat /var/lib/gpu_queue/running.json 2>/dev/null || echo "No running.json found"

echo "=== Error Logs ==="
dmesg | tail -20
```

> **Reminder:** `gpuq` is daemonless and does not write per-job log files. If your job
> failed, attach the output you captured yourself (e.g. the file from
> `... | tee train.log` or `> train.out 2> train.err`); there are no
> `job_XXXXX_*.log` files to fetch.

---

**Quick Reference Links**:
- [Best Practices Guide](best-practices.md)
- [Framework Guides](pytorch-guide.md): PyTorch, TensorFlow, JAX
- [GPU Queue Documentation](gpu-queue-guide.md)
- [H200 Specifications](h200-specs.md)
