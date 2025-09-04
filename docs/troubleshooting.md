# Troubleshooting Guide

Quick reference for resolving common issues on the Ruqola server's H200 GPUs.

## 🚨 Emergency Quick Fixes

### Immediate Actions for Critical Issues

```bash
# If system is unresponsive
sudo reboot  # Last resort - contact admin first

# Kill all your processes
pkill -u $USER

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Check system status
nvidia-smi
gpuq status
df -h
free -h
```

## 📋 Common Error Messages and Solutions

### CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 141.61 GiB total capacity; 139.20 GiB already allocated)
```

**Quick Fixes:**
```python
# 1. Reduce batch size immediately
batch_size = batch_size // 2

# 2. Clear cache
torch.cuda.empty_cache()

# 3. Enable gradient checkpointing
model.gradient_checkpointing = True

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
- All GPUs busy
- Requesting too much memory
- Syntax error in command
- Queue system issue

**Diagnostic Commands:**
```bash
gpuq status                    # Check queue state
nvidia-smi                     # Check GPU usage
gpuq submit --command "echo test" --gpus 1 --time 1  # Test submission
```

### Job Gets Killed

**Check logs:**
```bash
# View job logs
tail -100 /tmp/gpu_queue/logs/job_XXXXX_stderr.log
tail -100 /tmp/gpu_queue/logs/job_XXXXX_stdout.log

# Check system logs
dmesg | tail -50
journalctl -f  # Real-time system logs
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

# Common fix for CUDA version mismatch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

#### cuDNN Errors
```bash
# Check CUDA/cuDNN compatibility
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"

# Reinstall TensorFlow with specific CUDA version
pip install tensorflow[and-cuda]==2.15.0
```

### JAX Issues

#### Device Not Found
```python
import jax
print("Devices:", jax.devices())

# Check installation
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### Memory Preallococation
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
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
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

echo "Utilization should be >80% during training"
echo "If low, check:"
echo "1. Batch size too small"
echo "2. CPU bottleneck in data loading"  
echo "3. Model too simple for GPU"
echo "4. I/O bottleneck"
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
    pkill -u $USER python
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
   - GPU memory errors
   - System crashes or freezes

2. **Queue System Issues**:
   - Jobs not starting despite available resources
   - Queue commands not responding
   - Inconsistent resource reporting

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

echo "=== Recent Jobs ==="
tail -50 /tmp/gpu_queue/logs/job_*.log 2>/dev/null || echo "No job logs found"

echo "=== Error Logs ==="
dmesg | tail -20
```

---

**Quick Reference Links**:
- [Best Practices Guide](best-practices.md)
- [Framework Guides](pytorch-guide.md): PyTorch, TensorFlow, JAX
- [GPU Queue Documentation](gpu-queue-guide.md)
- [H200 Specifications](h200-specs.md)