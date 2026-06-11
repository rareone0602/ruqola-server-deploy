# PyTorch with NVIDIA H200 GPUs

Complete guide for optimizing PyTorch workflows on the Ruqola server's H200 GPUs.

The server (host `wsserver1`, "Mjolnir") has **4 x NVIDIA H200 NVL** GPUs (indices 0,1,2,3), each with ~141 GB of VRAM (~564 GB total). They are Hopper-class cards (compute capability 9.0, sm_90). The host runs Ubuntu 24.04 LTS with GPU driver 575.57.08 (CUDA 12.9), 256 logical CPUs, and 755 GiB of system RAM.

## 📖 Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Basic GPU Usage](#basic-gpu-usage)
3. [Memory Optimization](#memory-optimization)
4. [Performance Optimization](#performance-optimization)
5. [Multi-GPU Training](#multi-gpu-training)
6. [Large Model Training](#large-model-training)
7. [Advanced Techniques](#advanced-techniques)
8. [Debugging and Profiling](#debugging-and-profiling)
9. [Example Scripts](#example-scripts)

## Setup and Installation

### Recommended PyTorch Installation

```bash
# Current CUDA 12.x PyTorch wheel.
# The host driver is 575.57.08 (CUDA 12.9), so any cu12x wheel is forward-compatible,
# and all current PyTorch wheels include Hopper sm_90 kernels for the H200 NVL.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Or with conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

> The `cu126`/`cu128` wheels also work and may be preferable for the latest cuDNN/Hopper
> optimizations — pick whichever the current PyTorch release publishes. Avoid pinning to an
> old `cu121` build; while it still runs on a 12.9 driver, it leaves Hopper performance on the table.

### Verify Installation

```python
import torch

# Check PyTorch and CUDA versions
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Check H200 capabilities
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Multiprocessors: {props.multi_processor_count}")
```

On this server you should see 4 GPUs, each reporting ~140 GB and compute capability 9.0.

### Environment Setup

Add to your `~/.bashrc` or job submission script:

```bash
# CUDA environment variables for H200
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, or 0,1,2,3 for all 4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

> **Caution under gpuq:** Do **not** set `CUDA_VISIBLE_DEVICES` manually when running through
> `gpuq submit`. gpuq sets it for you to the GPU(s) it allocated, and overriding it makes your job
> run on a different physical GPU than the one you were granted — `gpuq audit` detects this rebind
> and will warn (and eventually kill) the job. Set `CUDA_VISIBLE_DEVICES` by hand only for
> interactive/non-gpuq experiments.

## Basic GPU Usage

### Device Management

```python
import torch

# Check available devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Specific GPU selection
device = torch.device('cuda:0')  # First H200

# Move tensors to GPU
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)
z = torch.mm(x, y)  # Matrix multiplication on GPU

# Move model to GPU
model = MyModel().to(device)
# or
model = MyModel().cuda()
```

### Memory Management

```python
# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")

# Clear cache when needed
torch.cuda.empty_cache()

# Memory-efficient context manager
with torch.cuda.device(0):
    # Operations on GPU 0
    x = torch.randn(1000, 1000, device='cuda')
```

### Simple Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model setup
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 10)
).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.cuda(), target.cuda()
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if batch_idx % 100 == 0:
        print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
```

## Memory Optimization

### Automatic Mixed Precision (AMP)

```python
from torch.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = optim.AdamW(model.parameters())
scaler = GradScaler('cuda')

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()
    
    optimizer.zero_grad()
    
    # Mixed precision forward pass
    with autocast('cuda'):
        output = model(data)
        loss = criterion(output, target)
    
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

> Use `torch.amp.autocast('cuda', ...)` / `torch.amp.GradScaler('cuda')`; the older
> `torch.cuda.amp.*` API is deprecated and emits `FutureWarning`. On Hopper the preferred
> choice is **BF16** autocast (`with autocast('cuda', dtype=torch.bfloat16): ...`), which has the
> same dynamic range as FP32 and therefore needs **no** `GradScaler` at all:
>
> ```python
> with autocast('cuda', dtype=torch.bfloat16):
>     output = model(data)
>     loss = criterion(output, target)
> loss.backward()       # no scaler needed for bf16
> optimizer.step()
> ```

### Gradient Checkpointing

```python
import torch.utils.checkpoint as checkpoint

class MemoryEfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1024, 1024) for _ in range(10)
        ])
    
    def forward(self, x):
        # Use checkpointing for memory efficiency
        for layer in self.layers:
            x = checkpoint.checkpoint(layer, x, use_reentrant=False)
        return x

# Or use checkpoint_sequential inside forward(): it returns the OUTPUT tensor of
# running `x` through the segments (it is not a reusable model object).
class SequentialCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(10)])

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint_sequential(
            self.layers, segments=4, input=x, use_reentrant=False
        )
```

> Pass `use_reentrant=False` explicitly — current PyTorch warns when it is omitted, and the
> non-reentrant implementation is the recommended one.

### Dynamic Memory Allocation

```python
# Monitor memory usage
def print_memory_stats():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Optimize memory allocation
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cudnn.enabled = True

# Memory-efficient data types
model = model.half()  # Use FP16
# or
model = model.bfloat16()  # Use BF16 (better for training, preferred on Hopper)
```

### Large Batch Sizes with Gradient Accumulation

```python
def train_with_gradient_accumulation(model, dataloader, accumulation_steps=4):
    model.train()
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        
        with autocast('cuda'):
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

# Effective batch size = actual_batch_size * accumulation_steps * num_gpus
```

## Performance Optimization

### Tensor Core Optimization

```python
# Align tensor dimensions for Tensor Core usage.
# Classic FP16 rule of thumb is multiples of 8; on Hopper with BF16/TF32, aligning to
# multiples of 16 (and ideally 64 for best memory coalescing) is the more current guideline.
def optimize_tensor_shapes(batch_size, seq_len, hidden_dim, align=16):
    def round_up(n):
        return ((n + align - 1) // align) * align
    return round_up(batch_size), round_up(seq_len), round_up(hidden_dim)

# Use Tensor Core friendly operations
x = torch.randn(64, 2048, dtype=torch.bfloat16, device='cuda')  # aligned to 16
linear = nn.Linear(2048, 4096).bfloat16().cuda()                # aligned to 16
output = linear(x)  # Uses Tensor Cores
```

> Each H200 NVL reaches roughly **~836 TFLOP/s** of dense BF16 tensor-core matmul (empirical on
> this server; the datasheet quotes ~989 TFLOP/s FP16 with sparsity). That is the figure to plan
> around — not the much smaller non-tensor CUDA-core number. Keep matmul dimensions aligned to
> make use of that throughput.

### Optimized Data Loading

```python
# H200-optimized DataLoader
def create_optimized_dataloader(dataset, batch_size=128):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,  # conservative; the box has 256 logical CPUs shared among users — scale up cautiously
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Reduce worker overhead
        prefetch_factor=2,  # Prefetch batches
        drop_last=True,  # Consistent batch sizes
    )

# Non-blocking transfers
for data, target in dataloader:
    data = data.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    # Process batch...
```

### Compilation and Optimization

```python
# PyTorch 2.0 compilation (if available)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='max-autotune')

# TensorRT optimization (for inference)
import torch_tensorrt

# Compile model for TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch.randn(1, 3, 224, 224).cuda()],
    enabled_precisions=torch.half
)
```

### Custom CUDA Kernels

```python
# Custom CUDA kernel example (advanced)
from torch.utils.cpp_extension import load

# Load custom CUDA kernel
custom_ops = load(
    'custom_ops',
    ['custom_kernel.cpp', 'custom_kernel.cu'],
    verbose=True
)

# Use custom kernel
result = custom_ops.my_function(input_tensor)
```

## Multi-GPU Training

### DataParallel (Simple Multi-GPU)

```python
# Basic DataParallel (not recommended for multiple nodes)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model = model.cuda()

# Training remains the same
for data, target in dataloader:
    data, target = data.cuda(), target.cuda()
    output = model(data)  # Automatically distributed
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### DistributedDataParallel (Recommended)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_ddp(rank, world_size):
    setup_ddp(rank, world_size)
    
    # Create model and move to GPU
    model = MyModel().cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        
        for data, target in dataloader:
            data, target = data.cuda(rank), target.cuda(rank)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Launch with torchrun; --nproc_per_node must equal the number of GPUs you were
# allocated (gpuq sets CUDA_VISIBLE_DEVICES for you) and match the --gpus value you
# passed to `gpuq submit`. Max 4 on this server.
# torchrun --nproc_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .) train_script.py
# (or just set it to your --gpus value, e.g. --nproc_per_node=4 for the full node)
```

### GPU Queue Multi-GPU Job

```bash
# Submit a multi-GPU job. gpuq runs it in the FOREGROUND in this terminal until it
# finishes; there is no daemon and no per-job log files — redirect output yourself if
# you want a log (e.g. append `> run.log 2>&1`).
gpuq submit \
  --command "torchrun --nproc_per_node=2 train_distributed.py" \
  --gpus 2 \
  --memory 60 \
  --time 12

# To use the whole node (all 4 H200s):
gpuq submit \
  --command "torchrun --nproc_per_node=4 train_distributed.py" \
  --gpus 4 \
  --memory 60 \
  --time 12
```

> `-m/--memory` is the **minimum free VRAM** (in GB) a GPU must have to be selected for your
> job — an admission requirement, not a hard cap your job is held to. gpuq sets
> `CUDA_VISIBLE_DEVICES` to the GPUs it picks, so keep `--nproc_per_node` equal to your `--gpus`
> count. You own the GPUs you are allocated: you may stack more of your own jobs on them, but GPUs
> held by other users are off-limits until they free them.

## Large Model Training

### Memory-Efficient Large Models

```python
# Large model with memory optimizations
class LargeModel(nn.Module):
    def __init__(self, vocab_size=50000, hidden_dim=4096, num_layers=24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Use ModuleList for checkpointing
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights efficiently
        self.apply(self._init_weights)
    
    def forward(self, x):
        x = self.embedding(x)
        
        # Gradient checkpointing for memory efficiency
        for layer in self.layers:
            x = checkpoint.checkpoint(layer, x, use_reentrant=False)
        
        return self.output(x)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

### DeepSpeed Integration

```python
# DeepSpeed configuration file (deepspeed_config.json)
deepspeed_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 2,
    "bf16": {
        "enabled": True  # BF16 preferred on Hopper (no loss scaling needed)
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO Stage 2 for memory efficiency
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True
    }
}

# DeepSpeed training
import deepspeed

def train_with_deepspeed():
    model = LargeModel()
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config="deepspeed_config.json"
    )
    
    for batch in dataloader:
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()
```

### Hugging Face Transformers with H200

```python
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer
)

# Load a model. With ~141 GB per H200, a single card holds DialoGPT-large easily —
# no sharding needed. (device_map="auto" requires the `accelerate` package and shards
# the model across GPUs for inference; it is generally not used with the Trainer for training.)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-large",
    torch_dtype=torch.bfloat16,  # BF16 preferred on Hopper
)

# Enable gradient checkpointing on the model (it is NOT a from_pretrained kwarg)...
model.gradient_checkpointing_enable()

# Training arguments optimized for H200
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    bf16=True,                     # BF16 on Hopper
    gradient_checkpointing=True,   # ...or enable it here instead
    eval_strategy="steps",         # NOTE: evaluation_strategy is deprecated/removed
    dataloader_pin_memory=True,
    dataloader_num_workers=8,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

> Set the Hugging Face cache via `HF_HOME` (the old `TRANSFORMERS_CACHE` is deprecated), e.g.
> `export HF_HOME=/path/to/large/scratch/hf_cache` so model downloads land on a roomy filesystem.

## Advanced Techniques

### Memory Mapping for Large Datasets

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MemoryMappedDataset(Dataset):
    def __init__(self, data_file, feature_size, mmap_mode='r'):
        # Memory-map large datasets
        self.feature_size = feature_size
        self.data = np.memmap(data_file, dtype='float32', mode=mmap_mode)
        self.length = len(self.data) // self.feature_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.feature_size
        end_idx = start_idx + self.feature_size
        return torch.from_numpy(self.data[start_idx:end_idx].copy())
    
    def __len__(self):
        return self.length
```

### Dynamic Loss Scaling

```python
class DynamicLossScaler:
    def __init__(self, init_scale=65536, scale_factor=2.0, scale_window=2000):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.counter = 0
        self.last_overflow_iter = 0
    
    def scale_loss(self, loss):
        return loss * self.scale
    
    def update_scale(self, overflow):
        if overflow:
            self.last_overflow_iter = self.counter
            self.scale = max(self.scale / self.scale_factor, 1.0)
        else:
            if (self.counter - self.last_overflow_iter) % self.scale_window == 0:
                self.scale *= self.scale_factor
        self.counter += 1
```

> In practice, prefer the built-in `torch.amp.GradScaler('cuda')` for FP16 — and remember that
> BF16 (the recommended dtype on Hopper) does not need loss scaling at all. The class above is
> illustrative.

### Custom Optimizers

```python
class AdamWWithWarmup(optim.AdamW):
    def __init__(self, params, lr=1e-3, warmup_steps=1000, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.base_lr = lr
    
    def step(self, closure=None):
        self.step_count += 1
        
        # Linear warmup
        if self.step_count <= self.warmup_steps:
            lr_scale = self.step_count / self.warmup_steps
            for param_group in self.param_groups:
                param_group['lr'] = self.base_lr * lr_scale
        
        return super().step(closure)
```

## Debugging and Profiling

### Memory Profiling

```python
import torch.profiler

def profile_memory_usage():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Your training code here
        for batch in dataloader:
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Save profiling results
    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

### GPU Utilization Monitoring

```python
import threading
import time
import pynvml as nvml  # pip install nvidia-ml-py (the official binding)

def monitor_gpu_utilization():
    nvml.nvmlInit()
    
    def monitor():
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        
        while training:
            # Get utilization
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_util = util.gpu
            mem_used = mem_info.used / 1024**3
            mem_total = mem_info.total / 1024**3
            
            print(f"GPU Util: {gpu_util}%, Memory: {mem_used:.1f}/{mem_total:.1f}GB")
            time.sleep(5)
    
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
```

> Use the official `nvidia-ml-py` package (imported as `pynvml`); the old `nvidia_ml_py3` fork is
> unmaintained and may be out of sync with the installed driver (575.x).

### Debugging CUDA Errors

```python
# Enable CUDA error checking
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Detect anomalies
torch.autograd.set_detect_anomaly(True)

try:
    # Training code
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
except RuntimeError as e:
    print(f"CUDA error: {e}")
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.1f}GB")
    
    # Print memory summary
    print(torch.cuda.memory_summary())
```

## Example Scripts

### Complete Training Script

```python
#!/usr/bin/env python3
"""
H200-Optimized PyTorch Training Script
Usage: gpuq submit --command "python train_h200.py --config config.yaml" --gpus 1 --memory 60
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import argparse
import yaml
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_model(config):
    """Create model based on config"""
    if config['model']['type'] == 'resnet':
        from torchvision.models import resnet50
        model = resnet50(num_classes=config['model']['num_classes'])
    else:
        # Custom model
        model = CustomModel(**config['model']['params'])
    
    return model.cuda()

def create_dataloader(config):
    """Create optimized dataloader"""
    # Your dataset creation here
    dataset = YourDataset(**config['dataset'])
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

def train_epoch(model, dataloader, optimizer, criterion, scaler, config):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=config['training']['mixed_precision']):
            output = model(data)
            loss = criterion(output, target)
        
        if config['training']['mixed_precision']:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % config['logging']['log_interval'] == 0:
            logging.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Memory usage
            if batch_idx % (config['logging']['log_interval'] * 10) == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                logging.info(f'Memory allocated: {allocated:.2f}GB')
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--resume', help='Resume from checkpoint')
    args = parser.parse_args()
    
    setup_logging()
    config = load_config(args.config)
    
    # Set random seeds
    torch.manual_seed(config['training']['seed'])
    torch.cuda.manual_seed_all(config['training']['seed'])
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    
    # Create model, optimizer, criterion
    model = create_model(config)
    optimizer = optim.AdamW(model.parameters(), **config['optimizer'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=config['training']['mixed_precision'])
    
    # Create dataloader
    train_loader = create_dataloader(config)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        logging.info(f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, config)
        logging.info(f'Epoch {epoch+1} completed, Average loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            checkpoint_path = Path(config['checkpoint']['dir']) / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            logging.info(f'Checkpoint saved to {checkpoint_path}')

if __name__ == '__main__':
    main()
```

### Configuration File (config.yaml)

```yaml
# H200 Training Configuration
model:
  type: "custom"  # or "resnet", "transformer", etc.
  num_classes: 1000
  params:
    hidden_dim: 4096
    num_layers: 12

dataset:
  root: "/path/to/dataset"
  transform: "standard"
  
training:
  batch_size: 128  # Optimize for H200 memory (~141 GB per card)
  epochs: 100
  seed: 42
  mixed_precision: true
  
optimizer:
  lr: 0.001
  weight_decay: 0.01
  
dataloader:
  num_workers: 8  # conservative; the box has 256 logical CPUs shared among users
  
checkpoint:
  dir: "./checkpoints"
  save_interval: 10
  
logging:
  log_interval: 100
```

### Job Submission Script

```bash
#!/bin/bash
# submit_job.sh - Submit H200 optimized PyTorch job

# Configuration
SCRIPT_PATH="train_h200.py"
CONFIG_PATH="config.yaml"
GPUS=1
MEMORY=60  # GB minimum free VRAM a candidate GPU must have
TIME=12    # hours

# Submit the job. gpuq runs it in the FOREGROUND in this terminal until it finishes;
# there is no daemon and no per-job log files — redirect output yourself if you want a log.
# gpuq notifies you on completion at the email read from your account; --notify only overrides it.
gpuq submit \
  --command "python $SCRIPT_PATH --config $CONFIG_PATH" \
  --gpus $GPUS \
  --memory $MEMORY \
  --time $TIME \
  --notify "your-email@example.com"

echo "Job finished! Check current state any time with: gpuq status"
```

---

**Next Steps**:
- For TensorFlow usage: [TensorFlow with H200 Guide](tensorflow-guide.md)
- For multi-framework comparisons: [Best Practices Guide](best-practices.md)
- For ready-to-use examples: [Example Scripts](../examples/)
