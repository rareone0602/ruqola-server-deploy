# NVIDIA H200 GPU Specifications and Capabilities

Technical specifications and optimization guidelines for the NVIDIA H200 NVL Tensor Core GPUs in the Ruqola server.

## 📖 Table of Contents

1. [Host and Software Summary](#host-and-software-summary)
2. [Hardware Specifications](#hardware-specifications)
3. [Memory Architecture](#memory-architecture)
4. [Compute Capabilities](#compute-capabilities)
5. [Performance Characteristics](#performance-characteristics)
6. [Optimization Guidelines](#optimization-guidelines)
7. [Comparison with Other GPUs](#comparison-with-other-gpus)
8. [Best Use Cases](#best-use-cases)

## Host and Software Summary

A quick reference for the live Ruqola server (host `wsserver1`, the NTU "Mjolnir" machine). These are the authoritative numbers — always prefer a live `nvidia-smi` over any datasheet figure if they ever disagree.

| Item | Value |
|------|-------|
| **GPUs** | 4 × NVIDIA H200 NVL (indices 0, 1, 2, 3) |
| **Per-GPU VRAM** | ~141 GB (nvidia-smi reports 143,771 MiB ≈ 140 GiB) |
| **Total GPU VRAM** | ~564 GB across the 4 cards |
| **GPU driver** | 575.57.08 |
| **CUDA (driver) version** | 12.9 |
| **Logical CPUs** | 256 |
| **System RAM** | 755 GiB |
| **Operating system** | Ubuntu 24.04.4 LTS |

> A 4th GPU was added on 2025-06-10. The server now has **exactly 4 GPUs**; older docs that say "3" are out of date. When you want to use every card, address indices `0,1,2,3` (e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3`, `torchrun --nproc_per_node=4`, `gpuq submit -g 4`).

## Hardware Specifications

### NVIDIA H200 NVL Overview

Our server is equipped with **4x NVIDIA H200 NVL** GPUs (compute capability 9.0, Hopper) with the following per-GPU specifications:

| Specification | Value |
|---------------|-------|
| **GPU Architecture** | Hopper (GH100 die) |
| **Process Node** | TSMC 4N (4nm) |
| **Transistors** | 80 billion |
| **SM (Streaming Multiprocessors)** | 132 (full GH100) |
| **CUDA Cores (FP32)** | 16,896 |
| **Tensor Cores** | 4th Gen (528 units) |
| **Base Clock** | ~1,365 MHz (NVL datasheet) |
| **Max SM (Boost) Clock** | ~1,785 MHz (live `clocks.max.sm`) |

> Datacenter Hopper GPUs (H100/H200) ship **without RT cores** — graphics ray-tracing units are not present on these compute parts. Any spec sheet that lists "RT Cores" for an H200 is describing a different (consumer) product.

### Memory Specifications

| Memory Feature | H200 NVL |
|----------------|----------|
| **Memory Type** | HBM3e |
| **Memory Capacity** | ~141 GB (143,771 MiB) |
| **Memory Bandwidth** | 4,800 GB/s |
| **Memory Bus Width** | 5,120-bit |
| **L2 Cache** | 50 MB |
| **Max Memory Clock** | 3,201 MHz (live `clocks.max.memory`) |

### Power and Thermal

| Specification | Value |
|---------------|-------|
| **Power Cap (max limit)** | 600W (live `power.max_limit`) |
| **Form Factor** | PCIe (NVL, dual-slot) |
| **Cooling** | Air-cooled in this NVL configuration |
| **Operating Temperature** | 0°C to 35°C ambient |

## Memory Architecture

### HBM3e Memory System

The H200's HBM3e memory system provides exceptional bandwidth and capacity:

```
┌─────────────────────────────────────┐
│        GPU Die (GH100 / Hopper)     │
├─────────────────────────────────────┤
│  L2 Cache: 50 MB (Shared)          │
├─────────────────────────────────────┤
│  HBM3e Memory: ~141 GB             │
│  Bandwidth: 4,800 GB/s             │
│  5,120-bit Memory Interface        │
└─────────────────────────────────────┘
```

### Memory Hierarchy

1. **Registers** (per thread): ~255 registers × 32-bit
2. **Shared Memory** (per SM): 228 KB configurable
3. **L1 Cache** (per SM): 256 KB
4. **L2 Cache** (global): 50 MB
5. **HBM3e** (global): ~141 GB at 4,800 GB/s

### Memory Bandwidth Utilization

```python
# Theoretical peak memory bandwidth test
import torch
import time

def memory_bandwidth_test(size_gb=10):
    device = torch.device('cuda')
    
    # Create tensors
    size = int(size_gb * 1024**3 / 4)  # float32 = 4 bytes
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    # Warmup
    for _ in range(10):
        c = a + b
    
    torch.cuda.synchronize()
    start = time.time()
    
    # Memory bandwidth test
    for _ in range(100):
        c = a + b  # Read 2 tensors, write 1 tensor
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Calculate bandwidth (3 * size_gb * 100 operations / elapsed time)
    bandwidth = (3 * size_gb * 100) / elapsed
    print(f"Memory bandwidth: {bandwidth:.1f} GB/s")
    print(f"Utilization: {bandwidth/4800*100:.1f}% of peak")

memory_bandwidth_test()
```

## Compute Capabilities

### CUDA Compute Capability

The H200 supports **CUDA Compute Capability 9.0 (Hopper)**:

```bash
# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Output: 9.0 (one line per GPU; 4 lines on this server)
```

### Tensor Core Capabilities

#### 4th Generation Tensor Cores (per-SM throughput)

The table below shows relative per-SM throughput by data type. These are **per-SM** figures; multiply by the SM count and clock for whole-GPU peaks (see the aggregate numbers below).

| Data Type | Matrix Size | Relative Per-SM Throughput |
|-----------|-------------|----------------------------|
| **FP16** | 16×16×16 | 256 TOPS |
| **BF16** | 16×16×16 | 256 TOPS |
| **TF32** | 16×16×16 | 128 TOPS |
| **FP8** | 16×16×16 | 512 TOPS |
| **INT8** | 16×16×16 | 512 TOPS |
| **INT4** | 16×16×16 | 1024 TOPS |

#### Aggregate Tensor-Core Peaks (whole GPU)

These are the headline numbers you should quote when comparing throughput:

| Metric | Value |
|--------|-------|
| **BF16/FP16 dense (datasheet)** | ~989 TFLOP/s |
| **BF16/FP16 with sparsity (datasheet)** | ~1,979 TFLOP/s |
| **BF16 dense (measured on this server)** | ~836 TFLOP/s |

> The empirical ~836 TFLOP/s is a realistic, sustained BF16 dense matmul number for these cards — somewhat below the datasheet peak, as is normal for real kernels. Use it when sizing expectations. Note that **134 TFLOPS is a CUDA-core (non-tensor) figure**, not the tensor-core throughput — don't quote it as the H200's headline FP16/BF16 number.

#### Tensor Core Usage in Deep Learning

```python
# PyTorch automatic mixed precision with H200
import torch
from torch.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())
scaler = GradScaler('cuda')

for data, target in dataloader:
    optimizer.zero_grad()
    
    # Use autocast for forward pass
    with autocast('cuda'):
        output = model(data)
        loss = criterion(output, target)
    
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### CUDA Cores Performance (non-tensor)

These are the **CUDA-core** (non-tensor) rates. Do not confuse them with the tensor-core peaks above — for deep learning you almost always want the tensor cores.

- **Single Precision (FP32)**: 67 TFLOPS
- **Half Precision (FP16, CUDA cores)**: ~67 TFLOPS (Hopper non-tensor FP16 runs at the same rate as FP32)
- **Double Precision (FP64)**: 34 TFLOPS

## Performance Characteristics

### Memory Bandwidth Scaling

```python
# Test memory bandwidth with different data sizes
import torch
import numpy as np
import matplotlib.pyplot as plt

def bandwidth_vs_size():
    sizes = [1, 2, 4, 8, 16, 32, 64, 100]  # GB
    bandwidths = []
    
    for size_gb in sizes:
        # Run bandwidth test
        bandwidth = memory_bandwidth_test(size_gb)
        bandwidths.append(bandwidth)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, bandwidths, 'b-o')
    plt.axhline(y=4800, color='r', linestyle='--', label='Peak Bandwidth')
    plt.xlabel('Data Size (GB)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('H200 Memory Bandwidth vs Data Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('h200_bandwidth.png')
```

### Compute Performance

#### Matrix Multiplication Performance

```python
# GEMM performance test
import torch
import time

def gemm_performance_test():
    device = torch.device('cuda')
    sizes = [1024, 2048, 4096, 8192, 16384]
    
    for size in sizes:
        a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
        b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
        
        # Warmup
        for _ in range(10):
            c = torch.mm(a, b)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            c = torch.mm(a, b)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate TFLOPS
        ops = 2 * size**3 * 100  # multiply-accumulate operations
        tflops = ops / elapsed / 1e12
        
        print(f"Matrix size {size}x{size}: {tflops:.2f} TFLOPS")
        # Large sizes should approach ~836 TFLOP/s (BF16 dense) on these cards

gemm_performance_test()
```

## Optimization Guidelines

### Memory Optimization

1. **Maximize Memory Utilization**:
   ```python
   # Use most of one card's ~141 GB, leaving headroom for fragmentation/activations
   batch_size = calculate_max_batch_size(model, input_size, memory_limit=130)  # ~11 GB buffer
   ```

2. **Memory-Efficient Training**:
   ```python
   # Gradient checkpointing for large models
   model = torch.utils.checkpoint.checkpoint_sequential(model, segments=4)
   
   # Gradient accumulation for large effective batch sizes
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### Compute Optimization

1. **Use Tensor Cores**:
   ```python
   # Enable automatic mixed precision (preferred over .half())
   with torch.autocast('cuda'):
       output = model(input)
   ```

2. **Optimize Tensor Shapes**:
   ```python
   # Ensure tensor dimensions are multiples of 8 for optimal Tensor Core usage
   batch_size = 64  # Multiple of 8
   hidden_dim = 4096  # Multiple of 8
   seq_length = 2048  # Multiple of 8
   ```

### Data Loading Optimization

```python
# Optimized data loading for H200
# The host has 256 logical CPUs, so num_workers can be generous;
# pick a value that suits your per-job CPU share, not the whole machine.
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,  # Tune up if you have spare CPU; the host has 256 logical cores
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Reduce worker restart overhead
    prefetch_factor=2,  # Prefetch batches
)
```

## Comparison with Other GPUs

### Performance Comparison

| GPU Model | Memory | Memory BW | FP16 Tensor TFLOPS (dense) | Architecture |
|-----------|--------|-----------|----------------------------|--------------|
| **H200 NVL** | **~141 GB** | **4,800 GB/s** | **~836 (measured) / ~989 (datasheet)** | **Hopper** |
| H100 SXM | 80 GB | 3,350 GB/s | ~989 | Hopper |
| A100 SXM | 80 GB | 2,039 GB/s | ~312 | Ampere |
| V100 SXM | 32 GB | 900 GB/s | ~125 | Volta |
| RTX 4090 | 24 GB | 1,008 GB/s | ~165 | Ada Lovelace |

> **Footnote:** The H200 and H100 share the same Hopper compute die, so their FP16/BF16 tensor-core throughput is essentially the same (~989 TFLOP/s dense). The H200's real advantage over the H100 is **memory** (~141 GB vs 80 GB) and **bandwidth** (4.8 vs 3.35 TB/s), not raw FP16 throughput. (FP16/BF16 dense peaks shown above; add sparsity for roughly 2× on Hopper/Ampere.)

### Memory Capacity Advantages

```python
# Models that benefit from H200's large memory.
# Sizes are approximate fp16 weight footprints; TRAINING needs much more
# (optimizer state + activations), so treat these as inference/loading guides.
models_by_memory = {
    "GPT-3 175B": "~350 GB",          # Needs multi-GPU model parallelism (use all 4 cards)
    "LLaMA 65B": "~130 GB fp16",      # Tight on one ~141 GB card for inference; no room to train
    "Stable Diffusion XL": "~12 GB",  # Much headroom for batch size
    "BERT Large": "~1.3 GB",          # Can run huge batch sizes
}
```

## Best Use Cases

### Ideal Workloads for H200

1. **Large Language Models**:
   ```python
   # A 70B model in fp16 is ~140 GB of weights ALONE, which does not leave
   # room on one ~141 GB card for activations/optimizer state. For ~70B:
   #   - inference: shard across multiple cards, or use 4-bit/8-bit quantization
   #   - training/finetuning: multi-GPU (this server has 4 cards) and/or quantization+offload
   from transformers import AutoModelForCausalLM

   # Multi-GPU inference example (shards across all 4 cards):
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-2-70b-hf",
       torch_dtype=torch.float16,
       device_map="auto",  # spreads layers across GPUs 0,1,2,3
   )
   ```

2. **Computer Vision with Large Images**:
   ```python
   # Process high-resolution images with large batch sizes
   batch_size = 128  # Much larger than possible on smaller GPUs
   image_size = 1024  # Higher resolution training
   ```

3. **Scientific Computing**:
   ```python
   # Large-scale numerical simulations (size grids to fit one card's ~141 GB,
   # or shard across the 4 GPUs for bigger problems)
   simulation_grid = torch.zeros(4096, 4096, 1024, device='cuda')
   ```

4. **Multi-Modal Models**:
   ```python
   # Train large vision-language models
   model = VisionLanguageModel(
       vision_dim=2048,
       text_dim=4096,
       hidden_dim=8192,  # Large hidden dimensions
       num_layers=48
   )
   ```

### Optimization Strategies by Use Case

#### Large Language Models
```python
# Memory-efficient LLM training
from transformers import AutoModelForCausalLM, TrainingArguments
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_cache=False,  # Save memory during training
)
# Enable gradient checkpointing on the model (not a from_pretrained kwarg):
model.gradient_checkpointing_enable()
# (Equivalently, TrainingArguments(gradient_checkpointing=True).)

# Use DeepSpeed ZeRO and/or multiple GPUs for even larger models
from deepspeed import initialize
model_engine, optimizer, _, _ = initialize(
    model=model,
    config="deepspeed_config.json"
)
```

#### Computer Vision
```python
# High-throughput image processing
def create_optimized_dataloader(dataset, batch_size=256):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,  # Large batch size utilizing full memory
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3
    )

# Mixed precision training for CNNs
with autocast('cuda'):
    output = model(images)
    loss = criterion(output, labels)

scaler.scale(loss).backward()
```

#### Scientific Computing
```python
# Large tensor operations
def scientific_simulation():
    # Large 3D simulation grids
    grid = torch.zeros(2048, 2048, 2048, device='cuda', dtype=torch.float32)
    
    # Physics-informed neural networks with large domains
    coordinates = torch.rand(1000000, 3, device='cuda')  # 1M sample points
    solution = physics_model(coordinates)
```

### Performance Monitoring

```python
# Monitor H200 utilization
import torch
import pynvml as nvml

def monitor_gpu_usage(index=0):
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(index)  # 0..3 on this server
    
    # Memory usage
    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
    memory_used = mem_info.used / 1024**3  # GiB
    memory_total = mem_info.total / 1024**3  # GiB
    
    # Utilization
    util = nvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu
    mem_util = util.memory
    
    # Temperature
    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
    
    print(f"Memory: {memory_used:.1f}/{memory_total:.1f} GiB ({memory_used/memory_total*100:.1f}%)")
    print(f"GPU Utilization: {gpu_util}%")
    print(f"Memory Utilization: {mem_util}%")
    print(f"Temperature: {temp}°C")

# Run periodically during training
import threading
import time

def monitoring_thread():
    while training:
        monitor_gpu_usage()
        time.sleep(10)

monitor = threading.Thread(target=monitoring_thread)
monitor.start()
```

---

**Next Steps**: 
- For PyTorch-specific optimizations: [PyTorch with H200 Guide](pytorch-guide.md)
- For TensorFlow optimizations: [TensorFlow with H200 Guide](tensorflow-guide.md)
- For general best practices: [Best Practices Guide](best-practices.md)
